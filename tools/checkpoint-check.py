#!/usr/bin/env python3
"""Is this checkpoint eligible for vLLM's Marlin path, and what will it cost?

Marlin is the kernel that executes a 4-bit weight-only GEMM (see the module
docstring in vllm/model_executor/kernels/linear/mixed_precision/marlin.py). A
checkpoint only reaches it if its quantization configuration is one of the shapes
the compiled kernel has a scalar type for, and getting that wrong costs a 20 GB
download and a boot that dies with a message most people read as a model problem.

The rules below come from the installed vLLM, not from memory:

  marlin_utils.py: res = [scalar_types.uint4b8, scalar_types.uint8b128]
                   quant_type not in supported_types -> "Marlin does not support
                   weight_bits = ...". So 4 and 8 bits only, whatever the
                   compressed-tensors schema accepts (it accepts 2..7).
  marlin_utils.py: MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]
  marlin_utils_fp4.py: FP4_MARLIN_SUPPORTED_GROUP_SIZES = [16]
  compressed_tensors_wNa16.py logs "Using MarlinLinearKernel for
                   CompressedTensorsWNA16" for the W4A16/W8A16 schemes.
  kernels/linear/mixed_precision/marlin.py get_min_capability -> 75 (sm_75+).

Run with no arguments to check every checkpoint under models/. The pool and
utilization numbers are projections from two measured points on this box, not
guarantees: the script prints the fit it uses so the arithmetic can be checked.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(REPO, "models")

# --- the kernel's rules (see the module docstring for where each comes from) ---
MARLIN_WEIGHT_BITS = {4: "uint4b8", 8: "uint8b128"}
MARLIN_GROUP_SIZES = {-1, 32, 64, 128}
FP4_GROUP_SIZES = {16}
MIN_CAPABILITY = 75
# Quant methods vLLM routes to a Marlin kernel (mixed_precision/marlin.py plus the
# nvfp4/mxfp4 variants).
MARLIN_METHODS = {"compressed-tensors", "gptq", "awq", "modelopt", "mxfp4"}

# --- cost model, fitted on this box (all of it measured, 2026-09-15, KVarN) ----
#   W4A16  util 0.92  resident weights 14.27 GiB -> 807,951 tokens of pool
#   NVFP4  util 0.85  resident weights 17.63 GiB -> 397,128 tokens of pool
# Solving both for pool bytes per token and for the non-pool overhead gives:
POOL_BYTES_PER_TOKEN = 14541   # effective, batch mode, KVarN, ~262144 ceiling
NON_POOL_MIB = 3812            # CUDA graphs + activation peak + non-torch
TOTAL_CUDA_MIB = 32202         # what the CUDA context sees on the RTX 5090
# Resident weights track the checkpoint size closely enough to plan with: measured
# 14.27 GiB for a 16.7 GB checkpoint and 17.63 GiB for a 20.6 GB one, i.e. ~0.92 of
# the file size in GiB. The repack into Marlin's layout is what makes loading take
# seconds and is already inside these numbers.
RESIDENT_FRACTION = 0.92
# The fp4 path pads and swizzles the scales at load, which needs transient room the
# fp16 paths do not: NVFP4 fails to boot at 0.92 here and comes up at 0.85, while
# the projection for the same weights asks for ~0.80.
FP4_UTIL_MARGIN = 0.05
TARGETS = [65536, 131072, 262144]


def sh(cmd: list[str], timeout: int = 60) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def capability() -> int | None:
    """Compute capability of the installed card, as a single integer (120 for 12.0)."""
    out = sh(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    if not out:
        return None
    try:
        major, _, minor = out.splitlines()[0].strip().partition(".")
        return int(major) * 10 + int(minor or 0)
    except ValueError:
        return None


def marlin_rules_from_vllm() -> str | None:
    """Read the supported bits and group sizes out of the installed vLLM.

    The image is the authority for these, so when it can be asked, ask it: a vLLM
    bump that adds a scalar type would otherwise leave this script lying.
    """
    image = "ghcr.io/syv-ai/qwen38-27b-rtx3090:latest"
    # device_capability has to be passed in: without it the query asks the running
    # GPU, and a probe container has none, so the list comes back empty.
    cap = capability() or 120
    probe = f"""
from vllm.model_executor.layers.quantization.utils import marlin_utils as m
from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4 as f
import vllm, json
print(json.dumps({{'vllm': vllm.__version__,
 'types': [str(t) for t in m.query_marlin_supported_quant_types(False, False, {cap})],
 'groups': m.MARLIN_SUPPORTED_GROUP_SIZES,
 'fp4_groups': f.FP4_MARLIN_SUPPORTED_GROUP_SIZES}}))
"""
    try:
        r = subprocess.run(["docker", "run", "--rm", "-i", "--entrypoint", "bash", image,
                            "-c", "venv/bin/python -"], input=probe, capture_output=True,
                           text=True, timeout=240)
        out = r.stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            return (f"vLLM {d['vllm']}: types {d['types']} | groups {d['groups']} | "
                    f"fp4 groups {d['fp4_groups']}")
    return None


def shard_bytes(path: str) -> tuple[int, int]:
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    return sum(os.path.getsize(f) for f in files), len(files)


def groups_of(cfg: dict) -> list[dict]:
    """Every quantization group: what it targets, at how many bits, with what group size."""
    q = cfg.get("quantization_config") or {}
    out = []
    for name, g in (q.get("config_groups") or {}).items():
        w = g.get("weights") or {}
        a = g.get("input_activations")
        targets = g.get("targets") or []
        out.append({
            "name": name,
            "bits": w.get("num_bits"),
            "type": w.get("type"),
            "group_size": w.get("group_size"),
            "sym": w.get("symmetric"),
            "activations": (a or {}).get("num_bits") if a else None,
            "scheme": g.get("scheme"),
            "target_count": len(targets) if isinstance(targets, list) else None,
            "targets_sample": targets[:2] if isinstance(targets, list) else [],
        })
    if not out:
        # Single-format configs put the numbers at the top level (gptq, awq).
        w = q
        out.append({
            "name": "(top level)", "bits": w.get("bits"), "type": w.get("type"),
            "group_size": w.get("group_size"), "sym": w.get("sym"),
            "activations": None, "scheme": q.get("format"),
            "target_count": None, "targets_sample": [],
        })
    return out


def classify(cfg: dict) -> tuple[bool, list[str], list[str]]:
    """Returns (eligible, findings, blockers)."""
    q = cfg.get("quantization_config") or {}
    method = (q.get("quant_method") or "").lower()
    fmt = (q.get("format") or "").lower()
    findings, blockers = [], []
    if not method:
        blockers.append("no quantization_config: this is not a quantized checkpoint, "
                        "so there is no Marlin path to take (it would load in bf16)")
        return False, findings, blockers
    if method not in MARLIN_METHODS:
        blockers.append(f"quant_method '{method}' is not one vLLM routes to Marlin "
                        f"(known: {sorted(MARLIN_METHODS)})")
    is_fp4 = "nvfp4" in fmt or "mxfp4" in fmt or "fp4" in str(fmt)
    if is_fp4:
        findings.append("fp4: this takes the nvfp4 Marlin kernel (W4A16_NVFP4), whose "
                        "group size must be 16 and which needs extra transient room at load")
    for g in groups_of(cfg):
        bits, gs = g["bits"], g["group_size"]
        # Targets here are patterns or type names, not a layer list: the shipped
        # checkpoint says {Linear} for the 4-bit group and {*lm_head$}, {*embed_tokens$},
        # {*} for the 8-bit ones, which is the head requantization the prepare scripts do.
        who = f" for {g['targets_sample'][0]}" if g["targets_sample"] else ""
        if g["target_count"] and g["target_count"] > 1:
            who += f" +{g['target_count'] - 1} more"
        label = f"{g['name']}: {bits}-bit group {gs}{who}"
        if bits is None:
            findings.append(f"{label} -> skipped (no num_bits in the group)")
            continue
        if not is_fp4:
            if bits not in MARLIN_WEIGHT_BITS:
                blockers.append(f"{label} -> NO KERNEL: Marlin's scalar types are "
                                f"{sorted(MARLIN_WEIGHT_BITS)} only. vLLM will refuse with "
                                f"\"Marlin does not support weight_bits = {bits}\"")
                continue
            ok_groups = MARLIN_GROUP_SIZES
        else:
            if bits != 4:
                blockers.append(f"{label} -> NO KERNEL: the fp4 Marlin path is 4-bit only")
                continue
            ok_groups = FP4_GROUP_SIZES
        if gs is not None and gs not in ok_groups:
            blockers.append(f"{label} -> NO KERNEL: group sizes allowed are {sorted(ok_groups)}"
                            + (" (use -1 for per-channel)" if -1 in ok_groups else ""))
            continue
        kind = "fp4 Marlin" if is_fp4 else MARLIN_WEIGHT_BITS[bits]
        findings.append(f"{label} -> Marlin OK ({kind})"
                        + ("  [int8 activations available]" if g["activations"] else ""))
    ignored = (q.get("ignore") or [])
    if ignored:
        findings.append(f"kept unquantized (ignore list): {len(ignored)} entries, "
                        f"e.g. {ignored[:3]}")
    return (not blockers), findings, blockers


def project(size_bytes: int, util: float, is_fp4: bool) -> dict:
    """Pool and ceiling projection from the measured fit."""
    weights_mib = size_bytes / 2**20 * RESIDENT_FRACTION
    pool_mib = util * TOTAL_CUDA_MIB - weights_mib - NON_POOL_MIB
    tokens = max(0, int(pool_mib * 2**20 / POOL_BYTES_PER_TOKEN))
    need_util = (weights_mib + NON_POOL_MIB + 262144 * POOL_BYTES_PER_TOKEN / 2**20) / TOTAL_CUDA_MIB
    if is_fp4:
        need_util += FP4_UTIL_MARGIN
    out = {"weights_gib": weights_mib / 1024, "pool_mib": pool_mib, "tokens": tokens,
           "min_util_for_262144": need_util, "rows": []}
    for t in TARGETS:
        need = t * POOL_BYTES_PER_TOKEN / 2**20
        out["rows"].append((t, need, need <= pool_mib))
    return out


def serve_flags(path: str, util: float, is_fp4: bool) -> list[str]:
    flags = []
    if is_fp4:
        flags.append(f"--gpu-util {util:.2f}   (the fp4 path needs a lower ceiling)")
    idx = glob.glob(os.path.join(path, "model.safetensors.index.json"))
    missing = []
    if idx:
        keys = set(json.load(open(idx[0]))["weight_map"])
        if "lm_head.weight_packed" not in keys:
            missing.append("lm_head")
        if not any(k.endswith("embed_tokens.weight_packed") for k in keys):
            missing.append("embed_tokens")
    if missing:
        flags.append(f"--skip-verify   (verify.sh fails on {'/'.join(missing)} not "
                     f"requantized: run prepare/quant_{'lm_head' if 'lm_head' in missing else 'embed'}.py "
                     f"to get those back)")
    return flags


def check(path: str, rules: str | None, cap: int | None) -> tuple[bool, list[str]]:
    name = os.path.basename(path.rstrip("/"))
    cfg_path = os.path.join(path, "config.json")
    if not os.path.exists(cfg_path):
        return False, [f"{name}: no config.json"]
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    size, shards = shard_bytes(path)
    q = cfg.get("quantization_config") or {}
    is_fp4 = "fp4" in (q.get("format") or "").lower()
    eligible, findings, blockers = classify(cfg)
    cap_ok = cap is None or cap >= MIN_CAPABILITY

    lines = [f"checkpoint   {name}",
             f"size         {size / 1e9:.1f} GB in {shards} shards",
             f"method       {q.get('quant_method') or '-'} / {q.get('format') or '-'}"]
    lines += [f"             {f}" for f in findings]
    lines.append(f"hardware     sm_{cap or '?'} vs sm_{MIN_CAPABILITY} minimum -> "
                 f"{'OK' if cap_ok else 'TOO OLD'}")
    if rules:
        lines.append(f"kernel rules {rules}")
    if not cap_ok:
        eligible = False
        blockers.append(f"compute capability {cap} is below Marlin's minimum {MIN_CAPABILITY}")

    if eligible:
        for util in (0.92, 0.85, 0.70):
            p = project(size, util, is_fp4)
            fits = [f"{t // 1024}k" for t, need, ok in p["rows"] if ok]
            lines.append(f"projection   util {util:.2f}: pool ~{p['tokens']:,} tokens "
                         f"({p['tokens'] / 262144:.2f}x at 262144); ceilings that fit: "
                         f"{' '.join(fits) or 'none'}")
        p = project(size, 0.92, is_fp4)
        lines.append(f"             min util for a 262144-token request: "
                     f"{p['min_util_for_262144']:.3f}"
                     + ("  (includes the fp4 load margin)" if is_fp4 else ""))
        flags = serve_flags(path, 0.85 if is_fp4 else 0.92, is_fp4)
        lines.append(f"serve        --mode batch --kv kvarn  {(' '.join(flags)).strip()}")
    for b in blockers:
        lines.append(f"BLOCKER      {b}")
    lines.append(f"verdict      {'ELIGIBLE for Marlin' if eligible else 'NOT ELIGIBLE'}")
    # Projections are estimates from a two-point fit; say so rather than letting
    # them read as measurements.
    lines.append("note         pool/util numbers are projections fitted on two measured "
                 "checkpoints on this box; the eligibility rules are the kernel's own")
    return eligible, lines


def self_test() -> int:
    """Prove the classification on synthetic configs, including the negative case."""
    def cfg(bits, group, fmt="pack-quantized", method="compressed-tensors", act=None):
        g = {"targets": ["Linear"], "weights": {"num_bits": bits, "group_size": group,
                                                "type": "float" if "fp4" in fmt else "int",
                                                "symmetric": True}}
        if act:
            g["input_activations"] = {"num_bits": act, "dynamic": "local", "group_size": group}
        return {"quantization_config": {"quant_method": method, "format": fmt,
                                        "config_groups": {"group_0": g}}}
    cases = [
        ("4-bit group 128 (the shipped W4A16)", cfg(4, 128), True),
        ("8-bit group 128 (W8A16)", cfg(8, 128), True),
        ("4-bit group 32", cfg(4, 32), True),
        ("6-bit group 64 (the INT6-Mixed shape)", cfg(6, 64), False),
        ("7-bit group 64", cfg(7, 64), False),
        ("5-bit group 128", cfg(5, 128), False),
        ("nvfp4 4-bit group 16", cfg(4, 16, "nvfp4-pack-quantized", "compressed-tensors", 4), True),
        ("nvfp4 4-bit group 128 (wrong group)",
         cfg(4, 128, "nvfp4-pack-quantized", "compressed-tensors", 4), False),
        ("gptq 4-bit group 128", cfg(4, 128, "gptq", "gptq"), True),
        ("awq 4-bit group 128", cfg(4, 128, "awq", "awq"), True),
        ("unquantized bf16", {"quantization_config": {}}, False),
    ]
    bad = 0
    for label, c, want in cases:
        got, _, blockers = classify(c)
        mark = "ok  " if got == want else "FAIL"
        if got != want:
            bad += 1
        why = "" if got == want else f" (expected {want}, got {got}: {blockers[:1]})"
        print(f"  {mark} {label:42s} eligible={got}{why}")
    print(f"self-test: {len(cases) - bad}/{len(cases)} correct")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", help="checkpoint dirs (default: every models/*)")
    ap.add_argument("--self-test", action="store_true", help="check the rules on synthetic configs")
    ap.add_argument("--no-probe", action="store_true", help="skip reading the rules from the image")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    paths = a.paths or sorted(d for d in glob.glob(os.path.join(MODELS, "*")) if os.path.isdir(d))
    if not paths:
        print(f"nothing to check under {MODELS}")
        return 1
    cap = capability()
    rules = None if a.no_probe else marlin_rules_from_vllm()
    if rules is None and not a.no_probe:
        print("(could not read the kernel rules from the image; using the constants baked in)\n")
    rc = 0
    for p in paths:
        ok, lines = check(p, rules, cap)
        print("\n".join(lines) + "\n")
        if not ok:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
