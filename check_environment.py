"""Check local EviHGT dependencies and hardware without loading data or models.

Exit codes: 0 = checks passed, 1 = missing/incompatible/unverified requirement,
2 = diagnostic/report failure. CUDA checks target the default BF16 recipe.
Only a uniquely named JSON diagnostic report is written; nothing is installed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import traceback
import uuid


ROOT = Path(__file__).resolve().parent
IMPORT_NAMES = {"PyYAML": "yaml"}
GIB = 1024 ** 3


def requirements(path):
    """Read the release's simple package/version requirements using stdlib."""
    result = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)((?:(?:>=|<=|==|!=|>|<)\d+(?:\.\d+)*(?:,|$))+)", line)
        if match is None:
            raise ValueError("Unsupported requirement expression: " + line)
        result.append(match.groups())
    if not result:
        raise ValueError("requirements.txt contains no required packages")
    return result


def satisfies(version, constraints):
    """Compare stable numeric releases; leave preview versions unverified.

    Local build suffixes such as +cu132 do not change the release ordering.
    This deliberately handles the expressions used by requirements.txt only.
    """
    release = version.split("+", 1)[0]
    if re.fullmatch(r"\d+(?:\.\d+)*", release) is None:
        return None
    actual = tuple(map(int, release.split(".")))
    for clause in constraints.split(","):
        match = re.fullmatch(r"(>=|<=|==|!=|>|<)(\d+(?:\.\d+)*)", clause)
        if match is None:
            return None
        operation, value = match.groups()
        expected = tuple(map(int, value.split(".")))
        width = max(len(actual), len(expected))
        left = actual + (0,) * (width - len(actual))
        right = expected + (0,) * (width - len(expected))
        accepted = {">=": left >= right, "<=": left <= right, "==": left == right,
                    "!=": left != right, ">": left > right, "<": left < right}[operation]
        if not accepted:
            return False
    return True


def physical_memory():
    """Read physical RAM counters without an optional monitoring dependency."""
    if os.name == "nt":
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32),
                        ("total_physical", ctypes.c_uint64), ("available_physical", ctypes.c_uint64),
                        ("total_pagefile", ctypes.c_uint64), ("available_pagefile", ctypes.c_uint64),
                        ("total_virtual", ctypes.c_uint64), ("available_virtual", ctypes.c_uint64),
                        ("available_extended_virtual", ctypes.c_uint64)]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return {"total_bytes": status.total_physical, "available_bytes": status.available_physical}
    page_size = os.sysconf("SC_PAGE_SIZE")
    total = page_size * os.sysconf("SC_PHYS_PAGES")
    try:
        available = page_size * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError):
        available = None
    return {"total_bytes": total, "available_bytes": available}


def collect(device):
    report = {"schema": "EVIHGT_ENVIRONMENT_CHECK_V1", "time": datetime.now(timezone.utc).isoformat(),
              "event": "environment_check_completed", "requested_device": device,
              "scope": "ENVIRONMENT_ONLY_NO_DATA_OR_MODEL_EXECUTION", "checks": [], "errors": [],
              "python": sys.version, "platform": platform.platform(), "gpu": [],
              "limits": "Capability and import checks only; no model, CUDA compute-kernel, or dataset validation."}
    def add(name, status, detail):
        report["checks"].append({"name": name, "status": status, "detail": detail})
    def error(name, exc):
        report["errors"].append({"check": name, "type": type(exc).__name__, "message": str(exc),
                                 "traceback": traceback.format_exc()})
    add("Python", "PASS" if sys.version_info >= (3, 11) else "FAIL",
        platform.python_version() + "; requires >=3.11")
    torch_module = None
    try:
        required = requirements(ROOT / "requirements.txt")
    except Exception as exc:
        error("requirements", exc)
        add("Requirements", "FAIL", "Cannot read supported requirements.txt: " + str(exc))
        required = []
    for package, bounds in required:
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            add(package, "FAIL", "Not installed; requires " + bounds)
            continue
        except Exception as exc:
            error(package, exc)
            add(package, "FAIL", "Package metadata unavailable: " + type(exc).__name__)
            continue
        compatible = satisfies(version, bounds)
        status = "PASS" if compatible is True else "FAIL" if compatible is False else "UNVERIFIED"
        detail = version + "; requires " + bounds
        try:
            module = importlib.import_module(IMPORT_NAMES.get(package, package.replace("-", "_")))
            detail += "; import OK"
            if package == "torch":
                torch_module = module
        except Exception as exc:
            error(package, exc)
            status, detail = "FAIL", detail + "; import failed: " + type(exc).__name__
        add(package, status, detail)
    add("CPU", "INFO", (platform.processor() or "Processor name unavailable")
        + "; logical CPUs=" + str(os.cpu_count()))
    try:
        memory = physical_memory()
        report["memory"] = memory
        detail = "{:.2f} GiB physical".format(memory["total_bytes"] / GIB)
        if memory["available_bytes"] is not None:
            detail += "; {:.2f} GiB available".format(memory["available_bytes"] / GIB)
        add("RAM", "INFO", detail)
    except Exception as exc:
        error("RAM", exc)
        add("RAM", "UNAVAILABLE", "OS memory counters unavailable: " + type(exc).__name__)
    if torch_module is None:
        add("CUDA", "UNAVAILABLE", "Cannot inspect CUDA because torch is unavailable")
        add("BF16", "UNAVAILABLE", "Cannot inspect BF16 because torch is unavailable")
    else:
        try:
            cuda = torch_module.cuda
            available = bool(cuda.is_available())
            report["cuda_build"] = torch_module.version.cuda
            add("CUDA", "PASS" if available else "FAIL" if device == "cuda" else "INFO",
                "available={}; PyTorch CUDA build={}".format(available, torch_module.version.cuda))
            if available:
                selected = int(cuda.current_device())
                report["selected_gpu"] = selected
                for index in range(cuda.device_count()):
                    properties = cuda.get_device_properties(index)
                    gpu = {"index": index, "name": properties.name,
                           "total_bytes": int(properties.total_memory),
                           "compute_capability": [properties.major, properties.minor], "free_bytes": None}
                    try:
                        free, _ = cuda.mem_get_info(index)
                        gpu["free_bytes"] = int(free)
                    except Exception as exc:
                        error("GPU memory " + str(index), exc)
                    report["gpu"].append(gpu)
                    free_text = "N/A" if gpu["free_bytes"] is None else "{:.2f} GiB".format(gpu["free_bytes"] / GIB)
                    add("GPU " + str(index), "INFO", "{}; {:.2f} GiB total; {} free; capability {}.{}{}".format(
                        properties.name, properties.total_memory / GIB, free_text, properties.major, properties.minor,
                        "; selected" if index == selected else ""))
                bf16 = bool(cuda.is_bf16_supported())
                report["bf16_supported"] = bf16
                add("BF16", "PASS" if bf16 else "FAIL" if device == "cuda" else "INFO",
                    "supported={}; {}".format(bf16, "required by the default CUDA recipe" if device == "cuda" else "not required for CPU/FP32"))
            else:
                add("BF16", "UNAVAILABLE" if device == "cuda" else "INFO",
                    "CUDA is unavailable" if device == "cuda" else "Not required for CPU/FP32")
        except Exception as exc:
            error("CUDA inspection", exc)
            add("CUDA inspection", "FAIL" if device == "cuda" else "UNAVAILABLE",
                "Hardware inspection failed: " + type(exc).__name__)
    report["exit_code"] = 1 if any(row["status"] in ("FAIL", "UNVERIFIED") for row in report["checks"]) else 0
    report["status"] = "NOT_READY" if report["exit_code"] else "READY"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda",
                        help="Target recipe: cuda requires BF16; cpu uses FP32 (default: cuda)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/environment_checks",
                        help="Directory for a new, uniquely named JSON report")
    args = parser.parse_args(argv)
    print("EviHGT | Environment check ({})".format(args.device), flush=True)
    try:
        report = collect(args.device)
    except Exception as exc:
        report = {"schema": "EVIHGT_ENVIRONMENT_CHECK_V1", "time": datetime.now(timezone.utc).isoformat(),
                  "event": "environment_check_failed", "requested_device": args.device,
                  "scope": "ENVIRONMENT_ONLY_NO_DATA_OR_MODEL_EXECUTION", "status": "DIAGNOSTIC_FAILED",
                  "exit_code": 2, "checks": [{"name": "Diagnostic", "status": "FAIL", "detail": type(exc).__name__}],
                  "errors": [{"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}]}
    print("{:<18} {:<12} {}".format("Check", "Status", "Details"))
    print("-" * 90)
    for row in report["checks"]:
        print("{:<18} {:<12} {}".format(row["name"], row["status"], row["detail"]))
    try:
        directory = args.output_dir.expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        name = "check_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8] + ".json"
        path = directory / name
        with path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=True, allow_nan=False, indent=2)
            handle.write("\n")
        print("JSON report: " + str(path))
    except Exception as exc:
        print("[Error] Cannot save the environment report: {}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return 2
    print("Result: {} (exit code {}).".format(report["status"], report["exit_code"]))
    if report["exit_code"]:
        print("Review failed checks and the JSON report. Install the required packages in this Python environment.")
    print("No datasets or models were loaded; no packages were installed or downloaded.")
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
