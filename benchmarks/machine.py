"""Hardware fingerprint and run conditions of this Mac, for benchmark records.

"Apple M4" alone does not pin down the hardware: each chip name ships in
several binned CPU/GPU core counts, the chassis decides sustained thermals
(a fanless MacBook Air throttles where a MacBook Pro does not), and every
macOS release brings a new Metal shader compiler. The record therefore keeps
all of these. Only stock macOS tools and the Foundation framework (through
ctypes) are used, so this adds no dependencies; serial numbers and hardware
UUIDs are deliberately not recorded.
"""

import ctypes
import json
import os
import re
import subprocess
from typing import Any

THERMAL_STATES = ("nominal", "fair", "serious", "critical")  # NSProcessInfo order
LIBOBJC = "/usr/lib/libobjc.A.dylib"
FOUNDATION = "/System/Library/Frameworks/Foundation.framework/Foundation"


def _run(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def _sysctl(name: str) -> str:
    return _run("sysctl", "-n", name).strip()


def machine() -> dict[str, Any]:
    """Hardware and OS of this Mac.

    Returns
    -------
    dict[str, Any]
        Model name and identifier (e.g. "MacBook Pro", "Mac16,1"), chip,
        CPU core counts (total, performance, efficiency), GPU core count,
        memory [GB], and the macOS version and build.
    """
    prof = json.loads(
        _run("system_profiler", "-json", "SPHardwareDataType", "SPDisplaysDataType")
    )
    hw = prof["SPHardwareDataType"][0]
    gpu = next(g for g in prof["SPDisplaysDataType"] if "sppci_cores" in g)
    return {
        "model_name": hw["machine_name"],
        "model_identifier": _sysctl("hw.model"),
        "chip": hw["chip_type"],
        "cpu_cores": int(_sysctl("hw.physicalcpu")),
        "cpu_performance_cores": int(_sysctl("hw.perflevel0.physicalcpu")),
        "cpu_efficiency_cores": int(_sysctl("hw.perflevel1.physicalcpu")),
        "gpu_cores": int(gpu["sppci_cores"]),
        "memory_gb": int(_sysctl("hw.memsize")) // 2**30,
        "macos": _run("sw_vers", "-productVersion").strip(),
        "macos_build": _run("sw_vers", "-buildVersion").strip(),
    }


def _process_info(selector: bytes, restype: type) -> Any:
    """Send a no-argument message to [NSProcessInfo processInfo]."""
    objc = ctypes.cdll.LoadLibrary(LIBOBJC)
    ctypes.cdll.LoadLibrary(FOUNDATION)  # registers NSProcessInfo
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]

    def send(obj: int, sel: bytes, res: type) -> Any:
        proto = ctypes.CFUNCTYPE(res, ctypes.c_void_p, ctypes.c_void_p)
        return ctypes.cast(objc.objc_msgSend, proto)(obj, objc.sel_registerName(sel))

    info = send(objc.objc_getClass(b"NSProcessInfo"), b"processInfo", ctypes.c_void_p)
    return send(info, selector, restype)


def conditions() -> dict[str, Any]:
    """Run conditions that move wall times: power, thermals, competing load.

    Returns
    -------
    dict[str, Any]
        ``power_source`` ("AC Power" or "Battery Power"), ``low_power_mode``
        (bool), macOS's system-wide ``thermal_state`` (one of
        THERMAL_STATES), and the 1-minute ``load_average`` (runnable
        processes). The thermal state is coarse: it stays "nominal" through
        the clock-down and the background load that move CPU timings by
        10-15%, which the load average does catch.
    """
    batt = _run("pmset", "-g", "batt")
    power = re.search(r"'([^']+)'", batt)
    return {
        "power_source": power.group(1) if power else "unknown",
        "low_power_mode": bool(_process_info(b"isLowPowerModeEnabled", ctypes.c_bool)),
        "thermal_state": THERMAL_STATES[_process_info(b"thermalState", ctypes.c_long)],
        "load_average": os.getloadavg()[0],
    }


def label(m: dict[str, Any]) -> str:
    """Human-readable hardware label, e.g. for a plot legend.

    Example: "MacBook Pro (Mac16,1) · Apple M4 · CPU 10 (4P+6E) ·
    GPU 10-core · 16 GB · macOS 26.6.2".
    """
    return (
        f"{m['model_name']} ({m['model_identifier']}) · {m['chip']} · "
        f"CPU {m['cpu_cores']} ({m['cpu_performance_cores']}P+"
        f"{m['cpu_efficiency_cores']}E) · GPU {m['gpu_cores']}-core · "
        f"{m['memory_gb']} GB · macOS {m['macos']}"
    )


def slug(m: dict[str, Any]) -> str:
    """Filesystem-safe hardware tag for run folder names, e.g. "Mac16-1_M4-10gpu"."""
    chip = m["chip"].removeprefix("Apple ").replace(" ", "")
    return f"{m['model_identifier'].replace(',', '-')}_{chip}-{m['gpu_cores']}gpu"


def hardware_key(m: dict[str, Any]) -> tuple[str | int, ...]:
    """The fields that identify a machine's hardware (macOS excluded)."""
    return (
        m["model_identifier"],
        m["chip"],
        m["cpu_performance_cores"],
        m["cpu_efficiency_cores"],
        m["gpu_cores"],
        m["memory_gb"],
    )
