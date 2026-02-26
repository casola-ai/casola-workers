"""
Pre-startup GPU health checks for Casola GPU workers.

Validates GPU hardware via nvidia-smi before launching the inference engine.
Catches bad hosts early (wrong GPU, insufficient VRAM, competing workloads,
ECC errors, power throttling) so the scheduler can block them immediately.

Usage:
    from casola_worker.gpu_health import run_gpu_health_checks

    result = run_gpu_health_checks(
        expected_gpu_name="RTX 4090",
        expected_vram_gb=24.0,
    )
    if not result.passed:
        print(result.error_message)
"""

import logging
import shutil
import subprocess
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    passed: bool = True
    checks_run: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def error_message(self) -> str:
        if self.passed:
            return ""
        details = "; ".join(self.failures)
        return f"gpu_health_check failed: {details}"


def run_gpu_health_checks(
    expected_gpu_name: str | None = None,
    expected_vram_gb: float | None = None,
) -> HealthCheckResult:
    """Run GPU health checks via nvidia-smi.

    Args:
        expected_gpu_name: Expected GPU name substring (case-insensitive match).
        expected_vram_gb: Expected total VRAM in GB (actual must meet or exceed).

    Returns:
        HealthCheckResult with pass/fail status and error details.
    """
    result = HealthCheckResult()

    if not shutil.which("nvidia-smi"):
        log.warning("nvidia-smi not found, skipping GPU health checks")
        return result

    fields = [
        "gpu_name",
        "memory.total",
        "memory.used",
        "utilization.gpu",
        "ecc.errors.uncorrected.aggregate.total",
        "power.limit",
        "power.default_limit",
    ]

    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        result.passed = False
        result.failures.append("nvidia-smi timed out after 15s")
        return result
    except Exception as e:
        result.passed = False
        result.failures.append(f"nvidia-smi execution error: {e}")
        return result

    if output.returncode != 0:
        result.passed = False
        stderr = output.stderr.strip()[:200]
        result.failures.append(f"nvidia-smi exited with code {output.returncode}: {stderr}")
        return result

    # Parse CSV output — one row per GPU, check each GPU
    for line in output.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(fields):
            result.passed = False
            result.failures.append(f"unexpected nvidia-smi output: {line[:200]}")
            continue

        gpu_name = parts[0]
        mem_total_mib = _parse_float(parts[1])
        mem_used_mib = _parse_float(parts[2])
        gpu_util_pct = _parse_float(parts[3])
        ecc_errors = parts[4].strip()
        power_limit = _parse_float(parts[5])
        power_default = _parse_float(parts[6])

        # Check GPU name
        if expected_gpu_name:
            result.checks_run += 1
            if expected_gpu_name.lower() not in gpu_name.lower():
                result.passed = False
                result.failures.append(
                    f"GPU name mismatch: expected '{expected_gpu_name}', got '{gpu_name}'"
                )

        # Check total VRAM
        if expected_vram_gb is not None and mem_total_mib is not None:
            result.checks_run += 1
            actual_vram_gb = mem_total_mib / 1024.0
            if actual_vram_gb < expected_vram_gb * 0.9:  # 10% tolerance
                result.passed = False
                result.failures.append(
                    f"VRAM too low: expected >={expected_vram_gb:.1f} GB, got {actual_vram_gb:.1f} GB"
                )

        # Check memory usage (competing workload)
        if mem_used_mib is not None:
            result.checks_run += 1
            if mem_used_mib >= 500:
                result.passed = False
                result.failures.append(
                    f"GPU memory already in use: {mem_used_mib:.0f} MiB (limit: 500 MiB)"
                )

        # Check GPU utilization (competing workload)
        if gpu_util_pct is not None:
            result.checks_run += 1
            if gpu_util_pct >= 10:
                result.passed = False
                result.failures.append(
                    f"GPU utilization too high: {gpu_util_pct:.0f}% (limit: 10%)"
                )

        # Check ECC errors
        if ecc_errors.strip().upper() not in ("N/A", "[N/A]", ""):
            ecc_count = _parse_float(ecc_errors)
            if ecc_count is not None:
                result.checks_run += 1
                if ecc_count > 0:
                    result.passed = False
                    result.failures.append(f"uncorrected ECC errors detected: {int(ecc_count)}")

        # Check power limit vs default
        if power_limit is not None and power_default is not None and power_default > 0:
            result.checks_run += 1
            ratio = power_limit / power_default
            if ratio < 0.7:
                result.passed = False
                result.failures.append(
                    f"power limit too low: {power_limit:.0f}W vs default {power_default:.0f}W ({ratio:.0%})"
                )

    return result


def _parse_float(value: str) -> float | None:
    """Parse a float from nvidia-smi output, returning None for N/A or invalid values."""
    v = value.strip()
    if v.upper() in ("N/A", "[N/A]", ""):
        return None
    try:
        return float(v)
    except ValueError:
        return None
