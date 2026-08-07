"""Reproducible compile/warm benchmarks for EROT's stable solvers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

import erot


def _memory_stats() -> dict[str, int] | None:
    stats = jax.devices()[0].memory_stats()
    if not stats:
        return None
    return {
        key: int(value)
        for key, value in stats.items()
        if key in {"bytes_in_use", "peak_bytes_in_use", "bytes_limit"}
    }


def _time_call(callable_: object) -> tuple[erot.SolveResult, float]:
    start = time.perf_counter()
    result = callable_()  # type: ignore[operator]
    result.coupling.block_until_ready()
    return result, time.perf_counter() - start


def benchmark_classical(n: int, dtype: str, device: str) -> dict[str, object]:
    x = np.linspace(-4.0, 4.0, n, dtype=dtype)
    cost = (x[:, None] - x[None, :]) ** 2
    a = np.exp(-0.5 * (x + 1.0) ** 2)
    b = np.exp(-0.5 * (x - 1.0) ** 2)
    a, b = a / a.sum(), b / b.sum()
    config = erot.SolverConfig(
        epsilon=0.5,
        tolerance=1e-8 if dtype == "float64" else 1e-5,
        max_iterations=20_000,
        dtype=dtype,  # type: ignore[arg-type]
        device=device,
    )

    def run() -> erot.SolveResult:
        return erot.solve(
            cost,
            [a, b],
            problem="classical",
            regularizer="shannon",
            method="sinkhorn",
            config=config,
        )

    compiled, compile_and_run = _time_call(run)
    warm, warm_seconds = _time_call(run)
    return {
        "problem": "classical-shannon",
        "size": n,
        "compile_and_run_seconds": compile_and_run,
        "warm_seconds": warm_seconds,
        "iterations": warm.iterations,
        "error": warm.error,
        "converged": warm.converged,
        "first_iterations": compiled.iterations,
    }


def benchmark_quantum(n: int, dtype: str, device: str) -> dict[str, object]:
    real_dtype = np.float64 if dtype == "float64" else np.float32
    marginal = np.eye(n, dtype=real_dtype) / n
    generator = np.random.default_rng(1729 + n)
    raw_cost = generator.normal(size=(n * n, n * n)).astype(real_dtype)
    cost = (raw_cost + raw_cost.T) / (2 * np.sqrt(n * n))
    config = erot.SolverConfig(
        epsilon=1.0,
        tolerance=1e-8 if dtype == "float64" else 1e-5,
        max_iterations=20_000,
        dtype=dtype,  # type: ignore[arg-type]
        device=device,
    )

    def run() -> erot.SolveResult:
        return erot.solve(
            cost,
            [marginal, marginal],
            problem="quantum",
            regularizer="quadratic",
            method="cyclic",
            config=config,
        )

    compiled, compile_and_run = _time_call(run)
    warm, warm_seconds = _time_call(run)
    return {
        "problem": "quantum-quadratic",
        "size": n,
        "compile_and_run_seconds": compile_and_run,
        "warm_seconds": warm_seconds,
        "iterations": warm.iterations,
        "error": warm.error,
        "converged": warm.converged,
        "first_iterations": compiled.iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--classical-sizes", nargs="*", type=int, default=[100, 1000, 5000]
    )
    parser.add_argument("--quantum-sizes", nargs="*", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--max-regression", type=float, default=0.10)
    args = parser.parse_args()

    def run_all() -> list[dict[str, object]]:
        records = [
            benchmark_classical(size, args.dtype, args.device)
            for size in args.classical_sizes
        ]
        records.extend(
            benchmark_quantum(size, args.dtype, args.device)
            for size in args.quantum_sizes
        )
        return records

    if args.profile_dir:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        with jax.profiler.trace(str(args.profile_dir)):
            records = run_all()
    else:
        records = run_all()
    report = {
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "dtype": args.dtype,
        "memory": _memory_stats(),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        baseline_times = {
            (record["problem"], record["size"]): record["warm_seconds"]
            for record in baseline["records"]
        }
        regressions = []
        for record in records:
            key = (record["problem"], record["size"])
            if key not in baseline_times:
                continue
            ratio = record["warm_seconds"] / baseline_times[key] - 1
            if ratio > args.max_regression:
                regressions.append((key, ratio))
        if regressions:
            details = ", ".join(
                f"{problem}[{size}] +{ratio:.1%}"
                for (problem, size), ratio in regressions
            )
            raise SystemExit(f"performance regression exceeds limit: {details}")


if __name__ == "__main__":
    main()
