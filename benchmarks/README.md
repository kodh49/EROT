# Benchmarks and profiling

Run the planned dense benchmark matrix on the target GPU:

```bash
python benchmarks/benchmark_solvers.py \
  --device cuda:0 \
  --dtype float64 \
  --output benchmark-results/cuda-float64.json
```

Pass `--profile-dir benchmark-results/trace` to capture a JAX profiler trace.
The JSON report separates first compile-and-run time from warm execution and
includes device memory statistics when the backend exposes them.

To enforce the 10% steady-state regression budget on the same reference
machine, retain an accepted JSON report and run:

```bash
python benchmarks/benchmark_solvers.py \
  --device cuda:0 \
  --baseline benchmark-results/accepted.json \
  --output benchmark-results/candidate.json
```

Only consider a custom Pallas/CuTe/C++ kernel after the trace identifies an
unsupported operation consuming at least 30% of total runtime. Keep it only if
it improves full-solve warm time by at least 1.5x or peak memory by at least
25% on the reference workload.
