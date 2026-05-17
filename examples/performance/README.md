# Performance

Examples that focus on JIT/vectorization performance:
- `benchmark_jit_matvec.py` — benchmark a JIT-compiled matvec.
- `benchmark_sharded_solve_scaling.py` — benchmark single-case CPU/GPU sharded RHSMode=1 solves. For current hot-solve scaling checks, use `--inner-warmup-solves 1 --sample-timeout-s 300 --rhs1-precond theta_schwarz --schwarz-coarse-levels 2`; add `--deterministic-output-probe` to record a bounded 1-vs-N residual/digest gate.
- `benchmark_multi_gpu_case_throughput.py` — benchmark the production GPU-throughput lane: one GPU per case, comparing sequential 1-GPU execution against two concurrent 1-GPU runs on a 2-GPU node.
- `benchmark_transport_parallel_scaling.py` — benchmark transport-worker scaling on CPU or GPU (`--backend cpu|gpu`). The current publication-grade multi-GPU result uses `--backend gpu` on `transport_parallel_2min.input.namelist`.
- `benchmark_transport_l11_vs_fortran.py` — reproduce the 2x2 L11 parity/runtime figure used in the README/docs.
- `benchmark_optional_lineax_implicit_solve.py` — optional Lineax gate for differentiable linear solves; it compares against the in-tree `custom_linear_solve` path on a synthetic nonsymmetric system, a tiny real SFINCS implicit-diff operator, and a repeated-RHS reuse case, and it skips cleanly when `lineax` is not installed.
- `benchmark_structured_solve.py` — bounded factor-once / repeated-RHS block-tridiagonal benchmark. It supports both deterministic synthetic systems and `--case sfincs-pas-block`, which extracts a real local PAS block from a SFINCS fixture. Use this as the admission gate before wiring structured velocity-space solves into production preconditioner or transport paths.
- `profile_transport_compile_runtime_cache.py` — profile transport-solve compile/runtime split with persistent JAX cache.
- `profile_reduced_examples.py` — batch profile the reduced upstream suite (runtime + memory summaries).
