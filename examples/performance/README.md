# Performance

Examples that focus on JIT/vectorization performance:
- `benchmark_jit_matvec.py` — benchmark a JIT-compiled matvec.
- `benchmark_sharded_solve_scaling.py` — benchmark single-case CPU/GPU sharded RHSMode=1 solves. For the current publication-style CPU benchmark, use `--rhs1-precond theta_schwarz --schwarz-coarse-levels 2`.
- `benchmark_multi_gpu_case_throughput.py` — benchmark the production GPU-throughput lane: one GPU per case, comparing sequential 1-GPU execution against two concurrent 1-GPU runs on a 2-GPU node.
- `benchmark_transport_parallel_scaling.py` — benchmark process-parallel transport workers.
- `benchmark_transport_l11_vs_fortran.py` — reproduce the 2x2 L11 parity/runtime figure used in the README/docs.
- `profile_transport_compile_runtime_cache.py` — profile transport-solve compile/runtime split with persistent JAX cache.
- `profile_reduced_examples.py` — batch profile the reduced upstream suite (runtime + memory summaries).
