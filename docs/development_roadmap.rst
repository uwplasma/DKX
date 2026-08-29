Development Roadmap
===================

The authoritative DKX 3 roadmap is the repository-root ``plan.md``. It records
the accepted product and scientific contracts, implementation phases,
performance and validation gates, pull-request sequence, and execution ledger.
This documentation page is only a navigation pointer; it does not define a
second roadmap.

The previous ``plan_final.md`` from merged pull request #8 is historical. Its
durable requirements are reconciled into ``plan.md``: coherent vertical
slices, one canonical owner per behavior, reusable solver state,
bounded-memory elimination, measured CPU/GPU scaling, and scientific admission
gates. Old branch names and campaign checklists are not current direction.

Machine-readable planning evidence lives under ``validation/``:

- ``capabilities.toml`` records capability status and evidence gaps;
- ``baseline.toml`` pins the DKX 2.3.1 starting tree, package sizes, CI timing,
  source/API inventory, and known drift from the originally audited commit;
- ``hardware.toml`` identifies measured hardware and unavailable lanes;
- ``benchmark_schema.toml`` defines the minimum record for later performance
  comparisons.

Until the planning pull request merges, runtime, API, solver, and physics
changes remain out of scope. The first implementation phase after merge is P1:
clean wheel/source-distribution tests and enforceable size contracts.
