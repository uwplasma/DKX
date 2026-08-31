Development Roadmap
===================

The authoritative DKX 3 roadmap is the repository-root ``plan.md``. It records
the accepted product and scientific contracts, the research-grade verification
and validation standard, the target public API and source architecture, the
ordered implementation phases, the pull-request sequence, and the definition of
done. This page is a pointer to that file, not a second roadmap.

Earlier roadmaps -- ``plan_final.md`` from merged pull request #8 and the
2200-line execution diary that ``plan.md`` replaced on 2026-08-30 -- are
historical. Their durable requirements are carried forward: coherent vertical
slices, one canonical owner per behavior, reusable solver state,
bounded-memory elimination, measured CPU/GPU scaling, and explicit scientific
admission criteria. Old branch names, phase letters, and campaign checklists do
not set direction.

Machine-readable planning and validation evidence lives under ``validation/``:

- ``registry.toml`` is the single index of registered evidence artifacts: one
  entry per artifact naming its capability, status, claim, inputs, generating
  command, checksum, and the limits of what it establishes. One runner,
  ``python -m tools.release.registry``, checks them all, and
  ``tests/test_validation.py`` is the one test module that consumes it;
- ``capabilities.toml`` records capability status and evidence gaps;
- ``baseline.toml`` pins the audited tree, package sizes, CI state, coverage,
  source and public-API inventory, and the known local test gaps;
- ``hardware.toml`` identifies measured hardware and the measurement work that
  no available host can run;
- ``benchmark_schema.toml`` defines the minimum record for later performance
  comparisons.

Phase A of ``plan.md`` is complete: the plan is replaced and the audited-state
inventory is frozen in ``baseline.toml``. Phase B is in progress. Its first
slice added the registry and its runner and replaced nineteen per-campaign test
modules with one. Its remaining slice moves the dated campaign directories and
raw suite outputs to release assets and reduces the summary and audit-script
counts themselves. No new resolution campaign starts before that lands.
