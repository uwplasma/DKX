Development Roadmap
===================

The authoritative DKX 3 roadmap is the repository-root ``plan.md``. It records
the accepted product and scientific contracts, the research-grade verification
and validation standard, the target public API and source architecture, the
ordered implementation phases, the pull-request sequence, and the definition of
done. This documentation page is only a navigation pointer; it does not define
a second roadmap.

Earlier roadmaps -- ``plan_final.md`` from merged pull request #8 and the
2200-line execution diary that ``plan.md`` replaced on 2026-08-30 -- are
historical. Their durable requirements are carried forward: coherent vertical
slices, one canonical owner per behavior, reusable solver state,
bounded-memory elimination, measured CPU/GPU scaling, and explicit scientific
admission gates. Old branch names, phase letters, and campaign checklists are
not current direction.

Machine-readable planning evidence lives under ``validation/``:

- ``capabilities.toml`` records capability status and evidence gaps;
- ``baseline.toml`` pins the audited tree, package sizes, CI state, coverage,
  source and public-API inventory, and the known local test gaps;
- ``hardware.toml`` identifies measured hardware and unavailable lanes;
- ``benchmark_schema.toml`` defines the minimum record for later performance
  comparisons.

The current phase is Phase A of ``plan.md``: replace the plan and freeze an
exact current-state inventory without changing runtime, API, or physics. The
next phase is Phase B, which consolidates the validation registry, the audit
runners, and the test tree before any new resolution campaign starts.
