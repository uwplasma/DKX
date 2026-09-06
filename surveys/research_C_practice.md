# Research C — Practice: methodology survey for DKX development process and README

Date: 2026-09-06. Scope: (a) development process, (b) README, for DKX (uwplasma/DKX), a JAX/GPU
differentiable neoclassical transport code targeting a JCP/CPC methods paper and a physics paper.

Problems this survey addresses:
- 17-PR dependency-stacked chain (+5,930/-2,889 lines in one day) that nobody can review.
- Plan grew to 16,000 words with an execution diary; replaced by a 5,800-word "focused" plan (PR #189) that still reads as a review report.
- README cut to 158 lines (30 Aug) -> 250 lines / 1,500 words of hedged prose in three days -> now proposed at 140 lines.
- Heavy investment in evidence/provenance infrastructure relative to scientific results.

Citation rule: every claim marked [Fetched] was read from the URL given; [Seen] = seen verbatim in a
search-result snippet; UNVERIFIED = from memory, not confirmed this session.

## Contents
1. V&V practice for scientific codes and code-paper structure
2. Benchmark reporting
3. Reproducibility standards for a small academic team
4. Workflow efficiency: PR size, stacking, trunk-based dev, CI tiers, planning docs
5. README design for scientific software (five READMEs analysed) + DKX proposal
6. How differentiable-physics projects present differentiability
7. Process rules for DKX
8. README blueprint for DKX
9. References

---

## 1. V&V practice for scientific codes and what a code paper looks like

### 1.1 The V&V vocabulary (Roache; Oberkampf & Roy; ASME V&V 20)

- Three tiers, kept separate in the literature: **code verification** (does the code solve the
  equations it claims to solve — exact/analytic solutions, method of manufactured solutions (MMS),
  observed order of accuracy), **solution verification** (how large is the numerical error of one
  particular solution — grid/resolution convergence, Richardson extrapolation, Roache's Grid
  Convergence Index), and **validation** (comparison with experiment, with uncertainty on both
  sides). Cross-code comparison sits between the first two: it is evidence of code verification
  only when the codes are independent discretizations of the same equations.
- Roache, "Code Verification by the Method of Manufactured Solutions", ASME J. Fluids Eng. 124(1),
  4-10 (2002). [Seen: ASME Digital Collection listing; abstract text: MMS "provides a
  straightforward and quite general procedure for generating benchmark solutions".]
- Roache's GCI (1994, 1997) is "a method for uniform reporting of grid convergence tests" and was
  "accepted as a publication standard (not required) by the ASME Journal of Fluids Engineering".
  [Seen: search snippets from C. J. Roy's JCP review "Review of Code and Solution Verification
  Procedures for Computational Simulation" and Roache 1998 book listing.]
- Oberkampf & Roy, *Verification and Validation in Scientific Computing*, Cambridge UP (2010),
  ISBN 9780521113601: "comprehensive and systematic development of the basic concepts, principles,
  and procedures for verification and validation of models and simulations". [Seen: publisher
  and bookseller listings; the chapter-level hierarchy is UNVERIFIED this session.]
- ASME V&V 20-2009 "Standard for Verification and Validation in Computational Fluid Dynamics and
  Heat Transfer": "describes the application of the internationally accepted approach for
  experimental uncertainty to total Validation uncertainty". [Seen: search snippet.] Relevance to
  DKX: it is the template for saying "validated against W7-X E_r data" honestly — a validation
  claim needs an uncertainty on the experimental side, not just a plotted overlay.

**What this means for a neoclassical code.** Analytic limits exist and should be the first tier:
Spitzer parallel conductivity (D33 normalized), tokamak banana/plateau/Pfirsch-Schlüter
coefficients, Onsager symmetry of the transport matrix (MONKES devotes an appendix to Onsager
symmetry as a check [Fetched: arXiv 2312.12248 outline]), and for a differentiable code the
adjoint/AD gradient against central finite differences. Cross-code parity with SFINCS (same
equations and discretization) is regression evidence, not independent verification; MONKES and
yancc are independent discretizations and therefore count as code verification at the ~1% level.

### 1.2 How existing codes structure verification (what could be confirmed)

- **SFINCS**: primary reference is Landreman, Smith, Mollén & Helander, Phys. Plasmas 21, 042503
  (2014), "Comparison of particle trajectories and collision operators for collisional transport
  in nonaxisymmetric plasmas" — the code paper is framed as a physics comparison (trajectory
  models, collision operators) and benchmarks against DKES. [Seen: github.com/landreman/sfincs
  README snippet and AIP listing.]
- **DESC**: repository ships `tests/` with separate UnitTests and RegressionTests CI badges and a
  `publications/` directory that "contains PDFs of publications by the DESC group, as well as
  scripts and data to reproduce the results of these papers". [Fetched: DESC README.rst.]
  Part I paper: "34 pages, 23 figures, 2 tables"; abstract claims force-balance accuracy vs VMEC,
  "an order of magnitude less time", exponential convergence with the Fourier-Zernike basis.
  [Fetched: arXiv 2203.17173 abstract page.]
- **GENE, GS2, COGENT**: UNVERIFIED this session (no fetch budget left). From memory: GS2 and
  GENE ship regression test suites of reference runs (linear growth rates on standard cases such
  as the Cyclone base case); COGENT papers report MMS/convergence studies. Treat as background,
  not as citable evidence.

### 1.3 Anatomy of recent neoclassical / equilibrium code papers

**yancc** (Conlin & Landreman, arXiv:2607.20861, July 2026) [Fetched: arXiv HTML v1]
- Sections: Introduction; Drift Kinetic Equation (BCs, properties); Discretization; Multigrid
  method and Krylov solver (grid hierarchy, smoothing, coarse grid, cycle type, Krylov);
  Benchmarks (5.1 W7-X monoenergetic DKE, 5.2 single-species density scan, 5.3 two-species E_rho
  scan); Conclusion with 6.1 Data Availability; Appendix on Rosenbluth potentials.
- Eight figures: 3 method figures (smoothing factor, multigrid cycle schematic, eigenvalue
  spectrum), 3 benchmark figures (D_ij vs MONKES on W7-X KJM; fluxes vs SFINCS on NCSX density
  scan; two-species E_rho scan vs SFINCS), 2 timing figures (vs JAX-MONKES on A100; vs SFINCS on
  128 CPUs).
- Verification statement: "both codes are converged to within 1%"; particle and heat flux
  agree to <1%, parallel current ~5% in the two-species case.
- Performance reporting: hardware named (A100 on Perlmutter; SFINCS on 128 cores of the same
  machine); memory stated (4 GB vs 1.4 GB; 6 GB vs >50 GB); explicit JIT convention: "times for
  yancc do not include the cost of just in time compilation in JAX which must be paid once for a
  given resolution, after which the JIT compiled code is re-used across the scan".
- Differentiability: claimed in abstract and conclusion, **not demonstrated** — no gradient
  check, no optimization example. Availability: github.com/f0uriest/yancc; CC BY 4.0.

**MONKES** (Escoto, Velasco, Calvo, Landreman, Parra, Nucl. Fusion 64, 076030 (2024);
arXiv:2312.12248) [Fetched: arXiv HTML]
- Sections: Introduction; Drift-kinetic equation and transport coefficients; Numerical method
  (Legendre expansion; spatial discretization and algorithm); Code performance and benchmark
  (4.1 convergence at low collisionality, 4.2 code performance, 4.3 benchmark of the coefficients);
  Conclusions; five appendices (Onsager symmetry, Legendre modes, invertibility, Fourier
  collocation, DKES convergence).
- Figures 1-6 are all convergence curves (D11, D31, D13, D33 vs number of Legendre modes) for
  W7-X EIM, W7-X KJM, CIEMAT-QI with/without E_r; timing is a table (22-89 s single core, Intel
  Xeon Gold 6254, 4-64x faster than DKES); the DKES/SFINCS benchmark is reported in text and
  tables with a 5-7% relative-error convergence criterion; complexity O(N_xi N_fs^3), memory
  O(N_fs^2). Code: github.com/JavierEscoto/MONKES.

**KNOSOS** (Velasco, Calvo, Parra, García-Regaña, J. Comput. Phys. 418, 109512 (2020)) [Seen:
search snippets; full text fetch failed on TLS] — headline: "reproduces the calculations of DKES
and EUTERPE in simulations that can be orders of magnitude faster"; code at
github.com/joseluisvelasco/KNOSOS. Section/figure structure UNVERIFIED.

**DESC Part I** (Panici, Conlin, Dudt, Unalmis, Kolemen, JPP 2023; arXiv:2203.17173) [Fetched:
abstract page] — 23 figures over 34 pages; the abstract itself carries three quantitative claims
(force-balance error vs VMEC, order-of-magnitude speed, exponential convergence).

### 1.4 What a JCP/CPC methods-paper reviewer therefore expects (synthesis)

1. Equations stated fully, then the discretization, then the solver — each with its own section.
2. **Convergence figures first** (solution verification) for every resolution axis, with the
   converged value and the criterion stated (MONKES: 5-7%; yancc: 1%).
3. **Cross-code figures** on named public configurations (W7-X EIM/KJM, NCSX, HSX, CIEMAT-QI) with
   tolerance stated in the text and both codes converged.
4. **Timing with hardware, memory, and the JIT convention in one sentence** (yancc), or a
   single-core table (MONKES). Ratios accompanied by absolute times.
5. A Data Availability section naming the repo and the tag that reproduces every figure.
6. CPC additionally requires a structured "Program summary" (title, licence, language, nature of
   problem, solution method, restrictions, running time) — UNVERIFIED this session, from memory
   of CPC author guidelines.
7. Differentiability, if claimed, must be demonstrated — yancc did not, which is the opening for
   DKX (see Topic 6).

Figure budget observed: 6-8 figures for a Nuclear Fusion / arXiv methods paper; ~20 for a JPP
"suite" paper. A JCP paper for DKX should plan 8-10 figures: 2 method, 3 convergence, 2-3
cross-code, 1 timing/memory, 1 gradient-check, 1 application teaser.

---

## 2. Benchmark reporting

### 2.1 Hoefler & Belli, "Scientific Benchmarking of Parallel Computing Systems: Twelve ways to
tell the masses when reporting performance results", SC '15, Austin, DOI 10.1145/2807591.2807644
[Fetched: PDF from htor.inf.ethz.ch, text extracted locally]

The paper surveys 120 HPC papers and introduces *interpretability* ("enough information to allow
scientists to understand the experiment, draw own conclusions, assess their certainty, and
possibly generalize results") as a weaker but achievable substitute for reproducibility. The
twelve rules, verbatim:

1. "When publishing parallel speedup, report if the base case is a single parallel process or
   best serial execution, as well as the absolute execution performance of the base case." The
   paper generalizes: "one should never report ratios without absolute values."
2. "Specify the reason for only reporting subsets of standard benchmarks or applications or not
   using all system resources." Corollary: "report all results, not just the best."
3. "Use the arithmetic mean only for summarizing costs. Use the harmonic mean for summarizing
   rates."
4. "Avoid summarizing ratios; summarize the costs or rates that the ratios base on instead. Only
   if these are not available use the geometric mean for summarizing ratios."
5. "Report if the measurement values are deterministic. For nondeterministic data, report
   confidence intervals of the measurement." Example given: "We collected measurements until the
   99% confidence interval was within 5% of our reported means."
6. "Do not assume normality of collected data (e.g., based on the number of samples) without
   diagnostic checking."
7. "Compare nondeterministic data in a statistically sound way, e.g., using non-overlapping
   confidence intervals or ANOVA."
8. "Carefully investigate if measures of central tendency such as mean or median are useful to
   report. Some problems, such as worst-case latency, may require other percentiles."
9. "Document all varying factors and their levels as well as the complete experimental setup
   (e.g., software, hardware, techniques) to facilitate reproducibility and provide
   interpretability."
10. "For parallel time measurements, report all measurement, (optional) synchronization, and
    summarization techniques."
11. "If possible, show upper performance bounds to facilitate interpretability of the measured
    results."
12. "Plot as much information as needed to interpret the experimental results. Only connect
    measurements by lines if they indicate trends and the interpolation is valid."

Applied to the current DKX README speed section: rules 1, 2 and 9 are met (absolute seconds,
the 38-deck table includes the 6 losses and the 6 did-not-complete, PETSc/MUMPS versions and
hardware are named). Rules 5-8 are not: every number is a single measurement with no repetition
count, no statement of determinism, no spread. One sentence fixes it: "median of 5 warm solves,
min-max within x%".

### 2.2 ACM artifact badging — UNVERIFIED (acm.org returned 403 twice)

From memory, for the record: badges are Artifacts Available, Artifacts Evaluated (Functional /
Reusable), Results Reproduced, Results Replicated; ACM's current definitions (aligned with NISO
in 2020) are repeatability = same team, same setup; reproducibility = different team, same
setup; replicability = different team, different setup. URL to verify:
https://www.acm.org/publications/policies/artifact-review-and-badging-current. For DKX the
useful takeaway is the ladder itself: aim for "Available" (tagged release + DOI) and "Results
Reproduced" (a second host regenerates the paper figures from the tag) — and note the README's
own admission that "The 0.93 GB figure above has not reproduced on a second registered host",
which is exactly the Reproduced criterion failing.

### 2.3 JAX-community norms for cold/warm/JIT reporting [Fetched: docs.jax.dev/en/latest/benchmarking.html]

- Asynchronous dispatch: "you need to call `.block_until_ready()` to ensure that computation has
  actually happened."
- Canonical pattern separating compile from run:
  `f_jit = jax.jit(f); %time f_jit(x).block_until_ready()  # compile time`
  `%timeit f_jit(x).block_until_ready()  # runtime` (their example: 193 ms compile, 485 us run).
- Match precision when comparing to NumPy: "JAX by default only uses 32-bit dtypes".
- Small problems measure overhead, not the solver: "if we switch this example to use 10x10
  input instead, JAX/GPU runs 10x slower than NumPy/CPU" — pick "large enough arrays" and
  "intensive enough computation".
- "apply `jax.jit()` on your outer-most function calls"; time `jax.device_put()` separately.
- Community convention in code papers (yancc, fetched): report warm times, state in one sentence
  that JIT is excluded and paid once per resolution. DKX's README currently spends ~30 lines on
  cold/warm; the norm is one table column ("compile, once") plus one sentence.

Concrete DKX benchmark-reporting rule set (used in Section 7): report per case {hardware, JAX and
jaxlib versions, x64 flag, compile time once, warm median of N>=5 with min-max, peak RSS, SFINCS
build string}, and never a speedup without the two absolute times beside it.

---

## 3. Reproducibility standards — what is worth doing for a small academic team

### 3.1 The sources

- **FAIR4RS** — Chue Hong, Katz, Barker et al., "FAIR Principles for Research Software (FAIR4RS
  Principles)", RDA, DOI 10.15497/RDA00068, 16 March 2022; also Scientific Data 9, 622 (2022),
  DOI 10.1038/s41597-022-01710-x. [Fetched: RDA group-output page for metadata; Nature blocked.]
  Motivation verbatim: "many of the FAIR Guiding Principles can be directly applied to research
  software by treating software and data as similar digital research objects", but software's
  "executability, composite nature, and continuous evolution and versioning" required revising
  them. The principle list (F1 persistent identifier incl. F1.1 components / F1.2 versions; F2
  rich metadata; F3 metadata carries the identifier; F4 metadata FAIR-searchable; A1 retrievable
  by identifier over a standard protocol; A2 metadata outlives the software; I1 uses community
  data standards; I2 qualified references to other objects; R1 rich description with R1.1 clear
  licence and R1.2 detailed provenance; R2 qualified references to other software; R3 meets
  domain community standards) is from memory — wording UNVERIFIED.
- **Wilson et al. 2017**, "Good enough practices in scientific computing", PLOS Comput. Biol.,
  DOI 10.1371/journal.pcbi.1005510 [Fetched]. Six areas; the ones that bite here: "Keep changes
  small", "Share changes frequently", "Add a CHANGELOG.txt file", "Create an overview of your
  project" (README with title, description, contact, "examples of how to run tasks" — "often the
  first thing users and collaborators ... will look at"), "Make the license explicit", "Make the
  project citable", "Provide a simple example or test data set". Notably the authors *exclude*
  unit testing, coverage and CI from "good enough" because they "usually aren't compelling for
  solo exploratory work" — DKX is past that stage, but the point stands that infrastructure is
  justified by use, not by principle.
- **Taschuk & Wilson 2017**, "Ten simple rules for making research software more robust", PLOS
  Comput. Biol., DOI 10.1371/journal.pcbi.1005412 [Fetched]. Rules: 1 version control; 2 document
  code and usage (README + `--help`); 3 make common operations easy to control (CLI flags, not
  code edits); 4 version your releases ("Increment your version number every time you release
  your software to other people"); 5 reuse software within reason; 6 rely on build tools and
  package managers; 7 no root; 8 eliminate hard-coded paths; 9 include a small test set;
  10 produce identical results given identical inputs (log parameters, seeds, versions).
- **JOSS review criteria** [Fetched: joss.readthedocs.io/en/latest/review_criteria.html]:
  statement of need ("what problems the software is designed to solve, who the target audience
  is, and its relation to other work"); installation with dependencies "handled by an automated
  procedure"; example usage "ideally to solve real-world analysis problems"; API docs for core
  functionality; community guidelines; "an automated test suite covering the core
  functionality"; an OSI licence as "an actual license file present in the repository"; evidence
  of "sustained development over time (preferably months or years)" rather than "all or most
  commits concentrated in the last few weeks before submission".
- **CITATION.cff + Zenodo** [Seen: The Turing Way "Software Citation with CITATION.cff",
  citation-file-format.github.io]: with a CITATION.cff in the repo GitHub shows "Cite this
  repository", and "Zenodo will use the information from CITATION.cff" when the GitHub-Zenodo
  integration mints a DOI per release. **The Turing Way** (book.the-turing-way.org) is the
  community handbook that hosts this guidance. [Seen.]

### 3.2 What is worth doing for a 1-3 person team with agents (cost-ranked)

Do now (each under two hours, permanent payoff):
1. `LICENSE` file (already present), `CITATION.cff`, GitHub-Zenodo integration, DOI badge in the
   README. Covers FAIR F1/F3/R1.1 and Wilson "make the project citable"; JOSS requires it.
2. Tagged releases with semantic versions and a `CHANGELOG.md`; the paper cites a tag. (Taschuk
   rule 4; Wilson CHANGELOG.)
3. Every output file records: package version, git SHA, JAX/jaxlib versions, x64 flag, command
   line/case_id, hostname/device. That *is* R1.2 provenance and Taschuk rule 10; DKX already has
   `Result ... provenance` and a deterministic `case_id` — stop there.
4. One `publications/<paper>/` directory with `make figures` that regenerates every figure from a
   pinned environment (DESC pattern, fetched). This replaces bespoke "sealed artifact" tooling.
5. A small, fast test set that doubles as the README example (Taschuk 9; Wilson "simple example").

Defer or drop:
- Custom evidence/provenance frameworks beyond item 3. FAIR4RS asks for *metadata and
  identifiers*, not for a provenance database; JOSS asks for tests and docs, not for artifacts.
- Badge-hunting (ACM-style artifact evaluation) before the paper exists.
- Multi-host reproduction of performance numbers as a *release gate*. Report it as Hoefler rule 9
  information instead; a second host is a nice-to-have paragraph, not a blocker.

Rule of thumb: infrastructure is justified when it removes a sentence of hedging from the README
or a caveat from the paper; otherwise it is deferred.

---

## 4. Workflow efficiency for a small team using AI coding agents

### 4.1 Evidence on change size and review effectiveness

- **Google eng-practices, "Small CLs"** [Fetched: google.github.io/eng-practices/review/developer/small-cls.html]
  - A small CL is "one self-contained change" addressing a single focused problem.
  - Size: "100 lines is usually a reasonable size for a CL, and 1000 lines is usually too large";
    a 200-line change in one file may be fine but "spread across 50 files it would usually be too
    large"; "When in doubt, write CLs that are smaller than you think you need."
  - Benefits listed: reviewed faster and more thoroughly, fewer bugs, "less wasted effort" when
    rejected, easier merges, easier rollbacks, and the author is "unblocked" while waiting.
  - Refactors separate from behaviour changes: "moving and renaming a class should be in a
    different CL from fixing a bug in that class."
  - Stacking is explicitly permitted: "Submit one CL for review, then immediately start a
    dependent CL"; other split strategies: by files, horizontally (stubs), vertically.
- **SmartBear / Cisco study** [Fetched: smartbear.com "Best Practices for Peer Code Review"]
  - "developers should review no more than 200 to 400 lines of code (LOC) at a time"; a review of
    that size "over 60 to 90 minutes should yield 70-90% defect discovery".
  - "a significant drop in defect density at rates faster than 500 LOC per hour".
  - "Do not review for more than 60 minutes at a time."
  - "lightweight code review takes less than 20% the time of formal reviews and finds just as
    many bugs".
- **Graphite, stacked diffs** [Fetched: graphite.com/guides/stacked-diffs] — "each diff should
  represent a single, coherent change"; prerequisites are tooling that rebases the stack
  automatically (`gt stack sync`), reviewers assessing each PR independently, and merging in
  dependency order. No numeric depth limit is given; the guide is silent on when not to stack.
- **Trunk-based development** [Fetched: trunkbaseddevelopment.com] — "all team members commit to
  trunk at least once every 24 hours"; feature branches are short-lived and "the product of a
  single dev-workstation"; incomplete work is hidden behind feature flags or branch-by-
  abstraction; "Shared branches off mainline/main/trunk are bad at any release cadence"; the
  codebase stays "always releasable on demand".

### 4.2 Arithmetic for the DKX incident

17 stacked PRs, +5,930/-2,889 = 8,819 changed lines in one day. At the Cisco ceiling of 500 LOC/h
that is ~18 reviewer-hours for one day's output; at the effective 200-400 LOC per 60-90 min
session it is 22-44 sessions. A dependency stack also converts every rejection into a rebase of
everything above it — the opposite of Google's "less wasted effort". The lesson is not that
stacking is wrong (Google and Graphite both endorse it) but that stacking without auto-rebase
tooling, independent reviewability, and a depth cap is a queue, not a workflow.

With coding agents the generation rate is unbounded, so **review capacity is the binding
constraint and must set the daily output budget**. If one human can review ~400 LOC/hour for at
most two hours a day (SmartBear's fatigue limit), the team's sustainable merge rate is roughly
800 reviewed lines per day, independent of how fast the agent writes.

### 4.3 CI tiers and test selection

No single fetched source prescribes tiers; the exemplar is DESC's split into separate `UnitTests`
and `RegressionTests` workflows with their own badges [Fetched: DESC README]. The user's own
notes record vmex CI at ~45 min per attempt and a standing rule to "fan out plan items ... focused
tests, not full CI per change" (memory: feedback-ci-first-attempt-discipline, feedback-
parallelize-dont-serialize-ci). Google's small/medium/large test-size taxonomy is the usual
reference — UNVERIFIED this session.

Proposed tiers for DKX:
- T0 (every push, <5 min): ruff/format, import smoke, unit tests on analytic geometries.
- T1 (PR gate, <=20 min): tests selected by touched module + one small SFINCS parity deck + the
  README example + docs build. A PR that needs more than T1 to be trusted is too big.
- T2 (nightly on `main`): the full 38-deck parity matrix, GPU job, benchmark timings appended to a
  CSV (Hoefler rule 9 metadata included).
- T3 (release / paper tag): cross-code MONKES/yancc comparisons, figure regeneration, wheel
  build, Zenodo deposit.

### 4.4 Figure-first planning and planning documents that stay useful

- **Whitesides, "Whitesides' Group: Writing a Paper", Adv. Mater. 16, 1375-1377 (2004), DOI
  10.1002/adma.200400767** [Seen: search snippets from the Whitesides group page and AcaWiki]:
  "A good outline for the paper is also a good plan for the research program, and you should
  write and rewrite these plans/outlines throughout the course of the research"; an outline
  "contains little text" — it is the ordered list of figures and tables with their captions and
  the conclusion each supports. This is the "figure-first" pattern: the plan for DKX *is* the
  figure lists of the JCP paper and the physics paper, each figure with a status and an
  acceptance criterion.
- **ADRs** [Fetched: adr.github.io]: an Architectural Decision is "a justified design choice that
  addresses a functional or non-functional requirement that is architecturally significant";
  Nygard's template is title / status / context / decision / consequences; ADRs accumulate into
  a "decision log" that is appended to, not rewritten (superseded records point forward).
- **Living roadmap vs changelog vs diary.** Wilson et al. separate the overview (README), the
  CHANGELOG and the shared to-do list [Fetched]. The DKX plan failed by merging all three plus a
  review report. The split that stays useful:
  - `ROADMAP.md`, <= 600 words, only the figure/table list with owner, status (todo / draft /
    final) and acceptance criterion. Reviewed weekly; finished items are deleted, not archived.
  - `docs/adr/NNNN-title.md`, <= 1 page each, immutable; one per non-obvious choice (solver
    route selection, x64 policy, SFINCS field-naming compatibility, gradient-check tolerance).
  - `CHANGELOG.md`, Keep-a-Changelog style (UNVERIFIED reference; format is uncontroversial).
  - Execution diary: PR descriptions and issues only. Never in the plan.
  - Review reports: a dated file under `docs/reviews/` that is never edited after the day it is
    written; the plan links to it in one line.

### 4.5 Concrete rules with citations (feeds Section 7)

- <= 400 changed lines per PR excluding generated data and lockfiles (SmartBear 200-400; Google
  100 reasonable / 1000 too large); reject above 800 without a written split plan.
- One behaviour change or one refactor per PR, never both (Google).
- Stack depth <= 3, only with auto-rebase tooling and each PR mergeable on its own (Google
  stacking; Graphite prerequisites); at most 3 open PRs per author.
- Branch lifetime <= 48 h; merge, split, or hide behind a flag (trunk-based: daily trunk commits).
- Review sessions <= 60 min; review is scheduled, not interrupt-driven (SmartBear).
- Daily merge budget = reviewer capacity (~800 reviewed lines/day per reviewer), which bounds
  agent output.

---

## 5. README design for scientific software

### 5.1 Diátaxis [Fetched: diataxis.fr]

Four documentation types: tutorials (learning-oriented), how-to guides (task-oriented), reference
(information-oriented), explanation (understanding-oriented), arranged on two axes — acquisition
vs application, and action vs cognition. The framework "prescribes approaches to content,
architecture and form that emerge from a systematic approach to understanding the needs of
documentation users". Consequence for a README: it is none of the four — it is the *front door*
that routes each reader to the right quadrant in one click. Prose that explains (cognition) or
warns (how-to caveats) belongs in the quadrant, not the door.

### 5.2 Eight READMEs measured (raw files fetched 2026-09-06; `wc -l` / `wc -w`)

| Project | Lines | Words | Sections (in order) | Showcase figure | Minimal example | Headline claim | Citation block |
|---|---:|---:|---|---|---|---|---|
| **DESC** (PlasmaControl/DESC, README.rst) | 137 | 623 | logo+badges (License, DOI, Issues, PyPI, Docs, UnitTests, RegressionTests, Codecov); one-sentence description; 4-paper cite list; Quick Start (pip, 8 tutorial links, CLI); Repository Contents; Contribute | logo only, no result figure | none in README (`pip install desc-opt`, `desc <inputfile>`); examples are linked tutorials | "solves for and optimizes 3D MHD equilibria using pseudo-spectral numerical methods and automatic differentiation" | 4 papers with DOIs + PDF links, Zenodo DOI badge; no BibTeX |
| **simsopt** (hiddenSymmetries/simsopt) | 85 | 403 | badges (license, codecov, Zenodo); logo + `coils_and_surfaces.png`; description + 5 component bullets; 4 design principles; install (pip, docker); separate module repos; citation; funding | yes — one coils-and-surfaces image directly under the logo | none (pip / docker one-liners) | "a framework for optimizing stellarators"; "Efficient implementations of the Biot-Savart law ... including derivatives" | JOSS reference in plain text with DOI; Zenodo badge |
| **Diffrax** (patrick-kidger/diffrax) | 82 | 315 | h1 + h2 tagline; 7 feature bullets; Installation; Documentation; Quick example; Citation; See also (JAX ecosystem) | none, no badges | **9 lines** of Python (`diffeqsolve(term, solver, ...)`) | "Numerical differential equation solvers in JAX. Autodifferentiable and GPU-capable."; "multiple adjoint methods for backpropagation" | BibTeX (PhD thesis) + arXiv link |
| **Optimistix** (patrick-kidger/optimistix) | 83 | 324 | same template as Diffrax; Quick example; Citation; See also; Credit | none | **13 lines** incl. two comment lines (`optx.fixed_point`) | "nonlinear solvers: root finding, minimisation, fixed points, and least squares"; "all the benefits of working with JAX: autodiff, autoparallelism, GPU/TPU" | BibTeX (arXiv 2402.09983) |
| **jax-cfd** (google/jax-cfd) | 151 | 693 | deprecation banner; authors; description + PNAS link; Getting started (6 Colab notebooks incl. 2 that reproduce the paper); Organization (submodules); Numerics; Projects using; Other projects; Citation (2 BibTeX); Local development | none | none in README; Colab notebooks | "exploring the potential of machine learning, automatic differentiation and hardware accelerators (GPU/TPU) for computational fluid dynamics" | 2 BibTeX entries keyed to which submodule you used |
| **JAX-MD** (jax-md/jax-md) | 180 | 1262 | logo + tagline; nav bar (Quickstart / Installation / Reference docs / Paper / NeurIPS); badges (Build, DOI, PyPI, License); 3 motivation paragraphs; Getting Started (talk video, 8 Colab notebooks incl. "Implicit Differentiation", "Meta Optimization", 6 example scripts, FEATURES.md); Installation (uv, pip); Development; Tests; Technical gotchas (GPU, 64-bit); Publications (19 papers using it); Citation (2 BibTeX) | logo + video thumbnail; no result figure | none inline; notebooks | "Accelerated, Differentiable, Molecular Dynamics"; "end-to-end differentiable" | 2 BibTeX (NeurIPS 2020; PNAS 2024 for RigidBody) + Zenodo DOI badge |
| **PhiFlow** (tum-pbs/PhiFlow) | 277 | 988 | logo; badges (build, pyversions, license, codecov, Colab); one paragraph; Examples as image-grid tables (Grids 12, Mesh, Particles, Optimization & Networks 7 — 44 images); Installation; Features (6 bullets); Documentation and Tutorials; Citation; Publications; Benchmarks & Data Sets; Version History; Contributions; Acknowledgements | yes — a gallery of ~30 thumbnails, each linking to an example page | none inline (gallery links) | "simulation toolkit built for optimization and machine learning applications ... end-to-end differentiable functions involving both learning models and physics simulations" | BibTeX (ICML 2024) |
| **JAX-Fluids** (tumaer/JAXFLUIDS, via GitHub API) | 179 | 885 | title; one paragraph; authors; Physical models and numerical methods (bullets); Example simulations (3 images); Pip installation (CPU / GPU); Quickstart; Documentation; Acknowledgements; Citation (2 BibTeX); Publications using (team / others); License | yes — 3 simulation images | link to notebooks | "fully-differentiable CFD solver for 3D, compressible single-phase and two-phase flows"; "tested on up to 512 NVIDIA A100 GPUs and on up to 2048 TPU-v3 cores" | 2 BibTeX (CPC 2023, arXiv 2024) |
| **DKX today** (uwplasma/DKX main) | 250 | 1508 | badges (PyPI, CI, Docs [still points at sfincs-jax], License, Python); one-liner; W7-X showcase figure; Install; Run (CLI + 21-line Python, then 12 lines of convergence caveat); Speed (table, then "Cold and warm solves" 30 lines, then 38-deck table); Accuracy (parity figure; "Against SFINCS, MONKES and YANCC" 33 lines); Capabilities table; Limitations (4 bullets); SFINCS compatibility; From an equilibrium (30 lines on density assumptions); Documentation links; License | yes — W7-X 4-panel figure, plus 3 more figures | 21 lines, prints a flux; **no `jax.grad` line** despite "every output is differentiable in every input" | "Outputs match SFINCS Fortran v3 field by field, and every output is differentiable in every input"; 27.2 s vs 463.6 s | **none**; no Zenodo DOI |

What makes the effective ones effective:
- **Diffrax/Optimistix**: <=330 words, tagline states the two differentiators (autodiff, GPU) in
  one line, a runnable example under 15 lines, BibTeX. Nothing hedged, nothing explained; every
  explanation is one click away in docs.
- **simsopt**: a single result image under the logo says "this is what you get" before any text.
- **DESC**: the citation list *is* the design document — four papers, each a DOI. Repository
  Contents points to `publications/` for figure reproduction.
- **JAX-Fluids**: one quantitative scale claim in the first paragraph (512 A100s / 2048 TPU cores)
  does the work of a whole performance section.
- **JAX-MD**: a one-line navigation bar (Quickstart | Installation | Reference docs | Paper) and
  a publications list as social proof; the cost is 1,262 words, most of it that list.
- **PhiFlow**: differentiability shown as a gallery of inverse problems, not asserted.
- **jax-cfd**: two notebooks explicitly labelled "Reproduce results from our PNAS paper".

Where DKX's README loses: 1,508 words with zero citation block; three long caveat passages
(cold/warm, convergence, density-from-pressure) that are how-to/explanation content on the front
door; the single most distinctive capability (gradients) is asserted in a table cell and never
shown; the Docs badge points at the legacy `sfincs-jax` project name. Its strengths to keep: the
W7-X showcase figure, the honest 38-deck table, the capability table, the two-tier accuracy
statement (SFINCS 1e-8 parity vs MONKES/yancc ~1% independence).

### 5.3 Proposed README structure for DKX (see Section 8 for the line-by-line blueprint)

Target: <= 120 lines, <= 650 words, 2 figures, 1 code block of <= 12 lines, 2 tables.

Sections and budgets: Title/tagline/badges (8 lines) -> Hero figure with 2 panels (5) -> Install
(4) -> Ten-line differentiable example (16) -> Why DKX: 4-code capability table (12) -> Verified
(8 + parity figure) -> Fast (10-row-max table, 12) -> Documentation (Diátaxis quartet, 6) -> Cite
(12) -> License / Contributing / Acknowledgements (5).

Moves to docs: cold/warm discussion -> `docs/performance.rst`; convergence caveat and `dkx
converge` narrative -> a how-to "Check your resolution"; "From an equilibrium" density assumption
-> how-to "Run from a VMEC wout"; Limitations -> `docs/limitations.rst` linked in one sentence;
SFINCS compatibility -> reference page; artifact regeneration, "five explicit exclusions", second-
host reproduction -> `docs/validation_matrix.rst`.

Impact elements, justified:
1. **A 10-line example that ends in `jax.grad`** — Diffrax and Optimistix prove that a sub-15-line
   runnable block is the norm for JAX scientific libraries; DKX's differentiator must appear in
   the code, not in a table cell.
2. **One hero figure with a gradient panel** — simsopt shows a single image works; PhiFlow shows
   gradients are convincing when *pictured*; yancc's paper shows nobody in neoclassical has done
   it yet.
3. **A four-code capability table (DKX / SFINCS / MONKES / yancc)** — extends the current
   two-column table to the codes reviewers will ask about; JOSS's "relation to other work".
4. **A cite block with BibTeX + Zenodo DOI** — every exemplar except DKX has one; JOSS and FAIR4RS
   require it; the future paper depends on it.
5. **One quantitative sentence per claim, no hedging**: "744k unknowns: 27 s warm / 24 s cold on
   one M3 Max vs 230-464 s for SFINCS v3; median of 5 runs" (JAX-Fluids pattern), with the full
   tables one click away.

---

## 6. How differentiable-physics projects present differentiability

| Project | Where differentiability appears | What is actually *shown* |
|---|---|---|
| DESC [Fetched README + arXiv abstract] | one-liner ("automatic differentiation"); tutorials "Basic optimization", "Advanced optimization" | In the README, nothing; in the papers (Part III, quasi-symmetry optimization) it is shown as optimization results. |
| JAX-MD [Fetched README] | tagline "Accelerated, Differentiable, Molecular Dynamics"; paragraph arguing that forces are gradients of energy so AD removes code; notebooks "Implicit Differentiation", "Meta Optimization" | No inline gradient figure; 19 downstream publications serve as evidence. |
| PhiFlow [Fetched README] | "end-to-end differentiable functions involving both learning models and physics simulations" | A dedicated **Optimization & Networks** gallery: Gradient Descent, Optimize throw, Learning to throw, PIV, Close packing, Learn potential, Differentiable pressure — seven thumbnails of inverse problems. |
| JAX-Fluids [Fetched README] | "fully-differentiable CFD solver"; "enables automatic differentiation for end-to-end optimization of numerical models" | Three forward-simulation images; differentiability demonstrated in the CPC 2023 paper (UNVERIFIED which figure). |
| Diffrax / Optimistix [Fetched READMEs] | "Autodifferentiable" in tagline; "multiple adjoint methods for backpropagation" in feature list | Nothing in README; adjoint options documented in docs. |
| yancc paper [Fetched arXiv] | abstract and conclusion | **Not demonstrated** — no gradient check, no optimization. |

Patterns that convince, in increasing cost:
1. **Code**: three lines — `J = lambda B_mn: dkx.run(case.with_geometry(B_mn)).bootstrap_current;
   dJ = jax.grad(J)(B_mn)`. Diffrax-style: the reader sees `jax.grad` applied to a physical output.
2. **Gradient-check figure**: |AD - FD| vs finite-difference step for two or three outputs; the
   V-shaped FD curve against a flat AD line is the standard verification plot (JAX's own
   `jax.test_util.check_grads` is the programmatic equivalent — UNVERIFIED citation, standard
   practice). This is *code verification* of the adjoint and belongs in the JCP paper as well.
3. **Sensitivity map**: d(bootstrap current)/d(B_mn) as a bar or heat map over Boozer harmonics
   for W7-X — physically interpretable, and a result no other neoclassical code has published.
4. **Inverse problem**: minimize a neoclassical objective over a handful of boundary modes with
   the convergence trace and before/after geometry (PhiFlow "Optimize throw", DESC Part III).

For DKX the hero figure should be panel (a) the existing W7-X result and panel (b) the sensitivity
map or the gradient check; the README example should be pattern 1; the JCP paper should carry
patterns 2 and 3; the physics paper carries pattern 4.

---

## 7. Process rules for DKX

1. **PR size cap: <= 400 changed lines, <= 10 files; hard reject above 800.** Generated data,
   lockfiles and pinned artifacts are excluded from the count but must be in their own PR.
   Justification: SmartBear/Cisco — defect discovery collapses above 400 LOC and above 500 LOC/h
   [Fetched]; Google — "100 lines is usually a reasonable size ... 1000 lines is usually too
   large" [Fetched]. Today's chain (8,819 lines/day) was 10-20x over any of these.
2. **One idea per PR; refactor and behaviour never share a PR.** Google: "moving and renaming a
   class should be in a different CL from fixing a bug in that class" [Fetched]. Corollary: a PR
   title must be a single verb phrase; "and" in a title is a split request.
3. **Stacks are allowed only with depth <= 3, auto-rebase tooling, and each PR independently
   mergeable.** Google permits stacking; Graphite lists auto-rebase, independent review and
   in-order merge as prerequisites [Fetched]. Without tooling, no stacks: finish and merge PR n
   before opening n+1. At most 3 open PRs per author at any time.
4. **Branch lifetime <= 48 h.** Trunk-based development: trunk commits "at least once every 24
   hours"; long-lived shared branches "are bad at any release cadence" [Fetched]. Incomplete
   features land behind a flag or as unused modules, not on a branch.
5. **Agent output is budgeted by review capacity, not generation speed.** One reviewer, two
   60-minute sessions per day (SmartBear fatigue limit) at <= 400 LOC each = ~800 reviewed
   lines/day. The agent stops opening PRs when the queue reaches that.
6. **CI in four tiers; the PR gate is T1 (<= 20 min).** T0 lint+unit on every push; T1 touched-
   module tests + one small SFINCS parity deck + README example; T2 nightly full 38-deck matrix
   and GPU; T3 release-tag cross-code + figure regeneration + wheel + Zenodo. Exemplar: DESC's
   separate UnitTests and RegressionTests workflows [Fetched]; internal precedent: vmex 45-min CI
   and the parallelize-don't-serialize rule (memory). A change that cannot be trusted after T1 is
   too large (rule 1).
7. **Figure-first planning: the roadmap is the ordered figure list of the two papers.** Whitesides:
   "A good outline for the paper is also a good plan for the research program" [Seen]. Each item =
   figure/table, owner, status (todo/draft/final), acceptance criterion (e.g. "D_ij within 1% of
   MONKES on W7-X KJM at nu* in [1e-5, 3e2]"). Work that maps to no figure and no bug is not
   scheduled.
8. **Three documents, three jobs, hard caps.** `ROADMAP.md` <= 600 words, deletions weekly.
   `docs/adr/NNNN-*.md` <= 1 page, Nygard template (title/status/context/decision/consequences),
   immutable, superseded by a new ADR [Fetched: adr.github.io]. `CHANGELOG.md` per release
   (Wilson et al. [Fetched]). Execution diary lives in PR descriptions and issues; review reports
   are dated, write-once files under `docs/reviews/`. The plan never contains status prose.
9. **Provenance is five fields, not a framework.** Every output records version, git SHA, JAX/
   jaxlib versions + x64 flag, case_id/command line, device/host (Taschuk rule 10 [Fetched];
   Hoefler rule 9 [Fetched]; FAIR4RS R1.2). Paper figures are regenerated by
   `publications/<paper>/make_figures.py` from a pinned environment (DESC pattern [Fetched]).
   No further evidence/sealing infrastructure until a reviewer asks for it.
10. **Benchmark reporting follows Hoefler rules 1, 2, 5, 9 and the JAX benchmarking page.**
    Absolute times beside every ratio; all decks reported including losses; "median of N >= 5
    warm solves after `block_until_ready()`, compile time reported once per resolution,
    min-max within x%"; hardware, JAX/jaxlib, x64, SFINCS build string in the table caption
    [Fetched: Hoefler & Belli; docs.jax.dev/benchmarking; yancc convention].
11. **Verification is labelled by tier in code, docs and paper.** Tier A code verification:
    analytic limits (Spitzer D33, tokamak banana/plateau coefficients), Onsager symmetry, AD-vs-FD
    gradient check. Tier B solution verification: convergence in each of theta/zeta/pitch/speed
    with stated criterion (MONKES 5-7%, yancc 1% [Fetched]). Tier C cross-code: SFINCS parity
    (same discretization, 1e-8 class) reported separately from MONKES/yancc independence (~1%).
    Tier D validation: W7-X E_r with experimental uncertainty (ASME V&V 20 framing [Seen]).
    Never call Tier C "validation".
12. **Release cadence: a tag at every paper milestone and at least monthly.** Semantic version,
    CHANGELOG entry, CITATION.cff, Zenodo DOI via GitHub integration [Seen: Turing Way];
    Taschuk rule 4 [Fetched]. The paper cites the tag, not `main`.
13. **README budget enforced by a test: <= 120 lines, <= 650 words, 2 figures, one code block.**
    Every hedging sentence ("has not reproduced on a second host", "is a survey, not a study")
    moves to the docs quadrant Diátaxis assigns it [Fetched]. Exemplars: Diffrax 82/315,
    Optimistix 83/324, simsopt 85/403, DESC 137/623 (measured).
14. **Results-to-infrastructure ratio: every week at least one merged PR adds or upgrades a paper
    figure.** Infrastructure PRs must link the figure or bug they unblock (rule 7). This is the
    direct counter to "heavy investment in evidence/provenance infrastructure relative to
    scientific results".
15. **Definition of done for a PR (<= 10-line description): what changed, why, which tier of CI
    proves it, which figure/ADR/issue it serves.** Tests and docs in the same PR (Google: related
    tests accompany the change [Fetched]); physics changes regenerate the affected figure by
    script and attach it.

---

## 8. README blueprint for DKX

Target: 100-120 lines, <= 650 words, 2 figures, 1 code block (<= 12 lines), 2 tables, 1 BibTeX.
Line budgets are for the rendered Markdown source.

| # | Section | Lines | Content | Exemplar evidence |
|---|---|---:|---|---|
| 1 | Title + tagline + badges | 8 | `# DKX` ; one line: "Differentiable neoclassical transport for stellarators and tokamaks, in JAX." ; badges: PyPI, CI, Docs (fix URL), Zenodo DOI, License | Diffrax h1/h2 tagline names both differentiators; DESC/JAX-MD carry a DOI badge |
| 2 | Hero figure | 5 | one image, two panels: (a) W7-X bootstrap current + E_r roots (existing), (b) d(J_bs)/d(B_mn) sensitivity bars or AD-vs-FD gradient check; caption <= 2 lines | simsopt single image; PhiFlow shows gradients as pictures; yancc shows nobody has |
| 3 | Install | 4 | `pip install dkx` / `pip install -U "jax[cuda12]"` | every exemplar |
| 4 | Ten-line example | 16 | build a `Case` from a shipped W7-X or tokamak file (1-2 lines, not 14 lines of dict), `result = dkx.run(case)`, print bootstrap current, then `jax.grad(lambda x: dkx.run(case.replace(...)).bootstrap_current)(x)` | Diffrax 9 lines; Optimistix 13 lines |
| 5 | Why DKX (capability table) | 12 | rows: full Fokker-Planck; Phi1; E_r root solve; SFINCS deck/HDF5 compatibility; GPU; exact gradients; warm-start scans — columns DKX / SFINCS v3 / MONKES / yancc | current two-column table, extended; JOSS "relation to other work" |
| 6 | Verified | 10 | three sentences: SFINCS parity 1e-10 to 1e-8 on matched decks (same discretization); MONKES and yancc within 1% on W7-X EIM/KJM and CIEMAT-QI (independent codes); gradients agree with finite differences to 1e-6. One parity figure. Link to validation matrix | yancc/MONKES tolerance statements; DESC abstract style |
| 7 | Fast | 12 | one table, <= 5 rows: HSX 744k unknowns — DKX CPU warm/cold, DKX GPU, SFINCS 1 and 2 ranks; caption: hardware, versions, "median of 5"; one sentence on the 38-deck score (9/9 structured, 7/23 Krylov) linking to `docs/performance.rst` | Hoefler rules 1, 2, 5, 9; JAX-Fluids one-sentence scale claim |
| 8 | Documentation | 6 | four links labelled Tutorial (first W7-X run), How-to (check resolution, run from a wout, SFINCS decks), Reference (case schema, CLI, API), Explanation (physics models, solver routes, limitations) | Diátaxis quartet |
| 9 | Cite | 12 | BibTeX for the JCP preprint (placeholder until arXiv) + Zenodo DOI line + "please also cite SFINCS (Landreman et al. 2014) when using its decks" | DESC, JAX-MD, jax-cfd multi-cite pattern |
| 10 | Contributing / License / Acknowledgements | 5 | one line each | DESC Contribute; simsopt funding line |

What leaves the README (and where it lands):
- "Cold and warm solves" (30 lines) -> `docs/performance.rst`, one table column + one sentence
  stays.
- Convergence caveat, `dkx converge`, the pitch=8 vs pitch=40 anecdote -> how-to "Check your
  resolution before trusting a case".
- "From an equilibrium" density-from-pressure discussion (30 lines) -> how-to "Run from a VMEC
  wout"; the README keeps `dkx wout.nc` as one line in the CLI list.
- Limitations bullets -> `docs/limitations.rst`; README carries "Known limitations are listed
  here" once.
- "sealed artifacts", "five explicit exclusions", second-host non-reproduction -> validation
  matrix and performance pages.
- SFINCS compatibility -> reference page; README keeps one capability-table row.

Acceptance test (add to T1 CI): README <= 120 lines, <= 650 words, exactly one ```python block
of <= 12 lines that runs in < 60 s on CPU and contains `jax.grad`, two image references, a
```bibtex block, and no occurrence of "has not", "cannot", "does not yet", "is not converged".

---

## 9. References

Fetched this session (content read):
- Google Engineering Practices, "Small CLs" — https://google.github.io/eng-practices/review/developer/small-cls.html
- SmartBear, "Best Practices for Peer Code Review" (Cisco study) — https://smartbear.com/learn/code-review/best-practices-for-peer-code-review/
- Hoefler & Belli, "Scientific Benchmarking of Parallel Computing Systems", SC '15, DOI 10.1145/2807591.2807644 — https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf
- JAX documentation, "Benchmarking JAX code" — https://docs.jax.dev/en/latest/benchmarking.html (and FAQ https://docs.jax.dev/en/latest/faq.html)
- Wilson et al. 2017, "Good enough practices in scientific computing", DOI 10.1371/journal.pcbi.1005510 — https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510
- Taschuk & Wilson 2017, "Ten simple rules for making research software more robust", DOI 10.1371/journal.pcbi.1005412 — https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005412
- JOSS review criteria — https://joss.readthedocs.io/en/latest/review_criteria.html
- Diátaxis — https://diataxis.fr/
- Graphite, "Stacked diffs" — https://graphite.com/guides/stacked-diffs
- Trunk Based Development — https://trunkbaseddevelopment.com/
- ADR GitHub organization — https://adr.github.io/
- FAIR4RS principles, RDA output page (metadata only), DOI 10.15497/RDA00068 — https://www.rd-alliance.org/group_output/fair-principles-for-research-software-fair4rs-principles/
- Conlin & Landreman, yancc, arXiv:2607.20861 — https://arxiv.org/html/2607.20861v1
- Escoto et al., MONKES, Nucl. Fusion 64, 076030 (2024), arXiv:2312.12248 — https://arxiv.org/html/2312.12248
- Panici et al., DESC Part I, JPP 2023, arXiv:2203.17173 — https://arxiv.org/abs/2203.17173
- READMEs (raw): DESC https://raw.githubusercontent.com/PlasmaControl/DESC/master/README.rst ; simsopt https://raw.githubusercontent.com/hiddenSymmetries/simsopt/master/README.md ; Diffrax https://raw.githubusercontent.com/patrick-kidger/diffrax/main/README.md ; jax-cfd https://raw.githubusercontent.com/google/jax-cfd/main/README.md ; Optimistix https://raw.githubusercontent.com/patrick-kidger/optimistix/main/README.md ; JAX-MD https://raw.githubusercontent.com/jax-md/jax-md/main/README.md ; PhiFlow https://raw.githubusercontent.com/tum-pbs/PhiFlow/master/README.md ; JAX-Fluids https://api.github.com/repos/tumaer/JAXFLUIDS/readme ; DKX https://raw.githubusercontent.com/uwplasma/DKX/main/README.md

Seen in search snippets only (title/abstract-level):
- Roache, "Code Verification by the Method of Manufactured Solutions", ASME J. Fluids Eng. 124, 4 (2002) — https://asmedigitalcollection.asme.org/fluidsengineering/article-abstract/124/1/4/462791
- Roy, "Review of Code and Solution Verification Procedures for Computational Simulation", JCP — https://www.aoe.vt.edu/content/dam/aoe_vt_edu/people/faculty/cjroy/Publications-Articles/cjr_jcp.revise.final-accepted.pdf
- Oberkampf & Roy, Verification and Validation in Scientific Computing, CUP 2010 — https://www.cambridge.org/core/books/abs/verification-and-validation-in-scientific-computing/index/EE029CB068531D278AB2631911F8BE42
- Velasco et al., KNOSOS, J. Comput. Phys. 418, 109512 (2020) — https://www.sciencedirect.com/science/article/abs/pii/S0021999120302862 ; code https://github.com/joseluisvelasco/KNOSOS
- Landreman, Smith, Mollén, Helander, Phys. Plasmas 21, 042503 (2014) (SFINCS) — https://pubs.aip.org/aip/pop/article-abstract/21/4/042503/818401 ; https://github.com/landreman/sfincs
- Whitesides, "Whitesides' Group: Writing a Paper", Adv. Mater. 16, 1375 (2004), DOI 10.1002/adma.200400767 — https://www.gmwgroup.harvard.edu/publications/whitesides-group-writing-paper
- The Turing Way, "Software Citation with CITATION.cff" — https://book.the-turing-way.org/communication/citable/citable-cff/ ; Citation File Format — https://citation-file-format.github.io/

UNVERIFIED (fetch failed or from memory):
- ACM Artifact Review and Badging v1.1 — https://www.acm.org/publications/policies/artifact-review-and-badging-current (HTTP 403)
- FAIR4RS principle wording; Chue Hong et al., Sci. Data 9, 622 (2022), DOI 10.1038/s41597-022-01710-x (Nature redirect loop)
- ASME V&V 20-2009 scope statement (snippet only)
- GENE / GS2 / COGENT verification suites; Google test-size taxonomy; Keep a Changelog; CPC "Program summary" requirement; `jax.test_util.check_grads`
