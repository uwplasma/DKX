Native cases
============

DKX schema version 1 defines an immutable, physically named ``Case``. TOML is
the primary human-authored format; JSON has exactly the same semantic fields
for generated inputs. Both formats pass through one validation boundary and
produce a deterministic SHA-256 case ID that is independent of table/key order
and the case file's location.

Start from the complete commented template:

.. code-block:: console

   dkx schema --format toml > case.toml
   dkx validate case.toml

Machine tooling can request JSON Schema instead:

.. code-block:: console

   dkx schema --format json > case-v1.schema.json

The checked full-schema example is
``examples/native/w7x_ambipolar_profile.toml``. Its
field names carry engineering units where a dimensional value appears, such as
``density_m3``, ``temperature_keV``, and ``search_kV_m``. Solver methods use
physical route names—``structured_direct``, ``recycled_krylov``, and
``sparse_direct_referee``—instead of numbered tiers.

Validation and portability
--------------------------

Validation errors identify the full input path, supplied value, expected form,
and a correction. Profile arrays must have one entry per requested surface;
surface values are unique and lie from zero through one; all physical profile
values are finite and positive; and an ambipolar search requires increasing
finite bounds.

Schema validation does not require a geometry file to exist. This keeps a case
portable and permits validation before data staging. ``Case.geometry_path``
resolves a relative geometry path beside the loaded case when execution begins.
The source location is provenance and is excluded from the semantic case ID.

Declarative scan preflight
--------------------------

Schema-v1 accepts Cartesian and zipped explicit-value axes. ``Case.scan``
computes the case count before launch and rejects counts above ``max_cases``;
zipped axes must have equal lengths. Resume metadata and append-safe result
storage belong to the native scan/result execution slice and are not simulated
by the validator.

Native execution and results
----------------------------

The directly executable route accepts built-in analytic geometry or a VMEC
``wout`` for prescribed-electric-field and ambipolar profiles. It consumes
``Case`` fields directly: it does not serialize or parse a SFINCS namelist
while constructing grids, geometry, species, collisions, or the operator. Run
a checked example from Python:

.. code-block:: python

   import dkx

   case = dkx.Case.from_file("examples/native/analytic_tokamak_profile.toml")
   result = dkx.run(case)
   result.print_summary()
   result.save()                         # the case's [output].file
   result.plot("profile.png")
   particle_flux = result.particle_flux_m2_s
   certificate = result.certificate()

For a VMEC equilibrium, change only the geometry source:

.. code-block:: toml

   [geometry]
   format = "vmec"
   file = "wout_my_device.nc"
   surfaces = [0.16, 0.25, 0.36]

The file is resolved through ``Case.geometry_path``, read once per profile, and
its exact SHA-256 is stored in the result. One phase-space grid is reused across
all surfaces because their array shapes are identical; each surface still gets
its own radially interpolated magnetic geometry and operator coefficients.
``value_kV_m`` is explicitly normalized using the pinned 1 keV and 1 m SFINCS
reference set (for which the numerical conversion to ``ErHat`` is one).

For native ambipolar execution, select the workflow and give physical search
controls:

.. code-block:: toml

   [run]
   workflow = "ambipolar_profile"

   [electric_field]
   mode = "ambipolar"
   search_kV_m = [-5.0, 5.0]
   search_points = 5
   root_tolerance_kV_m = 0.05
   max_root_iterations = 8
   find_all_roots = true
   continue_branches = true

Each surface performs one memory-bounded coarse electric-field batch, refines
every sign-changing bracket with real kinetic solves, and preserves all
evaluated fields, radial currents, fluxes, residuals, brackets, slopes, and
root classifications in ``Result``. A profile selects the root nearest zero on
its first surface and then the root nearest the selected branch on the
preceding surface.
When no root is bracketed, DKX retains the sampled point with the smallest
absolute radial current, labels it ``no_bracketed_root``, and never calls it
ambipolar.

Enable the existing convergence contract to insert every interval midpoint in
a deterministic bounded hierarchy:

.. code-block:: toml

   [convergence]
   enabled = true
   observables = ["particle_flux", "heat_flux", "electric_field"]
   relative_tolerance = 0.02
   max_refinements = 2

Every added kinetic solve records ``evaluation_reason`` and
``evaluation_refinement_level``. Each rung records its search and total solve
counts, discovered root count, root movement, requested-observable movement,
and maximum final bracket width. The preflight records a conservative retained
evaluation budget and rejects work or evidence storage beyond its fixed work
and requested memory bounds before allocating the hierarchy.

The bounded hierarchy runs through the configured ``max_refinements`` so an
early stable root cannot prevent a finer declared rung from exposing another
pair of crossings. ``ambipolar_refinement_status`` is ``resolved`` only when
the final two rungs retain the same nonzero root count and meet the declared
root, observable, and bracket-width tolerances. ``refinement_exhausted`` means
roots were observed but the final evidence did not stabilize.
``no_bracket_observed`` means the finite hierarchy observed no sign-changing
bracket. It is not a proof that no root exists: an even number of crossings can
remain hidden between the finest adjacent samples. ``find_all_roots`` therefore
means every bracket exposed by the declared finite hierarchy, not every
mathematically possible root. Branch creation/loss certificates remain the
next P8 promotion gate.

``Result`` copies its named arrays and makes them read-only. The
``dimensions`` map gives every array's named axes without requiring xarray;
``save`` writes schema-v1 NetCDF4, and ``Result.load`` reads it through the same
contract. Files contain the canonical case, normalization, geometry checksum,
package/runtime/device versions, selected route, residual, iteration and timing
evidence, and peak host memory.

The executable route supports ``workflow = "profile"`` with a prescribed
field or ``workflow = "ambipolar_profile"`` with a bounded search, ``format =
"analytic"`` or ``"vmec"`` geometry, ``magnetic_drifts = "dkes"``, ``phi1 =
"off"``, and at least two profile surfaces.
Unsupported native combinations fail with the exact case field and a
correction; they are not silently downgraded. Native Boozer execution,
resumable scan execution, phase-space convergence rungs, and SFINCS conversion
are subsequent vertical slices. Existing namelist workflows remain available
through ``dkx.run`` and the established CLI without a numerical-path change.
