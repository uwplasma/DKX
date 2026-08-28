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

The first directly executable route is an analytic, prescribed-electric-field
profile. It consumes ``Case`` fields directly: it does not serialize or parse a
SFINCS namelist while constructing grids, geometry, species, collisions, or the
operator. Run the checked example from Python:

.. code-block:: python

   import dkx

   case = dkx.Case.from_file("examples/native/analytic_tokamak_profile.toml")
   result = dkx.run(case)
   result.print_summary()
   result.save()                         # the case's [output].file
   result.plot("profile.png")
   particle_flux = result.particle_flux_m2_s
   certificate = result.certificate()

``Result`` copies its named arrays and makes them read-only. The
``dimensions`` map gives every array's named axes without requiring xarray;
``save`` writes schema-v1 NetCDF4, and ``Result.load`` reads it through the same
contract. Files contain the canonical case, normalization, geometry checksum,
package/runtime/device versions, selected route, residual, iteration and timing
evidence, and peak host memory.

The executable route supports only ``workflow = "profile"``, built-in
``format = "analytic"`` geometry, ``magnetic_drifts = "dkes"``, ``phi1 =
"off"``, a prescribed electric field, and at least two profile surfaces.
Unsupported native combinations fail with the exact case field and a
correction; they are not silently downgraded. Native VMEC/Boozer execution,
ambipolar roots, resumable scan execution, and SFINCS conversion are subsequent
vertical slices. Existing namelist workflows remain available through
``dkx.run`` and the established CLI without a numerical-path change.
