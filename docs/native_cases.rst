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

The checked example is ``examples/native/w7x_ambipolar_profile.toml``. Its
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

Current boundary
----------------

``Case.from_file()``, ``dkx validate``, and ``dkx schema`` are stable input
contracts. Native ``dkx.run(case)``, ``Result``, NetCDF output, SFINCS
conversion, and scan execution are the next vertical slices. Until those land,
existing namelist workflows continue to run through ``dkx.run.run_profile`` and
the established CLI without a numerical-path change.
