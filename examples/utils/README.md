# Example Utilities

This folder contains helper scripts used to generate or support example
documentation. It is not the first place users should look for SFINCS-JAX
workflows; start with `examples/README.md` or `examples/tutorials/`.

## Files

- `generate_utils_gallery.py`: builds the small utility-gallery figure used by
  the documentation from package plotting and postprocessing helpers.

Keep shared helper code here only when it is reused by multiple examples. If a
script is a standalone workflow, place it in the topic folder that matches the
user task instead.
