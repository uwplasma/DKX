"""Memoized grids and parsed equilibria must not be mutable through a shared result."""

import numpy as np
import pytest

from dkx.phase_space import make_speed_grid
from dkx.xgrid import make_x_grid


@pytest.mark.parametrize("make", [lambda: make_x_grid(n=5), lambda: make_speed_grid(n_x=5)])
def test_cached_grid_arrays_are_read_only(make):
    grid = make()
    with pytest.raises(ValueError):
        grid.x[0] = 99.0
    assert make().x[0] != 99.0
    assert not np.shares_memory(grid.dx_weights(), grid.x)
