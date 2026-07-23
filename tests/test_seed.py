import numpy as np

from action_segmentation.seed import set_global_seed


def test_numpy_seed_is_reproducible() -> None:
    set_global_seed(42)
    first = np.random.normal(size=5)

    set_global_seed(42)
    second = np.random.normal(size=5)

    np.testing.assert_allclose(first, second)
