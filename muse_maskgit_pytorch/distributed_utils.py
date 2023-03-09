"""
Utility functions for optional distributed execution.

To use,
1. set the `BACKENDS` to the ones you want to make available,
2. in the script, wrap the argument parser with `wrap_arg_parser`,
3. in the script, set and use the backend by calling
   `set_backend_from_args`.

You can check whether a backend is in use with the `using_backend`
function.
"""


is_distributed = None
"""Whether we are distributed."""
backend = None
"""Backend in usage."""


def require_set_backend():
    """Raise an `AssertionError` when the backend has not been set."""
    assert backend is not None, (
        "distributed backend is not set. Please call "
        "`distributed_utils.set_backend_from_args` at the start of your script"
    )


def using_backend(test_backend):
    """Return whether the backend is set to `test_backend`.

    `test_backend` may be a string of the name of the backend or
    its class.
    """
    require_set_backend()
    if isinstance(test_backend, str):
        return backend.BACKEND_NAME == test_backend
    return isinstance(backend, test_backend)
