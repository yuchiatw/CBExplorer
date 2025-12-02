"""
Package initialization for posthoc_generative_cbm.

This module exposes legacy short module names (e.g., `models`, `utils`,
`eval`) so that existing training/evaluation scripts that expect those
modules at the repository root continue to work without mutating
`sys.path` or changing the working directory.
"""

import sys

# Import subpackages so they can be aliased into sys.modules.
from . import models as _models  # noqa: F401
from . import utils as _utils  # noqa: F401
from . import eval as _eval  # noqa: F401
from . import torch_utils as _torch_utils  # noqa: F401
from . import dnnlib as _dnnlib  # noqa: F401

# Register aliases if they are not already set. This preserves backward
# compatibility for legacy imports such as `from models import ...`.
sys.modules.setdefault("models", _models)
sys.modules.setdefault("utils", _utils)
sys.modules.setdefault("eval", _eval)
sys.modules.setdefault("torch_utils", _torch_utils)
sys.modules.setdefault("dnnlib", _dnnlib)
