# -*- coding: utf-8 -*-

"""Code for Identifying core claims from a paragraph."""

import pip._internal

from .extract_claims import *  # noqa:F401,F403
from .utils import *  # noqa:F401,F403

try:
    import discourse  # noqa:F401
except Exception:
    pip._internal.main(["install", "git+https://github.com/titipata/detecting-scientific-claim.git"])
    import discourse  # noqa:F401
