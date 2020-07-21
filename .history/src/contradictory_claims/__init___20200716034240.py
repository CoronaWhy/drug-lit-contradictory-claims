# -*- coding: utf-8 -*-

"""A package for finding contradictory claims related to COVID-19 drug treatments in the CORD-19 literature."""

from .data.make_dataset import *  # noqa:F401,F403
from .models import train_model  # noqa:F401
from .claims  import extract_claims
