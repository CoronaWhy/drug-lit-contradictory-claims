# -*- coding: utf-8 -*-

"""A package for finding contradictory claims related to COVID-19 drug treatments in the CORD-19 literature."""

from .data.make_dataset import *  # noqa:F401,F403
#  from .models import train_model  # noqa:F401
from .models.train_model import *  # noqa:F401,F403
from .models.sbert_models import train_sbert_model  # noqa:F401
