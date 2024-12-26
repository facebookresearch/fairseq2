# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.jepa.classifier.factory import create_jepa_classifier_model as create_jepa_classifier_model
from fairseq2.models.jepa.classifier.factory import jepa_classifier_archs as jepa_classifier_archs
from fairseq2.models.jepa.classifier.factory import JEPA_CLASSIFIER_FAMILY as JEPA_CLASSIFIER_FAMILY
from fairseq2.models.jepa.classifier.factory import JepaClassifierConfig as JepaClassifierConfig
from fairseq2.models.jepa.classifier.factory import JepaClassifierBuilder as JepaClassifierBuilder
from fairseq2.models.jepa.classifier.model import JepaClassifierModel as JepaClassifierModel
from fairseq2.models.jepa.classifier.loader import load_jepa_classifier_model as load_jepa_classifier_model
