# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Module that implements FixedDialogs and IAD-prob dialog.

For details see 'Leaning Intelligent Dialogs for Bounding Box Annotation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import annotator


class FixedDialog(object):
  """Dialogs of the form V_D."""

  def __init__(self, num_verification_questions):
    """FixedDialog initialisation.

    Args:
      num_verification_questions: maximum number of verifications
          before manual drawing is done, int
      num_verification_questions=-1 leads to the always-verify agent.
    """
    self._num_v = num_verification_questions

  def get_next_action(self, state=None):
    """Gets the action to be executed.

    Args:
      state: None, keep for consistency with intelligent class (maybe remove?)

    Returns:
      0 or 1 depending if the chosen action is to verify or to draw
    """

    # keep the state argument here for consistency with IAD
    if self._num_v == 0:
      return annotator.Annotator.DRAW
    else:
      self._num_v -= 1
      return annotator.Annotator.VERIFY


# IAD-Prob agent
class DialogProb(object):
  """Intelligent annotation dialog IAD-Prob."""

  def __init__(self, model, annotator):
    """Initialisation of IAD-Prob agent.

    Args:
      model: sklearn classification model predicts the prob. of acceptance from state
      annotator: an instance of class Annotator,
          is needed fro the ratio between time_verify and time_draw
    """

    self.model = model
    self.threshold = annotator.time_verify/annotator.time_draw

  def get_next_action(self, state):
    """Gets the next action to execute.

    Args:
      state: features of the current bounding box under consideration
    Returns:
      next_action: next actions to execute (VERIFY or DRAW)
    """
    # if no box proposals are left, do extreme clicking
    if np.array_equal(state, state*0):
      next_action = annotator.Annotator.DRAW
    else:
      # model predicts the probability of acceptance
      prob_accept = self.model.predict_proba([state])[:, 1]
      if prob_accept < self.threshold:
        next_action = annotator.Annotator.DRAW
      else:
        next_action = annotator.Annotator.VERIFY

    return next_action
