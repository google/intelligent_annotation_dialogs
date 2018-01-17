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
"""The module that keeps and operates with detector's output.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Detector(object):
  """Keeps the detections and gets proposals for a given image-label pairs."""

  def __init__(self, detections, predictive_fields):
    """Initialisation of the detector.

    Args:
      detections: pandas dataframe with detections
      predictive_fields: list of features that will be used by IAD
    """

    self.predictive_fields = predictive_fields
    # We need to search in detection many times, sort them to make it faster
    self.detections = detections.sort_values('image_id')
    self.detections_nparray = self.detections.values[:, 0]

  def get_box_proposals(self, image_id, class_id):
    """Gets a list of proposals for a given image and class.

    Args:
      image_id: image id for verification, str
      class_id: class id for verification, int64

    Returns:
      coordinates: of all box proposals,
          pandas dataframe with columns 'xmin', 'xmax', 'ymin', 'ymax'
      features: corresponding to coordinates,
          pandas dataframe with columns stated in predictive fields
    """

    # as the images_id are sorted,
    # let's find the first and last occurrence of image_id
    # that will define the possible search range for proposals
    in1 = np.searchsorted(self.detections_nparray, image_id, side='left')
    in2 = np.searchsorted(self.detections_nparray, image_id, side='right')
    subset2search = self.detections.iloc[in1:in2]
    # now in this range find all class_ids
    all_proposals = subset2search[subset2search['class_id'] == class_id]

    all_proposals_sorted = all_proposals.sort_values(['prediction_score'],
                                                     ascending=0)

    # features of the boxes correcponding to image_id, class_id
    features = all_proposals_sorted[self.predictive_fields]
    coordinates = all_proposals_sorted[['xmin', 'xmax', 'ymin', 'ymax']]

    return coordinates, features
