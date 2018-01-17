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
"""Implementation of annotator.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import np_box_ops  # available from https://github.com/tensorflow/models/blob/master/research/object_detection/utils/np_box_ops.py


class Annotator(object):

  DRAW = 1
  VERIFY = 0
  """Annotator class."""

  def __init__(self, ground_truth, time_verify, time_draw, min_iou):
    """Basic class for Annotator.

    Args:
      ground_truth: image_id + class_id with coordinates of ground truth boxes,
                    pandas dataframe
      time_verify: time per box verification in seconds, float
      time_draw: time per drawing in seconds, float
      min_iou: minimal iou required for the box to be accepted, float
    """

    self.ground_truth = ground_truth
    self.time_verify = time_verify
    self.time_draw = time_draw
    self.min_iou = min_iou

  def compute_iou(self, box1, box2):
    """Copute intersection over union between 2 given boxes.

    Args:
      box1: first box in the form of pandas series with columns
           'xmin', 'xmax', 'ymin', 'ymax'
      box2: second box in the form of pandas series with columns
           'xmin', 'xmax', 'ymin', 'ymax'

    Returns:
      iou: IoU between box1 and box2
    """

    if (len(box1) != 1) or (len(box2) != 1):
      raise ValueError()

    box1_array = np.array([[box1['ymin'], box1['xmin'],
                            box1['ymax'], box1['xmax']]])
    box2_array = np.array([[box2['ymin'], box2['xmin'],
                            box2['ymax'], box2['xmax']]])

    iou = np_box_ops.iou(box1_array, box2_array)[0, 0]
    return iou

  def compute_iou_to_group(self, box1, boxes):

    """Compute intersection over union between 2 given boxes.

    Args:
     box1: first box in the form of pandas series with columns
           'xmin', 'xmax', 'ymin', 'ymax'
     boxes: second box in the form of pandas dataframe with columns
           'xmin', 'xmax', 'ymin', 'ymax'

    Returns:
      ious: the computed iou between box1 and boxes, numpy.ndarray 1xn where
            n is the number of rows in dataframe boxes
    """

    box1_array = np.array([[box1['ymin'], box1['xmin'],
                            box1['ymax'], box1['xmax']]])

    if np.shape(box1_array)[0] != 1:
      raise ValueError()

    boxes_array = boxes.as_matrix(columns=['ymin', 'xmin', 'ymax', 'xmax'])

    ious = np_box_ops.iou(box1_array, boxes_array)

    return ious


class AnnotatorSimple(Annotator):
  """Simple annotator model.

  It acceptes images with iou with ground truth highest than min_iou and
  rejects the others.
  """

  def __init__(self, ground_truth, random_seed, time_verify=3.5,
               time_draw=7, min_iou=0.7):
    Annotator.__init__(self, ground_truth, time_verify, time_draw, min_iou)
    self.random_seed = random_seed

  def do_box_verification(self, image_id, class_id, coord):
    """Do box verification.

    Args:
      image_id: image id for verification, str
      class_id: class id for verification, int64
      coord: coordinates of the box to verify in the form f pandas series
      {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1}

    Returns:
      time_verify: time that it took for operation, float
      is_accepted: boolean variable if it was accepted or not
    """

    is_accepted = False
    # find the relevant ground truth
    gt = self.ground_truth[(self.ground_truth['image_id'] == image_id) &
                           (self.ground_truth['class_id'] == class_id)]

    # try to rewrite:
    if np.amax(self.compute_iou_to_group(coord, gt)) >= self.min_iou:
      is_accepted = True

    return self.time_verify, is_accepted

  def do_extreme_clicking(self, image_id, class_id):
    """Do extreme clicking.

    Args:
      image_id: image id for verification, str
      class_id: class id for verification, int64

    Returns:
      time_draw: time that took for operation
      coordinates: dictionary of coordinates in the form
      {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1})
      for one of the boxes corresponding to gt
    """

    gt = self.ground_truth[(self.ground_truth['image_id'] == image_id) &
                           (self.ground_truth['class_id'] == class_id)]
    # if there are several boxes for this class in this image, get one at random
    selected_gt = gt.sample(1, random_state=self.random_seed)
    coordinates = {'xmin': int(selected_gt['xmin']),
                   'xmax': int(selected_gt['xmax']),
                   'ymin': int(selected_gt['ymin']),
                   'ymax': int(selected_gt['ymax'])}

    return self.time_draw, coordinates
