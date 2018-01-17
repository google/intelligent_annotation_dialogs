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
"""Environment for annotating data.

Following OpenAI gym conventions, create an environments that simulates
annotating image+class pairs.
- States are a combination of features of an image and a proposed box.
- Actions (0,1) correspond to (do box verification, do extreme clicking).
- Reward is negative time per iteration.
- Reward is 0 when annotation is obtained.
- Episode terminates when annotation for an image is obtained
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import pandas as pd


class AnnotatingDataset(object):
  """The environment that simulates image annotation by the annotator.

  There are 2 possible actions:
  - box verification and
  - manual drawing
  One of the actions is selected at every step of annotation process
  We follow the API of OpenAI here.
  """

  def __init__(self, annotator, detector, image_class):
    """Initialization of AnnotatingPASCAL environment.

    Args:
      annotator: instance of class Annotator
      detector: instance of class Detector
      image_class: image-level annotations (pairs images+classes to annotate),
          pandas dataframe with columns 'image_id' and 'class_id'
    """
    self.annotator = annotator
    self.detector = detector
    self.image_class = image_class
    # possible actions are: 0-box verification, 1-extreme clikcikng
    self._actions = [0, 1]
    self.action_space = gym.spaces.discrete.Discrete(len(self._actions))
    # observation space is the domain of feature values for the state
    # 5 numerical features with the indicates minimum and maximum values
    feat5 = gym.spaces.box.Box(np.array([0.0, 0.0, 0.0, -1.0, -1.0]),
                               np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    # 20 binary features that correspond to a categorical variable of class
    featcat = gym.spaces.MultiBinary(20)
    # observation space includes both of them
    self.observation_space = gym.spaces.Tuple((feat5, featcat))

  def reset(self, current_index=None):
    """Reset the environment to annotate a particular image+label pair.

    Args:
      current_index: if current_index is None, sample random image+class pair,
      if current_index is set (int), use it as index of image_class

    Returns:
      state: current state
    """

    # the steps count
    # for now steps is not used anywhere, but I tried experimenting with
    # 1) adding it as features
    # 2) stopping to long annotation rounds
    # It works sometimes, but needs more experimentation.
    self.steps = 0
    # the current_index can either indicate which image+label pair to use or
    # it can be None and then the pair is chosen at random
    if current_index is None:
      current_index = np.random.randint(0, len(self.image_class))
    self.current_image = self.image_class.iloc[current_index]['image_id']
    self.current_class = self.image_class.iloc[current_index]['class_id']
    # get all the box proposals with their coordinates
    # and features for the current image+class pair
    self.coordinates, self.features = self.detector.get_box_proposals(
        self.current_image, self.current_class)
    # keep a list of boxes that were rejected in this annotation round
    self.rejected_boxes = []
    # the index of the next box proposal to consider
    self.next_index = 0
    # set the current state
    self._compute_state()

    return self.state

  def _compute_state(self):
    """Compute the current state.

    For this, the function
    1) finds the next box: self.current_box
    2) gets its corresponding features
    3) returns the features of the selected box as the state

    Returns:
      current state
    """
    # if there are no more box proposals left, stop searching
    if len(self.coordinates) <= self.next_index:
      self.current_box = None
      # when we ran out ouf proposals, the state is set to a 0 vector
      self.state *= 0
    # if some proposals are found, pop them one by one
    else:
      # new potential box with coordinates and features
      new_coordinate = self.coordinates.iloc[self.next_index]
      new_feature = self.features.iloc[self.next_index]
      # the index of next box to consider is increased by 1
      self.next_index += 1

      # if some boxes were already rejected before,
      # we need to check if a new one doesn't overlap with them too much
      # this part can become much simpler if
      # 1) we do NMS at the beginning and
      # 2) we know th order of boxes to verify from the beginning
      # It is the case in out experiments because in the end we always use
      # the boxes in the order determined by the detector's score.
      if self.rejected_boxes:
        box_found = False
        # go through box proposals until a valid one is found
        while not box_found:

          # check if the max overlap with any of the rejected boxes is small
          # we convert rejected boxes to a dataframe to compatibility.
          # ideally, we would need to chamnge datatypes everywhere
          if np.amax(self.annotator.compute_iou_to_group(
              new_coordinate, pd.DataFrame(
                  self.rejected_boxes))) <= self.annotator.min_iou:

            # then this potential box becomes a real box to verify
            box_found = True
            self.current_box = new_coordinate
            self.state = new_feature
          # if the potential box is proned by the search space reduction
          else:
            # if some proposals are still left, get the next one
            if len(self.coordinates) > self.next_index:
              new_coordinate = self.coordinates.iloc[self.next_index]
              new_feature = self.features.iloc[self.next_index]
              self.next_index += 1
            # if we ran out of proposals, set the state to 0
            else:
              self.current_box = None
              self.state *= 0
              # and stop searching for a box
              break

      # if list of rejected boxes is empty
      else:
        self.current_box = new_coordinate
        self.state = new_feature

    return self.state

  def step(self, action):
    """Simulate one step of annotation.

    Args:
      action: 0 or 1 corresponding to the selected action

    Returns:
      state: new state
      reward: reward of performing action
      done: boolean variable indicating whether env is in terminal state
      coordinates: dictionary of the box that was positively verified or drawn
    """
    # boolean variable indicating if the environment reached terminal state
    done = False
    # coordinates of the aceepted or drawn bounding box
    coordinates = None
    # if requested action is box verification
    if action == 0:
      # if box proposal is available
      if self.current_box is not None:
        # run one box verification action
        iteration_time, is_accepted = self.annotator.do_box_verification(
            self.current_image, self.current_class, self.current_box)
        # if the box was rejected, save it for search space reduction
        if not is_accepted:
          self.rejected_boxes.append(self.current_box)
        # if the box is accepted, annotation is reached the terminal state
        if is_accepted:
          coordinates = {'xmin': int(self.current_box['xmin']),
                         'xmax': int(self.current_box['xmax']),
                         'ymin': int(self.current_box['ymin']),
                         'ymax': int(self.current_box['ymax'])}
          done = True
      else:
        # if no box porposals are available, do extreme clicking
        action = 1

    # if requested action is extreme clicking
    if action == 1:
      iteration_time, coordinates = self.annotator.do_extreme_clicking(
          self.current_image, self.current_class)
      # the terminal state is always reached after extreme clicking
      done = True

    # reward of the step is the negative duration of the step
    reward = -iteration_time
    # get the next state
    self._compute_state()
    # increase the counter of iterations
    self.steps += 1

    return self.state, reward, done, coordinates
