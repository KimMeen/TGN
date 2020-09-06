# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:32:19 2020

@author: Ming Jin
"""

import torch
import numpy as np


class TimeEncode(torch.nn.Module):
  """ 
  Time Encoding proposed by TGAT
  """
  def __init__(self, dimension):
    super(TimeEncode, self).__init__()

    self.dimension = dimension
    
    self.w = torch.nn.Linear(1, dimension)

    self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                       .float().reshape(dimension, -1))
    self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

  def forward(self, t):
    # t has shape [batch_size, seq_len], i.e. [source_nodes, num_temp_neighbors]
    # [batch_size, seq_len, 1], i.e. [source_nodes, num_temp_neighbors, 1]
    t = t.unsqueeze(dim=2)

    # output has shape [batch_size, seq_len, dimension], i.e. [source_nodes, num_temp_neighbors, dimension]
    output = torch.cos(self.w(t))

    return output