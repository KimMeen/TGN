# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 14:16:46 2020

@author: Ming Jin
"""


from collections import defaultdict
import torch
import numpy as np


class MessageAggregator(torch.nn.Module):
  """
  Abstract class for the message aggregator module
  """
  def __init__(self, device):
    super(MessageAggregator, self).__init__()
    self.device = device

  def aggregate(self, node_ids, messages):
      """
      Aggregate functions to be implemented
      """
      
  # def group_by_id(self, node_ids, messages, timestamps):
  #   """
  #   NOT HAS BEEN USED
  #   """
  #   node_id_to_messages = defaultdict(list)

  #   for i, node_id in enumerate(node_ids):
  #     node_id_to_messages[node_id].append((messages[i], timestamps[i]))

  #   return node_id_to_messages


class LastMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(LastMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """
    Given a list of node ids in a batch and associated messages m_i(t), aggregate different
    messages for the same id using the lastest message.
    
    
    INPUT: 
        node_ids: A list of node ids of length batch_size
        messages: A dictionary {node_id:[([message_1], timestamp_1), ([message_2], timestamp_2), ...]}
        
        P.S. timestamps: A tensor of shape [batch_size]
    
    OUTPUT:
        to_update_node_ids: A list of unique node ids
        unique_messages: A tensor of shape [unique_node_ids, aggregated message]
        unique_timestamps: A tensor contains corresponding timestamp for those aggregated messages
    
    
    EXAMPLE:
            node_ids = [1,2,2,3]
            messages = {1: [(tensor([1., 2., 3., 4., 5.]), tensor(1))], 
                        2: [(tensor([2., 3., 4., 5., 6.]), tensor(1)), (tensor([3., 4., 5., 6., 7.]), tensor(2))], 
                        3: [(tensor([4., 5., 6., 7., 8.]), tensor(2))]}
        
        ==>
            to_update_node_ids: [1, 2, 3]
            
            unique_messages: tensor([[1., 2., 3., 4., 5.],
                                     [3., 4., 5., 6., 7.],
                                     [4., 5., 6., 7., 8.]])
            
            unique_timestamps: tensor([1, 2, 2])
    
    """
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []
    
    to_update_node_ids = []
    
    for node_id in unique_node_ids:
        if len(messages[node_id]) > 0:
            to_update_node_ids.append(node_id)
            unique_messages.append(messages[node_id][-1][0])
            unique_timestamps.append(messages[node_id][-1][1])
    
    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps


class MeanMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(MeanMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """
    Given a list of node ids in a batch and associated messages m_i(t_j), aggregate different
    messages for the same id by averaging them.
    
    
    INPUT: 
        node_ids: A list of node ids of length batch_size
        messages: A dictionary {node_id:[([message_1], timestamp_1), ([message_2], timestamp_2), ...]}
        
        P.S. timestamps: A tensor of shape [batch_size]
    
    OUTPUT:
        to_update_node_ids: A list of unique node ids
        unique_messages: A tensor of shape [unique_node_ids, aggregated message]
        unique_timestamps: A tensor contains corresponding timestamp for those aggregated messages
    
    
    EXAMPLE:
            node_ids = [1,2,2,3]
            messages = {1: [(tensor([1., 2., 3., 4., 5.]), tensor(1))], 
                        2: [(tensor([2., 3., 4., 5., 6.]), tensor(1)), (tensor([3., 4., 5., 6., 7.]), tensor(2))], 
                        3: [(tensor([4., 5., 6., 7., 8.]), tensor(2))]}
        
        ==>
            to_update_node_ids: [1, 2, 3]
            
            unique_messages: tensor([[1.0000, 2.0000, 3.0000, 4.0000, 5.0000],
                                     [2.5000, 3.5000, 4.5000, 5.5000, 6.5000],
                                     [4.0000, 5.0000, 6.0000, 7.0000, 8.0000]])
            
            unique_timestamps: tensor([1, 2, 2])
    
    """
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []
    n_messages = 0

    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        n_messages += len(messages[node_id])
        to_update_node_ids.append(node_id)
        unique_messages.append(torch.mean(torch.stack([m[0] for m in messages[node_id]]), dim=0))  # This is the difference
        unique_timestamps.append(messages[node_id][-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps


'''
###############
REFERENCE ENTRY
###############
'''
def get_message_aggregator(aggregator_type, device):
  if aggregator_type == "last":
    return LastMessageAggregator(device=device)
  elif aggregator_type == "mean":
    return MeanMessageAggregator(device=device)
  else:
    raise ValueError("Message aggregator {} not implemented".format(aggregator_type))