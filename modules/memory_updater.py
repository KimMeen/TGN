# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:51:23 2020

@author: Ming Jin
"""

from torch import nn
import torch


class MemoryUpdater(nn.Module):
  """
  Abstract class for updating node memory
  """
  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    pass

class SequenceMemoryUpdater(MemoryUpdater):
  """
  RNN based memory updater. 
  Node's memory as the hidden state of RNN, aggregated message as the input, updated memory as the new hidden state
  
  INPUT:
      memory: Memory instance
      message_dimension: The dim of the message
      memory_dimension: The dim of the memory
      ---------------------------------------
      unique_node_ids: A list of unique node ids
      unique_messages: A tensor of shape [unique_node_ids, aggregated message]
      timestamps: A tensor contains corresponding timestamp for those aggregated messages

  OUTPUT:
      update_memory(): There is no output, we update node's memory by referring set_memory() method
      
      updated_memory: A tensor of shape [unique_nodes, memory_dimension]
      updated_last_update: A tensor of shape [unique_nodes]
      
  EXAMPLE (last_aggregator + memory_updator):
          node_ids = [0, 1, 1, 2]
          messages = {0: [(tensor([1., 2., 3., 4., 5.]), tensor(1))], 
                      1: [(tensor([2., 3., 4., 5., 6.]), tensor(1)), (tensor([3., 4., 5., 6., 7.]), tensor(2))], 
                      2: [(tensor([4., 5., 6., 7., 8.]), tensor(2))]}
        
      ==>
          to_update_node_ids: [0, 1, 2]
                
          unique_messages: tensor([[1., 2., 3., 4., 5.],
                                    [3., 4., 5., 6., 7.],
                                    [4., 5., 6., 7., 8.]])
                
          unique_timestamps: tensor([1., 2., 2.])
          
          ==>
          
              updated_memory: tensor([[-0.8096],
                                      [-0.9309],
                                      [-0.9771]], grad_fn=<IndexPutBackward>)
              
              updated_last_update: tensor([1., 2., 2.])    
  """
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(SequenceMemoryUpdater, self).__init__()
    self.memory = memory
    self.layer_norm = torch.nn.LayerNorm(memory_dimension)
    self.message_dimension = message_dimension
    self.device = device

  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    """
    Linked with Memory instance to update node's memory
    """
    if len(unique_node_ids) <= 0:  # This will happen at the very begining if memory_update_at_start
      return

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update memory to time in the past"

    memory = self.memory.get_memory(unique_node_ids)  # get the memory of these nodes
    self.memory.last_update[unique_node_ids] = timestamps  # set the last update timestamp of node ids

    # E.g. S_1(t1) <-- aggregated message m^line_1(t1) + previous memory S_1(t0)
    updated_memory = self.memory_updater(unique_messages, memory)

    self.memory.set_memory(unique_node_ids, updated_memory)

  def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
    """
    Detached from Memory instance to update node's memory and then return updated_memory and updated_last_update
    """
    if len(unique_node_ids) <= 0:  # This will happen at the very begining if memory_update_at_start
      return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update memory to time in the past"

    updated_memory = self.memory.memory.data.clone()
    updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])

    updated_last_update = self.memory.last_update.data.clone()
    updated_last_update[unique_node_ids] = timestamps

    return updated_memory, updated_last_update

class GRUMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(GRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

    self.memory_updater = nn.GRUCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)
    
class RNNMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(RNNMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

    self.memory_updater = nn.RNNCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)
    
'''
###############
REFERENCE ENTRY
###############
'''
def get_memory_updater(module_type, memory, message_dimension, memory_dimension, device):
  if module_type == "gru":
    return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
  elif module_type == "rnn":
    return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device)