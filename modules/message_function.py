# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 14:04:31 2020

@author: Ming Jin
"""

from torch import nn


class MessageFunction(nn.Module):
  """
  Abstract class
  
  Module which computes the message for a given interaction.
  """

  def compute_message(self, raw_messages):
    return None


class MLPMessageFunction(MessageFunction):
  """
  MLP message function to calculate the message m(t)
  
  INPUT:
      raw_message_dimension: Dimension of the raw_message
      message_dimension: Dimension of the message
      raw_messages: [S_i(t-1) || S_j(t-1) || delta_t || e(t)] for interaction events
      
  OUTPUT:
      message: m(t) <-- [S_i(t-1) || S_j(t-1) || delta_t || e(t)] for interation events
  
  """
  def __init__(self, raw_message_dimension, message_dimension):
    super(MLPMessageFunction, self).__init__()

    self.mlp = self.layers = nn.Sequential(
      nn.Linear(raw_message_dimension, raw_message_dimension // 2),
      nn.ReLU(),
      nn.Linear(raw_message_dimension // 2, message_dimension),
    )

  def compute_message(self, raw_messages):
    messages = self.mlp(raw_messages)

    return messages


class IdentityMessageFunction(MessageFunction):
  """
   message function returns m(t) = raw_message
  
  """

  def compute_message(self, raw_messages):

    return raw_messages


'''
###############
REFERENCE ENTRY
###############
'''
def get_message_function(module_type, raw_message_dimension, message_dimension):
  if module_type == "mlp":
    return MLPMessageFunction(raw_message_dimension, message_dimension)
  elif module_type == "identity":
    return IdentityMessageFunction()
  else:
    raise ValueError("Message function {} not implemented".format(module_type))