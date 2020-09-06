# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:52:24 2020

@author: Ming Jin
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def preprocess(data_name):
    """
    u: Users
    i: Items
    ts: Timestamps
    label: Dynamic node labels
    feat: Interaction features
    
    See JODIE paper for deatials about the meaning of user and item
    """
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []
    
    with open(data_name) as f:
        
        s = next(f)
        
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])
            ts = float(e[2])
            label = float(e[3])
            feat = np.array([float(x) for x in e[4:]])
        
            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)
            feat_l.append(feat)
        
    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list}), np.array(feat_l)


def reindex(df, bipartite=True):
    """
    Treat users and items as "nodes", their interactions as "temporal edges"
    Specifically, users are "source nodes", and items are "destination nodes" in a bipartite graph
    
    df looks like this:
        
               u    i         ts  label     idx
    0          0    0        0.0    0.0       0
    1          1    1       36.0    0.0       1
    2          1    1       77.0    0.0       2
    3          2    2      131.0    0.0       3
    4          1    1      150.0    0.0       4
    ...      ...  ...        ...    ...     ...
    157469  2003  632  2678155.0    0.0  157469
    157470  3762  798  2678158.0    0.0  157470
    157471  2399  495  2678293.0    0.0  157471
    157472  7479  920  2678333.0    0.0  157472
    157473  2399  495  2678373.0    0.0  157473    
    
    new_df looks like this:
               u     i         ts  label     idx
    0          1  8228        0.0    0.0       1
    1          2  8229       36.0    0.0       2
    2          2  8229       77.0    0.0       3
    3          3  8230      131.0    0.0       4
    4          2  8229      150.0    0.0       5
    ...      ...   ...        ...    ...     ...
    157469  2004  8860  2678155.0    0.0  157470
    157470  3763  9026  2678158.0    0.0  157471
    157471  2400  8723  2678293.0    0.0  157472
    157472  7480  9148  2678333.0    0.0  157473
    157473  2400  8723  2678373.0    0.0  157474    
    """
    new_df = df.copy()
    
    if bipartite:
        
        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))
      
        upper_u = df.u.max() + 1  # last source node index
        new_i = df.i + upper_u  # create dest node indeies after source nodes
      
        new_df.i = new_i  # dest node indeies
        new_df.u += 1  # source node indeies (start from 1)
        new_df.i += 1  
        new_df.idx += 1
        
    else:

        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
    
    return new_df

def run(data_name, bipartite=True):
  Path("data/").mkdir(parents=True, exist_ok=True)
  PATH = './data/{}.csv'.format(data_name)
  OUT_DF = './data/ml_{}.csv'.format(data_name)
  OUT_FEAT = './data/ml_{}.npy'.format(data_name)
  OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)

  df, feat = preprocess(PATH)  # get the interaction feature vectors and a dataframe which contains index, u, i, ts, label
  new_df = reindex(df, bipartite)  # get bipartite version of df
  
  empty = np.zeros(feat.shape[1])[np.newaxis, :]  # with shape [1, feat_dim]
  feat = np.vstack([empty, feat])  # with shape [interactions, feat_dim]

  max_idx = max(new_df.u.max(), new_df.i.max())  # number of nodes
  rand_feat = np.zeros((max_idx + 1, 172))  # initialize node features with fixed 172 dimension size

  new_df.to_csv(OUT_DF)  # temporal bipartite interaction graph
  np.save(OUT_FEAT, feat)  # initial node features
  np.save(OUT_NODE_FEAT, rand_feat)  # interaction (i.e. Temporal edge) features

### Entry
parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)', default='wikipedia')

args = parser.parse_args()
run(args.data)