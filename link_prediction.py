# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 16:31:51 2020

@author: Ming Jin

TGN for link prediction
"""

import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path

from utils.utils import eval_edge_prediction
from net.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics

torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)', default='wikipedia')
parser.add_argument('--batch', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--neighbors', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs for this script')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to backprop')
parser.add_argument('--use_memory', action='store_true', help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=["graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=["mlp", "identity"], help='Type of message function')
parser.add_argument('--aggregator', type=str, default="last", choices=["last", "mean"], help='Type of message aggregator')
parser.add_argument('--memory_update_at_start', action='store_true', help='Whether to update memory at start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for each node')
parser.add_argument('--different_new_nodes', action='store_true', help='Whether val and test set use different unseen nodes (to test inductiveness)')
parser.add_argument('--uniform', action='store_true', help='take random sampling from temporal neighbors')
         
try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

BATCH_SIZE = args.batch
NUM_NEIGHBORS = args.neighbors
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Extract data for training, validation and testing
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data = get_data(DATA, different_new_nodes_between_val_and_test=args.different_new_nodes)

# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# Initialize negative samplers
# Set seeds for validation and testing so negatives are the same across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=3)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

############################### START #################################
for i in range(args.n_runs):
      
    results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
    Path("results/").mkdir(parents=True, exist_ok=True)

    # Initialize Model
    tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
              edge_features=edge_features, device=device,
              n_layers=NUM_LAYER,
              n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
              message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
              memory_update_at_start=args.memory_update_at_start,
              embedding_module_type=args.embedding_module,
              message_function=args.message_function,
              aggregator_type=args.aggregator, n_neighbors=NUM_NEIGHBORS,
              mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
              mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst)
    
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
    tgn = tgn.to(device)
    
    num_instance = len(train_data.sources)  # n_interactions
    num_batch = math.ceil(num_instance / BATCH_SIZE)
    
    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)
    
    new_nodes_val_aps = []
    val_aps = []
    epoch_times = []
    total_epoch_times = []
    train_losses = []
    
    early_stopper = EarlyStopMonitor(max_round=args.patience)
    
    ###################### TRAINING ###################
    for epoch in range(NUM_EPOCH):
        
        start_epoch = time.time()
      
        # Reinitialize memory of the model at the start of each epoch
        if USE_MEMORY:
          tgn.memory.__init_memory__()
      
        # Train using only training graph
        tgn.set_neighbor_finder(train_ngh_finder)
        m_loss = []
      
        logger.info('start {} epoch'.format(epoch))
        
        ### Start to train on this epoch
        for k in range(0, num_batch, args.backprop_every):
            
            loss = 0
            optimizer.zero_grad()
        
            # Custom loop to allow to perform backpropagation only every a certain number of batches
            for j in range(args.backprop_every):
                
                batch_idx = k + j
          
                if batch_idx >= num_batch:
                  continue
                
                # get a src and dest node training batch
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(num_instance, start_idx + BATCH_SIZE)
                sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                                    train_data.destinations[start_idx:end_idx]
                
                # as well as the edge and timestamps for this batch                                    
                edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
                timestamps_batch = train_data.timestamps[start_idx:end_idx]
                
                # sample batch_size dest negatives (nodes)
                size = len(sources_batch)
                _, negatives_batch = train_rand_sampler.sample(size)
                
                # self-supervised labels setting
                with torch.no_grad():
                  pos_label = torch.ones(size, dtype=torch.float, device=device)
                  neg_label = torch.zeros(size, dtype=torch.float, device=device)
                
                # forward propagation
                tgn = tgn.train()
                pos_prob, neg_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch,
                                                                    timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)
          
                loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)
            
            # backward propagation
            loss /= args.backprop_every
            loss.backward()
            optimizer.step()
            m_loss.append(loss.item())
        
            ### Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to the start of time
            # If we don't do this, the "Trying backpropagate but buffers haven been freed" issue will occur because:
            # 1). For mem_update_at_end: Memory updated at the end may contain this batch information that loss will not cover,
            #     so we have to detach to ensure the memory has the information that loss has covered to backpropagate.
            # 2). For mem_update_at_start: We don't have the issue on (1) but some node messages may be removed after update_memory
            #     so we may try to backpropagate on those freed messages.
            if USE_MEMORY:
                tgn.memory.detach_memory()
      
        epoch_time = time.time() - start_epoch
        epoch_times.append(epoch_time)
      
        ####################### VALIDATION ######################
        # Validation uses the full graph
        tgn.set_neighbor_finder(full_ngh_finder)
      
        if USE_MEMORY:
            # Backup memory at the end of training, so later we can restore it and use it for the
            # validation on unseen nodes (since validation edges are strictly later in time than training edges)
            train_memory_backup = tgn.memory.backup_memory()
      
        val_ap, val_auc = eval_edge_prediction(model=tgn, negative_edge_sampler=val_rand_sampler, 
                                               data=val_data, n_neighbors=NUM_NEIGHBORS)
        if USE_MEMORY:
            # Backup memory after validation so it can be used for testing (since test edges are
            # strictly later in time than validation edges)
            val_memory_backup = tgn.memory.backup_memory()
            # Restore memory we had at the end of training to be used when validating on unseen nodes.
            tgn.memory.restore_memory(train_memory_backup)
      
        # Validate on unseen nodes
        nn_val_ap, nn_val_auc = eval_edge_prediction(model=tgn, negative_edge_sampler=val_rand_sampler, 
                                                     data=new_node_val_data, n_neighbors=NUM_NEIGHBORS)
      
        if USE_MEMORY:
            # Restore memory we had at the end of validation to get ready testing if:
            # 1). This is last epoch
            # 2). Early stopping happen on this epoch
            tgn.memory.restore_memory(val_memory_backup)
      
        new_nodes_val_aps.append(nn_val_ap)
        val_aps.append(val_ap)
        train_losses.append(np.mean(m_loss))
      
        # Save temporary results to disk
        pickle.dump({
          "val_aps": val_aps,
          "new_nodes_val_aps": new_nodes_val_aps,
          "train_losses": train_losses,
          "epoch_times": epoch_times,
          "total_epoch_times": total_epoch_times
        }, open(results_path, "wb"))
      
        total_epoch_time = time.time() - start_epoch
        total_epoch_times.append(total_epoch_time)
      
        logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
        logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info(
          'transductive val auc: {}, inductive val auc: {}'.format(val_auc, nn_val_auc))
        logger.info(
          'transductive val ap: {}, inductive val ap: {}'.format(val_ap, nn_val_ap))
      
        # Early stopping
        if early_stopper.early_stop_check(val_ap):
          logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
          logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
          best_model_path = get_checkpoint_path(early_stopper.best_epoch)
          tgn.load_state_dict(torch.load(best_model_path))
          logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
          tgn.eval()
          break
        else:
          torch.save(tgn.state_dict(), get_checkpoint_path(epoch))
          
    ############################ TESTING ####################################
    # Training has finished, we have loaded the best model, and we want to backup its current
    # memory (which has seen validation edges) so that it can also be used when testing on unseen
    # nodes
    if USE_MEMORY:
      val_memory_backup = tgn.memory.backup_memory()
    
    ### Test
    tgn.embedding_module.neighbor_finder = full_ngh_finder
    test_ap, test_auc = eval_edge_prediction(model=tgn, negative_edge_sampler=test_rand_sampler, 
                                             data=test_data, n_neighbors=NUM_NEIGHBORS)
    
    if USE_MEMORY:
      tgn.memory.restore_memory(val_memory_backup)
    
    # Test on unseen nodes
    nn_test_ap, nn_test_auc = eval_edge_prediction(model=tgn, negative_edge_sampler=nn_test_rand_sampler, 
                                                   data=new_node_test_data, n_neighbors=NUM_NEIGHBORS)
    
    logger.info('Test statistics: Transductive -- auc: {}, ap: {}'.format(test_auc, test_ap))
    logger.info('Test statistics: Inductive -- auc: {}, ap: {}'.format(nn_test_auc, nn_test_ap))
    
    # Save results for this run
    pickle.dump({
      "val_aps": val_aps,
      "new_nodes_val_aps": new_nodes_val_aps,
      "test_ap": test_ap,
      "new_node_test_ap": nn_test_ap,
      "epoch_times": epoch_times,
      "train_losses": train_losses,
      "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))
    
    logger.info('Saving TGN model')
    
    if USE_MEMORY:
      # Restore memory at the end of validation (save a model which is ready for testing)
      tgn.memory.restore_memory(val_memory_backup)
      
    torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
    logger.info('TGN model saved')