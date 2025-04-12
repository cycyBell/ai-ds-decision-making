import tensorflow as tf
import numpy as np
import time

from utils import are_consent,CPVs,normalization

def update_cons_tf(tensor1, tensor2, cons_matrix):

    # Let's create two masks. The first will be True for the upper triangular part of the tensor
    # and the latter will be True for lower part
    mask1 = tf.cast(tf.linalg.band_part(tf.ones_like(cons_matrix), 0, -1), dtype = tf.bool)  # Upper triangular mask (i < j)
    mask2 = tf.cast(tf.linalg.band_part(tf.ones_like(cons_matrix), -1, 0), dtype = tf.bool) # Lower part mask
    condition = tf.abs(cons_matrix) > tf.abs(tf.transpose(cons_matrix))

    #updating tensor1 and tensor2 based on the condition
    tensor1_new = tf.where(condition, tensor1 - cons_matrix, tensor1)
    tensor2_new = tf.where(condition, tensor2 + cons_matrix, tensor2)
    tensor1_new = tf.where((mask1 & condition) | (mask2 & condition), tensor1_new, tf.where((mask1 ^ condition) & (mask2 ^ condition), tensor1_new, 1.0/tf.transpose(tensor1_new)))
    tensor2_new = tf.where((mask1 & condition) | (mask2 & condition), tensor2_new, tf.where((mask1 ^ condition) & (mask2 ^ condition), tensor2_new, 1.0/tf.transpose(tensor2_new)))

    return tensor1_new, tensor2_new

def get_distance_matrix(criteria_comparison_data, DM_number):

    # Calculate pairwise distances for all pairs using broadcasting

    DMs = list(criteria_comparison_data.keys())
    pairwise_arrays = [criteria_comparison_data[DM] for DM in DMs]
    pairwise_tensor1 = tf.stack([tf.stack(pairwise_arrays, axis = 0)]*DM_number, axis = 0)
    pairwise_tensor2 = tf.stack([tf.stack([t]*DM_number, axis = 0) for t in pairwise_arrays], axis = 0)

    diff = pairwise_tensor1 - pairwise_tensor2  # Shape: (DM_number, DM_number, tensor_shape)
    distance_matrix = tf.linalg.norm(diff, ord='fro', axis=[-2, -1])   # Shape: (DM_number, DM_number)


    return distance_matrix


def consensus_operation(criteria_comparison_data, consensus_level, DMs, index_max_dist):
  consensus_index = consensus_level/2
  i,j = index_max_dist
  dm1 = DMs[i]
  dm2 = DMs[j]
  num_cr = criteria_comparison_data[dm1].shape[0]
  tensor1 = tf.Variable(tf.zeros((num_cr,num_cr)))
  tensor2 = tf.Variable(tf.zeros((num_cr,num_cr)))
  tensor1.assign(criteria_comparison_data[dm1])
  tensor2.assign(criteria_comparison_data[dm2])

  consensus_matrix = (tensor1 - tensor2)*consensus_index
  return update_cons_tf(tensor1, tensor2, consensus_matrix)

def criteria_priority_values(criteria_data ,criteria, DMs):
  priorities = list()
  for dm in DMs:
    normalized_PCM = normalization(criteria_data[dm])
    priorities.append(CPVs(normalized_PCM, criteria))
  return priorities

def consensus_training(criteria_comparison_data, consensus_level,scale_value, DM_number, epochs):
  max_allowed_dist = (scale_value-1/scale_value)*(1-consensus_level)
  DMs = list(criteria_comparison_data.keys())
  n = 0
  cons_loss = []
  while True:
    start_time = time.time()
    n += 1
    distance_matrix = get_distance_matrix(criteria_comparison_data, DM_number)
    index_max_dist = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
    i, j = index_max_dist
    dm1 = DMs[i]
    dm2 = DMs[j]
    tensor1 = criteria_comparison_data[dm1]
    tensor2 = criteria_comparison_data[dm2]
    if are_consent(tensor1, tensor2,max_allowed_dist):
      return cons_loss
    t1, t2 = consensus_operation(criteria_comparison_data, consensus_level, DMs,index_max_dist)
    tensor1.assign(t1)
    tensor2.assign(t2)
    end_time = time.time()
    exec_time = np.round(end_time - start_time, 2)
    dist = tf.reduce_sum(distance_matrix).numpy()/2
    cons_loss.append(dist)
    print("epoch : {}  time : {}s  distance : {}".format(n,exec_time,dist))
    if n == epochs:
      return cons_loss