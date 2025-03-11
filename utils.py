import tensorflow as tf
import numpy as np


def are_consent(pairwise_tensor1, pairwise_tensor2, max_dist):
  return not tf.reduce_any(tf.abs(pairwise_tensor1-pairwise_tensor2) > max_dist)


def CPVs(PC_tensor, criteria: list):
    PCM = normalization(PC_tensor)
    priorities = {}
    n = PCM.shape[0]
    sum_tab = tf.reduce_sum(PCM, axis = 1)
    for i in range(n):
        priorities[criteria[i]] = sum_tab[i] / n

    return priorities


def normalization(PC_tensor):
    return PC_tensor/tf.reduce_sum(PC_tensor, axis = 0)

def distance(pairwise_tensor1, pairwise_tensor2):
  return tf.linalg.norm(pairwise_tensor1-pairwise_tensor2)