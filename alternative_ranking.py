import tensorflow as tf
import pandas as pd



def get_alternative_scores(pca_data, inferred_axes, priority_tensor):
  reduced_data = tf.matmul(tf.cast(pca_data,tf.float32), inferred_axes)
  inferred_priorities = tf.nn.softmax(tf.matmul(tf.transpose(inferred_axes), tf.expand_dims(priorities, axis = 1)), axis = 0)
  final_scores = tf.nn.softmax(tf.matmul(reduced_data, inferred_priorities),axis=0)
  return final_scores

def get_n_best_alternatives_dataframe(final_scores, n):
  indices = tf.argsort(final_scores, axis = 0, direction = "DESCENDING")
  best_n_alt = indices[:n]
  best_n_scores = tf.reshape(tf.gather(final_scores, best_n_alt),shape = (n,-1))
  data = {'Alternative': best_n_alt.numpy().flatten(), 'Score': best_n_scores.numpy().flatten()}
  ranking_df = pd.DataFrame(data, index = range(1,n+1))
  return ranking_df