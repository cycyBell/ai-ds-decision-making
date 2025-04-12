import tensorflow as tf
import matplotlib.pyplot as plt


def get_sensitivity_index(pca_dataset, inferred_axes, priority_tensor, delta):
  '''
  '''
  n = inferred_axes.shape[0]
  initializer = tf.keras.initializers.Identity()
  n_identity = initializer(shape=(n, n))
  n_priority_tensor = tf.reshape(tf.stack([priority_tensor]*n, axis = 1), shape = (n,-1))
  shifted_priorities = []
  for i in range(n):
      current_priorities = n_priority_tensor[:, i]  
      reduction_amount = delta * current_priorities[i] 
      current_priorities = tf.tensor_scatter_nd_sub(current_priorities, [[i]], [reduction_amount])
      other_indices = tf.constant([j for j in range(n) if j != i])
      distribution_amount = reduction_amount / (n - 1)  
      current_priorities = tf.tensor_scatter_nd_add(current_priorities, tf.expand_dims(other_indices, axis=-1), [distribution_amount] * (n - 1))
      
      shifted_priorities.append(current_priorities)

  n_priority_tensor = tf.transpose(tf.stack(shifted_priorities))
  inferred_priorities = tf.nn.softmax(tf.matmul(tf.transpose(inferred_axes), n_priority_tensor), axis = 0)
  reduced_data = tf.matmul(tf.cast(pca_dataset,tf.float32), inferred_axes)
  final_scores = tf.nn.softmax(tf.matmul(reduced_data, inferred_priorities),axis=0)
  sum_scores = tf.reduce_sum(final_scores, axis=1, keepdims=True)  # Keep the dimension
  sensitivity_index = tf.reduce_mean(tf.reduce_mean(tf.abs(-n * final_scores + sum_scores) / delta, axis=1), axis=0)
  return sensitivity_index

def get_sensitivity_graph(pca_dataset, inferred_axes, priority_tensor):
  deltas = np.arange(0.01, 1, 0.001)
  sensitivity_indices = []
  for delta in deltas:
    sensitivity_indices.append(get_sensitivity_index(pca_dataset, inferred_axes, priority_tensor, delta))
  plt.plot(deltas, sensitivity_indices)
  plt.xlabel("Delta")
  plt.ylabel("Sensitivity Index")
  plt.title("Model's Sensitivity Graph")
  plt.savefig("sensitivity_analysis")
  plt.show()
