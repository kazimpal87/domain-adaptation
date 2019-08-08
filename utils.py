import tensorflow as tf

def make_dataset(X, y, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    return ds