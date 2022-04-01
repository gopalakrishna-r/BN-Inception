import tensorflow as tf
from d2l import tensorflow as d2l
from pathlib import Path

from Inception import build_graph

lr, num_epochs, batch_size, resize = 0.1, 10, 32, 224

mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
# Divide all numbers by 255 so that all pixel values are between
# 0 and 1, add a batch dimension at the last. And cast label to int32
process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                        tf.cast(y, dtype='int32'))
resize_fn = lambda X, y: (
    tf.image.resize_with_pad(X, resize, resize) if resize else X, y)
train_iter, test_iter =  (
    tf.data.Dataset.from_tensor_slices(process(*mnist_train)).batch(
        batch_size).shuffle(len(mnist_train[0])).map(resize_fn),
    tf.data.Dataset.from_tensor_slices(process(*mnist_test)).batch(
        batch_size).map(resize_fn))


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=lr,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)


optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
net = build_graph()
net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
net.fit(train_iter, epochs=num_epochs, verbose=1, validation_data=test_iter)