import tensorflow as tf

batch_size = 4
input = tf.random_normal(shape=[batch_size, 3, 6], dtype=tf.float32)
cell = tf.nn.rnn_cell.BasicLSTMCell(10, forget_bias=1.0, state_is_tuple=True)
init_state = cell.zero_state(batch_size, dtype=tf.float32)
output, final_state = tf.nn.dynamic_rnn(cell, input, sequence_length=[1, 2, 3, 2], initial_state=init_state)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([output, final_state]))
