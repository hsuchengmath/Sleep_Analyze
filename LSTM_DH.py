import tensorflow as tf
import numpy as np
import pyprind

def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx

def LSTM_DH(X_train,Y_train,user_Id,dict_user_X_test):
    tf.set_random_seed(1)
    np.random.seed(1)

    # Hyper Parameters
    BATCH_SIZE = 64
    TIME_STEP = 8          # rnn time step / image height
    INPUT_SIZE = 14        # rnn input size / image width
    LR = 0.01               # learning rate
    hidden_state_dim = 14

            

    # tensorflow placeholders
    tf_x = tf.placeholder(tf.float32, [None, TIME_STEP , INPUT_SIZE])     
    tf_y = tf.placeholder(tf.float32, [None, 1])                             

    # RNN
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_state_dim)
    outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
        rnn_cell,                   # cell you have chosen
        tf_x,                      # input
        initial_state=None,         # the initial hidden state
        dtype=tf.float32,           # must given if set initial_state = None
        time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
    )
    Weights = tf.Variable(tf.random_normal([TIME_STEP,hidden_state_dim]),dtype=tf.float32)
    outputs2 = Weights*outputs
    outputs3 = tf.reduce_sum(outputs2,axis=1,name="latent_vector")
    output = tf.layers.dense(outputs3, 1)             # output based on the last output step
    output = tf.sigmoid(output)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=tf_y)

    train_op = tf.train.AdamOptimizer(LR).minimize(loss)

    
    sess = tf.Session()

    

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
    sess.run(init_op)     # initialize var in graph

    for step in pyprind.prog_bar(range(1000)):    # training

        mb_idx = sample_idx(X_train.shape[0], BATCH_SIZE)
        X_mb = X_train[mb_idx,:]
        Y_mb = Y_train[mb_idx]

        _, loss_ = sess.run([train_op, loss], {tf_x: X_mb, tf_y: Y_mb})
    
    latent_vector =  sess.graph.get_operation_by_name("latent_vector").outputs[0]
    latent_train_vector = sess.run(latent_vector,{tf_x: X_train})   #%

    dict_user_X_latent_vector = {}
    for i in range(len(user_Id)):
        user_X_latent_vector = []
        for j in range(dict_user_X_test[user_Id[i]].shape[0]):
            user_X_latent_vector.append(sess.run(latent_vector,{tf_x: dict_user_X_test[user_Id[i]][j]}))
        user_X_latent_vector = np.array(user_X_latent_vector)
        dict_user_X_latent_vector[user_Id[i]] = user_X_latent_vector   



    return latent_train_vector,dict_user_X_latent_vector
