from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.util import nest
from tensorflow.contrib.distributions import Bernoulli
from tensorflow.contrib.layers import fully_connected
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple

import numpy as np
import copy
from collections import deque


def rnn_with_feed_prev(cell, inputs, is_training, config, initial_state=None):
    prev = None
    outputs = []
    state_list = []

    sample_prob = config.sample_prob # scheduled sampling probability

    is_sample = is_training and initial_state is not None # whether to use scheduled sampling  
 
    with tf.variable_scope("rnn") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        inputs_shape = inputs.get_shape().with_rank_at_least(3)
        batch_size = tf.shape(inputs)[0] 
        num_steps = inputs_shape[1]
        # input_size = int(inputs_shape[2])
        input_size = 1
        inp_steps = config.inp_steps
        output_size = cell.output_size
        # state_list = np.zeros(shape=(batch_size, num_steps-1, config.hidden_size),dtype=np.float32)

        # phased lstm input
        inp_t = tf.expand_dims(tf.range(1, batch_size+1), 1)

        dist = Bernoulli(probs=config.sample_prob)
        samples = dist.sample(sample_shape=num_steps)
        # with tf.Session() as sess:
        #     print('bernoulli',samples.eval())
        if initial_state is None:
            initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        state = initial_state



        for time_step in range(num_steps):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()

            inp = inputs[:, time_step, :]

            # print(inp[:,1].shape)
            
            if is_sample and time_step > 0: 
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    # inp = tf.cond(tf.cast(samples[time_step], tf.bool), lambda:tf.identity(inp), \
                    #    lambda:fully_connected(cell_output, input_size, activation_fn=tf.tanh))

                    inp = tf.cond(tf.cast(samples[time_step], tf.bool), lambda:tf.identity(inp),
                                  lambda:tf.concat([fully_connected(cell_output, input_size, activation_fn=tf.tanh),
                                                    tf.expand_dims(inp[:,1],1)],1))


                    # print(inp.get_shape())
                    
                    
            if not is_training and prev is not None  :
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    # inp = fully_connected(prev, input_size,  activation_fn=tf.tanh)
                    inp = tf.concat([fully_connected(prev, input_size, activation_fn=tf.tanh),
                                     tf.expand_dims(inp[:,1],1)],1)
                    #print("t", time_step, ">=", inp_steps, "--> feeding back output into input.")


            if isinstance(cell._cells[0], tf.contrib.rnn.PhasedLSTMCell):
                (cell_output, state) = cell((inp_t, inp), state)
            else:

                # with tf.variable_scope('fully_connected2') as scope:
                    # if time_step>0:
                    #     scope.reuse_variables()
                    # temp = tf.expand_dims(inp[:,1],1)
                    # inp = fully_connected(tf.expand_dims(inp[:,0],1),config.encodernum,activation_fn=tf.tanh)
                    # inp = tf.concat([inp,temp],axis=1)


                (cell_output, state) = cell(inp, state)
                state_list.append(cell_output)


            prev = cell_output

            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                output = fully_connected(cell_output, input_size, activation_fn=tf.tanh)
                outputs.append(output)
    outputs = tf.stack(outputs, 1)
    # print(state_list[0].shape)\

    # print(tf.get_variable_scope())
    return outputs, state, state_list

def _shift (input_list, new_item):
    """Update lag number of states"""
    output_list = copy.copy(input_list)
    output_list = deque(output_list)
    output_list.append(new_item) # deque = [1, 2, 3]
    output_list.popleft() # deque =[2, 3]
    return output_list

def _list_to_states(states_list):
    """Transform a list of state tuples into an augmented tuple state
    customizable function, depends on how long history is used"""
    num_layers = len(states_list[0])# state = (layer1, layer2...), layer1 = (c,h), c = tensor(batch_size, num_steps)
    output_states = ()
    for layer in range(num_layers):
        output_state = ()
        for states in states_list:
                #c,h = states[layer] for LSTM
                output_state += (states[layer],)
        output_states += (output_state,)
        # new cell has s*num_lags states
    return output_states

def tensor_rnn_with_feed_prev(cell, inputs, is_training, config, initial_states=None):
    """High Order Recurrent Neural Network Layer
    """
    #tuple of 2-d tensor (batch_size, s)
    outputs = []
    prev = None
    is_sample = is_training and initial_states is not None

    with tf.variable_scope("trnn") as varscope:
        if varscope.caching_device is None:
                    varscope.set_caching_device(lambda op: op.device)

        inputs_shape = inputs.get_shape().with_rank_at_least(3)
        batch_size = tf.shape(inputs)[0]
        num_steps = inputs_shape[1]
        # input_size = int(inputs_shape[2])
        input_size =1
        output_size = cell.output_size
        inp_steps =  config.inp_steps

        # Scheduled sampling
        dist = Bernoulli(probs=config.sample_prob)
        samples = dist.sample(sample_shape=num_steps)

        if initial_states is None:
            initial_states =[]
            for lag in range(config.num_lags):
                initial_state =  cell.zero_state(batch_size, dtype= tf.float32)
                initial_states.append(initial_state)

        states_list = initial_states #list of high order states

        for time_step in range(num_steps):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()

            inp = inputs[:, time_step, :]

            if is_sample and time_step > 0:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                #     inp = tf.cond(tf.cast(samples[time_step], tf.bool),  lambda:tf.identity(inp) , \
                #        lambda:fully_connected(cell_output, input_size, activation_fn=tf.sigmoid))
                    inp = tf.cond(tf.cast(samples[time_step], tf.bool), lambda: tf.identity(inp),
                                  lambda: tf.concat([fully_connected(cell_output, input_size, activation_fn=tf.tanh),
                                                     tf.expand_dims(inp[:, 1], 1)], 1))

            if not is_training and prev is not None and time_step >= inp_steps:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                #     inp = fully_connected(cell_output, input_size, activation_fn=tf.sigmoid)
                    #print("t", time_step, ">=", burn_in_steps, "--> feeding back output into input.")
                    inp = tf.concat([fully_connected(prev, input_size, activation_fn=tf.tanh),
                                 tf.expand_dims(inp[:, 1], 1)], 1)

            states = _list_to_states(states_list)
            """input tensor is [batch_size, num_steps, input_size]"""
            (cell_output, state)=cell(inp, states)

            states_list = _shift(states_list, state)

            prev = cell_output
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                output = fully_connected(cell_output, input_size, activation_fn=tf.sigmoid)
                outputs.append(output)

    outputs = tf.stack(outputs,1)
    return outputs, states_list



