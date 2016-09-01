'''
Created on 31 aug. 2016

@author: Robert-Jan
'''

import numpy as np;
from models.RecurrentModel import RecurrentModel
import tensorflow as tf;

class TensorflowRecurrentNeuralNetwork(RecurrentModel):
    '''
    classdocs
    '''


    def __init__(self, data_dim, hidden_dim, output_dim, minibatch_size, 
                 lstm=False, weight_values={}, single_digit=False, 
                 EOS_symbol_index=None, GO_symbol_index=None, n_max_digits=24, input_n_max_digits=24,
                 decoder=False, verboseOutputter=None, layers=1, mn=False, 
                 all_decoder_prediction=False):
        # Store settings in self since the initializing functions will need them
        self.layers = layers;
        self.single_digit = single_digit;
        self.minibatch_size = minibatch_size;
        self.n_max_digits = n_max_digits;
        self.input_n_max_digits = input_n_max_digits;
        self.lstm = lstm;
        self.mn = mn;
        self.single_digit = single_digit;
        self.decoder = decoder;
        self.verboseOutputter = verboseOutputter;
        self.all_decoder_prediction = all_decoder_prediction;
        
        self.EOS_symbol_index = EOS_symbol_index;
        self.GO_symbol_index = GO_symbol_index;
        
        # Feature support checks
        if (self.single_digit):
            raise ValueError("Feature single_digit = True not supported!");
        if (self.mn):
            raise ValueError("Feature memory networks not supported!");
        if (self.minibatch_size == 1):
            raise ValueError("Feature fake minibatching not supported!");
        if (self.layers > 1):
            raise ValueError("Feature multiple layers not supported!");
        
        # Set dimensions
        self.data_dim = data_dim;
        self.hidden_dim = hidden_dim;
        if (self.layers == 2):
            # For now we stick to layers of the same size
            self.hidden_dim_2 = hidden_dim;
        self.decoding_output_dim = output_dim;
        self.prediction_output_dim = output_dim;
        
        # Set up settings for fake minibatching
        self.fake_minibatch = False;
        if (self.minibatch_size == 1):
            self.fake_minibatch = True;
            self.minibatch_size = 2;
        
        # Set up Tensorflow model
        self.x = tf.placeholder(tf.float32, [self.input_n_max_digits, self.minibatch_size, self.data_dim], name='x');
        self.target = tf.placeholder(tf.float32, [self.n_max_digits, self.minibatch_size, self.decoding_output_dim], name='targets');
        self.learning_rate = tf.placeholder(tf.float32, ());
        
        # Set cell type
        init_cell_type = init_rnn_cell;
        cell_type = rnn_cell;
        if (self.lstm):
            init_cell_type = init_lstm_cell;
            cell_type = lstm_cell;
        
        # Encoding
        if (self.decoder):
            self.encoding_cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden_dim);
            init_hidden = tf.zeros([self.minibatch_size, self.hidden_dim]);
            _, state = tf.nn.rnn(self.encoding_cell, tf.unpack(self.x, axis=0), initial_state=init_hidden);
        else:
            init_cell_type('encoding_cell', self.data_dim, self.hidden_dim, self.prediction_output_dim);
            state = tf.zeros([self.minibatch_size, self.hidden_dim]);
            x_unpacked = tf.unpack(self.x, axis=0);
            
            for i in range(self.input_n_max_digits):
                state, _ = cell_type('encoding_cell', x_unpacked[i], state, withOutput=False);            
        
        if (not self.all_decoder_prediction):
            with tf.variable_scope('encoding_prediction'):
                hWo = tf.get_variable('hWo', [self.hidden_dim, self.decoding_output_dim], initializer=tf.truncated_normal_initializer(stddev=0.1));
                bo = tf.get_variable('bo', initializer=tf.zeros_initializer([self.decoding_output_dim]));
            input = tf.nn.softmax(tf.matmul(state,hWo) + bo);
        
        # Decoding
        if (self.decoder):
            init_cell_type('decoding_cell', self.prediction_output_dim, self.hidden_dim, self.decoding_output_dim);
            decoding_cell = lambda inp, st: cell_type('decoding_cell', inp, st);
        else:
            decoding_cell = lambda inp, st: cell_type('encoding_cell', inp, st);
        
        y_array = []
        if (self.all_decoder_prediction):
            input = tf.concat(0,[tf.zeros([self.decoding_output_dim-1, self.minibatch_size]),tf.ones([1, self.minibatch_size])]);
            input = tf.transpose(input);
            n_decoder_symbols = self.n_max_digits;
        else:
            n_decoder_symbols = self.n_max_digits - 1;
            y_array.append(input);
        
        for i in range(n_decoder_symbols):
            state, input = decoding_cell(input, state);
            y_array.append(input);
        self.y = tf.pack(y_array, axis=0);
        
        # Error computation
        # Cross entropy over 3 dimensions
        entropy_per_digit = -tf.reduce_sum(self.target * tf.log(self.y), reduction_indices=[2]);
        self.cross_entropy = tf.reduce_mean(entropy_per_digit);
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy);
        #train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy);

        # Accuracy computation
        #y_predictions = tf.argmax(tf.transpose(y, [1, 0, 2]),2);
        #labels = tf.argmax(tf.transpose(self.target, [1, 0, 2]), 2);
        #correct_predictions = tf.equal(y_predictions, labels);
        #accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32));
        
        # Initialize model
        self.session = tf.Session();
        with self.session.as_default():
            self.session.run(tf.initialize_all_variables());
    
    def sanityChecks(self, training_data, training_labels):
        if (not self.single_digit and training_labels.shape[1] > (self.n_max_digits+1)):
            raise ValueError("n_max_digits too small! Increase to %d" % training_labels.shape[1]);
    
    def sgd(self, dataset, data, labels, learning_rate=1e-4, nearestExpression=False, useFixedDecoderInputs=False):
        if (nearestExpression):
            raise ValueError("Feature nearestExpression = True not supported!");
        if (useFixedDecoderInputs):
            raise ValueError("Feature useFixedDecoderInputs = True not supported!");
        with self.session.as_default():
            _, error = self.session.run([self.train_step, self.cross_entropy], feed_dict={self.x: data, self.target: labels, self.learning_rate: learning_rate});
        return [error];
    
    def predict(self, data):
        data = np.swapaxes(data, 0, 1);
    
        with self.session.as_default():
            y_numpy = self.y.eval(feed_dict={self.x: data});
            y_unswapped = np.swapaxes(y_numpy, 0, 1);
            y_predictions = np.argmax(y_unswapped,2);
        
        return y_predictions, y_unswapped;
    
    def writeVerboseOutput(self):
        pass # TO DO: implement
    
    def getVars(self):
        if (self.decoder):
            if (self.lstm):
                vars = init_lstm_cell('decoding_cell', self.data_dim, self.hidden_dim, self.decoding_output_dim, justLoad=True);
            else:
                vars = init_rnn_cell('decoding_cell', self.data_dim, self.hidden_dim, self.decoding_output_dim, justLoad=True);
        else:
            if (self.lstm):
                vars = init_lstm_cell('encoding_cell', self.data_dim, self.hidden_dim, self.prediction_output_dim, justLoad=True);
            else:
                vars = init_rnn_cell('encoding_cell', self.data_dim, self.hidden_dim, self.prediction_output_dim, justLoad=True);
        
        vars_vals = self.session.run(vars);
        return {var.name: vars_vals[i] for (i,var) in enumerate(vars)};
        
def init_rnn_cell(scope, data_dim, hidden_dim, output_dim, justLoad=False):
    """
    Call with reuse=True if you want to get the variables instead of initializing them.
    """
    with tf.variable_scope(scope, reuse=justLoad):
        XWh = tf.get_variable('XWh', [data_dim, hidden_dim], initializer=tf.truncated_normal_initializer(stddev=0.1));
        hWh = tf.get_variable('hWh', [hidden_dim, hidden_dim], initializer=tf.truncated_normal_initializer(stddev=0.1));
        hWo = tf.get_variable('hWo', [hidden_dim, output_dim], initializer=tf.truncated_normal_initializer(stddev=0.1));
        bh = tf.get_variable('bh', initializer=tf.zeros_initializer([hidden_dim]));
        bo = tf.get_variable('bo', initializer=tf.zeros_initializer([output_dim]));
    return [XWh, hWh, hWo, bh, bo];

def init_lstm_cell(scope, data_dim, hidden_dim, output_dim, justLoad=False):
    with tf.variable_scope(scope, reuse=justLoad):
        XWf = tf.get_variable('XWf', [data_dim, hidden_dim], initializer=tf.truncated_normal_initializer(stddev=0.1));
        hWf = tf.get_variable('hWf', [hidden_dim, hidden_dim], initializer=tf.truncated_normal_initializer(stddev=0.1));
        XWi = tf.get_variable('XWi', [data_dim, hidden_dim], initializer=tf.truncated_normal_initializer(stddev=0.1));
        hWi = tf.get_variable('hWi', [hidden_dim, hidden_dim], initializer=tf.truncated_normal_initializer(stddev=0.1));
        XWc = tf.get_variable('XWc', [data_dim, hidden_dim], initializer=tf.truncated_normal_initializer(stddev=0.1));
        hWc = tf.get_variable('hWc', [hidden_dim, hidden_dim], initializer=tf.truncated_normal_initializer(stddev=0.1));
        XWo = tf.get_variable('XWo', [data_dim, hidden_dim], initializer=tf.truncated_normal_initializer(stddev=0.1));
        hWo = tf.get_variable('hWo', [hidden_dim, hidden_dim], initializer=tf.truncated_normal_initializer(stddev=0.1));
        hWy = tf.get_variable('hWy', [hidden_dim, output_dim], initializer=tf.truncated_normal_initializer(stddev=0.1));
    return [XWf, hWf, XWi, hWi, XWc, hWc, XWo, hWo, hWy];

def rnn_cell(scope, input, state, withOutput=True):
    with tf.variable_scope(scope, reuse=True):
        XWh = tf.get_variable('XWh');
        hWh = tf.get_variable('hWh');
        bh = tf.get_variable('bh');
        if (withOutput):
            hWo = tf.get_variable('hWo');
            bo = tf.get_variable('bo');
    input_gate = tf.matmul(input,XWh);
    state = tf.tanh(input_gate + tf.matmul(state,hWh) + bh);
    output = None;
    if (withOutput):
        output = tf.nn.softmax(tf.matmul(state, hWo) + bo);
    return state, output;

def lstm_cell(scope, input, state, withOutput=True):
    with tf.variable_scope(scope, reuse=True):
        XWf = tf.get_variable('XWf');
        hWf = tf.get_variable('hWf');
        XWi = tf.get_variable('XWi');
        hWi = tf.get_variable('hWi');
        XWc = tf.get_variable('XWc');
        hWc = tf.get_variable('hWc');
        XWo = tf.get_variable('XWo');
        hWo = tf.get_variable('hWo');
        if (withOutput):
            hWy = tf.get_variable('hWy');
    """
    forget_gate = T.nnet.sigmoid(previous_hidden.dot(hWf) + current_X.dot(XWf));
    input_gate = T.nnet.sigmoid(previous_hidden.dot(hWi) + current_X.dot(XWi));
    candidate_cell = T.tanh(previous_hidden.dot(hWc) + current_X.dot(XWc));
    cell = forget_gate * previous_hidden + input_gate * candidate_cell;
    output_gate = T.nnet.sigmoid(previous_hidden.dot(hWo) + current_X.dot(XWo));
    hidden = output_gate * cell;
    Y_output = T.nnet.softmax(hidden.dot(hWY));
    return Y_output, hidden;
    """
    forget_gate = tf.sigmoid(tf.matmul(state,hWf) + tf.matmul(input,XWf));
    input_gate = tf.sigmoid(tf.matmul(state,hWi) + tf.matmul(input,XWi));
    candidate_cell = tf.tanh(tf.matmul(state,hWc) + tf.matmul(input,XWc));
    cell = forget_gate * state + input_gate * candidate_cell;
    output_gate = tf.sigmoid(tf.matmul(state,hWo) + tf.matmul(input,XWo));
    state = output_gate * cell;
    
    output = None;
    if (withOutput):
        output = tf.nn.softmax(tf.matmul(state,hWy));
    
    return state, output;