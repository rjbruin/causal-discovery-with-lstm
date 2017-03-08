'''
Created on 22 feb. 2016

@author: Robert-Jan
'''

import os;

import theano;
import theano.tensor as T;
# from theano.compile.nanguardmode import NanGuardMode

import numpy as np;
from models.RecurrentModel import RecurrentModel
#from theano.compile.nanguardmode import NanGuardMode
import lasagne;

from profiler import profiler;

class TheanoRecurrentNeuralNetwork(RecurrentModel):
    '''
    Recurrent neural network model with one hidden layer. Models single class
    prediction based on regular recurrent model or LSTM model.
    '''

    SGD_OPTIMIZER = 0;
    RMS_OPTIMIZER = 1;
    NESTEROV_OPTIMIZER = 2;
    
    RNN_DECODESELFFEEDING = 0;
    RNN_ENCODEDECODEDATAFEEDING = 1;
    RNN_DECODESINGLEPREDICTION = 2;

    def __init__(self, data_dim, hidden_dim, output_dim, minibatch_size,
                 lstm=True, single_digit=False,
                 EOS_symbol_index=None, GO_symbol_index=None, n_max_digits=24,
                 decoder=False, verboseOutputter=None, finishExpressions=True,
                 optimizer=0, learning_rate=0.01,
                 operators=4, digits=10, only_cause_expression=False, seq2ndmarkov=False,
                 doubleLayer=False, tripleLayer=False, dropoutProb=0., outputBias=False,
                 crosslinks=True, appendAbstract=False, useAbstract=False, relu=False,
                 ignoreZeroDifference=False, peepholes=False, lstm_biases=False, lag=None,
                 rnn_version=0, nocrosslinks_hidden_factor=1., bottom_loss=True):
        '''
        Initialize all Theano models.
        '''
        # Store RNN version we are using
        self.rnn_version = rnn_version;
        
        # Store settings in self since the initializing functions will need them
        self.minibatch_size = minibatch_size;
        self.n_max_digits = n_max_digits;
        self.operators = operators;
        self.digits = digits;
        self.learning_rate = learning_rate;
        self.lstm = lstm;
        self.decoder = decoder;
        self.only_cause_expression = only_cause_expression;
        self.seq2ndmarkov = seq2ndmarkov;
        self.optimizer = optimizer;
        self.verboseOutputter = verboseOutputter;
        self.finishExpressions = finishExpressions;
        self.doubleLayer = doubleLayer;
        self.tripleLayer = tripleLayer;
        if (self.tripleLayer):
            self.doubleLayer = False;
        self.dropoutProb = dropoutProb;
        self.outputBias = outputBias;
        self.crosslinks = crosslinks;
        self.useAbstract = useAbstract;
        self.appendAbstract = appendAbstract;
        self.relu = relu;
        self.ignoreZeroDifference = ignoreZeroDifference;
        self.lag = lag;
        self.nocrosslinks_hidden_factor = nocrosslinks_hidden_factor;
        self.bottom_loss = bottom_loss;

#         self.peepholes = peepholes;
#         self.lstm_biases = lstm_biases;
        self.peepholes = True;
        self.lstm_biases = True;
        print("WARNING! Peepholes and LSTM biases are fixed to be on!");

        if (not self.lstm):
            raise ValueError("Feature LSTM = False is no longer supported!");

        self.EOS_symbol_index = EOS_symbol_index;
        self.GO_symbol_index = GO_symbol_index;

        if (self.GO_symbol_index is None):
            raise ValueError("GO symbol index not set!");

        # Set dimensions
        self.data_dim = data_dim;
        self.hidden_dim = hidden_dim;
        self.decoding_output_dim = output_dim;
        self.prediction_output_dim = output_dim;

        self.actual_data_dim = self.data_dim * 2;
        self.actual_prediction_output_dim = self.prediction_output_dim * 2;
        if (self.only_cause_expression or rnn_version == TheanoRecurrentNeuralNetwork.RNN_DECODESINGLEPREDICTION):
            self.actual_data_dim = self.data_dim;
            self.actual_prediction_output_dim = self.prediction_output_dim;
        if (not self.crosslinks):
            self.hidden_dim = int(self.hidden_dim * nocrosslinks_hidden_factor);
        
        self.RNNVars();
        if (self.rnn_version == 0):
            self.decodeSelfFeedingRNN();
        elif (self.rnn_version == 1):
            self.encodeDecodeDataFeedingRNN();
        elif (self.rnn_version == 2):
            self.decodeSinglePredictionRNN();
        
        super(TheanoRecurrentNeuralNetwork, self).__init__();

    def RNNVars(self, weight_values={}):
        # Set up shared variables
        self.varSettings = [];
        if (self.lstm):
            self.varSettings.append(('hWf',self.hidden_dim,self.hidden_dim));
            self.varSettings.append(('XWf',self.actual_data_dim,self.hidden_dim));
            self.varSettings.append(('hWi',self.hidden_dim,self.hidden_dim));
            self.varSettings.append(('XWi',self.actual_data_dim,self.hidden_dim));
            self.varSettings.append(('hWc',self.hidden_dim,self.hidden_dim));
            self.varSettings.append(('XWc',self.actual_data_dim,self.hidden_dim));
            self.varSettings.append(('hWo',self.hidden_dim,self.hidden_dim));
            self.varSettings.append(('XWo',self.actual_data_dim,self.hidden_dim));
            if (self.peepholes):
                # Peephole connections for forget, input and output gate
                self.varSettings.append(('Pf',False,self.hidden_dim));
                self.varSettings.append(('Pi',False,self.hidden_dim));
                self.varSettings.append(('Po',False,self.hidden_dim));
            if (self.lstm_biases):
                self.varSettings.append(('bc',False,self.hidden_dim));
                self.varSettings.append(('bf',False,self.hidden_dim));
                self.varSettings.append(('bi',False,self.hidden_dim));
                self.varSettings.append(('bo',False,self.hidden_dim));
        else:
            self.varSettings.append(('XWh',self.actual_data_dim,self.hidden_dim));
            self.varSettings.append(('Xbh',False,self.hidden_dim));
            self.varSettings.append(('hWh',self.hidden_dim,self.hidden_dim));
            self.varSettings.append(('hbh',False,self.hidden_dim));

        if (self.doubleLayer or self.tripleLayer):
            if (self.lstm):
                self.varSettings.append(('hWf2',self.hidden_dim,self.hidden_dim));
                self.varSettings.append(('XWf2',self.hidden_dim,self.hidden_dim));
                self.varSettings.append(('hWi2',self.hidden_dim,self.hidden_dim));
                self.varSettings.append(('XWi2',self.hidden_dim,self.hidden_dim));
                self.varSettings.append(('hWc2',self.hidden_dim,self.hidden_dim));
                self.varSettings.append(('XWc2',self.hidden_dim,self.hidden_dim));
                self.varSettings.append(('hWo2',self.hidden_dim,self.hidden_dim));
                self.varSettings.append(('XWo2',self.hidden_dim,self.hidden_dim));
                if (self.peepholes):
                    # Peephole connections for forget, input and output gate
                    self.varSettings.append(('Pf2',False,self.hidden_dim));
                    self.varSettings.append(('Pi2',False,self.hidden_dim));
                    self.varSettings.append(('Po2',False,self.hidden_dim));
                if (self.lstm_biases):     
                    self.varSettings.append(('bc2',False,self.hidden_dim));
                    self.varSettings.append(('bf2',False,self.hidden_dim));
                    self.varSettings.append(('bi2',False,self.hidden_dim));
                    self.varSettings.append(('bo2',False,self.hidden_dim));
            else:
                self.varSettings.append(('XWh2',self.hidden_dim,self.hidden_dim));
                self.varSettings.append(('Xbh2',False,self.hidden_dim));
                self.varSettings.append(('hWh2',self.hidden_dim,self.hidden_dim));
                self.varSettings.append(('hbh2',False,self.hidden_dim));
        if (self.tripleLayer):
            if (self.lstm):
                self.varSettings.append(('hWf3',self.hidden_dim,self.hidden_dim));
                self.varSettings.append(('XWf3',self.hidden_dim,self.hidden_dim));
                self.varSettings.append(('hWi3',self.hidden_dim,self.hidden_dim));
                self.varSettings.append(('XWi3',self.hidden_dim,self.hidden_dim));
                self.varSettings.append(('hWc3',self.hidden_dim,self.hidden_dim));
                self.varSettings.append(('XWc3',self.hidden_dim,self.hidden_dim));
                self.varSettings.append(('hWo3',self.hidden_dim,self.hidden_dim));
                self.varSettings.append(('XWo3',self.hidden_dim,self.hidden_dim));
                if (self.peepholes):
                    # Peephole connections for forget, input and output gate
                    self.varSettings.append(('Pf3',False,self.hidden_dim));
                    self.varSettings.append(('Pi3',False,self.hidden_dim));
                    self.varSettings.append(('Po3',False,self.hidden_dim));
                if (self.lstm_biases):     
                    self.varSettings.append(('bc3',False,self.hidden_dim));
                    self.varSettings.append(('bf3',False,self.hidden_dim));
                    self.varSettings.append(('bi3',False,self.hidden_dim));
                    self.varSettings.append(('bo3',False,self.hidden_dim));
            else:
                self.varSettings.append(('XWh3',self.hidden_dim,self.hidden_dim));
                self.varSettings.append(('Xbh3',False,self.hidden_dim));
                self.varSettings.append(('hWh3',self.hidden_dim,self.hidden_dim));
                self.varSettings.append(('hbh3',False,self.hidden_dim));

        self.varSettings.append(('hWY',self.hidden_dim,self.actual_prediction_output_dim));
        self.varSettings.append(('hbY',False,self.actual_prediction_output_dim));

        # Contruct variables
        self.vars = {};
        for (varName,dim1,dim2) in self.varSettings:
            # Get value for shared variable from constructor if present
            if (dim1 is not False):
                value = np.random.uniform(-np.sqrt(1.0/dim1),np.sqrt(1.0/dim1),(dim1,dim2)).astype('float32');
            else:
                value = np.random.uniform(-np.sqrt(1.0/dim2),-np.sqrt(1.0/dim2),(dim2)).astype('float32');
            if (varName in weight_values):
                value = weight_values[varName];
            self.vars[varName] = theano.shared(value, varName);

    def decodeSelfFeedingRNN(self):
        # Set up inputs to prediction and SGD
        # X is 3-dimensional: 1) index in sentence, 2) datapoint, 3) dimensionality of data
        X = T.ftensor3('X');
        # label is 3-dimensional: 1) index in sentence, 2) datapoint, 3) dimensionality of data
        label = T.ftensor3('label');
        intervention_locations = T.imatrix();
        nrSamples = T.iscalar();

        # Set the RNN cell to use for encoding and decoding
        decode_function = self.lstm_interventional_single;
        if (self.doubleLayer):
            decode_function = self.lstm_interventional_double;
        if (self.tripleLayer):
            decode_function = self.lstm_interventional_triple;

        if (self.dropoutProb > 0.):
            self.random_stream = T.shared_randomstreams.RandomStreams(seed=np.random.randint(10000));

        # Set the prediction parameters to be either the prediction
        # weights or the decoding weights depending on the setting
        decode_parameters = [intervention_locations] + [self.vars[k[0]] for k in self.varSettings];

        hidden = [T.zeros((self.minibatch_size,self.hidden_dim))];
        cell = [T.zeros((self.minibatch_size,self.hidden_dim))];
        if (self.doubleLayer or self.tripleLayer):
            hidden_2 = [T.zeros((self.minibatch_size,self.hidden_dim))];
            cell_2 = [T.zeros((self.minibatch_size,self.hidden_dim))];
        if (self.tripleLayer):
            hidden_3 = [T.zeros((self.minibatch_size,self.hidden_dim))];
            cell_3 = [T.zeros((self.minibatch_size,self.hidden_dim))];
        if (not self.crosslinks or self.only_cause_expression is not False):
            hidden_top = hidden;
            cell_top = [T.zeros((self.minibatch_size,self.hidden_dim))];
            hidden_bot = [T.zeros((self.minibatch_size,self.hidden_dim))];
            cell_bot = [T.zeros((self.minibatch_size,self.hidden_dim))];
            if (self.doubleLayer or self.tripleLayer):
                hidden_2_top = [T.zeros((self.minibatch_size,self.hidden_dim))];
                cell_2_top = [T.zeros((self.minibatch_size,self.hidden_dim))];
                hidden_2_bot = [T.zeros((self.minibatch_size,self.hidden_dim))];
                cell_2_bot = [T.zeros((self.minibatch_size,self.hidden_dim))];
            if (self.tripleLayer):
                hidden_3_top = [T.zeros((self.minibatch_size,self.hidden_dim))];
                cell_3_top = [T.zeros((self.minibatch_size,self.hidden_dim))];
                hidden_3_bot = [T.zeros((self.minibatch_size,self.hidden_dim))];
                cell_3_bot = [T.zeros((self.minibatch_size,self.hidden_dim))];

        # DECODING PHASE
        if (self.crosslinks and not self.only_cause_expression):
            init_values = ({'initial': T.zeros((self.minibatch_size,self.actual_data_dim)), 'taps': [-1]},
                           {'initial': hidden[-1], 'taps': [-1]}, 
                           {'initial': cell[-1], 'taps': [-1]}, 
                           {'initial': 0., 'taps': [-1]});
            if (self.doubleLayer):
                init_values = ({'initial': T.zeros((self.minibatch_size,self.actual_data_dim)), 'taps': [-1]},
                               {'initial': hidden[-1], 'taps': [-1]},
                               {'initial': hidden_2[-1], 'taps': [-1]},
                               {'initial': cell[-1], 'taps': [-1]}, 
                               {'initial': cell_2[-1], 'taps': [-1]}, 
                               {'initial': 0., 'taps': [-1]});
            if (self.tripleLayer):
                init_values = ({'initial': T.zeros((self.minibatch_size,self.actual_data_dim)), 'taps': [-1]},
                               {'initial': hidden[-1], 'taps': [-1]},
                               {'initial': hidden_2[-1], 'taps': [-1]},
                               {'initial': hidden_3[-1], 'taps': [-1]},
                               {'initial': cell[-1], 'taps': [-1]}, 
                               {'initial': cell_2[-1], 'taps': [-1]}, 
                               {'initial': cell_3[-1], 'taps': [-1]},
                               {'initial': 0., 'taps': [-1]});
            outputs, _ = theano.scan(fn=decode_function,
                                     sequences=label,
                                     outputs_info=init_values,
                                     non_sequences=decode_parameters + [0,self.actual_data_dim],
                                     name='decode_scan_1')
            right_hand = outputs[0];
        else:
            init_values = ({'initial': T.zeros((self.minibatch_size,self.data_dim)), 'taps': [-1]},
                           {'initial': hidden_top[-1], 'taps': [-1]}, 
                           {'initial': cell[-1], 'taps': [-1]},
                           {'initial': 0., 'taps': [-1]});
            if (self.doubleLayer):
                init_values = ({'initial': T.zeros((self.minibatch_size,self.data_dim)), 'taps': [-1]},
                               {'initial': hidden_top[-1], 'taps': [-1]},
                               {'initial': hidden_2_top[-1], 'taps': [-1]},
                               {'initial': cell_top[-1], 'taps': [-1]},
                               {'initial': cell_2_top[-1], 'taps': [-1]},
                               {'initial': 0., 'taps': [-1]});
            if (self.tripleLayer):
                init_values = ({'initial': T.zeros((self.minibatch_size,self.data_dim)), 'taps': [-1]},
                               {'initial': hidden_top[-1], 'taps': [-1]},
                               {'initial': hidden_2_top[-1], 'taps': [-1]},
                               {'initial': hidden_3_top[-1], 'taps': [-1]},
                               {'initial': cell_top[-1], 'taps': [-1]}, 
                               {'initial': cell_2_top[-1], 'taps': [-1]}, 
                               {'initial': cell_3_top[-1], 'taps': [-1]},
                               {'initial': 0., 'taps': [-1]});
            outputs_1, _ = theano.scan(fn=decode_function,
                                     sequences=label,
                                     outputs_info=init_values,
                                     non_sequences=decode_parameters + [0,self.data_dim],
                                     name='decode_scan_1')

            right_hand_1 = outputs_1[0];

            if (not self.only_cause_expression):
                init_values = ({'initial': T.zeros((self.minibatch_size,self.data_dim)), 'taps': [-1]},
                               {'initial': hidden_bot[-1], 'taps': [-1]}, {'initial': 0., 'taps': [-1]},
                               {'initial': cell[-1], 'taps': [-1]}, {'initial': 0., 'taps': [-1]});
                if (self.doubleLayer):
                    init_values = ({'initial': T.zeros((self.minibatch_size,self.data_dim)), 'taps': [-1]},
                                   {'initial': hidden_bot[-1], 'taps': [-1]},
                                   {'initial': hidden_2_bot[-1], 'taps': [-1]},
                                   {'initial': cell_bot[-1], 'taps': [-1]},
                                   {'initial': cell_2_bot[-1], 'taps': [-1]},
                                   {'initial': 0., 'taps': [-1]});
                if (self.tripleLayer):
                    init_values = ({'initial': T.zeros((self.minibatch_size,self.data_dim)), 'taps': [-1]},
                                   {'initial': hidden_bot[-1], 'taps': [-1]},
                                   {'initial': hidden_2_bot[-1], 'taps': [-1]},
                                   {'initial': hidden_3_bot[-1], 'taps': [-1]},
                                   {'initial': cell_bot[-1], 'taps': [-1]},
                                   {'initial': cell_2_bot[-1], 'taps': [-1]},
                                   {'initial': cell_3_bot[-1], 'taps': [-1]},
                                   {'initial': 0., 'taps': [-1]});
                outputs_2, _ = theano.scan(fn=decode_function,
                                         sequences=label,
                                         outputs_info=init_values,
                                         non_sequences=decode_parameters + [self.data_dim,self.data_dim*2],
                                         name='decode_scan_2')

                right_hand_2 = outputs_2[0];
                right_hand = T.concatenate([right_hand_1, right_hand_2], axis=2);
            else:
                right_hand = right_hand_1;

        right_hand_near_zeros = T.ones_like(right_hand) * 1e-15;
        right_hand = T.maximum(right_hand, right_hand_near_zeros);

        # Compute the prediction
        prediction_1 = T.argmax(right_hand[:,:,:self.data_dim], axis=2);
        if (not self.only_cause_expression):
            prediction_2 = T.argmax(right_hand[:,:,self.data_dim:], axis=2);

        # ERROR COMPUTATION AND PROPAGATION
        coding_dist = right_hand[:label.shape[0]]
        cat_cross = -T.sum(label * T.log(coding_dist), axis=coding_dist.ndim-1);
        mean_cross_per_sample = T.sum(cat_cross, axis=0) / (self.n_max_digits - (intervention_locations + 1.));
        error = T.mean(mean_cross_per_sample[:nrSamples]);
        summed_error = T.sum(mean_cross_per_sample[:nrSamples]);

        # Defining prediction function
        if (not self.only_cause_expression):
            self._predict = theano.function([X, label, intervention_locations, nrSamples], [prediction_1,
                                                                                prediction_2,
                                                                                right_hand,
                                                                                error,
                                                                                summed_error], on_unused_input='ignore');
        else:
            self._predict = theano.function([X, label, intervention_locations, nrSamples], [prediction_1,
                                                                                right_hand,
                                                                                error,
                                                                                summed_error], on_unused_input='ignore');

        # Defining stochastic gradient descent
        variables = filter(lambda name: name != 'hbY', self.vars.keys());
        if (self.outputBias):
            variables.append('hbY');
        var_list = map(lambda var: self.vars[var], variables)
        if (self.optimizer == self.SGD_OPTIMIZER):
            # Automatic backward pass for all models: gradients
            derivatives = T.grad(error, var_list);
            updates = [(var,var-self.learning_rate*der) for (var,der) in zip(var_list,derivatives)];
        elif (self.optimizer == self.RMS_OPTIMIZER):
            derivatives = T.grad(error, var_list);
            updates = lasagne.updates.rmsprop(derivatives,var_list,learning_rate=self.learning_rate).items();
        else:
            derivatives = T.grad(error, var_list);
            updates = lasagne.updates.nesterov_momentum(derivatives,var_list,learning_rate=self.learning_rate).items();

        # Defining SGD functuin
        self._sgd = theano.function([X, label, intervention_locations, nrSamples],
                                        [error, summed_error],
                                    updates=updates,
                                    allow_input_downcast=True,
                                    on_unused_input='ignore')

    def encodeDecodeDataFeedingRNN(self):
        """
        RNN for discreteprocess (f-subs new version).
        """
        # Set up inputs to prediction and SGD
        # X is 3-dimensional: 1) index in sentence, 2) datapoint, 3) dimensionality of data
        X = T.ftensor3('X');
        # label is 3-dimensional: 1) index in sentence, 2) datapoint, 3) dimensionality of data
        label = T.ftensor3('label');

        # Set the RNN cell to use for encoding and decoding
        encode_function = self.lstm_cell;
        decode_function = self.lstm_single;
        if (self.doubleLayer):
            encode_function = self.lstm_double_no_output;
            decode_function = self.lstm_double;
        if (self.tripleLayer):
            encode_function = self.lstm_triple_no_output;
            decode_function = self.lstm_triple;

        if (self.dropoutProb > 0.):
            self.random_stream = T.shared_randomstreams.RandomStreams(seed=np.random.randint(10000));

        # Set the prediction parameters to be either the prediction
        # weights or the decoding weights depending on the setting
        encode_parameters = [self.vars[k[0]] for k in self.varSettings if k[0][-1] != 'Y'];
        decode_parameters = [self.vars[k[0]] for k in self.varSettings];

        hidden = [T.zeros((self.minibatch_size,self.hidden_dim))];
        cell = [T.zeros((self.minibatch_size,self.hidden_dim))];
        if (self.doubleLayer):
            hidden_2 = [T.zeros((self.minibatch_size,self.hidden_dim))];
            cell_2 = [T.zeros((self.minibatch_size,self.hidden_dim))];
        if (self.tripleLayer):
            hidden_3 = [T.zeros((self.minibatch_size,self.hidden_dim))];
            cell_3 = [T.zeros((self.minibatch_size,self.hidden_dim))];
        if (not self.crosslinks or self.only_cause_expression is not False):
            hidden_top = hidden;
            cell_top = [T.zeros((self.minibatch_size,self.hidden_dim))];
            hidden_bot = [T.zeros((self.minibatch_size,self.hidden_dim))];
            cell_bot = [T.zeros((self.minibatch_size,self.hidden_dim))];
            if (self.doubleLayer):
                hidden_2_top = [T.zeros((self.minibatch_size,self.hidden_dim))];
                cell_2_top = [T.zeros((self.minibatch_size,self.hidden_dim))];
                hidden_2_bot = [T.zeros((self.minibatch_size,self.hidden_dim))];
                cell_2_bot = [T.zeros((self.minibatch_size,self.hidden_dim))];
            if (self.tripleLayer):
                hidden_3_top = [T.zeros((self.minibatch_size,self.hidden_dim))];
                cell_3_top = [T.zeros((self.minibatch_size,self.hidden_dim))];
                hidden_3_bot = [T.zeros((self.minibatch_size,self.hidden_dim))];
                cell_3_bot = [T.zeros((self.minibatch_size,self.hidden_dim))];

        # ENCODING PHASE (process random prefix of sample)
        if (self.crosslinks and not self.only_cause_expression):
            init_values = ({'initial': hidden[-1], 'taps': [-1]}, 
                           {'initial': cell[-1], 'taps': [-1]});
            if (self.doubleLayer):
                init_values = ({'initial': hidden[-1], 'taps': [-1]},
                               {'initial': hidden_2[-1], 'taps': [-1]},
                               {'initial': cell[-1], 'taps': [-1]}, 
                               {'initial': cell_2[-1], 'taps': [-1]});
            if (self.tripleLayer):
                init_values = ({'initial': hidden[-1], 'taps': [-1]},
                               {'initial': hidden_2[-1], 'taps': [-1]},
                               {'initial': hidden_3[-1], 'taps': [-1]},
                               {'initial': cell[-1], 'taps': [-1]}, 
                               {'initial': cell_2[-1], 'taps': [-1]}, 
                               {'initial': cell_3[-1], 'taps': [-1]});
            outputs, _ = theano.scan(fn=encode_function,
                                     sequences=label[:self.lag-1],
                                     outputs_info=init_values,
                                     non_sequences=encode_parameters + [0,self.actual_data_dim],
                                     name='decode_scan_1')
            learned_hidden = outputs[0];
            learned_cell = outputs[1];
            if (self.doubleLayer):
                learned_hidden_2 = outputs[1];
                learned_cell = outputs[2];
                learned_cell_2 = outputs[3];
            if (self.tripleLayer):
                learned_hidden_2 = outputs[1];
                learned_hidden_3 = outputs[2];
                learned_cell = outputs[3];
                learned_cell_2 = outputs[4];
                learned_cell_3 = outputs[5];
        else:
            init_values = ({'initial': hidden_top[-1], 'taps': [-1]}, 
                           {'initial': cell_top[-1], 'taps': [-1]});
            if (self.doubleLayer):
                init_values = ({'initial': hidden_top[-1], 'taps': [-1]},
                               {'initial': hidden_2_top[-1], 'taps': [-1]},
                               {'initial': cell_top[-1], 'taps': [-1]},
                               {'initial': cell_2_top[-1], 'taps': [-1]});
            if (self.tripleLayer):
                init_values = ({'initial': hidden_top[-1], 'taps': [-1]},
                               {'initial': hidden_2_top[-1], 'taps': [-1]},
                               {'initial': hidden_3_top[-1], 'taps': [-1]},
                               {'initial': cell_top[-1], 'taps': [-1]}, 
                               {'initial': cell_2_top[-1], 'taps': [-1]}, 
                               {'initial': cell_3_top[-1], 'taps': [-1]});
            outputs_1, _ = theano.scan(fn=encode_function,
                                     sequences=label[:self.lag-1,:,:self.data_dim],
                                     outputs_info=init_values,
                                     non_sequences=encode_parameters + [0,self.data_dim],
                                     name='decode_scan_1')

            learned_hidden_top = outputs_1[0];
            learned_cell_top = outputs_1[1];
            if (self.doubleLayer):
                learned_hidden_2_top = outputs_1[1];
                learned_cell_top = outputs_1[2];
                learned_cell_2_top = outputs_1[3];
            if (self.tripleLayer):
                learned_hidden_2_top = outputs_1[1];
                learned_hidden_3_top = outputs_1[2];
                learned_cell_top = outputs_1[3];
                learned_cell_2_top = outputs_1[4];
                learned_cell_3_top = outputs_1[5];

            if (not self.only_cause_expression):
                init_values = ({'initial': hidden_bot[-1], 'taps': [-1]},
                               {'initial': cell_bot[-1], 'taps': [-1]});
                if (self.doubleLayer):
                    init_values = ({'initial': hidden_bot[-1], 'taps': [-1]},
                                   {'initial': hidden_2_bot[-1], 'taps': [-1]},
                                   {'initial': cell_bot[-1], 'taps': [-1]},
                                   {'initial': cell_2_bot[-1], 'taps': [-1]});
                if (self.tripleLayer):
                    init_values = ({'initial': hidden_bot[-1], 'taps': [-1]},
                                   {'initial': hidden_2_bot[-1], 'taps': [-1]},
                                   {'initial': hidden_3_bot[-1], 'taps': [-1]},
                                   {'initial': cell_bot[-1], 'taps': [-1]},
                                   {'initial': cell_2_bot[-1], 'taps': [-1]},
                                   {'initial': cell_3_bot[-1], 'taps': [-1]});
                outputs_2, _ = theano.scan(fn=encode_function,
                                         sequences=label[:self.lag-1,:,self.data_dim:],
                                         outputs_info=init_values,
                                         non_sequences=encode_parameters + [self.data_dim,self.data_dim*2],
                                         name='decode_scan_2')

                learned_hidden_bot = outputs_2[0];
                learned_cell_bot = outputs_2[1];
                if (self.doubleLayer):
                    learned_hidden_2_bot = outputs_2[1];
                    learned_cell_bot = outputs_2[2];
                    learned_cell_2_bot = outputs_2[3];
                if (self.tripleLayer):
                    learned_hidden_2_bot = outputs_2[1];
                    learned_hidden_3_bot = outputs_2[2];
                    learned_cell_bot = outputs_2[3];
                    learned_cell_2_bot = outputs_2[4];
                    learned_cell_3_bot = outputs_2[5];

        # DECODING PHASE
        if (self.crosslinks and not self.only_cause_expression):
            init_values = (None,
                           {'initial': learned_hidden[-1], 'taps': [-1]}, 
                           {'initial': learned_cell[-1], 'taps': [-1]});
            if (self.doubleLayer):
                init_values = (None,
                               {'initial': learned_hidden[-1], 'taps': [-1]},
                               {'initial': learned_hidden_2[-1], 'taps': [-1]},
                               {'initial': learned_cell[-1], 'taps': [-1]}, 
                               {'initial': learned_cell_2[-1], 'taps': [-1]});
            if (self.tripleLayer):
                init_values = (None,
                               {'initial': learned_hidden[-1], 'taps': [-1]},
                               {'initial': learned_hidden_2[-1], 'taps': [-1]},
                               {'initial': learned_hidden_3[-1], 'taps': [-1]},
                               {'initial': learned_cell[-1], 'taps': [-1]}, 
                               {'initial': learned_cell_2[-1], 'taps': [-1]}, 
                               {'initial': learned_cell_3[-1], 'taps': [-1]});
            outputs, _ = theano.scan(fn=decode_function,
                                     sequences=label[self.lag-1:-1],
                                     outputs_info=init_values,
                                     non_sequences=decode_parameters + [0,self.actual_data_dim],
                                     name='decode_scan_1')
            right_hand = outputs[0];
        else:
            init_values = (None,
                           {'initial': learned_hidden_top[-1], 'taps': [-1]}, 
                           {'initial': learned_cell_top[-1], 'taps': [-1]});
            if (self.doubleLayer):
                init_values = (None,
                               {'initial': learned_hidden_top[-1], 'taps': [-1]},
                               {'initial': learned_hidden_2_top[-1], 'taps': [-1]},
                               {'initial': learned_cell_top[-1], 'taps': [-1]},
                               {'initial': learned_cell_2_top[-1], 'taps': [-1]});
            if (self.tripleLayer):
                init_values = (None,
                               {'initial': learned_hidden_top[-1], 'taps': [-1]},
                               {'initial': learned_hidden_2_top[-1], 'taps': [-1]},
                               {'initial': learned_hidden_3_top[-1], 'taps': [-1]},
                               {'initial': learned_cell_top[-1], 'taps': [-1]}, 
                               {'initial': learned_cell_2_top[-1], 'taps': [-1]}, 
                               {'initial': learned_cell_3_top[-1], 'taps': [-1]});
            outputs_1, _ = theano.scan(fn=decode_function,
                                     sequences=label[self.lag-1:-1,:,:self.data_dim],
                                     outputs_info=init_values,
                                     non_sequences=decode_parameters + [0,self.data_dim],
                                     name='decode_scan_1')

            right_hand_1 = outputs_1[0];

            if (not self.only_cause_expression):
                init_values = (None,
                               {'initial': learned_hidden_bot[-1], 'taps': [-1]},
                               {'initial': learned_cell_bot[-1], 'taps': [-1]});
                if (self.doubleLayer):
                    init_values = (None,
                                   {'initial': learned_hidden_bot[-1], 'taps': [-1]},
                                   {'initial': learned_hidden_2_bot[-1], 'taps': [-1]},
                                   {'initial': learned_cell_bot[-1], 'taps': [-1]},
                                   {'initial': learned_cell_2_bot[-1], 'taps': [-1]});
                if (self.tripleLayer):
                    init_values = (None,
                                   {'initial': learned_hidden_bot[-1], 'taps': [-1]},
                                   {'initial': learned_hidden_2_bot[-1], 'taps': [-1]},
                                   {'initial': learned_hidden_3_bot[-1], 'taps': [-1]},
                                   {'initial': learned_cell_bot[-1], 'taps': [-1]},
                                   {'initial': learned_cell_2_bot[-1], 'taps': [-1]},
                                   {'initial': learned_cell_3_bot[-1], 'taps': [-1]});
                outputs_2, _ = theano.scan(fn=decode_function,
                                         sequences=label[self.lag-1:-1,:,self.data_dim:],
                                         outputs_info=init_values,
                                         non_sequences=decode_parameters + [self.data_dim,self.data_dim*2],
                                         name='decode_scan_2')

                right_hand_2 = outputs_2[0];
                right_hand = T.concatenate([right_hand_1, right_hand_2], axis=2);
            else:
                right_hand = right_hand_1;

        right_hand_near_zeros = T.ones_like(right_hand) * 1e-15;
        right_hand = T.maximum(right_hand, right_hand_near_zeros);

        # Compute the prediction
        prediction_1 = T.argmax(right_hand[:,:,:self.data_dim], axis=2);
        if (not self.only_cause_expression):
            prediction_2 = T.argmax(right_hand[:,:,self.data_dim:], axis=2);

        # ERROR COMPUTATION AND PROPAGATION
        if (self.bottom_loss):
            coding_dist = right_hand[:,:,self.data_dim:];
            target_dist = label[self.lag:,:,self.data_dim:];
        else:
            coding_dist = right_hand;
            target_dist = label[self.lag:];
        cat_cross = T.nnet.categorical_crossentropy(coding_dist, target_dist);
        error = T.mean(cat_cross);
        summed_error = T.sum(T.mean(cat_cross, axis=cat_cross.ndim-1));
#         cat_cross = -T.mean(label[self.lag:] * T.log(coding_dist), axis=coding_dist.ndim-1);
#         mean_cross_per_sample = T.mean(cat_cross, axis=0);
#         error = T.mean(mean_cross_per_sample);
#         summed_error = T.sum(mean_cross_per_sample);

        # Defining prediction function
        if (not self.only_cause_expression):
            self._predict = theano.function([X, label], [prediction_1,
                                            prediction_2,
                                            right_hand,
                                            error,
                                            summed_error], on_unused_input='ignore');
        else:
            self._predict = theano.function([X, label], [prediction_1,
                                            right_hand,
                                            error,
                                            summed_error], on_unused_input='ignore');

        # Defining stochastic gradient descent
        variables = filter(lambda name: name != 'hbY', self.vars.keys());
        if (self.outputBias):
            variables.append('hbY');
        var_list = map(lambda var: self.vars[var], variables)
        if (self.optimizer == self.SGD_OPTIMIZER):
            # Automatic backward pass for all models: gradients
            derivatives = T.grad(error, var_list);
            updates = [(var,var-self.learning_rate*der) for (var,der) in zip(var_list,derivatives)];
        elif (self.optimizer == self.RMS_OPTIMIZER):
            derivatives = T.grad(error, var_list);
            updates = lasagne.updates.rmsprop(derivatives,var_list,learning_rate=self.learning_rate).items();
        else:
            derivatives = T.grad(error, var_list);
            updates = lasagne.updates.nesterov_momentum(derivatives,var_list,learning_rate=self.learning_rate).items();

        # Defining SGD functuin
        self._sgd = theano.function([X, label],
                                        [error, summed_error],
                                    updates=updates,
                                    allow_input_downcast=True,
                                    on_unused_input='ignore');

    def decodeSinglePredictionRNN(self):
        # Set up inputs to prediction and SGD
        # X is 3-dimensional: 1) index in sentence, 2) datapoint, 3) dimensionality of data
        X = T.ftensor3('X');
        # label is 3-dimensional: 1) index in sentence, 2) datapoint, 3) dimensionality of data
        label = T.fmatrix('label');
        
        # Set the RNN cell to use for encoding and decoding
        encode_function = self.lstm_cell;
        if (self.doubleLayer):
            encode_function = self.lstm_double_no_output;
        if (self.tripleLayer):
            encode_function = self.lstm_triple_no_output;
        
        if (self.dropoutProb > 0.):
            self.random_stream = T.shared_randomstreams.RandomStreams(seed=np.random.randint(10000));
        
        # Set the prediction parameters to be either the prediction 
        # weights or the decoding weights depending on the setting 
        encode_parameters = [self.vars[k[0]] for k in filter(lambda varset: varset[0][-1] != 'Y', self.varSettings)];
        
        hidden = [T.zeros((self.minibatch_size,self.hidden_dim))];
        cell = [T.zeros((self.minibatch_size,self.hidden_dim))];
        if (self.doubleLayer or self.tripleLayer):
            hidden_2 = [T.zeros((self.minibatch_size,self.hidden_dim))];
            cell_2 = [T.zeros((self.minibatch_size,self.hidden_dim))];
        if (self.tripleLayer):
            hidden_3 = [T.zeros((self.minibatch_size,self.hidden_dim))];
            cell_3 = [T.zeros((self.minibatch_size,self.hidden_dim))];
    
        # ENCODING PHASE
        init_values = ({'initial': hidden[-1], 'taps': [-1]},
                       {'initial': cell[-1], 'taps': [-1]});
        if (self.doubleLayer):
            init_values = ({'initial': hidden[-1], 'taps': [-1]}, 
                           {'initial': hidden_2[-1], 'taps': [-1]},
                           {'initial': cell[-1], 'taps': [-1]},
                           {'initial': cell_2[-1], 'taps': [-1]});
        if (self.tripleLayer):
            init_values = ({'initial': hidden[-1], 'taps': [-1]}, 
                           {'initial': hidden_2[-1], 'taps': [-1]},
                           {'initial': hidden_3[-1], 'taps': [-1]},
                           {'initial': cell[-1], 'taps': [-1]},
                           {'initial': cell_2[-1], 'taps': [-1]},
                           {'initial': cell_3[-1], 'taps': [-1]});
        outputs, _ = theano.scan(fn=encode_function,
                                          sequences=X,
                                          outputs_info=init_values,
                                          non_sequences=encode_parameters + [0,self.data_dim],
                                          name='encode_scan')
        encoding_hiddens = outputs[:1];
        if (self.doubleLayer):
            encoding_hiddens = outputs[:2];
        if (self.tripleLayer):
            encoding_hiddens = outputs[:3];
        right_hand = T.nnet.softmax(encoding_hiddens[-1][-1].dot(self.vars['hWY']) + self.vars['hbY']);
        right_hand_near_zeros = T.ones_like(right_hand) * 1e-15;
        right_hand = T.maximum(right_hand, right_hand_near_zeros);
        
        # Compute the prediction
        prediction = T.argmax(right_hand, axis=1);
        
        # ERROR COMPUTATION AND PROPAGATION
        coding_dist = right_hand;
        cat_cross = -T.sum(label * T.log(coding_dist), axis=coding_dist.ndim-1);
        error = T.mean(cat_cross);
        summed_error = T.sum(cat_cross);
        
        # Defining prediction function
        self._predict = theano.function([X, label], 
                                        [prediction, right_hand, error, summed_error], 
                                         on_unused_input='ignore');
        
        # Defining stochastic gradient descent
        variables = filter(lambda name: name != 'hbY', self.vars.keys());
        if (self.outputBias):
            variables.append('hbY');
        var_list = map(lambda var: self.vars[var], variables)
        if (self.optimizer == self.SGD_OPTIMIZER):
            # Automatic backward pass for all models: gradients
            derivatives = T.grad(error, var_list);
            updates = [(var,var-self.learning_rate*der) for (var,der) in zip(var_list,derivatives)];
        elif (self.optimizer == self.RMS_OPTIMIZER):
            derivatives = T.grad(error, var_list);
            updates = lasagne.updates.rmsprop(derivatives,var_list,learning_rate=self.learning_rate).items();
        else:
            derivatives = T.grad(error, var_list);
            updates = lasagne.updates.nesterov_momentum(derivatives,var_list,learning_rate=self.learning_rate).items();
        
        # Defining SGD functuin
        self._sgd = theano.function([X, label],
                                    [error, summed_error],
                                    updates=updates,
                                    allow_input_downcast=True,
                                    on_unused_input='ignore');
    
    def loadVars(self, variables, floats=False):
        """
        Provide vars as a dictionary matching the self.vars structure.
        """
        for key in variables:
            if (key not in self.vars):
                return False;
            if (floats):
                self.vars[key].set_value(variables[key].astype('float32'));
            else:
                self.vars[key].set_value(variables[key].get_value().astype('float32'));
        return True;

    def loadPartialDataDimVars(self, variables, offset, size):
        """
        Provide vars as a dictionary matching the self.vars structure.
        Data will be loaded into var[offset:offset+size].
        Only vars that have a data dimension will be overwritten.
        """
        for key in variables:
            if (key not in self.vars):
                return False;
            if (key[0] == "X" or key[:2] == "DX"):
                value = self.vars[key].get_value().astype('float32');
                new_part = variables[key].get_value().astype('float32');
                value[offset:offset+size,:] = new_part;
                self.vars[key].set_value(value);
        return True;

    # PREDICTION FUNCTIONS

    def lstm_cell(self, input, previous_hidden, previous_cell,
                  hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, Pf, Pi, Po, bc, bf, bi, bo, 
                  sd, ed):
        forget_gate = T.nnet.sigmoid(previous_hidden.dot(hWf) + input.dot(XWf[sd:ed,:]) + previous_cell * Pf + bf);
        input_gate = T.nnet.sigmoid(previous_hidden.dot(hWi) + input.dot(XWi[sd:ed,:]) + previous_cell * Pi + bi);
        candidate_cell = T.tanh(previous_hidden.dot(hWc) + input.dot(XWc[sd:ed,:]) + bc);
        cell = forget_gate * previous_cell + input_gate * candidate_cell;
        output_1 = previous_hidden.dot(hWo);
        output_2 = input.dot(XWo[sd:ed,:]);
        output_gate = T.nnet.sigmoid(output_1 + output_2 + previous_cell * Po + bo);
        hidden = output_gate * T.tanh(cell);
        
        return hidden, cell;

    def lstm_dropout(self, layer, dim):
        # Apply dropout (p = 1 - p because p  is chance of dropout and 1 is keep unit)
        if (self.dropoutProb > 0.):
            layer = lasagne.layers.dropout((self.minibatch_size, dim), self.dropoutProb).get_output_for(layer);
        
        return layer;

    def lstm_output(self, hidden, hWY, hbY):
        if (self.outputBias):
            return T.nnet.softmax(hidden.dot(hWY) + hbY);
        else:
            return T.nnet.softmax(hidden.dot(hWY));

    def lstm_single(self, given_X, previous_hidden, previous_cell,
                    hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, Pf, Pi, Po, bc, bf, bi, bo, 
                    hWY, hbY, sd, ed):
        hidden, cell = self.lstm_cell(given_X, previous_hidden, previous_cell, hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, \
                                      Pf, Pi, Po, bc, bf, bi, bo, sd, ed);
        hidden = self.lstm_dropout(hidden, self.hidden_dim);
        Y_output = self.lstm_output(hidden, hWY[:,sd:ed], hbY[sd:ed]);
        Y_output = self.lstm_dropout(Y_output, self.decoding_output_dim);        

        return Y_output, hidden, cell;

    def lstm_interventional_single(self, given_X, previous_output, previous_hidden, previous_cell, sentence_index, intervention_locations,
                            hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, Pf, Pi, Po, bc, bf, bi, bo, 
                            hWY, hbY, sd, ed):
        hidden, cell = self.lstm_cell(previous_output, previous_hidden, previous_cell, hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, \
                                      Pf, Pi, Po, bc, bf, bi, bo, sd, ed);
        hidden = self.lstm_dropout(hidden, self.hidden_dim);
        
        # Use given intervention locations to determine whether to use label
        # or previous prediction. This should allow for flexible minibatching
        comparison = T.le(sentence_index,intervention_locations).reshape((T.constant(2), T.constant(self.minibatch_size), T.constant(1)), ndim=3);
        
        Y_output = self.lstm_output(hidden, hWY, hbY);
        Y_output = self.lstm_dropout(Y_output, self.decoding_output_dim);   

        # Filter for intervention location
        if (not self.only_cause_expression):
            Y_output = T.concatenate([T.switch(comparison[0],given_X[:,:self.data_dim],Y_output[:,:self.data_dim]), T.switch(comparison[1],given_X[:,self.data_dim:],Y_output[:,self.data_dim:])], axis=1)[:,sd:ed];
        else:
            Y_output = T.switch(comparison[0],given_X,Y_output)[:,sd:ed];

        new_sentence_index = sentence_index + 1.;

        return Y_output, hidden, cell, new_sentence_index;

    def lstm_interventional_double(self, given_X, previous_output, previous_hidden_1,
                            previous_hidden_2, previous_cell_1, previous_cell_2, 
                            sentence_index, intervention_locations,
                            hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, Pf, Pi, Po, bc, bf, bi, bo,
                            hWf2, XWf2, hWi2, XWi2, hWc2, XWc2, hWo2, XWo2, Pf2, Pi2, Po2, bc2, bf2, bi2, bo2, 
                            hWY, hbY, sd, ed):
        hidden_1, cell = self.lstm_cell(previous_output, previous_hidden_1, previous_cell_1, hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, \
                                      Pf, Pi, Po, bc, bf, bi, bo, sd, ed);
        hidden_1 = self.lstm_dropout(hidden_1, self.hidden_dim);
        hidden_2, cell_2 = self.lstm_cell(hidden_1, previous_hidden_2, previous_cell_2, hWf2, XWf2, hWi2, XWi2, hWc2, XWc2, hWo2, \
                                        XWo2, Pf2, Pi2, Po2, bc2, bf2, bi2, bo2, 0, self.hidden_dim);
        hidden_2 = self.lstm_dropout(hidden_2, self.hidden_dim);
        
        # Use given intervention locations to determine whether to use label
        # or previous prediction. This should allow for flexible minibatching
        comparison_top = T.le(sentence_index,intervention_locations[0]).reshape((T.constant(self.minibatch_size), T.constant(1)), ndim=2);
        if (not self.only_cause_expression):
            comparison_bot = T.le(sentence_index,intervention_locations[1]).reshape((T.constant(self.minibatch_size), T.constant(1)), ndim=2);
        
        Y_output = self.lstm_output(hidden_2, hWY, hbY);
        Y_output = self.lstm_dropout(Y_output, self.decoding_output_dim);
        
        # Filter for intervention location
        if (not self.only_cause_expression):
            Y_output = T.concatenate([T.switch(comparison_top,given_X[:,:self.data_dim],Y_output[:,:self.data_dim]), T.switch(comparison_bot,given_X[:,self.data_dim:],Y_output[:,self.data_dim:])], axis=1)[:,sd:ed];
        else:
            Y_output = T.switch(comparison_top,given_X,Y_output)[:,sd:ed];

        new_sentence_index = sentence_index + 1.;      

        return Y_output, hidden_1, hidden_2, cell, cell_2, new_sentence_index;

    def lstm_interventional_triple(self, given_X, previous_output, previous_hidden_1,
                            previous_hidden_2, previous_hidden_3, previous_cell_1, previous_cell_2, previous_cell_3,  
                            sentence_index, intervention_locations,
                            hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, Pf, Pi, Po, bc, bf, bi, bo,
                            hWf2, XWf2, hWi2, XWi2, hWc2, XWc2, hWo2, XWo2, Pf2, Pi2, Po2, bc2, bf2, bi2, bo2,
                            hWf3, XWf3, hWi3, XWi3, hWc3, XWc3, hWo3, XWo3, Pf3, Pi3, Po3, bc3, bf3, bi3, bo3, 
                            hWY, hbY, sd, ed):
        hidden_1, cell = self.lstm_cell(previous_output, previous_hidden_1, previous_cell_1, hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, \
                                      Pf, Pi, Po, bc, bf, bi, bo, sd, ed);
        hidden_1 = self.lstm_dropout(hidden_1, self.hidden_dim);
        hidden_2, cell_2 = self.lstm_cell(hidden_1, previous_hidden_2, previous_cell_2, hWf2, XWf2, hWi2, XWi2, hWc2, XWc2, hWo2, \
                                        XWo2, Pf2, Pi2, Po2, bc2, bf2, bi2, bo2, 0, self.hidden_dim);
        hidden_2 = self.lstm_dropout(hidden_2, self.hidden_dim);
        hidden_3, cell_3 = self.lstm_cell(hidden_2, previous_hidden_3, previous_cell_3, hWf3, XWf3, hWi3, XWi3, hWc3, XWc3, hWo3, \
                                        XWo3, Pf3, Pi3, Po3, bc3, bf3, bi3, bo3, 0, self.hidden_dim);
        hidden_3 = self.lstm_dropout(hidden_3, self.hidden_dim);
        
        # Use given intervention locations to determine whether to use label
        # or previous prediction. This should allow for flexible minibatching
        comparison_top = T.le(sentence_index,intervention_locations[0]).reshape((T.constant(self.minibatch_size), T.constant(1)), ndim=2);
        if (not self.only_cause_expression):
            comparison_bot = T.le(sentence_index,intervention_locations[1]).reshape((T.constant(self.minibatch_size), T.constant(1)), ndim=2);
        
        Y_output = self.lstm_output(hidden_3, hWY, hbY);
        Y_output = self.lstm_dropout(Y_output, self.decoding_output_dim);
        
        # Filter for intervention location
        if (not self.only_cause_expression):
            Y_output = T.concatenate([T.switch(comparison_top,given_X[:,:self.data_dim],Y_output[:,:self.data_dim]), T.switch(comparison_bot,given_X[:,self.data_dim:],Y_output[:,self.data_dim:])], axis=1)[:,sd:ed];
        else:
            Y_output = T.switch(comparison_top,given_X,Y_output)[:,sd:ed];

        new_sentence_index = sentence_index + 1.;      

        return Y_output, hidden_1, hidden_2, hidden_3, cell, cell_2, cell_3, new_sentence_index;

    def lstm_double(self, given_X, previous_hidden_1,
                    previous_hidden_2, previous_cell_1, previous_cell_2, 
                    hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, Pf, Pi, Po, bc, bf, bi, bo,
                    hWf2, XWf2, hWi2, XWi2, hWc2, XWc2, hWo2, XWo2, Pf2, Pi2, Po2, bc2, bf2, bi2, bo2, 
                    hWY, hbY, sd, ed):
        hidden_1, cell = self.lstm_cell(given_X, previous_hidden_1, previous_cell_1, hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, \
                                      Pf, Pi, Po, bc, bf, bi, bo, sd, ed);
        hidden_1 = self.lstm_dropout(hidden_1, self.hidden_dim);
        hidden_2, cell_2 = self.lstm_cell(hidden_1, previous_hidden_2, previous_cell_2, hWf2, XWf2, hWi2, XWi2, hWc2, XWc2, hWo2, \
                                        XWo2, Pf2, Pi2, Po2, bc2, bf2, bi2, bo2, 0, self.hidden_dim);
        hidden_2 = self.lstm_dropout(hidden_2, self.hidden_dim);
        
        Y_output = self.lstm_output(hidden_2, hWY[:,sd:ed], hbY[sd:ed]);
        Y_output = self.lstm_dropout(Y_output, self.decoding_output_dim);
        
        return Y_output, hidden_1, hidden_2, cell, cell_2;

    def lstm_double_no_output(self, current_X, previous_hidden_1, previous_hidden_2,
                                      previous_cell_1, previous_cell_2,
                                      hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, Pf, Pi, Po, bc, bf, bi, bo,
                                      hWf2, XWf2, hWi2, XWi2, hWc2, XWc2, hWo2, XWo2, Pf2, Pi2, Po2, bc2, bf2, bi2, bo2, 
                                      sd, ed):
        hidden_1, cell = self.lstm_cell(current_X, previous_hidden_1, previous_cell_1, hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, \
                                      Pf, Pi, Po, bc, bf, bi, bo, sd, ed);
        hidden_1 = self.lstm_dropout(hidden_1, self.hidden_dim);
        hidden_2, cell_2 = self.lstm_cell(hidden_1, previous_hidden_2, previous_cell_2, hWf2, XWf2, hWi2, XWi2, hWc2, XWc2, hWo2, \
                                        XWo2, Pf2, Pi2, Po2, bc2, bf2, bi2, bo2, 0, self.hidden_dim);
        hidden_2 = self.lstm_dropout(hidden_2, self.hidden_dim);

        return hidden_1, hidden_2, cell, cell_2;
    
    def lstm_triple(self, given_X, previous_hidden_1, previous_hidden_2, previous_hidden_3,
                    previous_cell_1, previous_cell_2, previous_cell_3, 
                    hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, Pf, Pi, Po, bc, bf, bi, bo,
                    hWf2, XWf2, hWi2, XWi2, hWc2, XWc2, hWo2, XWo2, Pf2, Pi2, Po2, bc2, bf2, bi2, bo2, 
                    hWf3, XWf3, hWi3, XWi3, hWc3, XWc3, hWo3, XWo3, Pf3, Pi3, Po3, bc3, bf3, bi3, bo3,
                    hWY, hbY, sd, ed):
        hidden_1, cell = self.lstm_cell(given_X, previous_hidden_1, previous_cell_1, hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, \
                                      Pf, Pi, Po, bc, bf, bi, bo, sd, ed);
        hidden_1 = self.lstm_dropout(hidden_1, self.hidden_dim);
        hidden_2, cell_2 = self.lstm_cell(hidden_1, previous_hidden_2, previous_cell_2, hWf2, XWf2, hWi2, XWi2, hWc2, XWc2, hWo2, \
                                        XWo2, Pf2, Pi2, Po2, bc2, bf2, bi2, bo2, 0, self.hidden_dim);
        hidden_2 = self.lstm_dropout(hidden_2, self.hidden_dim);
        hidden_3, cell_3 = self.lstm_cell(hidden_2, previous_hidden_3, previous_cell_3, hWf3, XWf3, hWi3, XWi3, hWc3, XWc3, hWo3, \
                                        XWo3, Pf3, Pi3, Po3, bc3, bf3, bi3, bo3, 0, self.hidden_dim);
        hidden_3 = self.lstm_dropout(hidden_3, self.hidden_dim);
        
        Y_output = self.lstm_output(hidden_3, hWY[:,sd:ed], hbY[sd:ed]);
        Y_output = self.lstm_dropout(Y_output, self.decoding_output_dim);
        
        return Y_output, hidden_1, hidden_2, hidden_3, cell, cell_2, cell_3;
    
    def lstm_triple_no_output(self, current_X, previous_hidden_1, previous_hidden_2, previous_hidden_3,
                              previous_cell_1, previous_cell_2, previous_cell_3,
                              hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, Pf, Pi, Po, bc, bf, bi, bo,
                              hWf2, XWf2, hWi2, XWi2, hWc2, XWc2, hWo2, XWo2, Pf2, Pi2, Po2, bc2, bf2, bi2, bo2,
                              hWf3, XWf3, hWi3, XWi3, hWc3, XWc3, hWo3, XWo3, Pf3, Pi3, Po3, bc3, bf3, bi3, bo3, 
                              sd, ed):
        hidden_1, cell = self.lstm_cell(current_X, previous_hidden_1, previous_cell_1, hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, \
                                      Pf, Pi, Po, bc, bf, bi, bo, sd, ed);
        hidden_1 = self.lstm_dropout(hidden_1, self.hidden_dim);
        hidden_2, cell_2 = self.lstm_cell(hidden_1, previous_hidden_2, previous_cell_2, hWf2, XWf2, hWi2, XWi2, hWc2, XWc2, hWo2, \
                                        XWo2, Pf2, Pi2, Po2, bc2, bf2, bi2, bo2, 0, self.hidden_dim);
        hidden_2 = self.lstm_dropout(hidden_2, self.hidden_dim);
        hidden_3, cell_3 = self.lstm_cell(hidden_2, previous_hidden_3, previous_cell_3, hWf3, XWf3, hWi3, XWi3, hWc3, XWc3, hWo3, \
                                        XWo3, Pf3, Pi3, Po3, bc3, bf3, bi3, bo3, 0, self.hidden_dim);
        hidden_3 = self.lstm_dropout(hidden_3, self.hidden_dim);

        return hidden_1, hidden_2, hidden_3, cell, cell_2, cell_3;

    def rnn_predict_single(self, given_X, previous_output, previous_hidden, previous_cell, sentence_index, intervention_locations,
                            XWh, Xbh, hWh, hbh, hWY, hbY, sd, ed):
        hidden = T.tanh(previous_output.dot(XWh) + Xbh + previous_hidden.dot(hWh) + hbh);

        # Apply dropout (p = 1 - p because p  is chance of dropout and 1 is keep unit)
        if (self.dropoutProb > 0.):
            hidden = lasagne.layers.dropout((self.minibatch_size, self.hidden_dim), self.dropoutProb).get_output_for(hidden);

        # Use given intervention locations to determine whether to use label
        # or previous prediction. This should allow for flexible minibatching
        comparison = T.le(sentence_index,intervention_locations).reshape((T.constant(2), T.constant(self.minibatch_size), T.constant(1)), ndim=3);

        nonlin = T.nnet.softmax;

        if (self.outputBias):
            Y_output = nonlin(hidden.dot(hWY) + hbY);
        else:
            Y_output = nonlin(hidden.dot(hWY));

        # Apply dropout (p = 1 - p because p  is chance of dropout and 1 is keep unit)
        if (self.dropoutProb > 0.):
            Y_output = lasagne.layers.dropout((self.minibatch_size, self.decoding_output_dim), self.dropoutProb).get_output_for(Y_output);

        # Filter for intervention location
        if (not self.only_cause_expression):
            Y_output = T.concatenate([T.switch(comparison[0],given_X[:,:self.data_dim],Y_output[:,:self.data_dim]), T.switch(comparison[1],given_X[:,self.data_dim:],Y_output[:,self.data_dim:])], axis=1)[:,sd:ed];
        else:
            Y_output = T.switch(comparison[0],given_X,Y_output)[:,sd:ed];

        new_sentence_index = sentence_index + 1.;

        return Y_output, hidden, previous_cell, new_sentence_index;

    def rnn_predict_double(self, given_X, previous_output, previous_hidden_1,
                           previous_hidden_2, previous_cell_1, previous_cell_2, 
                           sentence_index, intervention_locations,
                           XWh, Xbh, hWh, hbh,
                           XWh2, Xbh2, hWh2, hbh2,
                           hWY, hbY,
                           sd, ed):
        hidden = T.tanh(previous_output.dot(XWh) + Xbh + previous_hidden_1.dot(hWh) + hbh);

        # Apply dropout (p = 1 - p because p  is chance of dropout and 1 is keep unit)
        if (self.dropoutProb > 0.):
            hidden = lasagne.layers.dropout((self.minibatch_size, self.hidden_dim), self.dropoutProb).get_output_for(hidden);

        hidden_2 = T.tanh(hidden.dot(XWh2) + Xbh2 + previous_hidden_2.dot(hWh2) + hbh2);

        # Use given intervention locations to determine whether to use label
        # or previous prediction. This should allow for flexible minibatching
        comparison = T.le(sentence_index,intervention_locations).reshape((T.constant(2), T.constant(self.minibatch_size), T.constant(1)), ndim=3);

        nonlin = T.nnet.softmax;

        if (self.outputBias):
            Y_output = nonlin(hidden_2.dot(hWY) + hbY);
        else:
            Y_output = nonlin(hidden_2.dot(hWY));

        # Apply dropout (p = 1 - p because p  is chance of dropout and 1 is keep unit)
        if (self.dropoutProb > 0.):
            Y_output = lasagne.layers.dropout((self.minibatch_size, self.decoding_output_dim), self.dropoutProb).get_output_for(Y_output);

        # Filter for intervention location
        if (not self.only_cause_expression):
            Y_output = T.concatenate([T.switch(comparison[0],given_X[:,:self.data_dim],Y_output[:,:self.data_dim]), T.switch(comparison[1],given_X[:,self.data_dim:],Y_output[:,self.data_dim:])], axis=1)[:,sd:ed];
        else:
            Y_output = T.switch(comparison[0],given_X,Y_output)[:,sd:ed];

        new_sentence_index = sentence_index + 1.;

        return Y_output, hidden, hidden_2, previous_cell_1, previous_cell_2, new_sentence_index;

    def rnn_predict_triple(self, given_X, previous_output, previous_hidden, previous_hidden_2, previous_hidden_3, previous_cell_1, previous_cell_2, previous_cell_3, sentence_index, intervention_locations,
                            XWh, Xbh, hWh, hbh,
                            XWh2, Xbh2, hWh2, hbh2,
                            XWh3, Xbh3, hWh3, hbh3,
                            hWY, hbY,
                            sd, ed):
        hidden = T.tanh(previous_output.dot(XWh) + Xbh + previous_hidden.dot(hWh) + hbh);

        # Apply dropout (p = 1 - p because p  is chance of dropout and 1 is keep unit)
        if (self.dropoutProb > 0.):
            hidden = lasagne.layers.dropout((self.minibatch_size, self.hidden_dim), self.dropoutProb).get_output_for(hidden);

        hidden_2 = T.tanh(hidden.dot(XWh2) + Xbh2 + previous_hidden_2.dot(hWh2) + hbh2);

        # Apply dropout (p = 1 - p because p  is chance of dropout and 1 is keep unit)
        if (self.dropoutProb > 0.):
            hidden_2 = lasagne.layers.dropout((self.minibatch_size, self.hidden_dim), self.dropoutProb).get_output_for(hidden_2);

        hidden_3 = T.tanh(hidden_2.dot(XWh3) + Xbh3 + previous_hidden_3.dot(hWh3) + hbh3);

        # Use given intervention locations to determine whether to use label
        # or previous prediction. This should allow for flexible minibatching
        comparison = T.le(sentence_index,intervention_locations).reshape((T.constant(2), T.constant(self.minibatch_size), T.constant(1)), ndim=3);

        nonlin = T.nnet.softmax;

        if (self.outputBias):
            Y_output = nonlin(hidden_3.dot(hWY) + hbY);
        else:
            Y_output = nonlin(hidden_3.dot(hWY));

        # Apply dropout (p = 1 - p because p  is chance of dropout and 1 is keep unit)
        if (self.dropoutProb > 0.):
            Y_output = lasagne.layers.dropout((self.minibatch_size, self.decoding_output_dim), self.dropoutProb).get_output_for(Y_output);

        # Filter for intervention location
        if (not self.only_cause_expression):
            Y_output = T.concatenate([T.switch(comparison[0],given_X[:,:self.data_dim],Y_output[:,:self.data_dim]), T.switch(comparison[1],given_X[:,self.data_dim:],Y_output[:,self.data_dim:])], axis=1)[:,sd:ed];
        else:
            Y_output = T.switch(comparison[0],given_X,Y_output)[:,sd:ed];

        new_sentence_index = sentence_index + 1.;

        return Y_output, hidden, hidden_2, hidden_3, previous_cell_1, previous_cell_2, previous_cell_3, new_sentence_index;

    # END OF INITIALIZATION

    def sanityChecks(self, training_data, training_labels):
        """
        Sanity checks to be called before training a batch. Throws exceptions
        if things are not right.
        """
        if (not self.single_digit and training_labels.shape[1] > (self.n_max_digits+1)):
            raise ValueError("n_max_digits too small! Increase to %d" % training_labels.shape[1]);

    def modelHealth(self):
        totalSum = 0.;
        for varname in self.vars.keys():
            totalSum += np.sum(self.vars[varname].get_value());
        
        return totalSum;

    def plotWeights(self, name='test'):
        import matplotlib;
        matplotlib.use('Agg');
        import matplotlib.pyplot as plt;
        
#         fig, axes = plt.subplots(nrows=3, ncols=4, squeeze=True);
        
        weights = [((0,0),'XWf'),
                   ((0,1),'XWi'),
                   ((0,2),'XWc'),
                   ((0,3),'XWo'),
                   ((1,0),'hWf'),
                   ((1,1),'hWi'),
                   ((1,2),'hWc'),
                   ((1,3),'hWo'),
                   ((2,0),'hWY')];
        for (x, y), title in weights:
            w = self.vars[title].get_value();
            norm = (w - np.min(w)) / float(np.max(w));
            
            ax = plt.subplot2grid((3,4), (x,y));
            ax.imshow(norm, cmap='gray');
            ax.set_title(title);
            ax.set_axis_off();  
        
        # Biases
        w = np.concatenate((self.vars['bf'].get_value().reshape((1,self.hidden_dim)),
                            self.vars['bi'].get_value().reshape((1,self.hidden_dim)),
                            self.vars['bc'].get_value().reshape((1,self.hidden_dim)),
                            self.vars['bo'].get_value().reshape((1,self.hidden_dim))), axis=0);
        norm = (w - np.min(w)) / float(np.max(w));
        ax = plt.subplot2grid((3,4), (2,0));
        ax.imshow(norm, cmap='gray');
        ax.set_title('b(f/i/c/o/Y)');
        ax.set_axis_off();
         
        # Peepholes
        w = np.concatenate((self.vars['Pf'].get_value().reshape((1,self.hidden_dim)),
                            self.vars['Pi'].get_value().reshape((1,self.hidden_dim)),
                            self.vars['Po'].get_value().reshape((1,self.hidden_dim))), axis=0);
        norm = (w - np.min(w)) / float(np.max(w));
        ax = plt.subplot2grid((3,4), (2,1));
        ax.imshow(norm, cmap='gray');
        ax.set_title('P(f/i/o)');
        ax.set_axis_off();
         
        # Output bias
        w = self.vars['hbY'].get_value().reshape((1,self.actual_prediction_output_dim));
        norm = (w - np.min(w)) / float(np.max(w));
        ax = plt.subplot2grid((3,4), (2,2));
        ax.imshow(norm, cmap='gray');
        ax.set_title('hbY');
        ax.set_axis_off();
        
#         plt.show();
        plt.savefig(os.path.join('.','figures',name+'.png'));

    def sgd(self, dataset, data, label, learning_rate, emptySamples=None,
            expressions=None,
            interventionLocations=0, topcause=True,
            bothcause=False, nrSamples=None):
        """
        The intervention location for finish expressions must be the same for
        all samples in this batch.
        """
        if (nrSamples is None):
            nrSamples = self.minibatch_size;

        data = np.swapaxes(data, 0, 1);
        if (self.rnn_version != 2):
            label = np.swapaxes(label, 0, 1);
        
        if (self.rnn_version == 0):
            return self._sgd(data, label, interventionLocations, nrSamples);
        else:
            return self._sgd(data, label);

    def predict(self, encoding_label, prediction_label, interventionLocations=None,
                intervention=True, fixedDecoderInputs=True, topcause=True, nrSamples=None):
        """
        Uses an encoding_label (formerly data) and a prediction_label (formerly
        label) to input to the predictive RNN.
        """
        if (nrSamples is None):
            nrSamples = self.minibatch_size;
        
        # Swap axes of index in sentence and datapoint for Theano purposes
        encoding_label = np.swapaxes(encoding_label, 0, 1);
        if (self.rnn_version != 2):            
            prediction_label = np.swapaxes(prediction_label, 0, 1);

        if (not self.only_cause_expression and self.rnn_version != TheanoRecurrentNeuralNetwork.RNN_DECODESINGLEPREDICTION):
            if (self.rnn_version == TheanoRecurrentNeuralNetwork.RNN_DECODESELFFEEDING):
                prediction_1, prediction_2, right_hand, error, summed_error = \
                        self._predict(encoding_label, prediction_label, interventionLocations, nrSamples);
            else:
                prediction_1, prediction_2, right_hand, error, summed_error = \
                        self._predict(encoding_label, prediction_label);
        else:
            if (self.rnn_version == TheanoRecurrentNeuralNetwork.RNN_DECODESELFFEEDING):
                prediction_1, right_hand, error, summed_error = \
                        self._predict(encoding_label, prediction_label, interventionLocations, nrSamples);
            else:
                prediction_1, right_hand, error, summed_error = \
                        self._predict(encoding_label, prediction_label);

        # Swap sentence index and datapoints back
        if (self.rnn_version != TheanoRecurrentNeuralNetwork.RNN_DECODESINGLEPREDICTION):
            prediction_1 = np.swapaxes(prediction_1, 0, 1);
            if (not self.only_cause_expression):
                prediction_2 = np.swapaxes(prediction_2, 0, 1);
            right_hand = np.swapaxes(right_hand, 0, 1);

        if (not self.only_cause_expression and self.rnn_version != TheanoRecurrentNeuralNetwork.RNN_DECODESINGLEPREDICTION):
            return [prediction_1, prediction_2], {'right_hand': right_hand, 'error': error, 'summed_error': summed_error};
        else:
            return prediction_1, {'right_hand': right_hand, 'error': error, 'summed_error': summed_error};

    def getVars(self):
        return self.vars.items();

    def batch_statistics(self, stats, prediction,
                         target_expressions, intervention_locations,
                         other, test_n, dataset, labels_to_use, dataset_data, parameters,
                         emptySamples=None,
                         training=False, topcause=True,
                         testInDataset=True, bothcause=False,
                         data=None):
        """
        Overriding for finish-target_expressions.
        expressions_with_interventions contains the label-target_expressions (in
        strings) that should be used to lookup the candidate labels for SGD (in
        finish_expression_find_labels)
        """
        notInDataset = [];
        
        if (self.rnn_version == TheanoRecurrentNeuralNetwork.RNN_DECODESINGLEPREDICTION):
            for j in range(0,test_n):
                if (emptySamples is not None and j in emptySamples):
                    continue;
                
                expression = None;
                if (data is not None):
                    expression = dataset.indicesToStr(np.argmax(data[j], axis=1));
                
                x_location = expression.index("x");
                if ('=' in expression):
                    equals_location = expression.index("=");
                    if (x_location < equals_location):
                        equals_location = 'left';
                    else:
                        equals_location = 'right';
                else:
                    equals_location = 'equals';
                stats['x_hand_side_size'][equals_location] += 1;
                stats['x_offset_size'][len(expression)-(x_location+1)] += 1;
                
                stats['symbol_confusion'][target_expressions[j],prediction[j]] += 1;
                
                # Check if cause sequence prediction is in dataset
                if (prediction[j] == target_expressions[j]):
                    stats['correct'] += 1.0;
                    stats['symbol_correct'][dataset.findSymbol[target_expressions[j]]] += 1;
                    stats['x_hand_side_correct'][equals_location] += 1;
                    stats['x_offset_correct'][len(expression)-(x_location+1)] += 1;
                    if (expression is not None):
                        stats['input_size_correct'][len(expression)] += 1;
           
                stats['prediction_1_histogram'][int(prediction[j])] += 1;
                stats['prediction_size'] += 1;
                if (expression is not None):
                    stats['input_sizes'][len(expression)] += 1;
                stats['symbol_size'][dataset.findSymbol[target_expressions[j]]] += 1;
        else:
            dont_switch = False;
            if (len(prediction) <= 1):
                # If we only have one prediction (only_cause_expression) we pad
                # the prediction with an empty one
                prediction.append([]);
                dont_switch = True;
    
            # Statistics
            causeIndex = 0;
            effectIndex = 1;
            if (not topcause and not dont_switch):
                causeIndex = 1;
                effectIndex = 0;
            if (self.only_cause_expression is not False):
                causeIndex = 0;
    
            for j in range(0,test_n):
                if (emptySamples is not None and j in emptySamples):
                    continue;
                
                if (intervention_locations[0,j] >= len(labels_to_use[j][causeIndex]) - 1):
                    stats['skipped_because_intervention_location'] += 1;
                    continue;
    
                # Taking argmax over symbols for each sentence returns
                # the location of the highest index, which is the first
                # EOS symbol
                eos_location = np.argmax(prediction[causeIndex][j]);
                # Check for edge case where no EOS was found and zero was returned
                eos_symbol_index = dataset.EOS_symbol_index;
                if (prediction[causeIndex][j,eos_location] != eos_symbol_index):
                    eos_location = prediction[causeIndex][j].shape[0];
    
                # Convert prediction to string expression
                causeExpressionPrediction = dataset.indicesToStr(prediction[causeIndex][j], ignoreEOS=parameters['dataset_type'] == 3);
                if (not self.only_cause_expression):
                    effectExpressionPrediction = dataset.indicesToStr(prediction[effectIndex][j], ignoreEOS=parameters['dataset_type'] == 3);

                # Prepare vars to save correct samples
                topCorrect = False;
                botCorrect = False;
                
                # Set up labels to use
                label_cause = labels_to_use[j][causeIndex];
                label_effect = labels_to_use[j][effectIndex];
                if (parameters['dataset_type'] == 3):
                    label_cause = label_cause[parameters['lag']:];
                    label_effect = label_effect[parameters['lag']:];

                stats['prediction_sizes'][len(causeExpressionPrediction)-(intervention_locations[0,j]+1)] += 1;
                stats['label_sizes'][len(labels_to_use[j][causeIndex])-(intervention_locations[0,j]+1)] += 1;
                stats['input_sizes'][intervention_locations[0,j]+1] += 1;
                stats['label_size_input_size_confusion_size'][len(labels_to_use[j][causeIndex])-(intervention_locations[0,j]+1),intervention_locations[0,j]+1] += 1;
                stats['correct_matrix_sizes'][len(label_cause)-(intervention_locations[0,j]+1)] += 1;
                
                scoredCorrectAlready = False;
                if (self.only_cause_expression is not False):
                    if (parameters['answering'] == False):
                        # Only for f-seqs
                        # Check for correct / semantically valid
                        valid, correct, left_hand_valid, right_hand_valid = dataset.valid_correct_expression(prediction[causeIndex][j], dataset.digits,dataset.operators);
                        if (valid):
                            stats['syntactically_valid'] += 1;
                        if (correct):
                            stats['semantically_valid'] += 1;
                            # Also score 
                            stats['prediction_size_correct'][len(causeExpressionPrediction)-(intervention_locations[0,j]+1)] += 1.;
                            stats['label_size_correct'][len(labels_to_use[j][causeIndex])-(intervention_locations[0,j]+1)] += 1;
                            stats['input_size_correct'][intervention_locations[0,j]+1] += 1.;
                            stats['label_size_input_size_confusion_correct'][len(labels_to_use[j][causeIndex])-(intervention_locations[0,j]+1),intervention_locations[0,j]+1] += 1;
                            stats['correct_matrix'][len(label_cause)-(intervention_locations[0,j]+1),len(label_cause)-(intervention_locations[0,j]+1)] += 1.;
                            scoredCorrectAlready = True;
                        if (intervention_locations[0,j] < labels_to_use[j][causeIndex].index('=')):
                            stats['left_hand_valid_with_prediction_size'] += 1;
                            if (left_hand_valid):
                                stats['valid_left_hand_valid_with_prediction_size'] += 1;
                                if (correct):
                                    stats['valid_left_hand_valid_with_prediction_correct'] += 1;
                                    stats['left_hand_valid_with_prediction_correct'] += 1;
                        if (left_hand_valid):
                            stats['left_hand_valid'] += 1;
                            if (correct):
                                stats['left_hand_valid_correct'] += 1;
                        if (right_hand_valid):
                            stats['right_hand_valid'] += 1;
    
                # Check if cause sequence prediction is in dataset
                causeMatchesLabel = False;
                if (causeExpressionPrediction == label_cause):
                    stats['structureCorrectCause'] += 1.0;
                    causeMatchesLabel = True;
    
                causeValid = False;
                # Check if cause sequence prediction is valid
                if (dataset.valid_checker(prediction[causeIndex][j],dataset.digits,dataset.operators)):
                    causeValid = True;
                    stats['structureValidCause'] += 1.0;
                
                effectMatchesLabel = False;
                effectValid = False;
                if (not self.only_cause_expression):
                    # Check if effect sequence prediction is in dataset
                    if (effectExpressionPrediction == label_effect):
                        stats['structureCorrectEffect'] += 1.0;
                        effectMatchesLabel = True;
        
                    if (parameters['dataset_type'] != 3):
                        # Check if effect sequence prediction is valid
                        if (dataset.valid_checker(prediction[effectIndex][j],dataset.digits,dataset.operators)):
                            effectValid = True;
                            stats['structureValidEffect'] += 1.0;
    
                # If correct = prediction matches label
                if ((causeMatchesLabel and self.only_cause_expression is not False) or (causeMatchesLabel and effectMatchesLabel)):
                    if (self.only_cause_expression is False):
                        stats['samplesCorrect'].append((True,True));
                    stats['correct'] += 1.0;
                    stats['valid'] += 1.0;
                    stats['inDataset'] += 1.0;
                    if (not scoredCorrectAlready):
                        stats['prediction_size_correct'][len(causeExpressionPrediction)-(intervention_locations[0,j]+1)] += 1.;
                        stats['label_size_correct'][len(labels_to_use[j][causeIndex])-(intervention_locations[0,j]+1)] += 1;
                        stats['input_size_correct'][intervention_locations[0,j]+1] += 1.;
                        stats['correct_matrix'][len(label_cause)-(intervention_locations[0,j]+1),len(label_cause)-(intervention_locations[0,j]+1)] += 1.;
    
                    # Do local scoring for seq2ndmarkov
                    if (self.seq2ndmarkov):
                        stats['localSize'] += float(len(causeExpressionPrediction)/3);
                        stats['localValid'] += float(len(causeExpressionPrediction)/3);
                        if (self.only_cause_expression is False):
                            stats['localValidCause'] += float(len(causeExpressionPrediction)/3);
                            stats['localValidEffect'] += float(len(effectExpressionPrediction)/3);
                else:
                    # If not correct
                    # Save sample correct
                    if (self.only_cause_expression is False):
                        stats['samplesCorrect'].append((topCorrect,botCorrect));
                    # Determine validity of sample if it is not correct
                    if ((causeValid and self.only_cause_expression is not False) or (causeValid and effectValid)):
                        stats['valid'] += 1.0;
                    
                    # Test whether sample appears in dataset
                    if (testInDataset and not training):
                        primeToUse = None;
                        if (self.only_cause_expression is False):
                            primeToUse = effectExpressionPrediction;
                        if (dataset.testExpressionsByPrefix.exists(causeExpressionPrediction, prime=primeToUse)):
                            stats['inDataset'] += 1.0;
                        elif (dataset.expressionsByPrefix.exists(causeExpressionPrediction, prime=primeToUse)):
                            stats['inDataset'] += 1.0;
                        elif (scoredCorrectAlready):
                            notInDataset.append(causeExpressionPrediction);
    
                    # Compute difference between prediction and label
                    difference1 = TheanoRecurrentNeuralNetwork.string_difference(causeExpressionPrediction, label_cause);
                    if (not self.only_cause_expression):
                        difference2 = TheanoRecurrentNeuralNetwork.string_difference(effectExpressionPrediction, label_effect);
                    else:
                        difference2 = 0;
                    difference = difference1 + difference2;
                    
                    # Difference can be larger than the answer size because the 
                    # prediction can be shorter or longer than the label
                    # This truncates difference to not overflow
                    if (difference > len(label_cause)-(intervention_locations[0,j]+1)):
                        difference = len(label_cause)-(intervention_locations[0,j]+1);
                    
                    # Catch glitches where difference = 0 but sample is incorrect
                    if (difference == 0):
                        if (not self.ignoreZeroDifference):
                            raise ValueError("Difference is 0 but sample is not correct! causeIndex: %d, cause: %s, effect: %s, difference1: %d, difference2: %d, cause matches label: %d, effect matches label: %d" %
                                            (causeIndex, causeExpressionPrediction, effectExpressionPrediction, difference1, difference2, int(causeMatchesLabel), int(effectMatchesLabel)));
                        else:
                            difference = 4; # Random digit outside of error margin computation range
                    
                    # Compute error histogram and correct_matrix using difference
                    stats['error_histogram'][difference] += 1;
                    stats['correct_matrix'][len(label_cause)-(intervention_locations[0,j]+1),len(label_cause)-(intervention_locations[0,j]+1)-difference] += 1.;
    
                    # Do local scoring for seq2ndmarkov
                    if (self.seq2ndmarkov):
                        for k in range(2,len(label_cause),3):
                            stats['localSize'] += 1.0;
                            localValidCause = False;
                            localValidEffect = False;
                            if (k+1 < len(causeExpressionPrediction) and dataset.valid_checker(prediction[causeIndex][j][k-2:k+2],dataset.digits,dataset.operators)):
                                if (self.only_cause_expression is False):
                                    localValidCause = True;
                                    stats['localValidCause'] += 1.0;
                                else:
                                    stats['localValid'] += 1.0;
                            if (self.only_cause_expression is False):
                                if (k+1 < len(effectExpressionPrediction) and dataset.valid_checker(prediction[effectIndex][j][k-2:k+2],dataset.digits,dataset.operators)):
                                    localValidEffect = True;
                                    stats['localValidEffect'] += 1.0;
                            if (self.only_cause_expression is False and localValidCause and localValidEffect):
                                stats['localValid'] += 1.0;
    
                # Digit precision and prediction size computation
                if (parameters['rnn_version'] == 0):
                    i = 0;
                    len_to_use = len(label_cause);
                    for i in range(intervention_locations[causeIndex,j]+1,len_to_use):
                        if (i < len(causeExpressionPrediction)):
                            if (causeExpressionPrediction[i] == label_cause[i]):
                                stats['digit_1_correct'][i] += 1.0;
                            stats['digit_1_prediction_size'][i] += 1;
        
                    if (not self.only_cause_expression):
                        i = 0;
                        len_to_use = len(label_effect);
                        for i in range(intervention_locations[effectIndex,j]+1,len_to_use):
                            if (i < len(effectExpressionPrediction)):
                                if (effectExpressionPrediction[i] == label_effect[i]):
                                    stats['digit_2_correct'] += 1.0;
                                stats['digit_2_prediction_size'] += 1;
        
        
                    stats['prediction_1_size_histogram'][int(eos_location)] += 1;
                    for digit_prediction in prediction[causeIndex][j][intervention_locations[causeIndex,j]+1:len(causeExpressionPrediction)]:
                        stats['prediction_1_histogram'][int(digit_prediction)] += 1;
        
                    if (not self.only_cause_expression):
                        stats['prediction_2_size_histogram'][int(eos_location)] += 1;
                        for digit_prediction in prediction[effectIndex][j][intervention_locations[effectIndex,j]+1:len(effectExpressionPrediction)]:
                            stats['prediction_2_histogram'][int(digit_prediction)] += 1;
                else:
                    i = 0;
                    len_to_use = len(label_cause);
                    for i in range(0,len_to_use):
                        if (i < len(causeExpressionPrediction)):
                            if (causeExpressionPrediction[i] == label_cause[i]):
                                stats['digit_1_correct'][i] += 1.0;
                            stats['digit_1_prediction_size'][i] += 1;
                    
                    stats['prediction_1_size_histogram'][int(eos_location)] += 1;
                    for digit_prediction in prediction[causeIndex][j][:len_to_use]:
                        stats['prediction_1_histogram'][int(digit_prediction)] += 1;
        
                    if (not self.only_cause_expression):
                        i = 0;
                        len_to_use = len(label_effect);
                        for i in range(intervention_locations[effectIndex,j]+1,len_to_use):
                            if (i < len(effectExpressionPrediction)):
                                if (effectExpressionPrediction[i] == label_effect[i]):
                                    stats['digit_2_correct'][i] += 1.0;
                                stats['digit_2_prediction_size'][i] += 1;
        
        
                    if (not self.only_cause_expression):
                        stats['prediction_2_size_histogram'][int(eos_location)] += 1;
                        for digit_prediction in prediction[effectIndex][j][:len_to_use]:
                            stats['prediction_2_histogram'][int(digit_prediction)] += 1;
    
                stats['prediction_size'] += 1;

        return stats, labels_to_use, notInDataset;

    @staticmethod
    def string_difference(string1, string2):
        # Compute string difference
        score = 0;
        string1len = len(string1);
        k = 0;
        for k,s in enumerate(string2):
            if (string1len <= k):
                score += 1;
            elif (s != string1[k]):
                score += 1;
        score += max(0,len(string1) - (k+1));
        return score;
