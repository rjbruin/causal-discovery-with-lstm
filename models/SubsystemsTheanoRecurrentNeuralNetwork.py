'''
Created on 22 feb. 2016

@author: Robert-Jan
'''

import theano;
import theano.tensor as T;
# from theano.compile.nanguardmode import NanGuardMode

import numpy as np;
from models.RecurrentModel import RecurrentModel
#from theano.compile.nanguardmode import NanGuardMode
import lasagne;

from profiler import profiler;

class SubsystemsTheanoRecurrentNeuralNetwork(RecurrentModel):
    '''
    Recurrent neural network model with one hidden layer. Models single class
    prediction based on regular recurrent model or LSTM model.
    '''

    SGD_OPTIMIZER = 0;
    RMS_OPTIMIZER = 1;
    NESTEROV_OPTIMIZER = 2;

    def __init__(self, data_dim, hidden_dim, output_dim, minibatch_size,
                 lstm=True, weight_values={}, single_digit=False,
                 EOS_symbol_index=None, GO_symbol_index=None, n_max_digits=24,
                 decoder=False, verboseOutputter=None, finishExpressions=True,
                 optimizer=0, learning_rate=0.01,
                 operators=4, digits=10, only_cause_expression=False, seq2ndmarkov=False,
                 doubleLayer=False, tripleLayer=False, dropoutProb=0., outputBias=False,
                 crosslinks=True, appendAbstract=False, useAbstract=False, relu=False,
                 ignoreZeroDifference=False, peepholes=False, lstm_biases=False, lag=None):
        '''
        Initialize all Theano models.
        '''
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
        self.dropoutProb = dropoutProb;
        self.outputBias = outputBias;
        self.crosslinks = crosslinks;
        self.useAbstract = useAbstract;
        self.appendAbstract = appendAbstract;
        self.relu = relu;
        self.ignoreZeroDifference = ignoreZeroDifference;
        self.peepholes = peepholes;
        self.lstm_biases = lstm_biases;
        self.lag = lag;

        if (not self.lstm):
            raise ValueError("Feature LSTM = False is no longer supported!");

        if (self.doubleLayer or self.tripleLayer):
            raise ValueError("NOT SUPPORTED!");
        
        self.EOS_symbol_index = EOS_symbol_index;
        self.GO_symbol_index = GO_symbol_index;

        if (self.GO_symbol_index is None):
            raise ValueError("GO symbol index not set!");

        # Set dimensions
        self.data_dim = data_dim;
        self.hidden_dim = hidden_dim;
        self.decoding_output_dim = output_dim;
        self.prediction_output_dim = output_dim;

        per_expression_dim = self.data_dim;

        actual_data_dim = per_expression_dim * 2;
        actual_prediction_output_dim = self.prediction_output_dim * 2;
        if (self.only_cause_expression):
            actual_data_dim = per_expression_dim;
            actual_prediction_output_dim = self.prediction_output_dim;

        # Set up shared variables
        varSettings = [];
        if (self.lstm):
            varSettings.append(('hWf',self.hidden_dim,self.hidden_dim));
            varSettings.append(('XWf',actual_data_dim,self.hidden_dim));
            varSettings.append(('hWi',self.hidden_dim,self.hidden_dim));
            varSettings.append(('XWi',actual_data_dim,self.hidden_dim));
            varSettings.append(('hWc',self.hidden_dim,self.hidden_dim));
            varSettings.append(('XWc',actual_data_dim,self.hidden_dim));
            varSettings.append(('hWo',self.hidden_dim,self.hidden_dim));
            varSettings.append(('XWo',actual_data_dim,self.hidden_dim));
            if (self.peepholes):
                # Peephole connections for forget, input and output gate
                varSettings.append(('Pf',False,self.hidden_dim));
                varSettings.append(('Pi',False,self.hidden_dim));
                varSettings.append(('Po',False,self.hidden_dim));
            if (self.lstm_biases):
                # Biases for candidate cell, forget, input and output gates
                varSettings.append(('bc',False,self.hidden_dim));
                varSettings.append(('bf',False,self.hidden_dim));
                varSettings.append(('bi',False,self.hidden_dim));
                varSettings.append(('bo',False,self.hidden_dim));
        else:
            varSettings.append(('XWh',actual_data_dim,self.hidden_dim));
            varSettings.append(('Xbh',False,self.hidden_dim));
            varSettings.append(('hWh',self.hidden_dim,self.hidden_dim));
            varSettings.append(('hbh',False,self.hidden_dim));

        if (self.doubleLayer or self.tripleLayer):
            if (self.lstm):
                varSettings.append(('hWf2',self.hidden_dim,self.hidden_dim));
                varSettings.append(('XWf2',self.hidden_dim,self.hidden_dim));
                varSettings.append(('hWi2',self.hidden_dim,self.hidden_dim));
                varSettings.append(('XWi2',self.hidden_dim,self.hidden_dim));
                varSettings.append(('hWc2',self.hidden_dim,self.hidden_dim));
                varSettings.append(('XWc2',self.hidden_dim,self.hidden_dim));
                varSettings.append(('hWo2',self.hidden_dim,self.hidden_dim));
                varSettings.append(('XWo2',self.hidden_dim,self.hidden_dim));
                if (self.peepholes):
                    # Peephole connections for forget, input and output gate
                    varSettings.append(('Pf2',False,self.hidden_dim));
                    varSettings.append(('Pi2',False,self.hidden_dim));
                    varSettings.append(('Po2',False,self.hidden_dim));
                if (self.lstm_biases):
                    # Biases for candidate cell, forget, input and output gates
                    varSettings.append(('bc2',False,self.hidden_dim));
                    varSettings.append(('bf2',False,self.hidden_dim));
                    varSettings.append(('bi2',False,self.hidden_dim));
                    varSettings.append(('bo2',False,self.hidden_dim));
            else:
                varSettings.append(('XWh2',self.hidden_dim,self.hidden_dim));
                varSettings.append(('Xbh2',False,self.hidden_dim));
                varSettings.append(('hWh2',self.hidden_dim,self.hidden_dim));
                varSettings.append(('hbh2',False,self.hidden_dim));
        if (self.tripleLayer):
            if (self.lstm):
                varSettings.append(('hWf3',self.hidden_dim,self.hidden_dim));
                varSettings.append(('XWf3',self.hidden_dim,self.hidden_dim));
                varSettings.append(('hWi3',self.hidden_dim,self.hidden_dim));
                varSettings.append(('XWi3',self.hidden_dim,self.hidden_dim));
                varSettings.append(('hWc3',self.hidden_dim,self.hidden_dim));
                varSettings.append(('XWc3',self.hidden_dim,self.hidden_dim));
                varSettings.append(('hWo3',self.hidden_dim,self.hidden_dim));
                varSettings.append(('XWo3',self.hidden_dim,self.hidden_dim));
                if (self.peepholes):
                    # Peephole connections for forget, input and output gate
                    varSettings.append(('Pf3',False,self.hidden_dim));
                    varSettings.append(('Pi3',False,self.hidden_dim));
                    varSettings.append(('Po3',False,self.hidden_dim));
                if (self.lstm_biases):
                    # Biases for candidate cell, forget, input and output gates
                    varSettings.append(('bc3',False,self.hidden_dim));
                    varSettings.append(('bf3',False,self.hidden_dim));
                    varSettings.append(('bi3',False,self.hidden_dim));
                    varSettings.append(('bo3',False,self.hidden_dim));
            else:
                varSettings.append(('XWh3',self.hidden_dim,self.hidden_dim));
                varSettings.append(('Xbh3',False,self.hidden_dim));
                varSettings.append(('hWh3',self.hidden_dim,self.hidden_dim));
                varSettings.append(('hbh3',False,self.hidden_dim));

        varSettings.append(('hWY',self.hidden_dim,actual_prediction_output_dim));
        varSettings.append(('hbY',False,actual_prediction_output_dim));

        # Contruct variables
        self.vars = {};
        for (varName,dim1,dim2) in varSettings:
            # Get value for shared variable from constructor if present
            if (dim1 is not False):
                value = np.random.uniform(-np.sqrt(1.0/dim1),np.sqrt(1.0/dim1),(dim1,dim2)).astype('float32');
            else:
                value = np.random.uniform(-np.sqrt(1.0/dim2),-np.sqrt(1.0/dim2),(dim2)).astype('float32');
            if (varName in weight_values):
                value = weight_values[varName];
            self.vars[varName] = theano.shared(value, varName);

        # Set up inputs to prediction and SGD
        # X is 3-dimensional: 1) index in sentence, 2) datapoint, 3) dimensionality of data
        X = T.ftensor3('X');
        # label is 3-dimensional: 1) index in sentence, 2) datapoint, 3) dimensionality of data
        label = T.ftensor3('label');
        lag = T.iscalar('lag');
        nrSamples = T.iscalar();

        # Set the RNN cell to use for encoding and decoding
        encode_function = self.lstm_predict_single_no_output;
        decode_function = self.lstm_predict_single if self.lstm else self.rnn_predict_single;
        if (self.doubleLayer):
            encode_function = self.lstm_predict_double_no_output;
            decode_function = self.lstm_predict_double if self.lstm else self.rnn_predict_double;
        if (self.tripleLayer):
            encode_function = self.lstm_predict_triple_no_output;
            decode_function = self.lstm_predict_triple if self.lstm else self.rnn_predict_triple;

        if (self.dropoutProb > 0.):
            self.random_stream = T.shared_randomstreams.RandomStreams(seed=np.random.randint(10000));

        # Set the prediction parameters to be either the prediction
        # weights or the decoding weights depending on the setting
        encode_parameters = [self.vars[k[0]] for k in varSettings if k[0][-1] != 'Y'];
        decode_parameters = [self.vars[k[0]] for k in varSettings];

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
                                     sequences=label[:lag],
                                     outputs_info=init_values,
                                     non_sequences=encode_parameters + [0,actual_data_dim],
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
                           {'initial': cell[-1], 'taps': [-1]});
            if (self.doubleLayer):
                init_values = ({'initial': hidden_top[-1], 'taps': [-1]},
                               {'initial': hidden_2_top[-1], 'taps': [-1]},
                               {'initial': cell_top[-1], 'taps': [-1]},
                               {'initial': cell_2_top[-1], 'taps': [-1]});
            if (self.tripleLayer):
                init_values = ({'initial': hidden[-1], 'taps': [-1]},
                               {'initial': hidden_2_top[-1], 'taps': [-1]},
                               {'initial': hidden_3_top[-1], 'taps': [-1]},
                               {'initial': cell[-1], 'taps': [-1]}, 
                               {'initial': cell_2_top[-1], 'taps': [-1]}, 
                               {'initial': cell_3_top[-1], 'taps': [-1]});
            outputs_1, _ = theano.scan(fn=encode_function,
                                     sequences=label[:lag],
                                     outputs_info=init_values,
                                     non_sequences=encode_parameters + [0,per_expression_dim],
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
                               {'initial': cell[-1], 'taps': [-1]});
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
                                         sequences=label[:lag],
                                         outputs_info=init_values,
                                         non_sequences=encode_parameters + [per_expression_dim,per_expression_dim*2],
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
                                     sequences=label,
                                     outputs_info=init_values,
                                     non_sequences=decode_parameters + [0,actual_data_dim],
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
                               {'initial': learned_hidden[-1], 'taps': [-1]},
                               {'initial': learned_hidden_2_top[-1], 'taps': [-1]},
                               {'initial': learned_hidden_3_top[-1], 'taps': [-1]},
                               {'initial': learned_cell_top[-1], 'taps': [-1]}, 
                               {'initial': learned_cell_2_top[-1], 'taps': [-1]}, 
                               {'initial': learned_cell_3_top[-1], 'taps': [-1]});
            outputs_1, _ = theano.scan(fn=decode_function,
                                     sequences=label,
                                     outputs_info=init_values,
                                     non_sequences=decode_parameters + [0,per_expression_dim],
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
                                         sequences=label,
                                         outputs_info=init_values,
                                         non_sequences=decode_parameters + [per_expression_dim,per_expression_dim*2],
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
        coding_dist = right_hand[lag:]
        cat_cross = -T.sum(label[lag:] * T.log(coding_dist), axis=coding_dist.ndim-1);
        mean_cross_per_sample = T.mean(cat_cross, axis=0);
        error = T.mean(mean_cross_per_sample[:nrSamples]);

        # Defining prediction function
        if (not self.only_cause_expression):
            self._predict = theano.function([X, label, lag, nrSamples], [prediction_1,
                                                                                prediction_2,
                                                                                right_hand,
                                                                                error], on_unused_input='ignore');
        else:
            self._predict = theano.function([X, label, lag, nrSamples], [prediction_1,
                                                                                right_hand,
                                                                                error], on_unused_input='ignore');

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
        self._sgd = theano.function([X, label, lag, nrSamples],
                                        [error],
                                    updates=updates,
                                    allow_input_downcast=True,
                                    on_unused_input='ignore')

        super(SubsystemsTheanoRecurrentNeuralNetwork, self).__init__();

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

    def lstm_predict_single(self, given_X, previous_hidden, previous_cell,
                            hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, Pf, Pi, Po, bc, bf, bi, bo, hWY, hbY, sd, ed):
        forget_gate = T.nnet.sigmoid(previous_hidden.dot(hWf) + given_X.dot(XWf[sd:ed,:]) + previous_cell * Pf + bf);
        input_gate = T.nnet.sigmoid(previous_hidden.dot(hWi) + given_X.dot(XWi[sd:ed,:]) + previous_cell * Pi + bi);
        candidate_cell = T.tanh(previous_hidden.dot(hWc) + given_X.dot(XWc[sd:ed,:]) + bc);
        cell = forget_gate * previous_cell + input_gate * candidate_cell;
        output_gate = T.nnet.sigmoid(previous_hidden.dot(hWo) + given_X.dot(XWo[sd:ed,:]) + previous_cell * Po + bo);
        hidden = output_gate * T.tanh(cell);

        # Apply dropout (p = 1 - p because p  is chance of dropout and 1 is keep unit)
        if (self.dropoutProb > 0.):
            hidden = lasagne.layers.dropout((self.minibatch_size, self.hidden_dim), self.dropoutProb).get_output_for(hidden);

        if (self.outputBias):
            Y_output = T.nnet.softmax(hidden.dot(hWY) + hbY);
        else:
            Y_output = T.nnet.softmax(hidden.dot(hWY));

        # Apply dropout (p = 1 - p because p  is chance of dropout and 1 is keep unit)
        if (self.dropoutProb > 0.):
            Y_output = lasagne.layers.dropout((self.minibatch_size, self.decoding_output_dim), self.dropoutProb).get_output_for(Y_output);

        return Y_output, hidden, cell;

    def lstm_predict_single_no_output(self, current_X, previous_hidden, previous_cell, hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, Pf, Pi, Po, bc, bf, bi, bo, sd, ed):
        forget_gate = T.nnet.sigmoid(previous_hidden.dot(hWf) + current_X.dot(XWf[sd:ed,:]) + previous_cell * Pf + bf);
        input_gate = T.nnet.sigmoid(previous_hidden.dot(hWi) + current_X.dot(XWi[sd:ed,:]) + previous_cell * Pi + bi);
        candidate_cell = T.tanh(previous_hidden.dot(hWc) + current_X.dot(XWc[sd:ed,:]) + bc);
        cell = forget_gate * previous_cell + input_gate * candidate_cell;
        output_gate = T.nnet.sigmoid(previous_hidden.dot(hWo) + current_X.dot(XWo[sd:ed,:]) + previous_cell * Po + bo);
        hidden = output_gate * T.tanh(cell);

        return hidden, cell;

    def lstm_predict_double(self, given_X, previous_output, previous_hidden_1,
                            previous_hidden_2, previous_cell_1, previous_cell_2,
                            hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, Pf, Pi, Po, bc, bf, bi, bo, 
                            hWf2, XWf2, hWi2, XWi2, hWc2, XWc2, hWo2, XWo2, Pf2, Pi2, Po2, bc2, bf2, bi2, bo2, hWY, hbY, sd, ed):
        forget_gate = T.nnet.sigmoid(previous_hidden_1.dot(hWf) + previous_output.dot(XWf) + previous_cell_1 * Pf + bf);
        input_gate = T.nnet.sigmoid(previous_hidden_1.dot(hWi) + previous_output.dot(XWi) + previous_cell_1 * Pi + bi);
        candidate_cell = T.tanh(previous_hidden_1.dot(hWc) + previous_output.dot(XWc) + bc);
        cell = forget_gate * previous_cell_1 + input_gate * candidate_cell;
        output_gate = T.nnet.sigmoid(previous_hidden_1.dot(hWo) + previous_output.dot(XWo) + previous_cell_1 * Po + bo);
        hidden_1 = output_gate * T.tanh(cell);

        # Apply dropout (p = 1 - p because p  is chance of dropout and 1 is keep unit)
        if (self.dropoutProb > 0.):
            hidden_1 = lasagne.layers.dropout((self.minibatch_size, self.hidden_dim), self.dropoutProb).get_output_for(hidden_1);

        forget_gate_2 = T.nnet.sigmoid(previous_hidden_2.dot(hWf2) + hidden_1.dot(XWf2) + previous_cell_2 * Pf2 + bf2);
        input_gate_2 = T.nnet.sigmoid(previous_hidden_2.dot(hWi2) + hidden_1.dot(XWi2) + previous_cell_2 * Pi2 + bi2);
        candidate_cell_2 = T.tanh(previous_hidden_2.dot(hWc2) + hidden_1.dot(XWc2) + bc2);
        cell_2 = forget_gate_2 * previous_cell_2 + input_gate_2 * candidate_cell_2;
        output_gate_2 = T.nnet.sigmoid(previous_hidden_2.dot(hWo2) + hidden_1.dot(XWo2) + previous_cell_2 * Po2 + bo2);
        hidden_2 = output_gate_2 * T.tanh(cell_2);

        # Apply dropout (p = 1 - p because p  is chance of dropout and 1 is keep unit)
        if (self.dropoutProb > 0.):
            hidden_2 = lasagne.layers.dropout((self.minibatch_size, self.hidden_dim), self.dropoutProb).get_output_for(hidden_2);

        if (self.outputBias):
            Y_output = T.nnet.softmax(hidden_2.dot(hWY) + hbY);
        else:
            Y_output = T.nnet.softmax(hidden_2.dot(hWY));

        # Apply dropout (p = 1 - p because p  is chance of dropout and 1 is keep unit)
        if (self.dropoutProb > 0.):
            Y_output = lasagne.layers.dropout((self.minibatch_size, self.decoding_output_dim), self.dropoutProb).get_output_for(Y_output);

        return Y_output, hidden_1, hidden_2, cell, cell_2;

    def lstm_predict_double_no_output(self, current_X, previous_hidden_1, previous_hidden_2,
                                      previous_cell_1, previous_cell_2,
                                      hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, Pf, Pi, Po, bc, bf, bi, bo,
                                      hWf2, XWf2, hWi2, XWi2, hWc2, XWc2, hWo2, XWo2, Pf2, Pi2, Po2, bc2, bf2, bi2, bo2, sd, ed):
        forget_gate = T.nnet.sigmoid(previous_hidden_1.dot(hWf) + current_X.dot(XWf[sd:ed,:]) + previous_cell_1 * Pf + bf);
        input_gate = T.nnet.sigmoid(previous_hidden_1.dot(hWi) + current_X.dot(XWi[sd:ed,:]) + previous_cell_1 * Pi + bi);
        candidate_cell = T.tanh(previous_hidden_1.dot(hWc) + current_X.dot(XWc[sd:ed,:]) + bc);
        cell = forget_gate * previous_cell_1 + input_gate * candidate_cell;
        output_gate = T.nnet.sigmoid(previous_hidden_1.dot(hWo) + current_X.dot(XWo[sd:ed,:]) + previous_cell_1 * Po + bo);
        hidden_1 = output_gate * T.tanh(cell);

        forget_gate_2 = T.nnet.sigmoid(previous_hidden_2.dot(hWf2) + hidden_1.dot(XWf2) + previous_cell_2 * Pf2 + bf2);
        input_gate_2 = T.nnet.sigmoid(previous_hidden_2.dot(hWi2) + hidden_1.dot(XWi2) + previous_cell_2 * Pi2 + bi2);
        candidate_cell_2 = T.tanh(previous_hidden_2.dot(hWc2) + hidden_1.dot(XWc2) + bc2);
        cell_2 = forget_gate_2 * previous_cell_2 + input_gate_2 * candidate_cell_2;
        output_gate_2 = T.nnet.sigmoid(previous_hidden_2.dot(hWo2) + hidden_1.dot(XWo2) + previous_cell_2 * Po2 + bo2);
        hidden_2 = output_gate_2 * T.tanh(cell_2);

        return hidden_1, hidden_2, cell, cell_2;

    def lstm_predict_triple(self, given_X, previous_output, previous_hidden_1,
                            previous_hidden_2, previous_hidden_3, 
                            previous_cell_1, previous_cell_2, previous_cell_3,
                            hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, Pf, Pi, Po, bc, bf, bi, bo,
                            hWf2, XWf2, hWi2, XWi2, hWc2, XWc2, hWo2, XWo2, Pf2, Pi2, Po2, bc2, bf2, bi2, bo2, 
                            hWf3, XWf3, hWi3, XWi3, hWc3, XWc3, hWo3, XWo3, Pf3, Pi3, Po3, bc3, bf3, bi3, bo3,
                            hWY, hbY, sd, ed):
        forget_gate = T.nnet.sigmoid(previous_hidden_1.dot(hWf) + previous_output.dot(XWf[sd:ed,:]) + previous_cell_1 * Pf + bf);
        input_gate = T.nnet.sigmoid(previous_hidden_1.dot(hWi) + previous_output.dot(XWi[sd:ed,:]) + previous_cell_1 * Pi + bi);
        candidate_cell = T.tanh(previous_hidden_1.dot(hWc) + previous_output.dot(XWc[sd:ed,:]) + bc);
        cell = forget_gate * previous_cell_1 + input_gate * candidate_cell;
        output_gate = T.nnet.sigmoid(previous_hidden_1.dot(hWo) + previous_output.dot(XWo[sd:ed,:]) + previous_cell_1 * Po + bo);
        hidden_1 = output_gate * T.tanh(cell);

        # Apply dropout (p = 1 - p because p  is chance of dropout and 1 is keep unit)
        if (self.dropoutProb > 0.):
            hidden_1 = lasagne.layers.dropout((self.minibatch_size, self.hidden_dim), self.dropoutProb).get_output_for(hidden_1);

        forget_gate_2 = T.nnet.sigmoid(previous_hidden_2.dot(hWf2) + hidden_1.dot(XWf2) + previous_cell_2 * Pf2 + bf2);
        input_gate_2 = T.nnet.sigmoid(previous_hidden_2.dot(hWi2) + hidden_1.dot(XWi2) + previous_cell_2 * Pi2 + bi2);
        candidate_cell_2 = T.tanh(previous_hidden_2.dot(hWc2) + hidden_1.dot(XWc2) + bc2);
        cell_2 = forget_gate_2 * previous_cell_2 + input_gate_2 * candidate_cell_2;
        output_gate_2 = T.nnet.sigmoid(previous_hidden_2.dot(hWo2) + hidden_1.dot(XWo2) + previous_cell_2 * Po2 + bo2);
        hidden_2 = output_gate_2 * T.tanh(cell_2);

        # Apply dropout (p = 1 - p because p  is chance of dropout and 1 is keep unit)
        if (self.dropoutProb > 0.):
            hidden_2 = lasagne.layers.dropout((self.minibatch_size, self.hidden_dim), self.dropoutProb).get_output_for(hidden_2);

        forget_gate_3 = T.nnet.sigmoid(previous_hidden_3.dot(hWf3) + hidden_2.dot(XWf3) + previous_cell_3 * Pf3 + bf3);
        input_gate_3 = T.nnet.sigmoid(previous_hidden_3.dot(hWi3) + hidden_2.dot(XWi3) + previous_cell_3 * Pi3 + bi3);
        candidate_cell_3 = T.tanh(previous_hidden_3.dot(hWc3) + hidden_2.dot(XWc3) + bc3);
        cell_3 = forget_gate_3 * previous_cell_3 + input_gate_3 * candidate_cell_3;
        output_gate_3 = T.nnet.sigmoid(previous_hidden_3.dot(hWo3) + hidden_2.dot(XWo3) + previous_cell_3 * Po3 + bo3);
        hidden_3 = output_gate_3 * T.tanh(cell_3);

        # Apply dropout (p = 1 - p because p  is chance of dropout and 1 is keep unit)
        if (self.dropoutProb > 0.):
            hidden_3 = lasagne.layers.dropout((self.minibatch_size, self.hidden_dim), self.dropoutProb).get_output_for(hidden_3);

        if (self.outputBias):
            Y_output = T.nnet.softmax(hidden_3.dot(hWY) + hbY);
        else:
            Y_output = T.nnet.softmax(hidden_3.dot(hWY));

        # Apply dropout (p = 1 - p because p  is chance of dropout and 1 is keep unit)
        if (self.dropoutProb > 0.):
            Y_output = lasagne.layers.dropout((self.minibatch_size, self.decoding_output_dim), self.dropoutProb).get_output_for(Y_output);

        return Y_output, hidden_1, hidden_2, hidden_3, cell, cell_2, cell_3;

    def lstm_predict_triple_no_output(self, given_X, previous_output, previous_hidden_1,
                            previous_hidden_2, previous_hidden_3, 
                            previous_cell_1, previous_cell_2, previous_cell_3,
                            hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, Pf, Pi, Po, bc, bf, bi, bo,
                            hWf2, XWf2, hWi2, XWi2, hWc2, XWc2, hWo2, XWo2, Pf2, Pi2, Po2, bc2, bf2, bi2, bo2, 
                            hWf3, XWf3, hWi3, XWi3, hWc3, XWc3, hWo3, XWo3, Pf3, Pi3, Po3, bc3, bf3, bi3, bo3,
                            hWY, hbY, sd, ed):
        forget_gate = T.nnet.sigmoid(previous_hidden_1.dot(hWf) + previous_output.dot(XWf[sd:ed,:]) + previous_cell_1 * Pf + bf);
        input_gate = T.nnet.sigmoid(previous_hidden_1.dot(hWi) + previous_output.dot(XWi[sd:ed,:]) + previous_cell_1 * Pi + bi);
        candidate_cell = T.tanh(previous_hidden_1.dot(hWc) + previous_output.dot(XWc[sd:ed,:]) + bc);
        cell = forget_gate * previous_cell_1 + input_gate * candidate_cell;
        output_gate = T.nnet.sigmoid(previous_hidden_1.dot(hWo) + previous_output.dot(XWo[sd:ed,:]) + previous_cell_1 * Po + bo);
        hidden_1 = output_gate * T.tanh(cell);

        forget_gate_2 = T.nnet.sigmoid(previous_hidden_2.dot(hWf2) + hidden_1.dot(XWf2) + previous_cell_2 * Pf2 + bf2);
        input_gate_2 = T.nnet.sigmoid(previous_hidden_2.dot(hWi2) + hidden_1.dot(XWi2) + previous_cell_2 * Pi2 + bi2);
        candidate_cell_2 = T.tanh(previous_hidden_2.dot(hWc2) + hidden_1.dot(XWc2) + bc2);
        cell_2 = forget_gate_2 * previous_cell_2 + input_gate_2 * candidate_cell_2;
        output_gate_2 = T.nnet.sigmoid(previous_hidden_2.dot(hWo2) + hidden_1.dot(XWo2) + previous_cell_2 * Po2 + bo2);
        hidden_2 = output_gate_2 * T.tanh(cell_2);

        forget_gate_3 = T.nnet.sigmoid(previous_hidden_3.dot(hWf3) + hidden_2.dot(XWf3) + previous_cell_3 * Pf3 + bf3);
        input_gate_3 = T.nnet.sigmoid(previous_hidden_3.dot(hWi3) + hidden_2.dot(XWi3) + previous_cell_3 * Pi3 + bi3);
        candidate_cell_3 = T.tanh(previous_hidden_3.dot(hWc3) + hidden_2.dot(XWc3) + bc3);
        cell_3 = forget_gate_3 * previous_cell_3 + input_gate_3 * candidate_cell_3;
        output_gate_3 = T.nnet.sigmoid(previous_hidden_3.dot(hWo3) + hidden_2.dot(XWo3) + previous_cell_3 * Po3 + bo3);
        hidden_3 = output_gate_3 * T.tanh(cell_3);

        return hidden_1, hidden_2, hidden_3, cell, cell_2, cell_3;

    def rnn_predict_single(self, given_X, previous_output, previous_hidden, sentence_index, intervention_locations,
                            XWh, Xbh, hWh, hbh, hWY, hbY, sd, ed, abstractExpressions):
        if (self.appendAbstract):
            previous_output = T.concatenate([previous_output, abstractExpressions], 1);

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

        return Y_output, hidden, new_sentence_index;

    def rnn_predict_double(self, given_X, previous_output, previous_hidden, previous_hidden_2, sentence_index, intervention_locations,
                            XWh, Xbh, hWh, hbh,
                            XWh2, Xbh2, hWh2, hbh2,
                            hWY, hbY,
                            sd, ed, abstractExpressions):
        if (self.appendAbstract):
            previous_output = T.concatenate([previous_output, abstractExpressions], 1);

        hidden = T.tanh(previous_output.dot(XWh) + Xbh + previous_hidden.dot(hWh) + hbh);

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

        return Y_output, hidden, hidden_2, new_sentence_index;

    def rnn_predict_triple(self, given_X, previous_output, previous_hidden, previous_hidden_2, previous_hidden_3, sentence_index, intervention_locations,
                            XWh, Xbh, hWh, hbh,
                            XWh2, Xbh2, hWh2, hbh2,
                            XWh3, Xbh3, hWh3, hbh3,
                            hWY, hbY,
                            sd, ed, abstractExpressions):
        if (self.appendAbstract):
            previous_output = T.concatenate([previous_output, abstractExpressions], 1);

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

        return Y_output, hidden, hidden_2, hidden_3, new_sentence_index;

    # END OF INITIALIZATION

    def sanityChecks(self, training_data, training_labels):
        """
        Sanity checks to be called before training a batch. Throws exceptions
        if things are not right.
        """
        if (not self.single_digit and training_labels.shape[1] > (self.n_max_digits+1)):
            raise ValueError("n_max_digits too small! Increase to %d" % training_labels.shape[1]);

    def sgd(self, dataset, data, label, learning_rate, emptySamples=None,
            expressions=None,
            interventionLocations=0, topcause=True,
            bothcause=False, nrSamples=None,
            abstractExpressions=None):
        """
        The intervention location for finish expressions must be the same for
        all samples in this batch.
        """
        if (nrSamples is None):
            nrSamples = self.minibatch_size;
        
        data = np.swapaxes(data, 0, 1);
        label = np.swapaxes(label, 0, 1);
        return self._sgd(data, label, self.lag, nrSamples);

    def predict(self, encoding_label, prediction_label, interventionLocations=None,
                intervention=True, fixedDecoderInputs=True, topcause=True, nrSamples=None,
                abstractExpressions=None):
        """
        Uses an encoding_label (formerly data) and a prediction_label (formerly
        label) to input to the predictive RNN.
        """
        if (nrSamples is None):
            nrSamples = self.minibatch_size;
        
        # Swap axes of index in sentence and datapoint for Theano purposes
        encoding_label = np.swapaxes(encoding_label, 0, 1);
        prediction_label = np.swapaxes(prediction_label, 0, 1);

        if (not self.only_cause_expression):
            prediction_1, prediction_2, right_hand, error = \
                    self._predict(encoding_label, prediction_label, self.lag, nrSamples);
        else:
            prediction_1, right_hand, error = \
                    self._predict(encoding_label, prediction_label, self.lag, nrSamples);

        # Swap sentence index and datapoints back
        prediction_1 = np.swapaxes(prediction_1, 0, 1);
        if (not self.only_cause_expression):
            prediction_2 = np.swapaxes(prediction_2, 0, 1);
        right_hand = np.swapaxes(right_hand, 0, 1);

        if (not self.only_cause_expression):
            return [prediction_1, prediction_2], {'right_hand': right_hand, 'error': error};
        else:
            return prediction_1, {'right_hand': right_hand, 'error': error};

    def getVars(self):
        return self.vars.items();

    def batch_statistics(self, stats, prediction,
                         target_expressions, intervention_locations,
                         other, test_n, dataset, labels_to_use,
                         emptySamples=None,
                         training=False, topcause=True,
                         testInDataset=True, bothcause=False):
        """
        Overriding for finish-target_expressions.
        expressions_with_interventions contains the label-target_expressions (in
        strings) that should be used to lookup the candidate labels for SGD (in
        finish_expression_find_labels)
        """
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
            causeExpressionPrediction = dataset.indicesToStr(prediction[causeIndex][j]);
            if (not self.only_cause_expression):
                effectExpressionPrediction = dataset.indicesToStr(prediction[effectIndex][j]);


            # Prepare vars to save correct samples
            topCorrect = False;
            botCorrect = False;

            # Check if cause sequence prediction is in dataset
            causeMatchesLabel = False;
            if (causeExpressionPrediction == labels_to_use[j][causeIndex]):
                stats['structureCorrectCause'] += 1.0;
                if (topcause):
                    topCorrect = True;
                    stats['structureCorrectTop'] += 1.0;
                else:
                    botCorrect = True;
                    stats['structureCorrectBot'] += 1.0;
                causeMatchesLabel = True;

            causeValid = False;
            # Check if cause sequence prediction is valid
            if (dataset.valid_checker(prediction[causeIndex][j],dataset.digits,dataset.operators)):
                causeValid = True;
                stats['structureValidCause'] += 1.0;
                if (topcause):
                    stats['structureValidTop'] += 1.0;
                else:
                    stats['structureValidBot'] += 1.0;

            effectMatchesLabel = False;
            effectValid = False;
            if (not self.only_cause_expression):
                # Check if effect sequence prediction is in dataset
                if (effectExpressionPrediction == labels_to_use[j][effectIndex]):
                    stats['structureCorrectEffect'] += 1.0;
                    if (topcause):
                        topCorrect = True;
                        stats['structureCorrectBot'] += 1.0;
                    else:
                        botCorrect = True;
                        stats['structureCorrectTop'] += 1.0;
                    effectMatchesLabel = True;

                # Check if effect sequence prediction is valid
                if (dataset.valid_checker(prediction[effectIndex][j],dataset.digits,dataset.operators)):
                    effectValid = True;
                    stats['structureValidEffect'] += 1.0;
                    if (topcause):
                        stats['structureValidBot'] += 1.0;
                    else:
                        stats['structureValidTop'] += 1.0;

            # Check if effect prediction is valid
            effectMatch = False;
            if (not self.only_cause_expression):
                effect = dataset.effect_matcher(prediction[causeIndex][j][:eos_location],
                                                prediction[effectIndex][j][:eos_location],
                                                self.digits,self.operators,topcause);
                if (effect == 1 or effect == 2):
                    stats['effectCorrect'] += 1.0;
                    if (effect == 2):
                        stats['noEffect'] += 1.0;
                    effectMatch = True;

            # Determine sample in dataset
            if ((causeMatchesLabel and self.only_cause_expression is not False) or (causeMatchesLabel and effectMatchesLabel)):
                stats['structureCorrect'] += 1.0;
            if ((causeMatchesLabel and self.only_cause_expression is not False) or (causeMatchesLabel and effectMatchesLabel and effectMatch)):
                if (self.only_cause_expression is False):
                    stats['samplesCorrect'].append((True,True));
                stats['correct'] += 1.0;
                stats['valid'] += 1.0;
                stats['inDataset'] += 1.0;

                # Do local scoring for seq2ndmarkov
                if (self.seq2ndmarkov):
                    stats['localSize'] += float(len(causeExpressionPrediction)/3);
                    stats['localValid'] += float(len(causeExpressionPrediction)/3);
                    if (self.only_cause_expression is False):
                        stats['localValidCause'] += float(len(causeExpressionPrediction)/3);
                        stats['localValidEffect'] += float(len(effectExpressionPrediction)/3);
            else:
                # Save sample correct
                if (self.only_cause_expression is False):
                    stats['samplesCorrect'].append((topCorrect,botCorrect));
                # Determine validity of sample if it is not correct
                if ((causeValid and self.only_cause_expression is not False) or (causeValid and effectValid and effectMatch)):
                    stats['valid'] += 1.0;
                if (testInDataset and not training):
                    primeToUse = None;
                    if (self.only_cause_expression is False):
                        primeToUse = effectExpressionPrediction;
                    if (dataset.testExpressionsByPrefix.exists(causeExpressionPrediction, prime=primeToUse)):
                        stats['inDataset'] += 1.0;
                    elif (dataset.expressionsByPrefix.exists(causeExpressionPrediction, prime=primeToUse)):
                        stats['inDataset'] += 1.0;

                difference1 = SubsystemsTheanoRecurrentNeuralNetwork.string_difference(causeExpressionPrediction, labels_to_use[j][causeIndex]);
                if (not self.only_cause_expression):
                    difference2 = SubsystemsTheanoRecurrentNeuralNetwork.string_difference(effectExpressionPrediction, labels_to_use[j][effectIndex]);
                else:
                    difference2 = 0;
                difference = difference1 + difference2;
                if (difference == 0):
                    if (not self.ignoreZeroDifference):
                        raise ValueError("Difference is 0 but sample is not correct! causeIndex: %d, cause: %s, effect: %s, difference1: %d, difference2: %d, cause matches label: %d, effect matches label: %d, effectMatch: %d" %
                                        (causeIndex, causeExpressionPrediction, effectExpressionPrediction, difference1, difference2, int(causeMatchesLabel), int(effectMatchesLabel), int(effectMatch)));
                    else:
                        difference = 4; # Random digit outside of error margin computation range
                stats['error_histogram'][difference] += 1;

                # Do local scoring for seq2ndmarkov
                for k in range(2,len(labels_to_use[j][causeIndex]),3):
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
            i = 0;
            len_to_use = len(labels_to_use[j][causeIndex]);
            for i in range(intervention_locations[causeIndex,j]+1,len_to_use):
                if (i < len(causeExpressionPrediction)):
                    if (causeExpressionPrediction[i] == labels_to_use[j][causeIndex][i]):
                        stats['digit_1_correct'] += 1.0;
            stats['digit_1_prediction_size'] += len_to_use - (intervention_locations[causeIndex,j]+1);

            if (not self.only_cause_expression):
                i = 0;
                len_to_use = len(labels_to_use[j][effectIndex]);
                for i in range(intervention_locations[effectIndex,j]+1,len_to_use):
                    if (i < len(effectExpressionPrediction)):
                        if (effectExpressionPrediction[i] == labels_to_use[j][effectIndex][i]):
                            stats['digit_2_correct'] += 1.0;
                stats['digit_2_prediction_size'] += len_to_use - (intervention_locations[effectIndex,j]+1);


            stats['prediction_1_size_histogram'][int(eos_location)] += 1;
            for digit_prediction in prediction[causeIndex][j][intervention_locations[causeIndex,j]+1:len(causeExpressionPrediction)]:
                stats['prediction_1_histogram'][int(digit_prediction)] += 1;

            if (not self.only_cause_expression):
                stats['prediction_2_size_histogram'][int(eos_location)] += 1;
                for digit_prediction in prediction[effectIndex][j][intervention_locations[effectIndex,j]+1:len(effectExpressionPrediction)]:
                    stats['prediction_2_histogram'][int(digit_prediction)] += 1;

            stats['prediction_size'] += 1;

        return stats, labels_to_use;

    @staticmethod
    def operator_scores(expression, correct, operators,
                        key_indices, op_scores):
        # Find operators in expression
        ops_in_expression = [];
        for literal in expression:
            if (literal in operators):
                ops_in_expression.append(literal);
        # For each operator, update statistics
        for op in set(ops_in_expression):
            if (correct):
                op_scores[key_indices[op],0] += 1;
            op_scores[key_indices[op],1] += 1;
        return op_scores;

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
