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

class SequenceRepairingRecurrentNeuralNetwork(RecurrentModel):
    '''
    Recurrent neural network model with one hidden layer. Models single class 
    prediction based on regular recurrent model or LSTM model. 
    '''
    
    SGD_OPTIMIZER = 0;
    RMS_OPTIMIZER = 1;
    NESTEROV_OPTIMIZER = 2;

    def __init__(self, data_dim, hidden_dim, output_dim, minibatch_size, 
                 weight_values={}, 
                 EOS_symbol_index=None, GO_symbol_index=None, n_max_digits=24, 
                 verboseOutputter=None,
                 optimizer=0, learning_rate=0.01,
                 operators=4, digits=10, seq2ndmarkov=False,
                 doubleLayer=False, tripleLayer=False, dropoutProb=0., outputBias=False):
        '''
        Initialize all Theano models.
        '''
        # Store settings in self since the initializing functions will need them
        self.minibatch_size = minibatch_size;
        self.n_max_digits = n_max_digits;
        self.operators = operators;
        self.digits = digits;
        self.learning_rate = learning_rate;
        self.seq2ndmarkov = seq2ndmarkov;
        self.optimizer = optimizer;
        self.verboseOutputter = verboseOutputter;
        self.doubleLayer = doubleLayer;
        self.tripleLayer = tripleLayer;
        self.dropoutProb = dropoutProb;
        self.outputBias = outputBias;
                
        self.EOS_symbol_index = EOS_symbol_index;
        self.GO_symbol_index = GO_symbol_index;
        
        if (self.GO_symbol_index is None):
            raise ValueError("GO symbol index not set!");
        
        # Set dimensions
        self.data_dim = data_dim;
        self.hidden_dim = hidden_dim;
        self.decoding_output_dim = output_dim;
        self.prediction_output_dim = output_dim;
        
        # Set up shared variables
        varSettings = [];
        varSettings.append(('hWf',self.hidden_dim,self.hidden_dim));
        varSettings.append(('XWf',self.data_dim,self.hidden_dim));
        varSettings.append(('hWi',self.hidden_dim,self.hidden_dim));
        varSettings.append(('XWi',self.data_dim,self.hidden_dim));
        varSettings.append(('hWc',self.hidden_dim,self.hidden_dim));
        varSettings.append(('XWc',self.data_dim,self.hidden_dim));
        varSettings.append(('hWo',self.hidden_dim,self.hidden_dim));
        varSettings.append(('XWo',self.data_dim,self.hidden_dim));
        varSettings.append(('DhWf',self.hidden_dim,self.hidden_dim));
        varSettings.append(('DXWf',self.data_dim,self.hidden_dim));
        varSettings.append(('DhWi',self.hidden_dim,self.hidden_dim));
        varSettings.append(('DXWi',self.data_dim,self.hidden_dim));
        varSettings.append(('DhWc',self.hidden_dim,self.hidden_dim));
        varSettings.append(('DXWc',self.data_dim,self.hidden_dim));
        varSettings.append(('DhWo',self.hidden_dim,self.hidden_dim));
        varSettings.append(('DXWo',self.data_dim,self.hidden_dim));
            
        if (self.doubleLayer or self.tripleLayer):
            varSettings.append(('hWf2',self.hidden_dim,self.hidden_dim));
            varSettings.append(('XWf2',self.hidden_dim,self.hidden_dim));
            varSettings.append(('hWi2',self.hidden_dim,self.hidden_dim));
            varSettings.append(('XWi2',self.hidden_dim,self.hidden_dim));
            varSettings.append(('hWc2',self.hidden_dim,self.hidden_dim));
            varSettings.append(('XWc2',self.hidden_dim,self.hidden_dim));
            varSettings.append(('hWo2',self.hidden_dim,self.hidden_dim));
            varSettings.append(('XWo2',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DhWf2',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DXWf2',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DhWi2',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DXWi2',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DhWc2',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DXWc2',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DhWo2',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DXWo2',self.hidden_dim,self.hidden_dim));
        if (self.tripleLayer):
            varSettings.append(('hWf3',self.hidden_dim,self.hidden_dim));
            varSettings.append(('XWf3',self.hidden_dim,self.hidden_dim));
            varSettings.append(('hWi3',self.hidden_dim,self.hidden_dim));
            varSettings.append(('XWi3',self.hidden_dim,self.hidden_dim));
            varSettings.append(('hWc3',self.hidden_dim,self.hidden_dim));
            varSettings.append(('XWc3',self.hidden_dim,self.hidden_dim));
            varSettings.append(('hWo3',self.hidden_dim,self.hidden_dim));
            varSettings.append(('XWo3',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DhWf3',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DXWf3',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DhWi3',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DXWi3',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DhWc3',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DXWc3',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DhWo3',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DXWo3',self.hidden_dim,self.hidden_dim));
            
        varSettings.append(('hWY',self.hidden_dim,self.prediction_output_dim));
        varSettings.append(('hbY',False,self.prediction_output_dim));
        
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
        nrSamples = T.iscalar();
        
        # Set the RNN cell to use for encoding and decoding
        encode_function = self.lstm_predict_single_no_output;
        decode_function = self.lstm_predict_single;
        if (self.doubleLayer):
            encode_function = self.lstm_predict_double_no_output;
            decode_function = self.lstm_predict_double;
        if (self.tripleLayer):
            encode_function = self.lstm_predict_triple_no_output;
            decode_function = self.lstm_predict_triple;
        
        if (self.dropoutProb > 0.):
            self.random_stream = T.shared_randomstreams.RandomStreams(seed=np.random.randint(10000));
        
        # Set the prediction parameters to be either the prediction 
        # weights or the decoding weights depending on the setting 
        encode_parameters = [self.vars[k[0]] for k in filter(lambda varset: varset[0][0] != 'D' and varset[0][-1] != 'Y', varSettings)];
        decode_parameters = [self.vars[k[0]] for k in filter(lambda varset: varset[0][0] == 'D' or varset[0][-1] == 'Y', varSettings)];
        
        hidden = [T.zeros((self.minibatch_size,self.hidden_dim))];
        if (self.doubleLayer or self.tripleLayer):
            hidden_2 = [T.zeros((self.minibatch_size,self.hidden_dim))];
        if (self.tripleLayer):
            hidden_3 = [T.zeros((self.minibatch_size,self.hidden_dim))];
    
        # ENCODING PHASE
        init_values = ({'initial': hidden[-1], 'taps': [-1]});
        if (self.doubleLayer):
            init_values = ({'initial': hidden[-1], 'taps': [-1]}, 
                           {'initial': hidden_2[-1], 'taps': [-1]});
        if (self.tripleLayer):
            init_values = ({'initial': hidden[-1], 'taps': [-1]}, 
                           {'initial': hidden_2[-1], 'taps': [-1]},
                           {'initial': hidden_3[-1], 'taps': [-1]});
        encoding_hidden, _ = theano.scan(fn=encode_function,
                                          sequences=X,
                                          outputs_info=init_values,
                                          non_sequences=encode_parameters,
                                          name='encode_scan')
        # Use the hidden state of the final layer for predicting the first symbol
        encoding_hiddens = [encoding_hidden];
        if (self.doubleLayer or self.tripleLayer):
            encoding_hiddens = encoding_hidden;
        first_prediction = T.nnet.softmax(encoding_hiddens[-1][-1].dot(self.vars['hWY']) + self.vars['hbY']);        
        
        # DECODING PHASE
        init_values = ({'initial': first_prediction, 'taps': [-1]}, 
                       {'initial': encoding_hiddens[-1][-1], 'taps': [-1]});
        if (self.doubleLayer):
            init_values = ({'initial': first_prediction, 'taps': [-1]}, 
                           {'initial': encoding_hiddens[0][-1], 'taps': [-1]}, 
                           {'initial': encoding_hiddens[1][-1], 'taps': [-1]});
        if (self.tripleLayer):
            init_values = ({'initial': first_prediction, 'taps': [-1]}, 
                           {'initial': encoding_hiddens[0][-1], 'taps': [-1]}, 
                           {'initial': encoding_hiddens[1][-1], 'taps': [-1]},
                           {'initial': encoding_hiddens[2][-1], 'taps': [-1]});
        outputs, _ = theano.scan(fn=decode_function,
                                 outputs_info=init_values,
                                 non_sequences=decode_parameters,
                                 name='decode_scan',
                                 n_steps=self.n_max_digits)
        
        #right_hand = T.concatenate([first_prediction.reshape((1,self.minibatch_size,self.prediction_output_dim)),outputs[0]],0);
        right_hand = T.concatenate([first_prediction.reshape((1,self.minibatch_size,self.prediction_output_dim)),outputs[0]],0);
        right_hand_near_zeros = T.ones_like(right_hand) * 1e-15;
        right_hand = T.maximum(right_hand, right_hand_near_zeros);
        
        # Compute the prediction
        prediction = T.argmax(right_hand, axis=2);
        
        # ERROR COMPUTATION AND PROPAGATION
        coding_dist = right_hand[:label.shape[0]];
        cat_cross = -T.sum(label * T.log(coding_dist), axis=coding_dist.ndim-1);
        # mean_cross_per_sample = T.sum(cat_cross, axis=0) / (self.n_max_digits - (intervention_locations + 1.));
#         mean_cross_per_sample = T.mean(cat_cross, axis=0);
#         error = T.mean(mean_cross_per_sample[:nrSamples]);
        error = T.mean(cat_cross);
        
        # Defining prediction function
        self._predict = theano.function([X, label, nrSamples], 
                                        [prediction, right_hand, error], 
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
            updates = lasagne.updates.rmsprop(derivatives,var_list).items();
        else:
            derivatives = T.grad(error, var_list);
            updates = lasagne.updates.nesterov_momentum(derivatives,var_list,learning_rate=self.learning_rate).items();
        
        # Defining SGD functuin
        self._sgd = theano.function([X, label, nrSamples],
                                    [error],
                                    updates=updates,
                                    allow_input_downcast=True,
                                    on_unused_input='ignore');
        
        super(SequenceRepairingRecurrentNeuralNetwork, self).__init__();
    
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
    
    def lstm_predict_single(self, previous_output, previous_hidden, 
                            hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, hWY, hbY):        
        forget_gate = T.nnet.sigmoid(previous_hidden.dot(hWf) + previous_output.dot(XWf));
        input_gate = T.nnet.sigmoid(previous_hidden.dot(hWi) + previous_output.dot(XWi));
        candidate_cell = T.tanh(previous_hidden.dot(hWc) + previous_output.dot(XWc));
        cell = forget_gate * previous_hidden + input_gate * candidate_cell;
        output_gate = T.nnet.sigmoid(previous_hidden.dot(hWo) + previous_output.dot(XWo));
        hidden = output_gate * cell;
        
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
        
        return Y_output, hidden;
    
    def lstm_predict_single_no_output(self, current_X, previous_hidden, hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo):
        forget_gate = T.nnet.sigmoid(previous_hidden.dot(hWf) + current_X.dot(XWf));
        input_gate = T.nnet.sigmoid(previous_hidden.dot(hWi) + current_X.dot(XWi));
        candidate_cell = T.tanh(previous_hidden.dot(hWc) + current_X.dot(XWc));
        cell = forget_gate * previous_hidden + input_gate * candidate_cell;
        output_gate = T.nnet.sigmoid(previous_hidden.dot(hWo) + current_X.dot(XWo));
        hidden = output_gate * cell;
        
        return hidden;
    
    def lstm_predict_double(self, previous_output, previous_hidden_1, 
                            previous_hidden_2,
                            hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo,
                            hWf2, XWf2, hWi2, XWi2, hWc2, XWc2, hWo2, XWo2, hWY, hbY):
        forget_gate = T.nnet.sigmoid(previous_hidden_1.dot(hWf) + previous_output.dot(XWf));
        input_gate = T.nnet.sigmoid(previous_hidden_1.dot(hWi) + previous_output.dot(XWi));
        candidate_cell = T.tanh(previous_hidden_1.dot(hWc) + previous_output.dot(XWc));
        cell = forget_gate * previous_hidden_1 + input_gate * candidate_cell;
        output_gate = T.nnet.sigmoid(previous_hidden_1.dot(hWo) + previous_output.dot(XWo));
        hidden_1 = output_gate * cell;
        
        # Apply dropout (p = 1 - p because p  is chance of dropout and 1 is keep unit)
        if (self.dropoutProb > 0.):
            hidden_1 = lasagne.layers.dropout((self.minibatch_size, self.hidden_dim), self.dropoutProb).get_output_for(hidden_1);
        
        forget_gate_2 = T.nnet.sigmoid(previous_hidden_2.dot(hWf2) + hidden_1.dot(XWf2));
        input_gate_2 = T.nnet.sigmoid(previous_hidden_2.dot(hWi2) + hidden_1.dot(XWi2));
        candidate_cell_2 = T.tanh(previous_hidden_2.dot(hWc2) + hidden_1.dot(XWc2));
        cell_2 = forget_gate_2 * previous_hidden_2 + input_gate_2 * candidate_cell_2;
        output_gate_2 = T.nnet.sigmoid(previous_hidden_2.dot(hWo2) + hidden_1.dot(XWo2));
        hidden_2 = output_gate_2 * cell_2;
        
        # Apply dropout (p = 1 - p because p  is chance of dropout and 1 is keep unit)
        if (self.dropoutProb > 0.):
            hidden_2 = lasagne.layers.dropout((self.minibatch_size, self.hidden_dim), self.dropoutProb).get_output_for(hidden_2);
        
        # Use given intervention locations to determine whether to use label
        # or previous prediction. This should allow for flexible minibatching
        # Use given intervention locations to determine whether to use label
        # or previous prediction. This should allow for flexible minibatching
        if (self.outputBias):
            Y_output = T.nnet.softmax(hidden_2.dot(hWY) + hbY);
        else:
            Y_output = T.nnet.softmax(hidden_2.dot(hWY));

        # Apply dropout (p = 1 - p because p  is chance of dropout and 1 is keep unit)
        if (self.dropoutProb > 0.):
            Y_output = lasagne.layers.dropout((self.minibatch_size, self.decoding_output_dim), self.dropoutProb).get_output_for(Y_output);
        
        return Y_output, hidden_1, hidden_2;
    
    def lstm_predict_double_no_output(self, current_X, previous_hidden_1, previous_hidden_2,
                                      hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo,
                                      hWf2, XWf2, hWi2, XWi2, hWc2, XWc2, hWo2, XWo2):
        forget_gate = T.nnet.sigmoid(previous_hidden_1.dot(hWf) + current_X.dot(XWf));
        input_gate = T.nnet.sigmoid(previous_hidden_1.dot(hWi) + current_X.dot(XWi));
        candidate_cell = T.tanh(previous_hidden_1.dot(hWc) + current_X.dot(XWc));
        cell = forget_gate * previous_hidden_1 + input_gate * candidate_cell;
        output_gate = T.nnet.sigmoid(previous_hidden_1.dot(hWo) + current_X.dot(XWo));
        hidden_1 = output_gate * cell;
        
        forget_gate_2 = T.nnet.sigmoid(previous_hidden_2.dot(hWf2) + hidden_1.dot(XWf2));
        input_gate_2 = T.nnet.sigmoid(previous_hidden_2.dot(hWi2) + hidden_1.dot(XWi2));
        candidate_cell_2 = T.tanh(previous_hidden_2.dot(hWc2) + hidden_1.dot(XWc2));
        cell_2 = forget_gate_2 * previous_hidden_2 + input_gate_2 * candidate_cell_2;
        output_gate_2 = T.nnet.sigmoid(previous_hidden_2.dot(hWo2) + hidden_1.dot(XWo2));
        hidden_2 = output_gate_2 * cell_2;
        
        return hidden_1, hidden_2;

    def lstm_predict_triple(self, previous_output, previous_hidden_1, 
                            previous_hidden_2, previous_hidden_3,
                            hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo,
                            hWf2, XWf2, hWi2, XWi2, hWc2, XWc2, hWo2, XWo2, 
                            hWf3, XWf3, hWi3, XWi3, hWc3, XWc3, hWo3, XWo3,
                            hWY, hbY):
        forget_gate = T.nnet.sigmoid(previous_hidden_1.dot(hWf) + previous_output.dot(XWf));
        input_gate = T.nnet.sigmoid(previous_hidden_1.dot(hWi) + previous_output.dot(XWi));
        candidate_cell = T.tanh(previous_hidden_1.dot(hWc) + previous_output.dot(XWc));
        cell = forget_gate * previous_hidden_1 + input_gate * candidate_cell;
        output_gate = T.nnet.sigmoid(previous_hidden_1.dot(hWo) + previous_output.dot(XWo));
        hidden_1 = output_gate * cell;
        
        # Apply dropout (p = 1 - p because p  is chance of dropout and 1 is keep unit)
        if (self.dropoutProb > 0.):
            hidden_1 = lasagne.layers.dropout((self.minibatch_size, self.hidden_dim), self.dropoutProb).get_output_for(hidden_1);
        
        forget_gate_2 = T.nnet.sigmoid(previous_hidden_2.dot(hWf2) + hidden_1.dot(XWf2));
        input_gate_2 = T.nnet.sigmoid(previous_hidden_2.dot(hWi2) + hidden_1.dot(XWi2));
        candidate_cell_2 = T.tanh(previous_hidden_2.dot(hWc2) + hidden_1.dot(XWc2));
        cell_2 = forget_gate_2 * previous_hidden_2 + input_gate_2 * candidate_cell_2;
        output_gate_2 = T.nnet.sigmoid(previous_hidden_2.dot(hWo2) + hidden_1.dot(XWo2));
        hidden_2 = output_gate_2 * cell_2;
        
        # Apply dropout (p = 1 - p because p  is chance of dropout and 1 is keep unit)
        if (self.dropoutProb > 0.):
            hidden_2 = lasagne.layers.dropout((self.minibatch_size, self.hidden_dim), self.dropoutProb).get_output_for(hidden_2);
        
        forget_gate_3 = T.nnet.sigmoid(previous_hidden_3.dot(hWf3) + hidden_2.dot(XWf3));
        input_gate_3 = T.nnet.sigmoid(previous_hidden_3.dot(hWi3) + hidden_2.dot(XWi3));
        candidate_cell_3 = T.tanh(previous_hidden_3.dot(hWc3) + hidden_2.dot(XWc3));
        cell_3 = forget_gate_3 * previous_hidden_3 + input_gate_3 * candidate_cell_3;
        output_gate_3 = T.nnet.sigmoid(previous_hidden_3.dot(hWo3) + hidden_2.dot(XWo3));
        hidden_3 = output_gate_3 * cell_3;
        
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
        
        return Y_output, hidden_1, hidden_2, hidden_3;
    
    # END OF INITIALIZATION
    
    def sanityChecks(self, training_data, training_labels):
        """
        Sanity checks to be called before training a batch. Throws exceptions 
        if things are not right.
        """
        if (not self.single_digit and training_labels.shape[1] > (self.n_max_digits+1)):
            raise ValueError("n_max_digits too small! Increase to %d" % training_labels.shape[1]);
    
    def sgd(self, dataset, data, label, learning_rate, nrSamples=None):
        """
        The intervention location for finish expressions must be the same for 
        all samples in this batch.
        """
        if (nrSamples is None):
            nrSamples = self.minibatch_size;
        
        data = np.swapaxes(data, 0, 1);
        label = np.swapaxes(label, 0, 1);
        return self._sgd(data, label, nrSamples);
    
    def predict(self, input_data, label,
                fixedDecoderInputs=True, topcause=True, nrSamples=None):
        """
        Uses an encoding_label (formerly data) and a prediction_label (formerly
        label) to input to the predictive RNN.
        """
        if (nrSamples is None):
            nrSamples = self.minibatch_size;
        
        # Swap axes of index in sentence and datapoint for Theano purposes
        input_data = np.swapaxes(input_data, 0, 1);
        label = np.swapaxes(label, 0, 1);
        
        prediction, right_hand, error = \
            self._predict(input_data, label, nrSamples);
        
        # Swap sentence index and datapoints back
        prediction = np.swapaxes(prediction, 0, 1);
        right_hand = np.swapaxes(right_hand, 0, 1);
        
        return prediction, {'right_hand': right_hand, 'error': error};

    def getVars(self):
        return self.vars.items();
    
    def batch_statistics(self, stats, prediction, 
                         target_expressions,
                         other, test_n, dataset,
                         emptySamples=None,
                         training=False, topcause=True,
                         testInDataset=True, bothcause=False):
        for j in range(0,test_n):
            if (emptySamples is not None and j in emptySamples):
                continue;
            
            # Taking argmax over symbols for each sentence returns 
            # the location of the highest index, which is the first 
            # EOS symbol
            eos_location = np.argmax(prediction[j]);
            # Check for edge case where no EOS was found and zero was returned
            eos_symbol_index = dataset.EOS_symbol_index;
            if (prediction[j,eos_location] != eos_symbol_index):
                eos_location = prediction[j].shape[0];
            
            # Convert prediction to string expression
            causeExpressionPrediction = dataset.indicesToStr(prediction[j]);
            
            # Check if cause sequence prediction is in dataset
            causeMatchesLabel = False;
            if (causeExpressionPrediction == target_expressions[j]):
                causeMatchesLabel = True;
            
            causeValid = False;
            # Check if cause sequence prediction is valid
            if (dataset.valid_checker(causeExpressionPrediction,dataset.digits,dataset.operators)):
                causeValid = True;
                stats['structureValidCause'] += 1.0;
            
            # Determine sample in dataset
            if (causeMatchesLabel):
                stats['correct'] += 1.0;
                stats['valid'] += 1.0;
                stats['inDataset'] += 1.0;
            else:
                # Determine validity of sample if it is not correct
                if (causeValid):
                    stats['valid'] += 1.0;
                if (testInDataset and not training):
                    primeToUse = None;
                    if (dataset.testExpressionsByPrefix.exists(causeExpressionPrediction, prime=primeToUse)):
                        stats['inDataset'] += 1.0;
                    elif (dataset.expressionsByPrefix.exists(causeExpressionPrediction, prime=primeToUse)):
                        stats['inDataset'] += 1.0;
                
                difference = SequenceRepairingRecurrentNeuralNetwork.string_difference(causeExpressionPrediction, target_expressions[j]);
                if (difference == 0):
                    raise ValueError("Difference is 0 but sample is not correct! expression: %s, difference: %d, cause matches label: %d" % 
                                     (causeExpressionPrediction, difference, int(causeMatchesLabel)));
                stats['error_histogram'][difference] += 1;
            
            # Digit precision and prediction size computation
            i = 0;
            len_to_use = len(target_expressions[j]);
            for i in range(len_to_use):
                if (i < len(causeExpressionPrediction)):
                    if (causeExpressionPrediction[i] == target_expressions[j][i]):
                        stats['digit_1_correct'] += 1.0;
            stats['digit_1_prediction_size'] += len_to_use;
       
            stats['prediction_1_size_histogram'][int(eos_location)] += 1;
            for digit_prediction in prediction[j][:len(causeExpressionPrediction)]:
                stats['prediction_1_histogram'][int(digit_prediction)] += 1;
            
            stats['prediction_size'] += 1;
        
        return stats, target_expressions;
    
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
