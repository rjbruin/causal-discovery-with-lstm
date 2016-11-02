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

class TheanoRecurrentNeuralNetwork(RecurrentModel):
    '''
    Recurrent neural network model with one hidden layer. Models single class 
    prediction based on regular recurrent model or LSTM model. 
    '''
    
    SGD_OPTIMIZER = 0;
    ADAM_OPTIMIZER = 1;

    def __init__(self, data_dim, hidden_dim, output_dim, minibatch_size, 
                 lstm=True, weight_values={}, single_digit=False, 
                 EOS_symbol_index=None, GO_symbol_index=None, n_max_digits=24, 
                 decoder=False, verboseOutputter=None, finishExpressions=True,
                 optimizer=0, learning_rate=0.01,
                 operators=4, digits=10, only_cause_expression=False, seq2ndmarkov=False,
                 doubleLayer=False, dropoutProb=0., useEncoder=True, outputBias=False,
                 crosslinks=True):
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
        self.dropoutProb = dropoutProb;
        self.useEncoder = useEncoder;
        self.outputBias = outputBias;
        self.crosslinks = crosslinks;
        
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
        
        actual_data_dim = self.data_dim * 2;
        actual_decoding_output_dim = self.decoding_output_dim * 2;
        actual_prediction_output_dim = self.prediction_output_dim * 2;
        if (self.only_cause_expression):
            actual_data_dim = self.data_dim;
            actual_decoding_output_dim = self.decoding_output_dim;
            actual_prediction_output_dim = self.prediction_output_dim;
        
        # Set up shared variables
        varSettings = [];
        varSettings.append(('hWf',self.hidden_dim,self.hidden_dim));
        varSettings.append(('XWf',actual_data_dim,self.hidden_dim));
        varSettings.append(('hWi',self.hidden_dim,self.hidden_dim));
        varSettings.append(('XWi',actual_data_dim,self.hidden_dim));
        varSettings.append(('hWc',self.hidden_dim,self.hidden_dim));
        varSettings.append(('XWc',actual_data_dim,self.hidden_dim));
        varSettings.append(('hWo',self.hidden_dim,self.hidden_dim));
        varSettings.append(('XWo',actual_data_dim,self.hidden_dim));
        if (self.doubleLayer):
            varSettings.append(('hWf2',self.hidden_dim,self.hidden_dim));
            varSettings.append(('XWf2',self.hidden_dim,self.hidden_dim));
            varSettings.append(('hWi2',self.hidden_dim,self.hidden_dim));
            varSettings.append(('XWi2',self.hidden_dim,self.hidden_dim));
            varSettings.append(('hWc2',self.hidden_dim,self.hidden_dim));
            varSettings.append(('XWc2',self.hidden_dim,self.hidden_dim));
            varSettings.append(('hWo2',self.hidden_dim,self.hidden_dim));
            varSettings.append(('XWo2',self.hidden_dim,self.hidden_dim));
            
        if (self.decoder):
            # Add variables for the decoding phase
            # All these variables begin with 'D' so they can be 
            # automatically filtered to be used as parameters
            varSettings.append(('DhWf',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DXWf',actual_data_dim,self.hidden_dim));
            varSettings.append(('DhWi',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DXWi',actual_data_dim,self.hidden_dim));
            varSettings.append(('DhWc',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DXWc',actual_data_dim,self.hidden_dim));
            varSettings.append(('DhWo',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DXWo',actual_data_dim,self.hidden_dim));
            if (self.doubleLayer):
                varSettings.append(('DhWf2',self.hidden_dim,self.hidden_dim));
                varSettings.append(('DXWf2',self.hidden_dim,self.hidden_dim));
                varSettings.append(('DhWi2',self.hidden_dim,self.hidden_dim));
                varSettings.append(('DXWi2',self.hidden_dim,self.hidden_dim));
                varSettings.append(('DhWc2',self.hidden_dim,self.hidden_dim));
                varSettings.append(('DXWc2',self.hidden_dim,self.hidden_dim));
                varSettings.append(('DhWo2',self.hidden_dim,self.hidden_dim));
                varSettings.append(('DXWo2',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DhWY',self.hidden_dim,actual_decoding_output_dim));
            varSettings.append(('DhbY',False,actual_decoding_output_dim));
        else:
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
        intervention_locations = T.imatrix();
        nrSamples = T.iscalar();
        
        # Set the RNN cell to use for encoding and decoding
        encode_function = self.lstm_predict_single_no_output;
        decode_function = self.lstm_predict_single;
        if (self.doubleLayer):
            encode_function = self.lstm_predict_double_no_output;
            decode_function = self.lstm_predict_double;
        
        if (self.dropoutProb > 0.):
            self.random_stream = T.shared_randomstreams.RandomStreams(seed=np.random.randint(10000));
        
        # Set the prediction parameters to be either the prediction 
        # weights or the decoding weights depending on the setting 
        encode_parameters = [self.vars[k[0]] for k in filter(lambda name: name[0][0] != 'D' and name[0] != 'hWY' and name[0] != 'hbY', varSettings)];
        if (self.decoder):
            decode_parameters = [intervention_locations] + [self.vars[k[0]] for k in filter(lambda name: name[0][0] == 'D', varSettings)];
        else:
            decode_parameters = [intervention_locations] + encode_parameters + [self.vars['hWY'], self.vars['hbY']];
        
        # ENCODING PHASE
        initial_encode = ({'initial': T.zeros((self.minibatch_size,self.hidden_dim)), 'taps': [-1]});
        if (self.doubleLayer):
            initial_encode = ({'initial': T.zeros((self.minibatch_size,self.hidden_dim)), 'taps': [-1]},{'initial': T.zeros((self.minibatch_size,self.hidden_dim)), 'taps': [-1]});
        
        if (self.useEncoder):
            if (self.crosslinks and not self.only_cause_expression):
                hiddens, _ = theano.scan(fn=encode_function,
                                        sequences=X,
                                        # Input a zero hidden layer
                                        outputs_info=initial_encode,
                                        non_sequences=encode_parameters + [0,actual_data_dim],
                                        name='encode_scan_1');
                hidden = hiddens;
                if (self.doubleLayer):
                    hidden = hiddens[0];
                    hidden_2 = hiddens[1];
            else:
                hiddens_top, _ = theano.scan(fn=encode_function,
                                           sequences=X,
                                           # Input a zero hidden layer
                                           outputs_info=initial_encode,
                                           non_sequences=encode_parameters + [0,self.data_dim],
                                           name='encode_scan_1');
                hidden_top = hiddens_top;
                if (self.doubleLayer):
                    hidden_top = hiddens_top[0];
                    hidden_2_top = hiddens_top[1];
                
                if (not self.only_cause_expression):
                    initial_encode_bot = ({'initial': T.zeros((self.minibatch_size,self.hidden_dim)), 'taps': [-1]});
                    if (self.doubleLayer):
                        initial_encode_bot = ({'initial': T.zeros((self.minibatch_size,self.hidden_dim)), 'taps': [-1]},{'initial': T.zeros((self.minibatch_size,self.hidden_dim)), 'taps': [-1]});
                    hiddens_bot, _ = theano.scan(fn=encode_function,
                                               sequences=X,
                                               # Input a zero hidden layer
                                               outputs_info=initial_encode_bot,
                                               non_sequences=encode_parameters + [self.data_dim,self.data_dim*2],
                                               name='encode_scan_2');
                    hidden_bot = hiddens_bot;
                    if (self.doubleLayer):
                        hidden_bot = hiddens_bot[0];
                        hidden_2_bot = hiddens_bot[1];
            
        else:
            hidden = [T.zeros((self.minibatch_size,self.hidden_dim))];
            if (self.doubleLayer):
                hidden_2 = [T.zeros((self.minibatch_size,self.hidden_dim))];
            if (not self.crosslinks or self.only_cause_expression is not False):
                hidden_top = hidden;
                hidden_bot = [T.zeros((self.minibatch_size,self.hidden_dim))];
                if (self.doubleLayer):
                    hidden_2_top = [T.zeros((self.minibatch_size,self.hidden_dim))];
                    hidden_2_bot = [T.zeros((self.minibatch_size,self.hidden_dim))];
    
        # DECODING PHASE
        if (self.crosslinks and not self.only_cause_expression):
            init_values = ({'initial': T.zeros((self.minibatch_size,actual_data_dim)), 'taps': [-1]}, 
                           {'initial': hidden[-1], 'taps': [-1]}, {'initial': 0., 'taps': [-1]});
            if (self.doubleLayer):
                init_values = ({'initial': T.zeros((self.minibatch_size,actual_data_dim)), 'taps': [-1]}, 
                               {'initial': hidden[-1], 'taps': [-1]}, 
                               {'initial': hidden_2[-1], 'taps': [-1]}, 
                               {'initial': 0., 'taps': [-1]});
            outputs, _ = theano.scan(fn=decode_function,
                                     sequences=label,
                                     outputs_info=init_values,
                                     non_sequences=decode_parameters + [0,actual_data_dim],
                                     name='decode_scan_1')
            if (self.doubleLayer):
                right_hand, _, _, _ = outputs;
            else:
                right_hand, _, _ = outputs;
        else:
            init_values = ({'initial': T.zeros((self.minibatch_size,self.data_dim)), 'taps': [-1]}, 
                           {'initial': hidden_top[-1], 'taps': [-1]}, {'initial': 0., 'taps': [-1]});
            if (self.doubleLayer):
                init_values = ({'initial': T.zeros((self.minibatch_size,self.data_dim)), 'taps': [-1]}, 
                               {'initial': hidden_top[-1], 'taps': [-1]}, 
                               {'initial': hidden_2_top[-1], 'taps': [-1]}, 
                               {'initial': 0., 'taps': [-1]});
            outputs_1, _ = theano.scan(fn=decode_function,
                                     sequences=label,
                                     outputs_info=init_values,
                                     non_sequences=decode_parameters + [0,self.data_dim],
                                     name='decode_scan_1')
            
            if (self.doubleLayer):
                right_hand_1, _, _, _ = outputs_1;
            else:
                right_hand_1, _, _ = outputs_1;
            
            if (not self.only_cause_expression):
                init_values = ({'initial': T.zeros((self.minibatch_size,self.data_dim)), 'taps': [-1]}, 
                               {'initial': hidden_bot[-1], 'taps': [-1]}, {'initial': 0., 'taps': [-1]});
                if (self.doubleLayer):
                    init_values = ({'initial': T.zeros((self.minibatch_size,self.data_dim)), 'taps': [-1]}, 
                                   {'initial': hidden_bot[-1], 'taps': [-1]}, 
                                   {'initial': hidden_2_bot[-1], 'taps': [-1]}, 
                                   {'initial': 0., 'taps': [-1]});
                outputs_2, _ = theano.scan(fn=decode_function,
                                         sequences=label,
                                         outputs_info=init_values,
                                         non_sequences=decode_parameters + [self.data_dim,self.data_dim*2],
                                         name='decode_scan_2')
                if (self.doubleLayer):
                    right_hand_2, _, _, _ = outputs_2;
                else:
                    right_hand_2, _, _ = outputs_2;
            
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
        
        # Defining prediction function
        if (not self.only_cause_expression):
            self._predict = theano.function([X, label, intervention_locations, nrSamples], [prediction_1,
                                                                                prediction_2,
                                                                                right_hand,
                                                                                error], on_unused_input='ignore');
        else:
            self._predict = theano.function([X, label, intervention_locations, nrSamples], [prediction_1,
                                                                                right_hand,
                                                                                error], on_unused_input='ignore');
        
        # Defining stochastic gradient descent
        variables = filter(lambda name: name != 'hbY' and name != 'DhbY', self.vars.keys());
        if (self.outputBias):
            if (self.decoder):
                variables.append('DhbY');
            else:
                variables.append('hbY');
        var_list = map(lambda var: self.vars[var], variables)
        if (self.optimizer == self.SGD_OPTIMIZER):
            # Automatic backward pass for all models: gradients
            derivatives = T.grad(error, var_list);
            updates = [(var,var-self.learning_rate*der) for (var,der) in zip(var_list,derivatives)];
        else:
            #updates, derivatives = self.adam(error, map(lambda var: self.vars[var], variables), learning_rate);
            derivatives = T.grad(error, var_list);
            updates = lasagne.updates.nesterov_momentum(derivatives,var_list,learning_rate=self.learning_rate).items();
        
        # Defining SGD functuin
        self._sgd = theano.function([X, label, intervention_locations, nrSamples],
                                        [error],
                                    updates=updates,
                                    allow_input_downcast=True,
                                    on_unused_input='ignore')
        
        super(TheanoRecurrentNeuralNetwork, self).__init__();
    
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
    
    def lstm_predict_single(self, given_X, previous_output, previous_hidden, sentence_index, intervention_locations, hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, hWY, hbY, sd, ed):
        forget_gate = T.nnet.sigmoid(previous_hidden.dot(hWf) + previous_output.dot(XWf[sd:ed,:]));
        input_gate = T.nnet.sigmoid(previous_hidden.dot(hWi) + previous_output.dot(XWi[sd:ed,:]));
        candidate_cell = T.tanh(previous_hidden.dot(hWc) + previous_output.dot(XWc[sd:ed,:]));
        cell = forget_gate * previous_hidden + input_gate * candidate_cell;
        output_gate = T.nnet.sigmoid(previous_hidden.dot(hWo) + previous_output.dot(XWo[sd:ed,:]));
        hidden = output_gate * cell;
        
        # Apply dropout (p = 1 - p because p  is chance of dropout and 1 is keep unit)
        if (self.dropoutProb > 0.):
            hidden = lasagne.layers.dropout((self.minibatch_size, self.hidden_dim), self.dropoutProb).get_output_for(hidden);
        
        # Use given intervention locations to determine whether to use label
        # or previous prediction. This should allow for flexible minibatching
        comparison = T.le(sentence_index,intervention_locations).reshape((T.constant(2), T.constant(self.minibatch_size), T.constant(1)), ndim=3);
        
        if (self.outputBias):
            Y_output = T.nnet.softmax(hidden.dot(hWY) + hbY);
        else:
            Y_output = T.nnet.softmax(hidden.dot(hWY));
        
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
    
    def lstm_predict_single_no_output(self, current_X, previous_hidden, hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, sd, ed):
        forget_gate = T.nnet.sigmoid(previous_hidden.dot(hWf) + current_X.dot(XWf));
        input_gate = T.nnet.sigmoid(previous_hidden.dot(hWi) + current_X.dot(XWi));
        candidate_cell = T.tanh(previous_hidden.dot(hWc) + current_X.dot(XWc));
        cell = forget_gate * previous_hidden + input_gate * candidate_cell;
        output_gate = T.nnet.sigmoid(previous_hidden.dot(hWo) + current_X.dot(XWo));
        hidden = output_gate * cell;
        
        return hidden;
    
    def lstm_predict_double(self, given_X, previous_output, previous_hidden_1, 
                            previous_hidden_2, sentence_index, intervention_locations,
                            hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo,
                            hWf2, XWf2, hWi2, XWi2, hWc2, XWc2, hWo2, XWo2, hWY, hbY, sd, ed):
        forget_gate = T.nnet.sigmoid(previous_hidden_1.dot(hWf) + given_X.dot(XWf));
        input_gate = T.nnet.sigmoid(previous_hidden_1.dot(hWi) + given_X.dot(XWi));
        candidate_cell = T.tanh(previous_hidden_1.dot(hWc) + given_X.dot(XWc));
        cell = forget_gate * previous_hidden_1 + input_gate * candidate_cell;
        output_gate = T.nnet.sigmoid(previous_hidden_1.dot(hWo) + given_X.dot(XWo));
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
        comparison_top = T.le(sentence_index,intervention_locations[0]).reshape((T.constant(self.minibatch_size), T.constant(1)), ndim=2);
        if (not self.only_cause_expression):
            comparison_bot = T.le(sentence_index,intervention_locations[1]).reshape((T.constant(self.minibatch_size), T.constant(1)), ndim=2);
        
        if (self.outputBias):
            Y_output = T.nnet.softmax(hidden_2.dot(hWY) + hbY);
        else:
            Y_output = T.nnet.softmax(hidden_2.dot(hWY));

        # Apply dropout (p = 1 - p because p  is chance of dropout and 1 is keep unit)
        if (self.dropoutProb > 0.):
            Y_output = lasagne.layers.dropout((self.minibatch_size, self.decoding_output_dim), self.dropoutProb).get_output_for(Y_output);
        
        # Filter for intervention location
        if (not self.only_cause_expression):
            Y_output = T.concatenate([T.switch(comparison_top,given_X[:,:self.data_dim],Y_output[:,:self.data_dim]), T.switch(comparison_bot,given_X[:,self.data_dim:],Y_output[:,self.data_dim:])], axis=1);
        else:
            Y_output = T.switch(comparison_top,given_X,Y_output);
        
        new_sentence_index = sentence_index + 1.;
        
        return Y_output, hidden_1, hidden_2, new_sentence_index;
    
    def lstm_predict_double_no_output(self, current_X, previous_hidden_1, previous_hidden_2,
                                      hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo,
                                      hWf2, XWf2, hWi2, XWi2, hWc2, XWc2, hWo2, XWo2, sd, ed):
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
            bothcause=False, nrSamples=None):
        """
        The intervention location for finish expressions must be the same for 
        all samples in this batch.
        """
        if (nrSamples is None):
            nrSamples = self.minibatch_size;
        
        data = np.swapaxes(data, 0, 1);
        label = np.swapaxes(label, 0, 1);
        return self._sgd(data, label, interventionLocations, nrSamples);
        
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
        prediction_label = np.swapaxes(prediction_label, 0, 1);
        
        if (not self.only_cause_expression):
            prediction_1, prediction_2, right_hand, error = \
                    self._predict(encoding_label, prediction_label, interventionLocations, nrSamples);
        else:
            prediction_1, right_hand, error = \
                    self._predict(encoding_label, prediction_label, interventionLocations, nrSamples);
        
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
                         testValidity=True, bothcause=False):
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
            
            if (intervention_locations[0,j] >= len(labels_to_use[j][causeIndex])):
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
            
            if (self.seq2ndmarkov):
                # Check if cause sequence prediction is in dataset
                causeInDataset = False;
                if (causeExpressionPrediction == labels_to_use[j][causeIndex]):
                    stats['structureCorrectCause'] += 1.0;
                    if (topcause):
                        stats['structureCorrectTop'] += 1.0;
                    else:
                        stats['structureCorrectBot'] += 1.0;
                    causeInDataset = True;
                
                # Check if cause sequence prediction is valid
                causeValid = False;
                if (dataset.valid_seq2ndmarkov(prediction[causeIndex][j][:eos_location],dataset.digits,dataset.operators)):
                    causeValid = True;
                    stats['structureValidCause'] += 1.0;
                    if (topcause):
                        stats['structureValidTop'] += 1.0;
                    else:
                        stats['structureValidBot'] += 1.0;
                
                effectInDataset = False;
                effectValid = False;
                if (not self.only_cause_expression):
                    # Check if effect sequence prediction is in dataset
                    if (effectExpressionPrediction == labels_to_use[j][effectIndex]):
                        stats['structureCorrectEffect'] += 1.0;
                        if (topcause):
                            stats['structureCorrectBot'] += 1.0;
                        else:
                            stats['structureCorrectTop'] += 1.0;
                        effectInDataset = True;
                    
                    # Check if effect sequence prediction is valid
                    if (dataset.valid_seq2ndmarkov(prediction[effectIndex][j][:eos_location],dataset.digits,dataset.operators)):
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
                
                # Determine validity of sample
                if ((causeValid and self.only_cause_expression is not False) or (causeValid and effectValid and effectMatch)):
                    stats['valid'] += 1.0;
                
                # Determine sample in dataset
                if ((causeInDataset and self.only_cause_expression is not False) or (causeInDataset and effectInDataset)):
                    stats['structureCorrect'] += 1.0;
                if ((causeInDataset and self.only_cause_expression is not False) or (causeInDataset and effectInDataset and effectMatch)):
                    stats['correct'] += 1.0;
                else:
                    difference1 = TheanoRecurrentNeuralNetwork.string_difference(causeExpressionPrediction, labels_to_use[j][causeIndex]);
                    if (not self.only_cause_expression):
                        difference2 = TheanoRecurrentNeuralNetwork.string_difference(effectExpressionPrediction, labels_to_use[j][effectIndex]);
                    else:
                        difference2 = 0;
                    if (difference1 + difference2 == 0):
                        print("%s vs %s" % (causeExpressionPrediction, labels_to_use[j][causeIndex]));
                        print("%s vs %s" % (effectExpressionPrediction, labels_to_use[j][effectIndex]));
                        print("causeValid: %s, effectValid: %s, effectMatch: %s, self.only_cause_exression: %s" % \
                              (str(causeValid), str(effectValid), str(effectMatch), str(self.only_cause_expression)));
                        raise ValueError("Difference is zero!");
                    stats['error_histogram'][difference1 + difference2] += 1;
            else:
                if (causeExpressionPrediction == labels_to_use[j][causeIndex]):
                    stats['structureCorrect'] += 1.0;
                    stats['structureValidCause'] += 1.0;
                    if (self.only_cause_expression is not False):
                        stats['correct'] += 1.0;
                        stats['valid'] += 1.0;
                    elif (effectExpressionPrediction == labels_to_use[j][effectIndex]):
                        stats['correct'] += 1.0;
                        stats['valid'] += 1.0;
                        stats['effectCorrect'] += 1.0;
                        stats['structureValidEffect'] += 1.0;
                else:
                    if (testValidity and not training and self.only_cause_expression is not False):
                        if (dataset.testExpressionsByPrefix.exists(causeExpressionPrediction)):
                            stats['valid'] += 1.0;
                        elif (dataset.expressionsByPrefix.exists(causeExpressionPrediction)):
                            stats['valid'] += 1.0;
                    
                    difference1 = TheanoRecurrentNeuralNetwork.string_difference(causeExpressionPrediction, labels_to_use[j][causeIndex]);
                    if (not self.only_cause_expression):
                        difference2 = TheanoRecurrentNeuralNetwork.string_difference(effectExpressionPrediction, labels_to_use[j][effectIndex]);
                    else:
                        difference2 = 0;
                    stats['error_histogram'][difference1 + difference2] += 1;
                    # Defer matching of effect prediction and what it should be to dataset effect matcher (dependent on dataset)
                    if (not self.only_cause_expression and dataset.effect_matcher(prediction[causeIndex][j][:eos_location],prediction[effectIndex][j][:eos_location],self.digits,self.operators,topcause)):
                        stats['effectCorrect'] += 1.0;
            
            # Digit precision and prediction size computation
            i = 0;
            len_to_use = min(len(causeExpressionPrediction),len(labels_to_use[j][causeIndex]));
            for i in range(intervention_locations[causeIndex,j]+1,len_to_use):
                if (causeExpressionPrediction[i] == labels_to_use[j][causeIndex][i]):
                    stats['digit_1_correct'] += 1.0;
            stats['digit_1_prediction_size'] += len_to_use;
            
            if (not self.only_cause_expression):
                i = 0;
                len_to_use = min(len(effectExpressionPrediction),len(labels_to_use[j][effectIndex]));
                for i in range(intervention_locations[effectIndex,j]+1,len_to_use):
                    if (effectExpressionPrediction[i] == labels_to_use[j][effectIndex][i]):
                        stats['digit_2_correct'] += 1.0;
                stats['digit_2_prediction_size'] += len_to_use;

       
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
