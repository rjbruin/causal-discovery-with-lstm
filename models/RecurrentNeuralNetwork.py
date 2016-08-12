'''
Created on 22 feb. 2016

@author: Robert-Jan
'''

import theano;
import theano.tensor as T;
import numpy as np;
from models.RecurrentModel import RecurrentModel

class RecurrentNeuralNetwork(RecurrentModel):
    '''
    Recurrent neural network model with one hidden layer. Models single class 
    prediction based on regular recurrent model or LSTM model. 
    '''


    def __init__(self, data_dim, hidden_dim, output_dim, minibatch_size, 
                 lstm=False, weight_values={}, single_digit=True, EOS_symbol_index=None,
                 n_max_digits=24, time_training_batch=False, decoder=False,
                 verboseOutputter=None, layers=1, mn=False):
        '''
        Initialize all Theano models.
        '''
        # Store settings in self since the initializing functions will need them
        self.layers = layers;
        self.single_digit = single_digit;
        self.minibatch_size = minibatch_size;
        self.n_max_digits = n_max_digits;
        self.time_training_batch = time_training_batch;
        self.lstm = lstm;
        self.mn = mn;
        self.single_digit = single_digit;
        self.decoder = decoder;
        self.verboseOutputter = verboseOutputter;
        
        # Set dimensions
        self.data_dim = data_dim;
        self.hidden_dim = hidden_dim;
        if (self.layers == 2):
            # For now we stick to layers of the same size
            self.hidden_dim_2 = hidden_dim;
        self.decoding_output_dim = output_dim;
        self.prediction_output_dim = output_dim;
        # Change output dim of prediction phase if decoding
        if (self.decoder):
            self.prediction_output_dim = self.hidden_dim;
        
        # Set up settings for fake minibatching
        self.fake_minibatch = False;
        if (self.minibatch_size == 1):
            self.fake_minibatch = True;
            self.minibatch_size = 2;
        
        
        
        # Set up shared variables
        varSettings = [];
        if (self.layers == 1):
            if (self.mn):
                varSettings.append(('qWu_B',self.query_dim,self.memory_dim));
                varSettings.append(('xWm_A',self.data_dim,self.memory_dim));
                varSettings.append(('xWc_C',self.data_dim,self.memory_dim));
                varSettings.append(('ouWy_W',self.memory_dim,self.output_dim));
            elif (not self.lstm):
                varSettings.append(('XWh',self.data_dim,self.hidden_dim));
                varSettings.append(('hWh',self.hidden_dim,self.hidden_dim));
                varSettings.append(('hWo',self.hidden_dim,self.decoding_output_dim));
            else:
                varSettings.append(('hWf',self.hidden_dim,self.hidden_dim));
                varSettings.append(('XWf',self.data_dim,self.hidden_dim));
                varSettings.append(('hWi',self.hidden_dim,self.hidden_dim));
                varSettings.append(('XWi',self.data_dim,self.hidden_dim));
                varSettings.append(('hWc',self.hidden_dim,self.hidden_dim));
                varSettings.append(('XWc',self.data_dim,self.hidden_dim));
                varSettings.append(('hWo',self.hidden_dim,self.hidden_dim));
                varSettings.append(('XWo',self.data_dim,self.hidden_dim));
                varSettings.append(('hWY',self.hidden_dim,self.prediction_output_dim));
                if (decoder):
                    # Add variables for the decoding phase
                    # All these variables begin with 'D' so they can be 
                    # automatically filtered to be used as parameters
                    varSettings.append(('DhWf',self.hidden_dim,self.hidden_dim));
                    varSettings.append(('DXWf',self.data_dim,self.hidden_dim));
                    varSettings.append(('DhWi',self.hidden_dim,self.hidden_dim));
                    varSettings.append(('DXWi',self.data_dim,self.hidden_dim));
                    varSettings.append(('DhWc',self.hidden_dim,self.hidden_dim));
                    varSettings.append(('DXWc',self.data_dim,self.hidden_dim));
                    varSettings.append(('DhWo',self.hidden_dim,self.hidden_dim));
                    varSettings.append(('DXWo',self.data_dim,self.hidden_dim));
                    varSettings.append(('DhWY',self.hidden_dim,self.decoding_output_dim));
        elif (self.layers == 2):
            varSettings.append(('hWf',self.hidden_dim,self.hidden_dim));
            varSettings.append(('XWf',self.data_dim,self.hidden_dim));
            varSettings.append(('hWi',self.hidden_dim,self.hidden_dim));
            varSettings.append(('XWi',self.data_dim,self.hidden_dim));
            varSettings.append(('hWc',self.hidden_dim,self.hidden_dim));
            varSettings.append(('XWc',self.data_dim,self.hidden_dim));
            varSettings.append(('hWo',self.hidden_dim,self.hidden_dim));
            varSettings.append(('XWo',self.data_dim,self.hidden_dim));
            varSettings.append(('hWh2',self.hidden_dim,self.hidden_dim_2));
            varSettings.append(('sh2Wf',self.hidden_dim_2,self.hidden_dim_2));
            varSettings.append(('shWf',self.hidden_dim,self.hidden_dim_2));
            varSettings.append(('sh2Wi',self.hidden_dim_2,self.hidden_dim_2));
            varSettings.append(('shWi',self.hidden_dim,self.hidden_dim_2));
            varSettings.append(('sh2Wc',self.hidden_dim_2,self.hidden_dim_2));
            varSettings.append(('shWc',self.hidden_dim,self.hidden_dim_2));
            varSettings.append(('sh2Wo',self.hidden_dim_2,self.hidden_dim_2));
            varSettings.append(('shWo',self.hidden_dim,self.hidden_dim_2));
            varSettings.append(('sh2WY',self.hidden_dim_2,self.prediction_output_dim));
        
        # Contruct variables
        self.vars = {};
        for (varName,dim1,dim2) in varSettings:
            # Get value for shared variable from constructor if present
            value = np.random.uniform(-np.sqrt(1.0/dim1),np.sqrt(1.0/dim1),(dim1,dim2));
            if (varName in weight_values):
                value = weight_values[varName];
            self.vars[varName] = theano.shared(name=varName, value=value);
        
        
        
        # Set up inputs to prediction and SGD
        # X is 3-dimensional: 1) index in sentence, 2) datapoint, 3) dimensionality of data
        X = T.dtensor3('X');
        if (single_digit):
            # targets is 2-dimensional: 1) datapoint, 2) label for each answer
            label = T.imatrix('label');
        else:
            # targets is 3-dimensional: 1) index in sentence, 2) datapoint, 3) encodings
            label = T.dtensor3('label');
        
        
        
        # Call the specific initializing function for this specific model
        if (self.layers > 1):
            prediction, right_hand, padded_label, summed_error, error = self.init_multiple_layers(X, label, varSettings);
        else:
            prediction, right_hand, padded_label, summed_error, error = self.init_other(X, label, varSettings);
          
        
        
        # Automatic backward pass for all models: gradients    
        derivatives = T.grad(error, self.vars.values());
           
        # Defining prediction
        if (single_digit):
            self._predict = theano.function([X], prediction);
        else:
            right_hand_symbol_indices = T.argmax(right_hand,axis=2);
            self._predict = theano.function([X], [prediction, 
                                                  right_hand_symbol_indices,
                                                  right_hand]);
        
        # Defining stochastic gradient descent
        learning_rate = T.dscalar('learning_rate');
        updates = [(var,var-learning_rate*der) for (var,der) in zip(self.vars.values(),derivatives)];
        self._sgd = theano.function([X, label, learning_rate], [], 
                                   updates=updates,
                                   allow_input_downcast=True)
        
        super(RecurrentNeuralNetwork, self).__init__();
    
    def init_other(self, X, label, varSettings):
        # Set scan functions and arguments
        if (self.lstm):
            recurrence_function = self.lstm_predict_single;
            predict_function = self.lstm_predict_sequence;
            # Set the prediction parameters to be either the prediction 
            # weights or the decoding weights depending on the setting 
            predict_parameters = [self.vars[k[0]] for k in filter(lambda name: name[0][0] != 'D', varSettings)];
            if (self.decoder):
                decode_parameters = [self.vars[k[0]] for k in filter(lambda name: name[0][0] == 'D', varSettings)];
            else:
                decode_parameters = predict_parameters;
        else:
            recurrence_function = self.rnn_predict_single;
            predict_function = self.rnn_predict_sequence;
            predict_parameters = [self.vars[k] for k in self.vars];
            decode_parameters = predict_parameters;
        
        first_hidden = np.zeros((self.minibatch_size,self.hidden_dim));
        [Y_1, hidden], _ = theano.scan(fn=recurrence_function,
                                sequences=X,
                                # Input a zero hidden layer
                                outputs_info=(None,first_hidden),
                                non_sequences=predict_parameters)
        
        # If X is shorter than Y we are testing and thus need to predict
        # multiple digits at the end by inputting the previously predicted
        # digit at each step as X
        if (self.single_digit):
            # We only predict on the final Y because for now we only predict the final digit in the expression
            # Takes the argmax over the last dimension, resulting in a vector of predictions
            prediction = T.argmax(Y_1[-1], 1);
            #error = T.nnet.categorical_crossentropy(Y_1[-1].reshape((1,self.minibatch_size)), label);
            # Perform crossentropy calculation for each datapoint and take the mean
            error = T.mean(T.nnet.categorical_crossentropy(Y_1[-1], label));
        else:
            # Add predictions until EOS to Y
            # The maximum number of digits to predict is n_max_digits
            init_values = ({'initial': Y_1[-1], 'taps': [-1]},
                           {'initial': hidden[-1], 'taps': [-1]});
            if (self.decoder):
                # When decoding we start the decoder with zeros as input and
                # the last prediction output state as first hidden state
                # This is why we had to change the prediction output dimension
                # size to match the hidden dimension size
                init_values = ({'initial': T.zeros((self.minibatch_size,self.data_dim), dtype='float64'), 'taps': [-1]},
                               {'initial': Y_1[-1], 'taps': [-1]});
            [Ys, _], _ = theano.scan(fn=predict_function,
                                     # Inputs the last hidden layer and the last predicted symbol
                                     outputs_info=init_values,
                                     non_sequences=decode_parameters,
                                     n_steps=self.n_max_digits)
            
            if (self.decoder):
                right_hand = Ys;
            else: 
                # The right hand is now the last output of the recurrence function
                # joined with the sequential output of the prediction function
                right_hand = T.join(0,Y_1[-1].reshape((1,self.minibatch_size,self.decoding_output_dim)),Ys);
            
            # We predict the final n symbols (all symbols predicted as output from input '=')
            prediction = T.argmax(right_hand, axis=2);
            padded_label = T.join(0, label, T.zeros((self.n_max_digits - label.shape[0],self.minibatch_size,self.decoding_output_dim)));
            
            accumulator = theano.shared(np.float64(0.), name='accumulatedError');
            summed_error, _ = theano.scan(fn=self.crossentropy_2d,
                                          sequences=(right_hand,padded_label),
                                          outputs_info={'initial': accumulator, 'taps': [-1]})
            error = summed_error[-1];
        
        return prediction, right_hand, padded_label, summed_error, error;
    
    def init_multiple_layers(self, X, label, varSettings):
        # Set scan functions and arguments
        
        if (not self.lstm):
            raise ValueError("Multiple layers can only be used with LSTM!");
        
        if (self.layers != 2):
            raise ValueError("Multiple layers for now only works with 2 layers.");
        
        recurrence_function = self.lstm_predict_single;
        predict_function = self.lstm_double_predict_sequence;
        predict_parameters = [self.vars[k[0]] for k in filter(lambda name: name[0][0] != 's', varSettings)];
        predict_parameters_2 = [self.vars[k[0]] for k in filter(lambda name: name[0][0] == 's', varSettings)];
        
        # Because the extra layers cannot communicate back up but only receive 
        # we can perform the scans after each other 
        
        first_hidden = np.zeros((self.minibatch_size,self.hidden_dim));
        [to_hidden_2, hidden], _ = theano.scan(fn=recurrence_function,
                                sequences=X,
                                # Input a zero hidden layer
                                outputs_info=(None,first_hidden),
                                non_sequences=predict_parameters)
        
        first_hidden_2 = np.zeros((self.minibatch_size,self.hidden_dim_2));
        [Y_1, hidden_2], _ = theano.scan(fn=recurrence_function,
                            sequences=to_hidden_2,
                            # Input a zero hidden layer
                            outputs_info=(None,first_hidden_2),
                            non_sequences=predict_parameters_2)
        
        # If X is shorter than Y we are testing and thus need to predict
        # multiple digits at the end by inputting the previously predicted
        # digit at each step as X
        if (self.single_digit):
            raise ValueError("Single digit not implemented yet for multiple layers!");
        else:
            # Add predictions to Y
            # We have to do both layers in one scan, since the output of the 
            # last layer becomes the input for the first one
            init_values = ({'initial': Y_1[-1], 'taps': [-1]},
                           {'initial': hidden[-1], 'taps': [-1]},
                           {'initial': hidden_2[-1], 'taps': [-1]});
            if (self.decoder):
                raise ValueError("Decoder not implemented yet for multiple layers!");
            [Ys, _, _], _ = theano.scan(fn=predict_function,
                                     # Inputs the last hidden layer and the last predicted symbol
                                     outputs_info=init_values,
                                     non_sequences=predict_parameters + predict_parameters_2,
                                     n_steps=self.n_max_digits)
            
            if (self.decoder):
                raise ValueError("Decoder not implemented yet for multiple layers!");
            else: 
                # The right hand is now the last output of the recurrence function
                # joined with the sequential output of the prediction function
                right_hand = T.join(0,Y_1[-1].reshape((1,self.minibatch_size,self.decoding_output_dim)),Ys);
            
            # We predict the final n symbols (all symbols predicted as output from input '=')
            prediction = T.argmax(right_hand, axis=2);
            padded_label = T.join(0, label, T.zeros((self.n_max_digits - label.shape[0],self.minibatch_size,self.decoding_output_dim)));
            
            accumulator = theano.shared(np.float64(0.), name='accumulatedError');
            summed_error, _ = theano.scan(fn=self.crossentropy_2d,
                                          sequences=(right_hand,padded_label),
                                          outputs_info={'initial': accumulator, 'taps': [-1]})
            error = summed_error[-1];
        
        return prediction, right_hand, padded_label, summed_error, error;
    
    def loadVars(self, variables):
        """
        Provide vars as a dictionary matching the self.vars structure.
        """
        for key in variables:
            if (key not in self.vars):
                return False;
            self.vars[key].set_value(variables[key].get_value());
        return True;
    
    def crossentropy_2d(self, coding_dist, true_dist, accumulated_score):
        return accumulated_score + T.mean(T.nnet.categorical_crossentropy(coding_dist, true_dist));
    
    # PREDICTION FUNCTIONS
    
    def rnn_predict_single(self, current_X, previous_hidden, hWo, hWh, XWh):
        hidden = T.nnet.sigmoid(previous_hidden.dot(hWh) + current_X.dot(XWh));
        Ys = T.nnet.softmax(hidden.dot(hWo));
        return Ys, hidden;
    
    def rnn_predict_sequence(self, current_X, previous_hidden, hWo, hWh, XWh):
        Ys, hidden = self.rnn_predict_single(current_X, previous_hidden, hWo, hWh, XWh);
        return [Ys, hidden];
    
    def lstm_predict_single(self, current_X, previous_hidden, hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, hWY):
        forget_gate = T.nnet.sigmoid(previous_hidden.dot(hWf) + current_X.dot(XWf));
        input_gate = T.nnet.sigmoid(previous_hidden.dot(hWi) + current_X.dot(XWi));
        candidate_cell = T.tanh(previous_hidden.dot(hWc) + current_X.dot(XWc));
        cell = forget_gate * previous_hidden + input_gate * candidate_cell;
        output_gate = T.nnet.sigmoid(previous_hidden.dot(hWo) + current_X.dot(XWo));
        hidden = output_gate * cell;
        Y_output = T.nnet.softmax(hidden.dot(hWY));
        return Y_output, hidden;
     
    def lstm_predict_sequence(self, current_X, previous_hidden, hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, hWY):
        Y_output, hidden = self.lstm_predict_single(current_X, previous_hidden, hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, hWY);
        return [Y_output, hidden];
    
    def lstm_double_predict_sequence(self, current_X, previous_hidden, previous_hidden_2, hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, hWh2, sh2Wf, shWf, sh2Wi, shWi, sh2Wc, shWc, sh2Wo, shWo, sh2WY):
        to_hidden_2, hidden = self.lstm_predict_single(current_X, previous_hidden, hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, hWh2);
        Y_output, hidden_2 = self.lstm_predict_single(to_hidden_2, previous_hidden_2, sh2Wf, shWf, sh2Wi, shWi, sh2Wc, shWc, sh2Wo, shWo, sh2WY);
        return [Y_output, hidden, hidden_2];
    
    # END OF INITIALIZATION
    
    def sanityChecks(self, training_data, training_labels):
        """
        Sanity checks to be called before training a batch. Throws exceptions 
        if things are not right.
        """
        if (not self.single_digit and training_labels.shape[1] > self.n_max_digits):
            raise ValueError("n_max_digits too small! Increase to %d" % training_labels.shape[1]);
    
    def sgd(self, data, label, learning_rate):
        return self._sgd(data, label, learning_rate);
        
    def predict(self, data):
        """
        Perform necessary models-specific transformations and call the actual 
        prediction function of the model.
        """
        if (self.single_digit):
            # Swap axes of index in sentence and datapoint for Theano purposes
            prediction = self.predict(np.swapaxes(data, 0, 1));
        else:
            prediction, right_hand_symbol_indices, right_hand = \
                self._predict(np.swapaxes(data, 0, 1));
        
        prediction = np.swapaxes(prediction, 0, 1);
        # Swap sentence index and datapoints back
        right_hand_symbol_indices = np.swapaxes(right_hand_symbol_indices, 0, 1);
        
        return prediction, {'right_hand_symbol_indices': right_hand_symbol_indices,
                            'right_hand': right_hand};
        
    def verboseOutput(self, prediction, other):
        self.verboseOutputter['write']("Prediction: %s\nright_hand_symbol_indices: %s\nright_hand: %s" 
                                       % (str(prediction), str(other['right_hand_symbol_indices']), 
                                          str(other['right_hand'])));
    
    def batch_statistics(self, stats, prediction, labels, targets, expressions, 
                         other, test_n, dataset,
                         excludeStats=None, no_print_progress=False,
                         eos_symbol_index=None, print_sample=False):
        # Statistics
        for j in range(0,test_n):
            if (self.single_digit):
                if (prediction[j] == np.argmax(labels[j])):
                    stats['correct'] += 1;
            else:
                # Get the labels
                argmax_target = np.argmax(targets[j],axis=1);
                # Compute the length of the target answer
                target_length = np.argmax(argmax_target);
                if (target_length == 0):
                    # If no EOS is found, the target is the entire length
                    target_length = targets[j].shape[1];
                # Compute the length of the prediction answer
                prediction_length = np.argmax(prediction[j]);
                if (prediction_length == target_length and np.array_equal(prediction[j][:target_length],argmax_target[:target_length])):
                    # Correct if prediction and target length match and 
                    # prediction and target up to target length are the same
                    stats['correct'] += 1.0;
                for k,digit in enumerate(prediction[j][:len(argmax_target)]):
                    if (digit == np.argmax(targets[j][k])):
                        stats['digit_correct'] += 1.0;
                    stats['digit_prediction_size'] += 1;
                    
            if (self.single_digit):
                stats['prediction_histogram'][int(prediction[j])] += 1;
                stats['groundtruth_histogram'][np.argmax(labels[j])] += 1;
                stats['prediction_confusion_matrix']\
                    [np.argmax(labels[j]),int(prediction[j])] += 1;
                if ('operator_scores' not in excludeStats):
                    stats['operator_scores'] = \
                        self.operator_scores(expressions[j], 
                                             int(prediction[j])==np.argmax(labels[j]),
                                             dataset.operators,
                                             dataset.key_indices,
                                             stats['operator_scores']);
            else:
                # Taking argmax over symbols for each sentence returns 
                # the location of the highest index, which is the first 
                # EOS symbol
                eos_location = np.argmax(other['right_hand_symbol_indices'][j]);
                # Check for edge case where no EOS was found and zero was returned
                if (eos_symbol_index is None):
                    eos_symbol_index = dataset.EOS_symbol_index;
                if (other['right_hand_symbol_indices'][j,eos_location] != eos_symbol_index):
                    stats['prediction_size_histogram'][other['right_hand_symbol_indices'][j].shape[0]] += 1;
                else:
                    stats['prediction_size_histogram'][int(eos_location)] += 1;
                for digit_prediction in prediction[j]:
                    stats['prediction_histogram'][int(digit_prediction)] += 1;
            stats['prediction_size'] += 1;
        
        return stats;
        
    def total_statistics(self, stats):
        """
        Adds general statistics to the statistics generated per batch.
        """
        stats['score'] = stats['correct'] / float(stats['prediction_size']);
        if (not self.single_digit):
            stats['digit_score'] = stats['digit_correct'] / float(stats['digit_prediction_size']);
        
        return stats;
    
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