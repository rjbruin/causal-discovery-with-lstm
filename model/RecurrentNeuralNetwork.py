'''
Created on 22 feb. 2016

@author: Robert-Jan
'''

import theano;
import theano.tensor as T;
import numpy as np;

class RecurrentNeuralNetwork(object):
    '''
    Recurrent neural network model with one hidden layer. Models single class 
    prediction based on regular recurrent model or LSTM model. 
    '''


    def __init__(self, data_dim, hidden_dim, output_dim, minibatch_size, 
                 lstm=False, weight_values={}, single_digit=True, EOS_symbol_index=None,
                 n_max_digits=24):
        '''
        Initialize all Theano models.
        '''
        self.single_digit = single_digit;
        
        # Store dimensions        
        self.data_dim = data_dim;
        self.hidden_dim = hidden_dim;
        self.output_dim = output_dim;
        self.minibatch_size = minibatch_size;
        self.n_max_digits = n_max_digits;
        
        self.fake_minibatch = False;
        if (self.minibatch_size == 1):
            self.fake_minibatch = True;
            self.minibatch_size = 2;
        
        if (not single_digit):
            if (EOS_symbol_index is None):
                # EOS symbol is last index by default
                EOS_symbol_index = self.data_dim-1;
            EOS_symbol = T.constant(EOS_symbol_index);
        
        varSettings = [];
        # Set up shared variables
        if (not lstm):
            varSettings.append(('XWh',self.data_dim,self.hidden_dim));
            varSettings.append(('hWh',self.hidden_dim,self.hidden_dim));
            varSettings.append(('hWo',self.hidden_dim,self.output_dim));
        else:
            varSettings.append(('hWf',self.hidden_dim,self.hidden_dim));
            varSettings.append(('XWf',self.data_dim,self.hidden_dim));
            varSettings.append(('hWi',self.hidden_dim,self.hidden_dim));
            varSettings.append(('XWi',self.data_dim,self.hidden_dim));
            varSettings.append(('hWc',self.hidden_dim,self.hidden_dim));
            varSettings.append(('XWc',self.data_dim,self.hidden_dim));
            varSettings.append(('hWo',self.hidden_dim,self.hidden_dim));
            varSettings.append(('XWo',self.data_dim,self.hidden_dim));
            varSettings.append(('hWY',self.hidden_dim,self.output_dim));
        
        # Contruct variables
        self.vars = {};
        for (varName,dim1,dim2) in varSettings:
            # Get value for shared variable from constructor if present
            value = np.random.uniform(-np.sqrt(1.0/dim1),np.sqrt(1.0/dim1),(dim1,dim2));
            if (varName in weight_values):
                value = weight_values[varName];
            self.vars[varName] = theano.shared(name=varName, value=value);
        
        # Forward pass
        # X is 3-dimensional: 1) index in sentence, 2) datapoint, 3) dimensionality of data
        X = T.dtensor3('X');
        
        if (single_digit):
            # targets is 2-dimensional: 1) datapoint, 2) label for each answer
            label = T.imatrix('label');
        else:
            # targets is 3-dimensional: 1) index in sentence, 2) datapoint, 3) encodings
            label = T.dtensor3('label');
        
        if (lstm):
            recurrence_function = self.lstm_predict_single;
            predict_function = self.lstm_predict_sequence;
        else:
            recurrence_function = self.rnn_predict_single;
            predict_function = self.rnn_predict_sequence;
        
        first_hidden = np.zeros((self.minibatch_size,self.hidden_dim));
        [Y_1, hidden], _ = theano.scan(fn=recurrence_function,
                                sequences=X,
                                # Input a zero hidden layer
                                outputs_info=(None,first_hidden))
        
        # If X is shorter than Y we are testing and thus need to predict
        # multiple digits at the end by inputting the previously predicted
        # digit at each step as X
        if (single_digit):
            # DEBUG
#             predicted_size = T.constant(1);
#             Ys = Y_1;
#             unfinished_sentence = Y_1;
             
            # We only predict on the final Y because for now we only predict the final digit in the expression
            # Takes the argmax over the last dimension, resulting in a vector of predictions
            prediction = T.argmax(Y_1[-1], 1);
            #error = T.nnet.categorical_crossentropy(Y_1[-1].reshape((1,self.minibatch_size)), label);
            # Perform crossentropy calculation for each datapoint and take the mean
            error = T.mean(T.nnet.categorical_crossentropy(Y_1[-1], label));
        else:
            # Add predictions until EOS to Y
            # The maximum number of digits to predict is n_max_digits
            [Ys, _], _ = theano.scan(fn=predict_function,
                                     # Inputs the last hidden layer and the last predicted symbol
                                     outputs_info=({'initial': Y_1[-1], 'taps': [-1]},
                                                   {'initial': hidden[-1], 'taps': [-1]}),
                                     non_sequences=EOS_symbol,
                                     n_steps=self.n_max_digits)
             
            # The right hand is now the last output of the recurrence function
            # joined with the sequential output of the prediction function
            accumulator = theano.shared(np.float64(0.), name='accumulatedError');
            right_hand = T.join(0,Y_1[-1].reshape((1,self.minibatch_size,self.output_dim)),Ys);
            # We predict the final n symbols (all symbols predicted as output from input '=')
            prediction = T.argmax(right_hand, axis=2);
            padded_label = T.join(0, label, T.zeros((self.n_max_digits - label.shape[0],self.minibatch_size,self.output_dim)));
            summed_error, _ = theano.scan(fn=self.crossentropy_2d,
                                          sequences=(right_hand,padded_label),
                                          outputs_info={'initial': accumulator, 'taps': [-1]})
            error = summed_error[-1] / T.constant(float(self.n_max_digits), dtype='float64');
          
        # Backward pass: gradients    
        derivatives = T.grad(error, self.vars.values());
           
        # Functions
        if (single_digit):
            self.predict = theano.function([X], prediction);
        else:
            right_hand_symbol_indices = T.argmax(right_hand,axis=2);
            self.predict = theano.function([X, label], [prediction, 
                                                 right_hand_symbol_indices,
                                                 right_hand,
                                                 padded_label,
                                                 summed_error]);
        
        # Stochastic Gradient Descent
        learning_rate = T.dscalar('learning_rate');
        # Always perform updates for all vars in self.vars
        updates = [(var,var-learning_rate*der) for (var,der) in zip(self.vars.values(),derivatives)];
        self.sgd = theano.function([X, label, learning_rate], [], 
                                   updates=updates,
                                   allow_input_downcast=True)
        
        # Sequence repairing - disabled until we find the bug in regular 
        # prediction
#         missing_X = T.iscalar();
#         dX = T.grad(error, X);
#         self.find_x_gradient = theano.function([X, label, missing_X], [dX[missing_X]]);
#         missing_X_digit = T.argmin(dX[missing_X]);
#         self.find_x = theano.function([X, label, missing_X], [missing_X_digit]);
    
    def crossentropy_2d(self, coding_dist, true_dist, accumulated_score):
        return accumulated_score + T.mean(T.nnet.categorical_crossentropy(coding_dist, true_dist));
    
    # PREDICTION FUNCTIONS
    
    def rnn_predict_single(self, current_X, previous_hidden):
        hidden = T.nnet.sigmoid(previous_hidden.dot(self.vars['hWh']) + current_X.dot(self.vars['XWh']));
        Ys = T.nnet.softmax(hidden.dot(self.vars['hWo']));
        return Ys, hidden;
    
    def rnn_predict_sequence(self, current_X, previous_hidden, EOS_symbol):
        Ys, hidden = self.rnn_predict_single(current_X, previous_hidden);
        #return [Ys, hidden], {}, theano.scan_module.until(T.eq(T.argmax(Ys),EOS_symbol));
        return [Ys, hidden];
    
    def lstm_predict_single(self, current_X, previous_hidden):
        forget_gate = T.nnet.sigmoid(previous_hidden.dot(self.vars['hWf']) + current_X.dot(self.vars['XWf']));
        input_gate = T.nnet.sigmoid(previous_hidden.dot(self.vars['hWi']) + current_X.dot(self.vars['XWi']));
        candidate_cell = T.tanh(previous_hidden.dot(self.vars['hWc']) + current_X.dot(self.vars['XWc']));
        cell = forget_gate * previous_hidden + input_gate * candidate_cell;
        output_gate = T.nnet.sigmoid(previous_hidden.dot(self.vars['hWo']) + current_X.dot(self.vars['XWo']));
        hidden = output_gate * cell;
        Y_output = T.nnet.softmax(hidden.dot(self.vars['hWY']));
        return Y_output, hidden;
     
    def lstm_predict_sequence(self, current_X, previous_hidden, EOS_symbol):
        Y_output, hidden = self.lstm_predict_single(current_X, previous_hidden);
        #return [Y_output, hidden], {}, theano.scan_module.until(T.eq(T.argmax(Y_output),EOS_symbol));
        return [Y_output, hidden];
    
    # END OF INITIALIZATION
    
    def train(self, training_data, training_labels, learning_rate, no_print=False):
        """
        Takes data and trains the model with it. DOES NOT handle batching or 
        any other project-like structures.
        """
        # Sanity checks
        if (not self.single_digit and training_labels.shape[1] > self.n_max_digits):
            raise ValueError("n_max_digits too small! Increase to %d" % training_labels.shape[1]);
        
        # Set printing interval
        total = len(training_data);
        printing_interval = 1000;
        if (total <= printing_interval * 10):
            # Make printing interval always at least one
            printing_interval = max(total / 10,1);
        
        # Train model per minibatch
        batch_range = range(0,total,self.minibatch_size);
        if (self.fake_minibatch):
            batch_range = range(0,total);
        for k in batch_range:
            data = training_data[k:k+self.minibatch_size];
            label = training_labels[k:k+self.minibatch_size];
            
            if (self.fake_minibatch):
                data = training_data[k:k+1];
                label = training_labels[k:k+1];
            
            if (len(data) < self.minibatch_size):
                missing_datapoints = self.minibatch_size - data.shape[0];
                data = np.concatenate((data,np.zeros((missing_datapoints, training_data.shape[1], training_data.shape[2]))), axis=0);
                label = np.concatenate((label,np.zeros((missing_datapoints, training_labels.shape[1], training_labels.shape[2]))), axis=0);
            # Swap axes of index in sentence and datapoint for Theano purposes
            data = np.swapaxes(data, 0, 1);
            label = np.swapaxes(label, 0, 1);
            # Run training
            self.sgd(data, label, learning_rate);
            
            if (not no_print and k % printing_interval == 0):
                print("# %d / %d" % (k, total));
        
    def test(self, test_data, test_labels, test_targets, test_expressions, dataset, stats, excludeStats=None, no_print=False):
        """
        Run test data through model. Output percentage of correctly predicted
        test instances. DOES NOT handle batching. DOES output testing 
        statistics.
        """
        # Set printing interval
        total = len(test_data);
        printing_interval = 1000;
        if (total <= printing_interval * 10):
            # Make printing interval always at least one
            printing_interval = max(total / 10,1);
        
        batch_range = range(0,len(test_data),self.minibatch_size);
        if (self.fake_minibatch):
            batch_range = range(0,len(test_data));
        for j in batch_range:
            data = test_data[j:j+self.minibatch_size];
            targets = test_targets[j:j+self.minibatch_size];
            labels = test_labels[j:j+self.minibatch_size];
            test_n = self.minibatch_size;
            
            if (self.fake_minibatch):
                data = test_data[j:j+1];
                targets = test_targets[j:j+1];
                labels = test_labels[j:j+1];
                test_n = 1;
            
            # Add zeros to minibatch if the batch is too small
            if (len(data) < self.minibatch_size):
                test_n = data.shape[0];
                missing_datapoints = self.minibatch_size - test_n;
                data = np.concatenate((data,np.zeros((missing_datapoints, test_data.shape[1], test_data.shape[2]))), axis=0);
                targets = np.concatenate((targets,np.zeros((missing_datapoints, test_targets.shape[1], test_targets.shape[2]))), axis=0);
            
            if (self.single_digit):
                # Swap axes of index in sentence and datapoint for Theano purposes
                prediction = self.predict(np.swapaxes(data, 0, 1));
            else:
                prediction, right_hand_symbol_indices, right_hand, padded_label, summed_error = \
                    self.predict(np.swapaxes(data, 0, 1), np.swapaxes(targets, 0, 1));
            
            prediction = np.swapaxes(prediction, 0, 1);
            # Swap sentence index and datapoints back
            right_hand_symbol_indices = np.swapaxes(right_hand_symbol_indices, 0, 1);
            
            # Statistics
            for js in range(j,j+test_n):
                if (self.single_digit):
                    if (prediction[js-j] == np.argmax(labels[js-j])):
                        stats['correct'] += 1;
                else:
                    # Get the labels
                    argmax_target = np.argmax(targets[js-j],axis=1);
                    # Compute the length of the target answer
                    target_length = np.argmax(argmax_target);
                    if (target_length == 0):
                        # If no EOS is found, the target is the entire length
                        target_length = targets[js-j].shape[1];
                    # Compute the length of the prediction answer
                    prediction_length = np.argmax(prediction[js-j]);
                    if (prediction_length == target_length and np.array_equal(prediction[js-j][:target_length],argmax_target[:target_length])):
                        # Correct if prediction and target length match and 
                        # prediction and target up to target length are the same
                        stats['correct'] += 1.0;
                    for k,digit in enumerate(prediction[js-j][:len(argmax_target)]):
                        if (digit == np.argmax(targets[js-j][k])):
                            stats['digit_correct'] += 1.0;
                        stats['digit_prediction_size'] += 1;
                        
                if (self.single_digit):
                    stats['prediction_histogram'][int(prediction[js-j])] += 1;
                    stats['groundtruth_histogram'][np.argmax(labels[js-j])] += 1;
                    stats['prediction_confusion_matrix']\
                        [np.argmax(labels[js-j]),int(prediction[js-j])] += 1;
                    if ('operator_scores' not in excludeStats):
                        stats['operator_scores'] = \
                            self.operator_scores(test_expressions[j], 
                                                 int(prediction[js-j])==np.argmax(labels[js-j]),
                                                 dataset.operators,
                                                 dataset.key_indices,
                                                 stats['operator_scores']);
                else:
                    # Taking argmax over symbols for each sentence returns 
                    # the location of the highest index, which is the first 
                    # EOS symbol
                    eos_location = np.argmax(right_hand_symbol_indices[js-j]);
                    stats['prediction_size_histogram'][int(eos_location)] += 1;
                    if (int(eos_location) > self.n_max_digits):
                        print('score');
                    for digit_prediction in prediction[js-j]:
                        stats['prediction_histogram'][int(digit_prediction)] += 1;
                stats['prediction_size'] += 1;
            
            if (not no_print and stats['prediction_size'] % printing_interval == 0):
                print("# %d / %d" % (stats['prediction_size'], total));
        
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
