'''
Created on 22 feb. 2016

@author: Robert-Jan
'''

import theano;
import theano.tensor as T;
import theano.scan_module;
import numpy as np;
import time;

class RecurrentNeuralNetwork(object):
    '''
    Recurrent neural network model with one hidden layer. Models single class 
    prediction based on regular recurrent model or LSTM model. 
    '''


    def __init__(self, data_dim, hidden_dim, output_dim, lstm=False, weight_values={}, single_digit=True, EOS_symbol_index=None,
                 time_training_batch=False):
        '''
        Initialize all Theano models.
        '''
        self.single_digit = single_digit;
        
        # Store dimensions        
        self.data_dim = data_dim;
        self.hidden_dim = hidden_dim;
        self.output_dim = output_dim;
        self.time_training_batch = time_training_batch;
        
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
        # X is 2-dimensional: 1) index in sentence, 2) dimensionality of data 
        X = T.dmatrix('X');
        
        if (single_digit):
            # targets is 1-dimensional: 1) label for each answer
            label = T.ivector('label');
        else:
            # targets is 2-dimensional: 1) answers, 2) encodings
            label = T.dmatrix('label');
        
        if (lstm):
            recurrence_function = self.lstm_predict_single;
            predict_function = self.lstm_predict_sequence;
        else:
            recurrence_function = self.rnn_predict_single;
            predict_function = self.rnn_predict_sequence;
        
        [Y_1, hidden], _ = theano.scan(fn=recurrence_function,
                                sequences=X,
                                # Input a zero hidden layer
                                outputs_info=(None,np.zeros(self.hidden_dim)))
        
        # If X is shorter than Y we are testing and thus need to predict
        # multiple digits at the end by inputting the previously predicted
        # digit at each step as X
        if (single_digit):
            # DEBUG
            predicted_size = T.constant(1);
            Ys = Y_1;
            unfinished_sentence = Y_1;
            # KEEP THIS
            sentence = Y_1;
            
            # We only predict on the final Y because for now we only predict the final digit in the expression
            prediction = T.argmax(Y_1[-1]);
            error = T.nnet.categorical_crossentropy(Y_1[-1].reshape((1,self.output_dim)), label)[0];
        else:
            # Add predictions until EOS to Y
            # The maximum number of digits to predict is:
            n_max_digits = 24;
            [Ys, _], _ = theano.scan(fn=predict_function,
                                     # Inputs the last hidden layer and the last predicted symbol
                                     outputs_info=({'initial': Y_1[-1], 'taps': [-1]},
                                                   {'initial': hidden[-1], 'taps': [-1]}),
                                     non_sequences=EOS_symbol,
                                     n_steps=n_max_digits)
            # After predicting digits, check if we predicted the right amount of digits
            unfinished_sentence = T.join(0,Y_1,Ys); # DEBUG
            #full_sentence_size = unfinished_sentence.shape[0]; # DEBUG
            predicted_size = Ys.shape[0]; # DEBUG
            
            # Add zero scores to fill up the answer to the maximum size
            zero_scores = T.zeros((n_max_digits - Ys.shape[0],self.output_dim));
            full_right_hand = T.join(0,Ys,zero_scores);
            
            # Because we added zeros to pad the answer to be the maximum length
            # we predicted too many digits - throw away digits we don't need
            # The algorithm will be punished as the final symbol should be EOS
            sentence_size = X.shape[0];
            total_size = sentence_size + label.shape[0];
            right_hand = full_right_hand[:label.shape[0]];
            sentence = T.join(0,Y_1,right_hand);
            sentence = sentence[:total_size];
            
            # We predict the final n symbols (all symbols predicted as output from input '=')
            prediction = T.argmax(right_hand, axis=1);
            error = T.mean(T.nnet.categorical_crossentropy(right_hand, label));
          
        # Backward pass: gradients    
        derivatives = T.grad(error, self.vars.values());
          
        # Functions
        if (single_digit):
            self.predict = theano.function([X], prediction);
        else:
            self.predict = theano.function([X, label], [prediction, predicted_size, T.argmax(Ys,axis=1), 
                                                        total_size, sentence_size, T.argmax(sentence,axis=1),
                                                        T.argmax(unfinished_sentence,axis=1), label.shape[0],
                                                        T.argmax(Y_1,axis=1), T.argmax(Ys,axis=1), T.lt(sentence_size,total_size),
                                                        T.argmax(full_right_hand,axis=1)]);
        
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
    
    # PREDICTION FUNCTIONS
    
    def rnn_predict_single(self, current_X, previous_hidden):
        hidden = T.nnet.sigmoid(previous_hidden.dot(self.vars['hWh']) + current_X.dot(self.vars['XWh']));
        Ys = T.nnet.softmax(hidden.dot(self.vars['hWo']));
        return Ys.flatten(ndim=1), hidden;
    
    def rnn_predict_sequence(self, current_X, previous_hidden, EOS_symbol):
        Ys, hidden = self.rnn_predict_single(current_X, previous_hidden);
        return [Ys, hidden], {}, theano.scan_module.until(T.eq(T.argmax(Ys),EOS_symbol));
    
    def lstm_predict_single(self, current_X, previous_hidden):
        forget_gate = T.nnet.sigmoid(previous_hidden.dot(self.vars['hWf']) + current_X.dot(self.vars['XWf']));
        input_gate = T.nnet.sigmoid(previous_hidden.dot(self.vars['hWi']) + current_X.dot(self.vars['XWi']));
        candidate_cell = T.tanh(previous_hidden.dot(self.vars['hWc']) + current_X.dot(self.vars['XWc']));
        cell = forget_gate * previous_hidden + input_gate * candidate_cell;
        output_gate = T.nnet.sigmoid(previous_hidden.dot(self.vars['hWo']) + current_X.dot(self.vars['XWo']));
        hidden = output_gate * cell;
        Y_output = T.nnet.softmax(hidden.dot(self.vars['hWY']));
        return Y_output.flatten(ndim=1), hidden;
     
    def lstm_predict_sequence(self, current_X, previous_hidden, EOS_symbol):
        Y_output, hidden = self.lstm_predict_single(current_X, previous_hidden);
        return [Y_output, hidden], {}, theano.scan_module.until(T.eq(T.argmax(Y_output),EOS_symbol));
    
    # END OF INITIALIZATION
    
    def train(self, training_data, training_labels, learning_rate):
        """
        Takes data and trains the model with it. DOES NOT handle batching or 
        any other project-like structures.
        """
        # Set printing interval
        total = len(training_data);
        printing_interval = 1000;
        if (total <= printing_interval * 10):
            # Make printing interval always at least one
            printing_interval = max(total / 5,1);
        
        # Train model per datapoint
        if (self.time_training_batch):
            start = time.clock();
        for k in range(total):
            data = np.array(training_data[k]);
            label = np.array(training_labels[k]);
            if (self.single_digit):
                label = np.array([label]);
            # Run training
            self.sgd(data, label, learning_rate);
            
            if (k % printing_interval == 0):
                print("# %d / %d" % (k, total));
                if (self.time_training_batch):
                    duration = time.clock() - start;
                    print("%d seconds" % duration);
                    start = time.clock();
        
    def test(self, test_data, test_labels, test_targets, test_expressions, dataset, stats, excludeStats=None):
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
        
        for j in range(len(test_data)):
            data = test_data[j];
            
            if (self.single_digit):
                prediction = self.predict(data);
            else:
                prediction, right_hand_size, predicted_right_hand,\
                total_size, sentence_size, sentence, unf_sentence,\
                label_size, Y_1, Ys, lt, full_right_hand = \
                    self.predict(data,test_targets[j]);
            
            # Statistics
            if (self.single_digit):
                if (prediction == test_labels[j]):
                    stats['correct'] += 1;
            else:
                if (np.array_equal(prediction,np.argmax(test_targets[j],axis=1))):
                    stats['correct'] += 1.0;
                for k,digit in enumerate(prediction):
                    if (digit == np.argmax(test_targets[j][k])):
                        stats['digit_correct'] += 1.0;
                    stats['digit_prediction_size'] += 1;
                    
            if (self.single_digit):
                stats['prediction_histogram'][int(prediction)] += 1;
                stats['groundtruth_histogram'][test_labels[j]] += 1;
                stats['prediction_confusion_matrix']\
                    [test_labels[j],int(prediction)] += 1;
                if ('operator_scores' not in excludeStats):
                    stats['operator_scores'] = \
                        self.operator_scores(test_expressions[j], 
                                             int(prediction)==test_labels[j],
                                             dataset.operators,
                                             dataset.key_indices,
                                             stats['operator_scores']);
            else:
                stats['prediction_size_histogram'][int(right_hand_size)] += 1;
            stats['prediction_size'] += 1;
            
            if (stats['prediction_size'] % printing_interval == 0):
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
