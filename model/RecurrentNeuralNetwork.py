'''
Created on 22 feb. 2016

@author: Robert-Jan
'''

import theano;
import theano.tensor as T;
import theano.scan_module;
import numpy as np;

class RecurrentNeuralNetwork(object):
    '''
    Recurrent neural network model with one hidden layer. Models single class 
    prediction based on regular recurrent model or LSTM model. 
    '''


    def __init__(self, data_dim, hidden_dim, output_dim, lstm=False, single_digit=True, EOS_symbol_index=None):
        '''
        Constructor
        '''
        self.single_digit = single_digit;
        
        # Store dimensions        
        self.data_dim = data_dim;
        self.hidden_dim = hidden_dim;
        self.output_dim = output_dim;
        
        if (not single_digit and EOS_symbol_index is None):
            # EOS symbol is last index by default
            EOS_symbol_index = self.data_dim-1;
        
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
        
        self.vars = {};
        for (varName,dim1,dim2) in varSettings:
            self.vars[varName] = theano.shared(name=varName, value=np.random.uniform(-np.sqrt(1.0/dim1),np.sqrt(1.0/dim1),(dim1,dim2)));
        
        # Forward pass
        # X is 2-dimensional: 1) index in sentence, 2) dimensionality of data 
        X = T.dmatrix('X');
        # targets is 1-dimensional: 1) answer encoding
        
        if (single_digit):
            label = T.ivector('label');
        else:
            label = T.dmatrix('label');
        
        if (lstm):
            recurrence_function = self.lstm_predict;
            if (not single_digit):
                Y_function = self.lstm_predict_sequence;
        else:
            recurrence_function = self.rnn_predict;
            if (not single_digit):
                Y_function = self.rnn_predict_sequence;
        
        [Y, hidden], _ = theano.scan(fn=recurrence_function,
                                sequences=X,
                                # Input a zero hidden layer
                                outputs_info=(None,np.zeros(self.hidden_dim)))
        
        # If X is shorter than Y we are testing and thus need to predict
        # multiple digits at the end by inputting the previously predicted
        # digit at each step as X
        sentence_size = Y.shape[0];
        total_size = label.shape[0];
        if (not single_digit and T.lt(sentence_size,total_size)):
            # Add predictions until EOS to Y
            [Ys, _], _ = theano.scan(fn=recurrence_function,
                                     # Inputs the last hidden layer and the last predicted symbol
                                     outputs_info=({'initial': Y[-1], 'taps': [-1]},
                                                   {'initial': hidden[-1], 'taps': [-1]}),
                                     #outputs_info=(np.zeros(self.output_dim),np.zeros(self.hidden_dim)),
                                     #non_sequences=EOS_symbol_index,
                                     n_steps=5)
            # After predicting digits, check if we predicted the right amount of digits
            sentence = T.concatenate([Y,Ys],axis=1);
            sentence_size = sentence.shape[0];
            if (T.lt(sentence_size,total_size)):
                # We did not predict enough digits - add zero scores
                x = total_size - sentence_size;
                zero_scores = T.zeros((x,self.output_dim));
                right_hand = T.concatenate([Ys,zero_scores],axis=1)
                sentence = T.concatenate([Y,right_hand],axis=1);
            elif (T.gt(sentence_size,total_size)):
                # We predicted too many digits - throw away digits we don't need
                # The algorithm will be punished as the final symbol should be EOS
                sentence = sentence[:total_size];
                right_hand = right_hand[:(total_size-Y.shape[0])]
        
        if (single_digit):
            # We only predict on the final Y because for now we only predict the final digit in the expression
            prediction = T.argmax(Y[-1]);
            error = T.nnet.categorical_crossentropy(Y[-1], label)[0];
        else:
            # We predict the final n symbols (all symbols predicted as output from input '=')
            prediction = T.argmax(right_hand, axis=1);
            error = T.mean(T.nnet.categorical_crossentropy(right_hand, label));
        
        # Backward pass: gradients
        derivatives = T.grad(error, self.vars.values());
        
        # Functions
        self.predict = theano.function([X, label], prediction);
        
        # Stochastic Gradient Descent
        learning_rate = T.dscalar('learning_rate');
        updates = [(var,var-learning_rate*der) for (var,der) in zip(self.vars.values(),derivatives)];
        self.sgd = theano.function([X, label, learning_rate], [], 
                                   updates=updates,
                                   allow_input_downcast=False,
                                   mode='DebugMode')
    
    def rnn_predict(self, current_X, previous_hidden):
        hidden = T.nnet.sigmoid(previous_hidden.dot(self.vars['hWh']) + current_X.dot(self.vars['XWh']));
        Ys = T.nnet.softmax(hidden.dot(self.vars['hWo']));
        return Ys.flatten(ndim=1), hidden;
        #return Ys, hidden;
    
    #def rnn_predict_sequence(self, current_X, previous_hidden, EOS_symbol_index):
    def rnn_predict_sequence(self, current_X, previous_hidden):
        Ys, hidden = self.rnn_predict(current_X, previous_hidden);
        #return [hidden, Ys], {}, theano.scan_module.until(T.eq(T.argmax(Ys),EOS_symbol_index));
        #return [hidden, Ys], {}, theano.scan_module.until(T.eq(EOS_symbol_index,T.nnet.softmax(T.nnet.sigmoid(previous_hidden.dot(self.vars['hWh']) + current_X.dot(self.vars['XWh']))).dot(self.vars['hWo'])));
        # TODO: debug - using random termination criterion now
        return (Ys, hidden), theano.scan_module.until(T.sum(T.nnet.softmax(current_X[-1].dot(previous_hidden[-1]))) > 2.0);
        #return hidden, Ys;
    
#     def lstm_predict(self, current_X, previous_hidden):
#         forget_gate = T.nnet.sigmoid(previous_hidden.dot(self.vars['hWf']) + current_X.dot(self.vars['XWf']));
#         input_gate = T.nnet.sigmoid(previous_hidden.dot(self.vars['hWi']) + current_X.dot(self.vars['XWi']));
#         candidate_cell = T.tanh(previous_hidden.dot(self.vars['hWc']) + current_X.dot(self.vars['XWc']));
#         cell = forget_gate * previous_hidden + input_gate * candidate_cell;
#         output_gate = T.nnet.sigmoid(previous_hidden.dot(self.vars['hWo']) + current_X.dot(self.vars['XWo']));
#         hidden = output_gate * cell;
#         Y_output = T.nnet.softmax(hidden.dot(self.vars['hWY']));
#         return hidden, Y_output;
#     
#     def lstm_predict_sequence(self, current_X, previous_hidden, EOS_symbol_index):
#         hidden, Y_output = self.lstm_predict(current_X, previous_hidden);
#         prediction = T.eq(T.argmax(Y_output),EOS_symbol_index);
#         return [hidden, Y_output], theano.scan_module.until(prediction);
#     
#     def lstm_predict_sequence_from_Y(self, previous_hidden, previous_Y, EOS_symbol_index):
#         return self.lstm_predict_seqence(previous_Y, previous_hidden, EOS_symbol_index);
    
    def train(self, training_data, training_labels, learning_rate):
        total = len(training_data);
        printing_interval = 1000;
        if (total <= printing_interval * 10):
            printing_interval = total / 10;
        
        for k in range(total):
            data = np.array(training_data[k]);
            label = np.array(training_labels[k]);
            if (self.single_digit):
                label = np.array([label]);
            # Run training
            
            self.sgd(data, label, learning_rate);
            
            if (k % printing_interval == 0):
                print("# %d / %d" % (k, total));
        
    def test(self, test_data, test_labels, test_expressions, operators, key_indices, dataset, max_testing_size=None, single_digit=True):
        """
        Run test data through model. Output percentage of correctly predicted
        test instances.
        """
        print("Testing...");
        
        total = len(test_data);
        printing_interval = 1000;
        if (max_testing_size is not None):
            total = max_testing_size;
        if (total < printing_interval * 10):
            printing_interval = total / 10;
        
        # Set up statistics
        correct = 0.0;
        prediction_size = 0;
        if (single_digit):
            prediction_histogram = {k: 0 for k in range(self.output_dim)};
            groundtruth_histogram = {k: 0 for k in range(self.output_dim)};
            # First dimension is actual class, second dimension is predicted dimension
            prediction_confusion_matrix = np.zeros((dataset.output_dim,dataset.output_dim));
            # For each non-digit symbol keep correct and total predictions
            operator_scores = np.zeros((len(key_indices),2));
        
        # Predict
        for j in range(len(test_data)):
            if (j % printing_interval == 0):
                print("# %d / %d" % (j, total));
            data = test_data[j];
            prediction = self.predict(data);
            
            # Statistics
            if (prediction == test_labels[j]):
                correct += 1.0;
            if (single_digit):
                prediction_histogram[int(prediction)] += 1;
                groundtruth_histogram[test_labels[j]] += 1;
                prediction_confusion_matrix[test_labels[j],int(prediction)] += 1;
                operator_scores = dataset.operator_scores(test_expressions[j],int(prediction)==test_labels[j],operators,key_indices,operator_scores);
            prediction_size += 1;
        
        if (single_digit):
            return (correct / float(prediction_size), prediction_histogram, groundtruth_histogram, prediction_confusion_matrix, operator_scores);
        else:
            return correct / float(prediction_size);