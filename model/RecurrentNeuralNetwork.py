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


    def __init__(self, data_dim, hidden_dim, output_dim, lstm=False):
        '''
        Constructor
        '''
        # Store dimensions        
        self.data_dim = data_dim;
        self.hidden_dim = hidden_dim;
        self.output_dim = output_dim;
        
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
        label = T.ivector('label');
        
        if (not lstm):
            recurrence_function = self.rnn_recurrence;
        else:
            recurrence_function = self.lstm_recurrence;
        
        [_, Y], _ = theano.scan(fn=recurrence_function,
                                 sequences=X,
                                 # Input a zero hidden layer
                                 outputs_info=(np.zeros(self.hidden_dim),None))
        
        # We only predict on the final Y because for now we only predict the final digit in the expression
        prediction = T.argmax(Y[-1]);
        error = T.nnet.categorical_crossentropy(Y[-1], label)[0];
        
        # Backward pass: gradients
        derivatives = T.grad(error, self.vars.values());
        
        # Functions
        self.predict = theano.function([X], prediction);
        
        # Stochastic Gradient Descent
        learning_rate = T.dscalar('learning_rate');
        updates = [(var,var-learning_rate*der) for (var,der) in zip(self.vars.values(),derivatives)];
        self.sgd = theano.function([X, label, learning_rate], [], 
                                   updates=updates,
                                   allow_input_downcast=False)
    
    def rnn_recurrence(self, current_X, previous_hidden):
        hidden = T.nnet.sigmoid(previous_hidden.dot(self.vars['hWh']) + current_X.dot(self.vars['XWh']));
        Ys = T.nnet.softmax(hidden.dot(self.vars['hWo']));
        return hidden, Ys;
    
    def lstm_recurrence(self, current_X, previous_hidden):
        forget_gate = T.nnet.sigmoid(previous_hidden.dot(self.vars['hWf']) + current_X.dot(self.vars['XWf']));
        input_gate = T.nnet.sigmoid(previous_hidden.dot(self.vars['hWi']) + current_X.dot(self.vars['XWi']));
        candidate_cell = T.tanh(previous_hidden.dot(self.vars['hWc']) + current_X.dot(self.vars['XWc']));
        cell = forget_gate * previous_hidden + input_gate * candidate_cell;
        output_gate = T.nnet.sigmoid(previous_hidden.dot(self.vars['hWo']) + current_X.dot(self.vars['XWo']));
        hidden = output_gate * cell;
        Y_output = T.nnet.softmax(hidden.dot(self.vars['hWY']));
        return hidden, Y_output;
        
    def train(self, training_data, training_labels, learning_rate, max_training_size=None):
        total = len(training_data);
        printing_interval = 1000;
        if (max_training_size is not None):
            total = max_training_size;
        if (total <= printing_interval * 10):
            printing_interval = total / 100;
        
        for k in range(total):
            data = training_data[k];
            label = training_labels[k];
            # Run training
            self.sgd(data, np.array([label]), learning_rate);
            
            if (k % printing_interval == 0):
                print("%d / %d" % (k, total));
        
    def test(self, test_data, test_labels, test_expressions, operators, key_indices, dataset, max_testing_size=None):
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
            printing_interval = total / 100;
        
        # Set up statistics
        correct = 0.0;
        prediction_size = 0;
        prediction_histogram = {k: 0 for k in range(self.output_dim)};
        groundtruth_histogram = {k: 0 for k in range(self.output_dim)};
        # First dimension is actual class, second dimension is predicted dimension
        prediction_confusion_matrix = np.zeros((dataset.output_dim,dataset.output_dim));
        # For each non-digit symbol keep correct and total predictions
        operator_scores = np.zeros((len(key_indices),2));
        
        # Predict
        for j in range(len(test_data)):
            if (j % printing_interval == 0):
                print("%d / %d" % (j, total));
            data = test_data[j];                
            prediction = self.predict(data);
            
            # Statistics
            if (prediction == test_labels[j]):
                correct += 1.0;
            prediction_histogram[int(prediction)] += 1;
            groundtruth_histogram[test_labels[j]] += 1;
            prediction_confusion_matrix[test_labels[j],int(prediction)] += 1;
            prediction_size += 1;
            operator_scores = dataset.operator_scores(test_expressions[j],int(prediction)==test_labels[j],operators,key_indices,operator_scores);
                
        return (correct / float(prediction_size), prediction_histogram, groundtruth_histogram, prediction_confusion_matrix, operator_scores);