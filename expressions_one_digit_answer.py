'''
Created on 16 feb. 2016

@author: Robert-Jan
'''

import numpy as np;
import theano;
import theano.tensor as T;
import time;
import sys;

from statistic_tools import confusion_matrix;

#theano.config.mode = 'FAST_COMPILE'

class RecurrentNeuralNetwork(object):
    '''
    classdocs
    '''


    def __init__(self, data_dim, hidden_dim, output_dim):
        '''
        Constructor
        '''
        # Store dimensions        
        self.data_dim = data_dim;
        self.hidden_dim = hidden_dim;
        self.output_dim = output_dim;
        
        # X weights to hidden
        self.init_XWh = np.random.uniform(-np.sqrt(1.0/self.data_dim),np.sqrt(1.0/self.data_dim),(self.data_dim,self.hidden_dim));
        # hidden weights to hidden
        self.init_hWh = np.random.uniform(-np.sqrt(1.0/self.hidden_dim),np.sqrt(1.0/self.hidden_dim),(self.hidden_dim,self.hidden_dim));
        # hidden weights to output
        self.init_hWo = np.random.uniform(-np.sqrt(1.0/self.hidden_dim),np.sqrt(1.0/self.hidden_dim),(self.hidden_dim,self.output_dim));
        
        # Set up shared variables
        self.XWh = theano.shared(name='XWh', value=self.init_XWh);
        self.hWh = theano.shared(name='hWh', value=self.init_hWh);
        self.hWo = theano.shared(name='hWo', value=self.init_hWo);
        
        # Forward pass
        # X is 2-dimensional: 1) index in sentence, 2) dimensionality of data 
        X = T.dmatrix('X');
        # targets is 1-dimensional: 1) answer encoding
        label = T.ivector('label');
        
        def rnn_recurrence(current_X, previous_hidden):
            hidden = T.nnet.sigmoid(previous_hidden.dot(self.hWh) + current_X.dot(self.XWh));
            Ys = T.nnet.softmax(hidden.dot(self.hWo));
            return hidden, Ys;
        
        [_, Y], _ = theano.scan(fn=rnn_recurrence,
                                 sequences=X,
                                 # Input a zero hidden layer
                                 outputs_info=(np.zeros(self.hidden_dim),None))
        
        # We only predict on the final Y because for now we only predict the final digit in the expression
        prediction = T.argmax(Y[-1]);
        error = T.nnet.categorical_crossentropy(Y[-1], label)[0];
        
        # Backward pass: gradients
        dXWh, dhWh, dhWo = T.grad(error, [self.XWh, self.hWh, self.hWo]);
        
        # Functions
        self.predict = theano.function([X], prediction);
        
        # Stochastic Gradient Descent
        learning_rate = T.dscalar('learning_rate');
        self.sgd = theano.function([X, label, learning_rate], [dXWh, dhWh, dhWo, Y[-1]], 
                                   updates=[(self.XWh,self.XWh - learning_rate * dXWh),
                                            (self.hWh,self.hWh - learning_rate * dhWh),
                                            (self.hWo,self.hWo - learning_rate * dhWo)],
                                   allow_input_downcast=False)
        
    def train(self, training_data, training_labels, learning_rate, max_training_size=None):
        k = 0;
        total = len(training_data);
        printing_interval = 1000;
        if (max_training_size is not None):
            total = max_training_size;
        if (total <= printing_interval * 10):
            printing_interval = total / 100;
        
        while k < total:
            data = training_data[k];
            label = training_labels[k];
            # Run training
            dXWh_value, dhWh_value, dhWo_value, Y_last = self.sgd(data, np.array([label]), learning_rate);
            
            if (k % printing_interval == 0):
                print("%d / %d" % (k, total));
            
            if (np.isnan(dXWh_value[0,0])):
                print("NaN!");
            
            k += 1; #minibatch_size;
        
    def test(self, test_data, test_labels, max_testing_size=None):
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
        
        correct = 0.0;
        prediction_size = 0;
        prediction_histogram = {k: 0 for k in range(self.output_dim)};
        groundtruth_histogram = {k: 0 for k in range(self.output_dim)};
        # First dimension is actual class, second dimension is predicted dimension
        prediction_confusion_matrix = np.zeros((dataset.output_dim,dataset.output_dim));
        # Predict
        for j in range(len(test_data)):
            if (j % printing_interval == 0):
                print("%d / %d" % (j, total));
            data = test_data[j];                
            prediction = self.predict(data);
            # For each index in sentence
            if (prediction == test_labels[j]):
                correct += 1.0;
            prediction_histogram[int(prediction)] += 1;
            groundtruth_histogram[test_labels[j]] += 1;
            prediction_confusion_matrix[test_labels[j],int(prediction)] += 1;
            prediction_size += 1;
                
        return (correct / float(prediction_size), prediction_histogram, groundtruth_histogram, prediction_confusion_matrix);

class GeneratedExpressionDataset(object):
    
    def __init__(self, sourceFolder):
        self.train_source = sourceFolder + '/train.txt';
        self.test_source = sourceFolder + '/test.txt';
        
        # Setting one-hot encoding
        self.oneHot = {'+': 10, '-': 11, '*': 12, '/': 13, '(': 14, ')': 15, '=': 16};
        # Digits are pre-assigned 0-9
        for digit in range(10):
            self.oneHot[str(digit)] = digit;
        # Data dimension = number of symbols + 1
        self.data_dim = max(self.oneHot.values()) + 1;
        self.output_dim = self.data_dim;
        
        self.load();
    
    def load(self):
        self.train, self.train_targets, self.train_labels = self.loadFile(self.train_source);
        self.test, self.test_targets, self.test_labels = self.loadFile(self.test_source);
    
    def loadFile(self, source):
        # Importing data
        f_data = open(source,'r');
        data = [];
        targets = [];
        labels = [];
        for line in f_data:
            # Get expression from line
            expression = line.strip();
            left_hand = expression[:-1];
            right_hand = expression[-1];
            # Generate encodings for data and target
            X = np.zeros((len(left_hand),self.data_dim));
            for i, literal in enumerate(left_hand):
                X[i,self.oneHot[literal]] = 1.0;
            target = np.zeros(self.data_dim);
            target[self.oneHot[right_hand]] = 1.0;
            # Set training variables
            data.append(X);
            targets.append(np.array([target]));
            labels.append(self.oneHot[right_hand]);
        
        return data, targets, np.array(labels);
        
if (__name__ == '__main__'):
    
    dataset_path = './data/expressions_one_digit_answer_large';
    repetitions = 1;
    hidden_dim = 16;
    learning_rate = 0.01;
    
    if (len(sys.argv) > 1):
        dataset_path = sys.argv[1];
        if (len(sys.argv) > 2):
            repetitions = int(sys.argv[2]);
            if (len(sys.argv) > 3):
                hidden_dim = int(sys.argv[3]);
                if (len(sys.argv) > 4):
                    learning_rate = float(sys.argv[4]);
    
    # Debug settings
    max_training_size = 1000;
    
    dataset = GeneratedExpressionDataset(dataset_path);
    rnn = RecurrentNeuralNetwork(dataset.data_dim, hidden_dim, dataset.output_dim);

    start = time.clock();
     
    # Train
    for r in range(repetitions):
        print("Repetition %d of %d" % (r+1,repetitions));
        rnn.train(dataset.train, dataset.train_labels, learning_rate, max_training_size);
     
    # Test
    score, prediction_histogram, groundtruth_histogram, prediction_confusion_matrix = rnn.test(dataset.test, dataset.test_labels)
     
    print
     
    # Print statistics
    duration = time.clock() - start;
    print("Duration: %d seconds" % duration);
    print("Score: %.2f percent" % (score*100));
    print("Prediction histogram:   %s" % (str(prediction_histogram)));
    print("Ground truth histogram: %s" % (str(groundtruth_histogram)));
    
    print("Confusion matrix:");
    confusion_matrix(prediction_confusion_matrix)
