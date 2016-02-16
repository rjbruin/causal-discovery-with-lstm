'''
Created on 11 jan. 2016

@author: Robert-Jan
'''

import numpy as np;
import theano;
import theano.tensor as T;
from sklearn.datasets import fetch_mldata
import time;

# theano.config.mode = 'FAST_COMPILE'

class NeuralNetwork(object):
    '''
    classdocs
    '''


    def __init__(self, training_dim, hidden_dim, output_dim):
        '''
        Constructor
        '''
        
        self.words_dim = training_dim;
        self.hidden_dim = hidden_dim;
        self.output_dim = output_dim;
        
        self.init_W = np.random.uniform(-np.sqrt(1.0/self.words_dim),np.sqrt(1.0/self.words_dim),(self.words_dim,self.hidden_dim));
        self.init_V = np.random.uniform(-np.sqrt(1.0/self.hidden_dim),np.sqrt(1.0/self.hidden_dim),(self.hidden_dim,self.output_dim));
        
        self.W = theano.shared(name='W', value=self.init_W);
        self.V = theano.shared(name='V', value=self.init_V);
        
        # Forward pass
        X = T.fmatrix('X');
        target = T.fmatrix('target');
        unsigmoided_hidden = X.dot(self.W);
        hidden = T.nnet.sigmoid(unsigmoided_hidden);
        unmaxed_Y = hidden.dot(self.V);
        Y = T.nnet.softmax(unmaxed_Y);
        prediction = T.argmax(Y, axis=1);
        # http://deeplearning.net/tutorial/logreg.html
        unsummed_error = T.nnet.categorical_crossentropy(Y, target);
        error = T.mean(unsummed_error);
#         unsummed_error = T.log(Y)[T.arange(target.shape[0]), T.argmax(target,axis=1)];
#         error = -T.mean(unsummed_error);
        
        # Backward pass: gradients
        dW, dV = T.grad(error, [self.W, self.V]);
        
        # Functions
        self.predict = theano.function([X], prediction);
        self.error = theano.function([X, target], error);
        self.propagation = theano.function([X, target], [dW, dV]);
        
        # Stochastic Gradient Descent
        learning_rate = T.dscalar('learning_rate');
        self.sgd = theano.function([X, target, learning_rate], [dW, dV, unsigmoided_hidden, unmaxed_Y, unsummed_error], 
                                   updates=[(self.W,self.W - learning_rate * dW),
                                            (self.V,self.V - learning_rate * dV)],
                                   allow_input_downcast=False)
        
    def train(self, training_data, training_labels, learning_rate, max_training_size=None):
        minibatch_size = 20;
        k = 0;
        total = np.size(training_data,0);
        printing_interval = 1000;
        if (max_training_size is not None):
            total = max_training_size;
        if (total < printing_interval * 10):
            printing_interval = total / 100;
        
        while k < total:
            # Set data
            data = training_data[k:k+minibatch_size,:];
            # Set labels
            labels = np.zeros((minibatch_size,self.output_dim));
            for j in range(k,k+minibatch_size):
                # Rows are local so subtract i from j
                labels[j-k,training_labels[j]] = 1.0;
            # Run training
            dW_value, dV_value, hidden_value, Y_value, error_value = self.sgd(data, labels.astype(np.float32), learning_rate);
            
            if (k % printing_interval == 0):
                print("%d / %d" % (k, total));
            
            if (np.isnan(dW_value[0,0])):
                print("NaN!");
            
            k += minibatch_size;
        
    def test(self, test_data, test_labels):
        """
        Run test data through model. Output percentage of correctly predicted
        test instances.
        """
        correct = 0.0;
        prediction_histogram = {k: 0 for k in range(10)};
        
        # Predict
        prediction = self.predict(test_data);
        
        # Compute statistics
        for j in range(np.size(prediction)):
            prediction_histogram[prediction[j]] += 1;
            if (prediction[j] == test_labels[j]):
                correct += 1.0;
                
        return (correct / np.size(test_data,0), prediction_histogram, (self.W, self.V));
        
if (__name__ == '__main__'):
    datasplit = 60000;
    max_training_size = 60000;
    repetitions = 2;
    
    mnist = fetch_mldata('MNIST original', data_home='../data/nn_mnist')
    
    words_dim = np.size(mnist.data,1);
    hidden_dim = words_dim * 1;
    output_classes = 10;
    
    training_data = mnist.data[:datasplit,:];
    training_labels = mnist.target[:datasplit];
    test_data = mnist.data[datasplit:,:];
    test_labels = mnist.target[datasplit:];
    
    # Shuffle
    shuffle_order = np.array(range(datasplit));
    np.random.shuffle(shuffle_order);
    training_data = training_data[shuffle_order,:];
    training_labels = training_labels[shuffle_order];
    
    # Create train and test labels histogram
    test_histogram = {k: 0 for k in range(10)};
    for i in range(np.size(test_labels)):
        test_histogram[test_labels[i]] += 1;
    train_histogram = {k: 0 for k in range(10)};
    for i in range(np.size(training_labels[:max_training_size])):
        train_histogram[training_labels[i]] += 1;
    
    nn = NeuralNetwork(words_dim, hidden_dim, output_classes);
    
    start = time.clock();
    
    # Train
    for r in range(repetitions):
        nn.train(training_data, training_labels, 0.01, max_training_size);
    
    # Test
    score, prediction_histogram, weights = nn.test(test_data, test_labels)
    W, V = weights
    
    print
    
    # Print statistics
    duration = time.clock() - start;
    print("Duration: %d seconds" % duration);
    print("Summed weight difference:");
    print("W:" + str(np.sum(W.get_value() - nn.init_W)));
    print("V:" + str(np.sum(V.get_value() - nn.init_V)));
    print("Score: %.2f percent" % (score*100));
    print("Training histogram:     %s" % (", ".join(map(lambda (k,v): "%d:%d" % (k,v), train_histogram.items()))));
    print("Prediction histogram:   %s" % (", ".join(map(lambda (k,v): "%d:%d" % (k,v), prediction_histogram.items()))));
    print("Real testset histogram: %s" % (", ".join(map(lambda (k,v): "%d:%d" % (k,v), test_histogram.items()))));