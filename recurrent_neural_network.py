'''
Created on 12 feb. 2016

@author: Robert-Jan
'''

import numpy as np;
import theano;
import theano.tensor as T;
import time;
import cPickle
import operator;

theano.config.mode = 'FAST_COMPILE'

class NeuralNetwork(object):
    '''
    classdocs
    '''


    def __init__(self, training_dim, hidden_dim, output_dim, nr_embeddings):
        '''
        Constructor
        '''
        
        self.data_dim = training_dim;
        self.hidden_dim = hidden_dim;
        self.output_dim = output_dim;
        self.embedding_dim = nr_embeddings+1;
        
        # X weights to hidden
        self.init_XWh = np.random.uniform(-np.sqrt(1.0/self.data_dim),np.sqrt(1.0/self.data_dim),(self.data_dim,self.hidden_dim));
        # hidden weights to hidden
        self.init_hWh = np.random.uniform(-np.sqrt(1.0/self.hidden_dim),np.sqrt(1.0/self.hidden_dim),(self.hidden_dim,self.hidden_dim));
        # initial hidden layer
        self.init_h0 = np.zeros(self.hidden_dim);
        # hidden weights to output
        self.init_hWo = np.random.uniform(-np.sqrt(1.0/self.hidden_dim),np.sqrt(1.0/self.hidden_dim),(self.hidden_dim,self.output_dim));
        
        # Word embeddings
        self.init_emb = np.random.uniform(-0.2, 0.2, (self.embedding_dim, self.data_dim));
        
        # Set up shared variables
        self.XWh = theano.shared(name='XWh', value=self.init_XWh);
        self.hWh = theano.shared(name='hWh', value=self.init_hWh);
        self.hWo = theano.shared(name='hWo', value=self.init_hWo);
        self.h0 = theano.shared(name='h0', value=self.init_h0);
        self.emb = theano.shared(name='emb', value=self.init_emb);
        
        # Forward pass
        # X is 2-dimensional: 1) index in sentence, 2) dimensionality of data 
        X_idx = T.ivector('X_idx')
        # We are not using a content window for this validation dataset. If it doesn't work without it we'll reconsider
        X = self.emb[X_idx].reshape((X_idx.shape[0],self.embedding_dim));
        # targets is 2-dimensional: 1) index in sentence, 3) nr of classification classes
        targets = T.fmatrix('targets');
        
        def rnn_recurrence(previous_hidden, current_X):
            hidden = T.nnet.sigmoid(previous_hidden.dot(self.hWh) + current_X.dot(self.XWh));
            Ys = T.nnet.softmax((previous_hidden.dot(self.hWh) + current_X.dot(self.XWh)).dot(self.hWo));
            return hidden, Ys;
        
        [_, Y], _ = theano.scan(fn=rnn_recurrence,
                                 sequences=X,
                                 # Replicate hidden layer input along horizontal axis
                                 outputs_info=(self.h0,None))
        
        Y = Y.reshape((X.shape[0],self.output_dim))
        predictions = T.argmax(Y, axis=1);
        error = T.mean(T.nnet.categorical_crossentropy(Y, targets));
        
        # Backward pass: gradients
        dXWh, dhWh, dhWo, dh0, demb = T.grad(error, [self.XWh, self.hWh, self.hWo, self.h0, self.emb]);
        
        # Functions
        self.predict = theano.function([X], predictions);
        
        # Stochastic Gradient Descent
        learning_rate = T.dscalar('learning_rate');
        self.sgd = theano.function([X_idx, targets, learning_rate], [dXWh, dhWh, dhWo, dh0, demb], 
                                   updates=[(self.XWh,self.XWh - learning_rate * dXWh),
                                            (self.hWh,self.hWh - learning_rate * dhWh),
                                            (self.hWo,self.hWo - learning_rate * dhWo),
                                            (self.h0,self.h0 - learning_rate * dh0),
                                            (self.emb,self.emb - learning_rate * demb)],
                                   allow_input_downcast=False)
        
    def train(self, training_data, training_labels, learning_rate, max_training_size=None):
#         minibatch_size = 10;
        k = 0;
        total = len(training_data);
        printing_interval = 1000;
        if (max_training_size is not None):
            total = max_training_size;
        if (total < printing_interval * 10):
            printing_interval = total / 100;
        
        while k < total:
            # Set data
            data = training_data[k];
            # Set label
            label = np.zeros(self.output_dim);
            label_index = training_labels[k];
            label[label_index] = 1.0;
            # Run training
            dXWh_value, dhWh_value, dhWo_value, dh0_value, demb_value = self.sgd(data, label.astype(np.float32), learning_rate);
            
            if (k % printing_interval == 0):
                print("%d / %d" % (k, total));
            
            if (np.isnan(dXWh_value[0,0])):
                print("NaN!");
            
            k += 1; #minibatch_size;
        
    def test(self, test_data, test_labels):
        """
        Run test data through model. Output percentage of correctly predicted
        test instances.
        """
        correct = 0.0;        
        # Predict
        for j in range(np.size(test_data,0)):
            prediction = self.predict(test_data[j,:,:]);
            # For each index in sentence
            for s in range(np.size(prediction,0)):
                if (prediction[s] == test_labels[j,s]):
                    correct += 1.0;
                
        return (correct / np.size(test_data,0), prediction_histogram, (self.W, self.V));
        
if (__name__ == '__main__'):
    max_training_size = 1000;
        
    train, test, dicts = cPickle.load(open("atis.pkl"));
    
    train, _, train_labels = train;
    test, _, test_labels = test;
    
    # Generate labels
    max_labels_index = max(dicts['labels2idx'].iteritems(),key=operator.itemgetter(1))[1];
    max_words_index = max(dicts['words2idx'].iteritems(),key=operator.itemgetter(1))[1];
    #idx2labels = {v: k for (k,v) in dicts['labels2idx'].items()};
    #idx2words = {v: k for (k,v) in dicts['words2idx'].items()};
    
    # Define dimensions
    words_dim = 100;
    hidden_dim = words_dim*2;
    output_classes = max_labels_index+1;
     
    nn = NeuralNetwork(words_dim, hidden_dim, output_classes, max_words_index);

    start = time.clock();
     
    # Train
    nn.train(train, train_labels, 0.1, max_training_size);
     
    # Test
    score, prediction_histogram, weights = nn.test(test, test_labels)
    W, V = weights
     
    print
     
    # Print statistics
    duration = time.clock() - start;
    print("Duration: %d seconds" % duration);
    print("Summed weight difference:");
    print("W:" + str(np.sum(W.get_value() - nn.init_W)));
    print("V:" + str(np.sum(V.get_value() - nn.init_V)));
    print("Score: %.2f percent" % (score*100));