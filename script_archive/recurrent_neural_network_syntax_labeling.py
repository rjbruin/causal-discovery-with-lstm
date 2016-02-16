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

#theano.config.mode = 'FAST_COMPILE'

class RecurrentNeuralNetwork(object):
    '''
    classdocs
    '''


    def __init__(self, training_dim, hidden_dim, output_dim, nr_embeddings, embeddings):
        '''
        Constructor
        '''
        # Temp
        self.embeddings = embeddings;
        
        self.data_dim = training_dim;
        self.hidden_dim = hidden_dim;
        self.output_dim = output_dim;
        self.embedding_dim = nr_embeddings+1;
        
        # X weights to hidden
        self.init_XWh = np.random.uniform(-np.sqrt(1.0/self.data_dim),np.sqrt(1.0/self.data_dim),(self.data_dim,self.hidden_dim));
        # hidden weights to hidden
        self.init_hWh = np.random.uniform(-np.sqrt(1.0/self.hidden_dim),np.sqrt(1.0/self.hidden_dim),(self.hidden_dim,self.hidden_dim));
        # initial hidden layer
        self.init_h0 = np.random.uniform(-np.sqrt(1.0/self.hidden_dim),np.sqrt(1.0/self.hidden_dim),self.hidden_dim);
        # hidden weights to output
        self.init_hWo = np.random.uniform(-np.sqrt(1.0/self.hidden_dim),np.sqrt(1.0/self.hidden_dim),(self.hidden_dim,self.output_dim));
        
        # Set up shared variables
        self.XWh = theano.shared(name='XWh', value=self.init_XWh);
        self.hWh = theano.shared(name='hWh', value=self.init_hWh);
        self.hWo = theano.shared(name='hWo', value=self.init_hWo);
        self.h0 = theano.shared(name='h0', value=self.init_h0);
        
        # Forward pass
        # X is 2-dimensional: 1) index in sentence, 2) dimensionality of data 
        # X_idx = T.ivector('X_idx')
        # We are not using a content window for this validation dataset. If it doesn't work without it we'll reconsider
        #X = self.emb[X_idx,:];#.reshape((X_idx.shape[0],self.embedding_dim));
        X = T.dmatrix('X');
        # targets is 2-dimensional: 1) index in sentence, 3) nr of classification classes
        targets = T.fmatrix('targets');
        
        def rnn_recurrence(current_X, previous_hidden):
            hidden = T.nnet.sigmoid(previous_hidden.dot(self.hWh) + current_X.dot(self.XWh));
            Ys = T.nnet.softmax(hidden.dot(self.hWo));
            return hidden, Ys;
        
        [_, Y], _ = theano.scan(fn=rnn_recurrence,
                                 sequences=X,
                                 # Replicate hidden layer input along horizontal axis
                                 outputs_info=(self.h0,None))
        
        Y = Y.reshape((X.shape[0],self.output_dim))
        predictions = T.argmax(Y, axis=1);
        error = T.mean(T.nnet.categorical_crossentropy(Y, targets));
        
        # Backward pass: gradients
        #dXWh, dhWh, dhWo, dh0, demb = T.grad(error, [self.XWh, self.hWh, self.hWo, self.h0, self.emb]);
        dXWh, dhWh, dhWo, dh0 = T.grad(error, [self.XWh, self.hWh, self.hWo, self.h0]);
        
        # Functions
        self.predict = theano.function([X], predictions);
        
        # Stochastic Gradient Descent
        learning_rate = T.dscalar('learning_rate');
        self.sgd = theano.function([X, targets, learning_rate], [dXWh, dhWh, dhWo, dh0], 
                                   updates=[(self.XWh,self.XWh - learning_rate * dXWh),
                                            (self.hWh,self.hWh - learning_rate * dhWh),
                                            (self.hWo,self.hWo - learning_rate * dhWo),
                                            (self.h0,self.h0 - learning_rate * dh0)],
                                            #(self.emb,self.emb - learning_rate * demb)],
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
            # Get embeddings for each word in sentence
            data_embedding = np.zeros((len(data),self.data_dim), dtype=np.float64);
            for data_index, e_index in enumerate(data):
                data_embedding[data_index,:] = self.embeddings[e_index,:];
            # Set label
            label = np.zeros((len(data),self.output_dim));
            label_indices = training_labels[k];
            for y_index in range(label.shape[0]):
                label[y_index,label_indices[y_index]] = 1.0;
            # Run training
            dXWh_value, dhWh_value, dhWo_value, dh0_value = self.sgd(data_embedding, label.astype(np.float32), learning_rate);
            
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
        prediction_size = 0;
        prediction_histogram = {k: 0 for k in range(self.output_dim)};    
        # Predict
        for j in range(len(test_data)):
            data = test_data[j];
            data_embedding = np.zeros((len(data),self.data_dim), dtype=np.float64);
            for data_index, e_index in enumerate(data):
                data_embedding[data_index,:] = self.embeddings[e_index,:];
                
            prediction = self.predict(data_embedding);
            # For each index in sentence
            for s in range(np.size(prediction,0)):
                if (prediction[s] == test_labels[j][s]):
                    correct += 1.0;
                prediction_histogram[test_labels[j][s]] += 1;
                prediction_size += 1;
                
        return (correct / float(prediction_size), prediction_histogram);
        
if (__name__ == '__main__'):
    max_training_size = None;
    repetitions = 10;
        
    train, test, dicts = cPickle.load(open("../data/rnn_syntax_labeling/atis.pkl"));
    
    train, _, train_labels = train;
    test, _, test_labels = test;
    
    # Generate labels
    max_labels_index = max(dicts['labels2idx'].iteritems(),key=operator.itemgetter(1))[1];
    max_words_index = max(dicts['words2idx'].iteritems(),key=operator.itemgetter(1))[1];
    
    # Define dimensions
    words_dim = 100;
    hidden_dim = words_dim*2;
    output_classes = max_labels_index+1;
    
    # Use dummy embeddings for now
    emb = np.random.uniform(-0.2, 0.2, (max_words_index+1, words_dim));
     
    nn = RecurrentNeuralNetwork(words_dim, hidden_dim, output_classes, max_words_index, emb);

    start = time.clock();
     
    # Train
    for r in range(repetitions):
        nn.train(train, train_labels, 0.1, max_training_size);
     
    # Test
    score, prediction_histogram = nn.test(test, test_labels)
     
    print
     
    # Print statistics
    duration = time.clock() - start;
    print("Duration: %d seconds" % duration);
    print("Score: %.2f percent" % (score*100));
    print("Prediction histogram: %s" % (str(prediction_histogram)))