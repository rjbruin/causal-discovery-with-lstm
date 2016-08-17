'''
Created on 17 aug. 2016

@author: Robert-Jan
'''
import sys;

import theano;
import theano.tensor as T;
from theano.ifelse import ifelse;
import numpy as np;

from profiler import profiler;

def rnn_predict(current_X, previous_hidden, hWo, hWh, XWh):
    if (clipping):
        zero = T.constant(0.0, dtype='float64');
        summation = T.sum(current_X);
        check = T.eq(summation,zero);
        hidden = ifelse(check,T.nnet.sigmoid(previous_hidden.dot(hWh) + current_X.dot(XWh)),previous_hidden);
    else:
        hidden = T.nnet.sigmoid(previous_hidden.dot(hWh) + current_X.dot(XWh));
    Ys = T.nnet.softmax(hidden.dot(hWo));
    return Ys, hidden;

def crossentropy_2d(coding_dist, true_dist, accumulated_score):
    return accumulated_score + T.mean(T.nnet.categorical_crossentropy(coding_dist, true_dist));

if __name__ == '__main__':
    repetitions = 1000000;
    minibatch_size = 10;
    learning_rate = 0.05;
    input_length = 5;
    output_length = 5;
    data_dim = 10;
    hidden_dim = 16;
    output_dim = 10;
    clipping = sys.argv[1] == 'True';
    
    print("Initializing model...");
    
    # X is 3-dimensional: 1) index in sentence, 2) datapoint, 3) dimensionality of data
    X = T.dtensor3('X');
    # targets is 3-dimensional: 1) index in sentence, 2) datapoint, 3) encodings
    label = T.dtensor3('label');
    lr = T.dscalar('learning_rate');
    
    # Define variables
    XWh = theano.shared(name='XWh', value=np.random.uniform(-np.sqrt(1.0/data_dim),np.sqrt(1.0/data_dim),(data_dim,hidden_dim)));
    hWh = theano.shared(name='XWh', value=np.random.uniform(-np.sqrt(1.0/hidden_dim),np.sqrt(1.0/hidden_dim),(hidden_dim,hidden_dim)));
    hWo = theano.shared(name='XWh', value=np.random.uniform(-np.sqrt(1.0/hidden_dim),np.sqrt(1.0/hidden_dim),(hidden_dim,output_dim)));
    
    # Forward pass: compile encoding phase
    init_values = (None, {'initial': np.zeros((minibatch_size,hidden_dim))});
    [Y_1, hidden], _ = theano.scan(fn=rnn_predict,
                            sequences=({'input': X}),
                            # Input a zero hidden layer
                            outputs_info=init_values,
                            non_sequences=(hWo, hWh, XWh));
    
    # Forward pass: compile decoding phase
    init_values = ({'initial': Y_1[-1], 'taps': [-1]},
                   {'initial': hidden[-1], 'taps': [-1]});
#     init_values = ({'initial': Y_1[-1]},
#                    {'initial': hidden[-1]});
    [Ys, _], _ = theano.scan(fn=rnn_predict,
                             # Inputs the last hidden layer and the last predicted symbol
                             outputs_info=init_values,
                             non_sequences=(hWo, hWh, XWh),
                             n_steps=output_length-1)
    
    
    # Forward pass: determine prediction
    right_hand = T.join(0,Y_1[-1].reshape((1,minibatch_size,output_dim)),Ys);
    prediction = T.argmax(right_hand, axis=2);
    
    # Determine error
    padded_label = T.join(0, label, T.zeros((output_length - label.shape[0],minibatch_size,output_dim)));
    accumulator = theano.shared(np.float64(0.), name='accumulatedError');
    summed_error, _ = theano.scan(fn=crossentropy_2d,
                                  sequences=(right_hand,padded_label),
                                  outputs_info={'initial': accumulator, 'taps': [-1]})
    error = summed_error[-1];
    
    # Backward pass for all variables   
    derivatives = T.grad(error, [hWo, hWh, XWh]);
    
    # Computing stochastic gradient descent updates
    updates = [(var,var-lr*der) for (var,der) in zip([hWo, hWh, XWh],derivatives)];
    
    # Compile functions
    predict = theano.function([X], [prediction,
                                    right_hand]);
    sgd = theano.function([X, label, lr], [], 
                          updates=updates,
                          allow_input_downcast=True);
    
    print("Creating data...");
    
    # Create data
    n_data_samples = 10000;
    data = np.zeros((n_data_samples, input_length, data_dim));
    labels = np.zeros((n_data_samples, input_length));
    targets = np.zeros((n_data_samples, input_length, data_dim));
    for i in range(len(data)):
        onehot_indices = np.random.random_integers(0,data_dim-1,(input_length));
        input_sample_length = np.random.randint(1,input_length);
        data[i,range(input_sample_length),onehot_indices[:input_sample_length]] = 1.0;
        targets[i,range(input_sample_length),onehot_indices[:input_sample_length]] = 1.0;
        labels = np.argmax(targets,2);
    
    print("Training model...");
    
    # Train model
    profiler.start('training');
    for i in range(repetitions):
        sample_inds = np.random.randint(0,len(data),(minibatch_size));
        train_data = np.swapaxes(data[sample_inds],0,1);
        train_targets = np.swapaxes(targets[sample_inds],0,1);
        sgd(train_data, train_targets, learning_rate);
        
        if (i > 0 and i % (repetitions/10) == 0):
            print("Iteration %d of %d" % (i, repetitions));
    profiler.stop('training');
    
    print("Testing model...");
    
    # Test model
    score = 0.0;
    digit_score = 0.0;
    for i in range(0,len(data),minibatch_size):
        prediction, right_hand = predict(np.swapaxes(data[i:i+minibatch_size],0,1));
        prediction = np.swapaxes(prediction,0,1);
        for j in range(minibatch_size):
            if (np.array_equal(prediction[j],labels[i+j])):
                score += 1.0;
            for k in range(output_length):
                if (prediction[j,k] == labels[i+j,k]):
                    digit_score += 1.0;
    
    print("Model %s clipping" % ("with" if clipping else "without"));
    print("After %d training repetitions:" % repetitions);
    print("Scores: %d of %d right = %.4f percent" % (score, len(data)*minibatch_size, (score / (len(data)*minibatch_size))*100))
    print("Digit scores: %d of %d right = %.4f percent" % (digit_score, len(data)*minibatch_size*output_length, (digit_score / (len(data)*minibatch_size*output_length))*100))
    
    profiler.profile();