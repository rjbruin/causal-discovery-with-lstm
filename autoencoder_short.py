'''
Created on 4 nov. 2016

@author: Robert-Jan
'''

from tools.model import  set_up_statistics;
from models.SequencesByPrefix import SequencesByPrefix;

import theano;
import theano.tensor as T;
import lasagne;

import numpy as np;

def lstm_predict_single(previous_output, previous_hidden, hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, hWY, hbY):
    forget_gate = T.nnet.sigmoid(previous_hidden.dot(hWf) + previous_output.dot(XWf));
    input_gate = T.nnet.sigmoid(previous_hidden.dot(hWi) + previous_output.dot(XWi));
    candidate_cell = T.tanh(previous_hidden.dot(hWc) + previous_output.dot(XWc));
    cell = forget_gate * previous_hidden + input_gate * candidate_cell;
    output_gate = T.nnet.sigmoid(previous_hidden.dot(hWo) + previous_output.dot(XWo));
    hidden = output_gate * cell;
    
    Y_output = T.nnet.softmax(hidden.dot(hWY) + hbY);
    
    return Y_output, hidden;

def lstm_predict_single_no_output(current_X, previous_hidden, hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo):
    forget_gate = T.nnet.sigmoid(previous_hidden.dot(hWf) + current_X.dot(XWf));
    input_gate = T.nnet.sigmoid(previous_hidden.dot(hWi) + current_X.dot(XWi));
    candidate_cell = T.tanh(previous_hidden.dot(hWc) + current_X.dot(XWc));
    cell = forget_gate * previous_hidden + input_gate * candidate_cell;
    output_gate = T.nnet.sigmoid(previous_hidden.dot(hWo) + current_X.dot(XWo));
    hidden = output_gate * cell;
    
    return hidden;

def precision_from_digits_correct(data, digits_correct):
    correct = 0;
    d_correct = 0;
    digits_total = 0;
    for i in range(data.shape[0]):
        seqCorrect = True;
        for j in range(data[i].shape[0]):
            if (digits_correct[i,j] == 0.):
                seqCorrect = False;
            else:
                d_correct += 1;
            if (np.argmax(data[i,j]) == EOS_symbol_index):
                # Stop if we encounter EOS
                if (seqCorrect):
                    correct += 1;
                break;
        if (seqCorrect):
            correct += 1;
        digits_total += j+1;
    return correct / float(i+1), d_correct / float(digits_total);

def get_batch(storage, debug=False):    
    # Reseed the random generator to prevent generating identical batches
    np.random.seed();
    
    batch = [];
    while (len(batch) < minibatch_size):
        interventionLocation = np.random.randint(0, n_max_digits-1);
        
        subbatch = [];
        while (len(subbatch) < subbatch_size):
            branch = storage.get_random_by_length(interventionLocation, getStructure=True);
            
            randomPrefix = np.random.randint(0,len(branch.fullExpressions));
            subbatch.append(branch.fullExpressions[randomPrefix]);
        
        # Add subbatch to batch
        batch.extend(subbatch);
    
    data = np.zeros((len(batch), n_max_digits, data_dim), dtype='float32');
    for i, expression in enumerate(batch):
        for j, literal in enumerate(expression):
            # Encode symbol into data
            data[i,j,fromSymbols[literal]] = 1.0;
        
        if (j+1 < n_max_digits):
            # Add EOS if it fits in the encoding
            data[i,j+1,EOS_symbol_index] = 1.0;
    
    return data, batch;

if __name__ == '__main__':
    theano.config.floatX = 'float32';
    np.set_printoptions(precision=3, threshold=10000000);
    
    # MODEL PARAMETERS
    symbols = [str(i) for i in range(10)] + ['+','-','*','/','(',')','=','_','G'];
    fromSymbols = {symbols[k]: k for k in range(len(symbols))};
    data_dim = len(symbols);
    hidden_dim = 128;
    minibatch_size = 64;
    subbatch_size = 64;
    n_max_digits = 17;
    learning_rate = 0.005;
    repetitions = 300;
    
    # DATASET PARAMETERS
    train_source = './data/subsystems_shallow_simple_topcause/train.txt';
    test_source = './data/subsystems_shallow_simple_topcause/test.txt';
    EOS_symbol_index = data_dim - 2;
    GO_symbol_index = data_dim - 1;
    
    # READ IN ALL EXPRESSIONS (TRAIN AND TEST) TO TREE-STRUCTURE STORAGE
    
    # Read training part of dataset
    expressionsByPrefix = SequencesByPrefix();
    f = open(train_source,'r');
    line = f.readline().strip();
    train_n = 0;
    while (line != ""):
        result = line.split(";");
        expression, expression_prime = result;
        expression_prime = "";
        
        expressionsByPrefix.add(expression, expression_prime);
        line = f.readline().strip();
        train_n += 1;
    f.close();
    
    # Read testing part of dataset
    testExpressionsByPrefix = SequencesByPrefix();
    f = open(test_source,'r');
    line = f.readline().strip();
    test_n = 0;
    while (line != ""):
        result = line.split(";");
        expression, expression_prime = result;
        expression_prime = "";
        
        testExpressionsByPrefix.add(expression, expression_prime);
        line = f.readline().strip();
        test_n += 1;
    f.close();
    
    
    # CONSTRUCT THEANO COMPUTATIONAL GRAPH
    
    # Initialize shared variables (AKA weights)
    # We use two LSTM cells: one for encoding and one for decoding (prefixed by 'D')
    
    # Encoding variables
    vars = {};
    vars['hWf'] = theano.shared(np.random.uniform(-np.sqrt(1.0/hidden_dim),np.sqrt(1.0/hidden_dim),(hidden_dim,hidden_dim)).astype('float32'), 'hWf');
    vars['XWf'] = theano.shared(np.random.uniform(-np.sqrt(1.0/data_dim),np.sqrt(1.0/data_dim),(data_dim,hidden_dim)).astype('float32'), 'XWf')
    vars['hWi'] = theano.shared(np.random.uniform(-np.sqrt(1.0/hidden_dim),np.sqrt(1.0/hidden_dim),(hidden_dim,hidden_dim)).astype('float32'), 'hWi')
    vars['XWi'] = theano.shared(np.random.uniform(-np.sqrt(1.0/data_dim),np.sqrt(1.0/data_dim),(data_dim,hidden_dim)).astype('float32'), 'XWi')
    vars['hWc'] = theano.shared(np.random.uniform(-np.sqrt(1.0/hidden_dim),np.sqrt(1.0/hidden_dim),(hidden_dim,hidden_dim)).astype('float32'), 'hWc')
    vars['XWc'] = theano.shared(np.random.uniform(-np.sqrt(1.0/data_dim),np.sqrt(1.0/data_dim),(data_dim,hidden_dim)).astype('float32'), 'XWc')
    vars['hWo'] = theano.shared(np.random.uniform(-np.sqrt(1.0/hidden_dim),np.sqrt(1.0/hidden_dim),(hidden_dim,hidden_dim)).astype('float32'), 'hWo')
    vars['XWo'] = theano.shared(np.random.uniform(-np.sqrt(1.0/data_dim),np.sqrt(1.0/data_dim),(data_dim,hidden_dim)).astype('float32'), 'XWo')
    encode_params = [vars['hWf'], vars['XWf'], vars['hWi'], vars['XWi'], vars['hWc'], vars['XWc'], vars['hWo'], vars['XWo']];
    
    # Decoding variables
    vars['DhWf'] = theano.shared(np.random.uniform(-np.sqrt(1.0/hidden_dim),np.sqrt(1.0/hidden_dim),(hidden_dim,hidden_dim)).astype('float32'), 'DhWf')
    vars['DXWf'] = theano.shared(np.random.uniform(-np.sqrt(1.0/data_dim),np.sqrt(1.0/data_dim),(data_dim,hidden_dim)).astype('float32'), 'DXWf')
    vars['DhWi'] = theano.shared(np.random.uniform(-np.sqrt(1.0/hidden_dim),np.sqrt(1.0/hidden_dim),(hidden_dim,hidden_dim)).astype('float32'), 'DhWi')
    vars['DXWi'] = theano.shared(np.random.uniform(-np.sqrt(1.0/data_dim),np.sqrt(1.0/data_dim),(data_dim,hidden_dim)).astype('float32'), 'DXWi')
    vars['DhWc'] = theano.shared(np.random.uniform(-np.sqrt(1.0/hidden_dim),np.sqrt(1.0/hidden_dim),(hidden_dim,hidden_dim)).astype('float32'), 'DhWc')
    vars['DXWc'] = theano.shared(np.random.uniform(-np.sqrt(1.0/data_dim),np.sqrt(1.0/data_dim),(data_dim,hidden_dim)).astype('float32'), 'DXWc')
    vars['DhWo'] = theano.shared(np.random.uniform(-np.sqrt(1.0/hidden_dim),np.sqrt(1.0/hidden_dim),(hidden_dim,hidden_dim)).astype('float32'), 'DhWo')
    vars['DXWo'] = theano.shared(np.random.uniform(-np.sqrt(1.0/data_dim),np.sqrt(1.0/data_dim),(data_dim,hidden_dim)).astype('float32'), 'DXWo')
    vars['DhWY'] = theano.shared(np.random.uniform(-np.sqrt(1.0/hidden_dim),np.sqrt(1.0/hidden_dim),(hidden_dim,data_dim)).astype('float32'), 'DhWY')
    vars['DhbY'] = theano.shared(np.random.uniform(-np.sqrt(1.0/data_dim),np.sqrt(1.0/data_dim),(data_dim)).astype('float32'), 'DhbY')
    decode_params = [vars['DhWf'], vars['DXWf'], vars['DhWi'], vars['DXWi'], vars['DhWc'], vars['DXWc'], vars['DhWo'], vars['DXWo'], vars['DhWY'], vars['DhbY']];
    
    # Initialize input to the graph
    # Since for the autoencoder the label is the same as the input we only need
    # one input variable 'data'
    data = T.ftensor3('data');
    
    # In order to be able to use scan for this problem we need to swap the axis
    # of any sequence input to the algorithm.
    # For example: if sequence length = 5, nr of samples in batch = 10 and data
    # dimenstionality = 20, the size of the original Theano input would be
    # (10, 5, 20). We need to swap the first two axes to obtain a size of
    # (5, 10, 20).
    
    # Encoding phase: input propagates through LSTM cells to update a hidden state
    # `hiddens_encode` contains the hidden state after each new input in the sequence
    # Weights are passed using the `encode_params` list constructed above
    # initial inputs:
    #     1) Empty hidden layer for each sample
    hiddens_encode, _ = theano.scan(fn=lstm_predict_single_no_output,
                                    sequences=data,
                                    # Input a zero hidden layer
                                    outputs_info=({'initial': T.zeros((minibatch_size, hidden_dim), dtype=theano.config.floatX), 'taps': [-1]}),
                                    non_sequences=encode_params,
                                    name='encode_scan');
    
    # Decoding phase: LSTM cells use the hidden layer to predict a fixed number of 
    # symbols.
    # `output` contains the predicted sequence, with a symbol for each sequence index.
    # Weights are passed using the `decode_params` list constructed above
    # initial inputs:
    #     1) An input representation of the `GO` symbol to start of the prediction
    #     2) The last hidden layer of the encoding phase
    first_decode_x = T.zeros((minibatch_size, data_dim), dtype=theano.config.floatX);
    T.set_subtensor(first_decode_x[:,GO_symbol_index],T.ones((minibatch_size), dtype=theano.config.floatX));
    (output, _), _ = theano.scan(fn=lstm_predict_single,
                                 # Input a zero hidden layer
                                 outputs_info=({'initial': first_decode_x, 'taps': [-1]}, {'initial': hiddens_encode[-1], 'taps': [-1]}),
                                 non_sequences=decode_params,
                                 name='decode_scan',
                                 n_steps=n_max_digits);
    
    # Error computation: we use categorical crossentropy. We make sure to not
    # include the empty predictions of shorter samples in the averaging of the
    # prediction error. Detailed comments should make this clear.
    
    # First, we sum the categorical categorical crossentropy over the last axis.
    cat_cross = -T.sum(data * T.log(output), axis=output.ndim-1);
    # Second, we average the resulting loss over the first axis, which is the axis
    # corresponding to sequence index. (See the explanation on axis swapping just
    # below the definition of `data`.)
    mean_cross_per_sample = T.sum(cat_cross, axis=0) / float(n_max_digits);
    # Finally we take the average over the corrected mean of loss per sample
    error = T.mean(mean_cross_per_sample);
    
    # Compute predictions per sample and per digit
    predictions = T.argmax(output, axis=2);
    # We can directly see which sequence indices (`digits`) are correct by 
    # comparing with `data`
    digits_correct = T.eq(predictions, T.argmax(data, axis=2));
    
    # Compute updates for SGD using Nesterov momentum
    var_list = encode_params + decode_params;
    updates = lasagne.updates.nesterov_momentum(error,var_list,learning_rate=learning_rate).items();
    
    # Defining functions
    # We output some extra outputs for debugging purposes
    _sgd = theano.function([data], [error],
                                updates=updates);
    _predict = theano.function([data], [output, predictions, digits_correct, error]);
    _encode = theano.function([data], [hiddens_encode[-1]]);    
    # _decode is missing because I couldn't get it working at this moment
    
    
    
    # TRAINING PHASE
    
    repetition_size = 1000;
    for r in range(repetitions):
        total_error = 0.0;
        # Print repetition progress and save to raw results file
        print("Batch %d (repetition %d of %d, dataset 1 of 1) (samples processed after batch: %d)" % \
                (r+1,r+1,repetitions,(r+1)*repetition_size));
        
        # Train model per minibatch
        k = 0;
        printedProgress = -1;
        while k < repetition_size:
            # Call get_batch to get a random batch from the tree-structured storage
            # Provides the storage to use as we have separate ones for training
            # and testing
            data, target_expressions = get_batch(expressionsByPrefix);
            
            # Call SGD
            # We swap the data (see explanation below definition of `data` in 
            # model init code
            swapped_data = np.swapaxes(data, 0, 1);
            error = _sgd(swapped_data);
            total_error += error[0];
            
            # Print batch progress - does something slightly complicated to 
            # print after each printing threshold (just ignore this, it works ;) ) 
            if ((k+minibatch_size) % (minibatch_size*4) < minibatch_size and \
                (k+minibatch_size) / (minibatch_size*4) > printedProgress):
                printedProgress = (k+minibatch_size) / (minibatch_size*4);
                print("# %d / %d (error = %.2f)" % (k+minibatch_size, repetition_size, total_error));
            
            k += minibatch_size;
        
        # Report on error
        print("Total error: %.2f" % total_error);
        
        # We perform intermediate testing after each repetition
        print("Testing...");
        intermediate_testing_n = 10000;
        printing_interval = 1000;
        
        totalError = 0.0;
        k = 0;
        precisions = [];
        digit_precisions = [];
        printedProgress = -1;
        while k < intermediate_testing_n:
            # Call get_batch with test storage
            test_data, test_expressions = get_batch(testExpressionsByPrefix);
            
            # Swap data before prediction (see explanation above)
            swapped_test_data = np.swapaxes(test_data, 0, 1);
            output, predictions, digits_correct, error = _predict(swapped_test_data);
            
            # Swap back outputs
            predictions = np.swapaxes(predictions, 0, 1);
            output = np.swapaxes(output, 0, 1);
            digits_correct = np.swapaxes(digits_correct, 0, 1);
            
            # Compute precision using digit_precision computed by Theano code
            precision, digit_precision = precision_from_digits_correct(test_data, digits_correct);
            
            # Store results
            precisions.append(precision);
            digit_precisions.append(digit_precision);
            totalError += error;
    
            k += minibatch_size;
            
            if ((k+minibatch_size) % printing_interval < minibatch_size and \
                (k+minibatch_size) / printing_interval > printedProgress):
                printedProgress = (k+minibatch_size) / printing_interval;
                print("# %d / %d" % (k, intermediate_testing_n));
        
        # Print overall testing results for this iteration
        print("Total testing error: %.2f" % totalError);
        print("Score: %.2f percent" % (np.mean(precisions)));
        print("Digit-based score: %.2f percent\n" % (np.mean(digit_precisions)));
    
    print("Training finished!");
    
    