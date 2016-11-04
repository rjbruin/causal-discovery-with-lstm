'''
Created on 4 nov. 2016

@author: Robert-Jan
'''

import numpy as np;
import theano;
import theano.tensor as T;
import lasagne;

class Autoencoder(object):
    '''
    classdocs
    '''


    def __init__(self, data_dim, hidden_dim, minibatch_size, n_max_digits, 
                 learning_rate, GO_symbol_index, EOS_symbol_index, only_cause_expression):
        '''
        Constructor
        '''
        self.data_dim = data_dim;
        self.hidden_dim = hidden_dim;
        self.minibatch_size = minibatch_size;
        self.EOS_symbol_index = EOS_symbol_index;
        self.n_max_digits = n_max_digits;
        self.only_cause_expression = only_cause_expression;
        
        self.vars = {};
        self.vars['hWf'] = theano.shared(np.random.uniform(-np.sqrt(1.0/hidden_dim),np.sqrt(1.0/hidden_dim),(hidden_dim,hidden_dim)).astype('float32'), 'hWf');
        self.vars['XWf'] = theano.shared(np.random.uniform(-np.sqrt(1.0/data_dim),np.sqrt(1.0/data_dim),(data_dim,hidden_dim)).astype('float32'), 'XWf')
        self.vars['hWi'] = theano.shared(np.random.uniform(-np.sqrt(1.0/hidden_dim),np.sqrt(1.0/hidden_dim),(hidden_dim,hidden_dim)).astype('float32'), 'hWi')
        self.vars['XWi'] = theano.shared(np.random.uniform(-np.sqrt(1.0/data_dim),np.sqrt(1.0/data_dim),(data_dim,hidden_dim)).astype('float32'), 'XWi')
        self.vars['hWc'] = theano.shared(np.random.uniform(-np.sqrt(1.0/hidden_dim),np.sqrt(1.0/hidden_dim),(hidden_dim,hidden_dim)).astype('float32'), 'hWc')
        self.vars['XWc'] = theano.shared(np.random.uniform(-np.sqrt(1.0/data_dim),np.sqrt(1.0/data_dim),(data_dim,hidden_dim)).astype('float32'), 'XWc')
        self.vars['hWo'] = theano.shared(np.random.uniform(-np.sqrt(1.0/hidden_dim),np.sqrt(1.0/hidden_dim),(hidden_dim,hidden_dim)).astype('float32'), 'hWo')
        self.vars['XWo'] = theano.shared(np.random.uniform(-np.sqrt(1.0/data_dim),np.sqrt(1.0/data_dim),(data_dim,hidden_dim)).astype('float32'), 'XWo')
        encode_params = [self.vars['hWf'], self.vars['XWf'], self.vars['hWi'], self.vars['XWi'], self.vars['hWc'], self.vars['XWc'], self.vars['hWo'], self.vars['XWo']];
        
        self.vars['DhWf'] = theano.shared(np.random.uniform(-np.sqrt(1.0/hidden_dim),np.sqrt(1.0/hidden_dim),(hidden_dim,hidden_dim)).astype('float32'), 'DhWf')
        self.vars['DXWf'] = theano.shared(np.random.uniform(-np.sqrt(1.0/data_dim),np.sqrt(1.0/data_dim),(data_dim,hidden_dim)).astype('float32'), 'DXWf')
        self.vars['DhWi'] = theano.shared(np.random.uniform(-np.sqrt(1.0/hidden_dim),np.sqrt(1.0/hidden_dim),(hidden_dim,hidden_dim)).astype('float32'), 'DhWi')
        self.vars['DXWi'] = theano.shared(np.random.uniform(-np.sqrt(1.0/data_dim),np.sqrt(1.0/data_dim),(data_dim,hidden_dim)).astype('float32'), 'DXWi')
        self.vars['DhWc'] = theano.shared(np.random.uniform(-np.sqrt(1.0/hidden_dim),np.sqrt(1.0/hidden_dim),(hidden_dim,hidden_dim)).astype('float32'), 'DhWc')
        self.vars['DXWc'] = theano.shared(np.random.uniform(-np.sqrt(1.0/data_dim),np.sqrt(1.0/data_dim),(data_dim,hidden_dim)).astype('float32'), 'DXWc')
        self.vars['DhWo'] = theano.shared(np.random.uniform(-np.sqrt(1.0/hidden_dim),np.sqrt(1.0/hidden_dim),(hidden_dim,hidden_dim)).astype('float32'), 'DhWo')
        self.vars['DXWo'] = theano.shared(np.random.uniform(-np.sqrt(1.0/data_dim),np.sqrt(1.0/data_dim),(data_dim,hidden_dim)).astype('float32'), 'DXWo')
        self.vars['DhWY'] = theano.shared(np.random.uniform(-np.sqrt(1.0/hidden_dim),np.sqrt(1.0/hidden_dim),(hidden_dim,data_dim)).astype('float32'), 'DhWY')
        self.vars['DhbY'] = theano.shared(np.random.uniform(-np.sqrt(1.0/data_dim),np.sqrt(1.0/data_dim),(data_dim)).astype('float32'), 'DhbY')
        decode_params = [self.vars['DhWf'], self.vars['DXWf'], self.vars['DhWi'], self.vars['DXWi'], self.vars['DhWc'], self.vars['DXWc'], self.vars['DhWo'], self.vars['DXWo'], self.vars['DhWY'], self.vars['DhbY']];
        
        data = T.ftensor3('data');
        
        hiddens_encode, _ = theano.scan(fn=self.lstm_predict_single_no_output,
                                        sequences=data,
                                        # Input a zero hidden layer
                                        outputs_info=({'initial': T.zeros((minibatch_size, hidden_dim), dtype=theano.config.floatX), 'taps': [-1]}),
                                        non_sequences=encode_params,
                                        name='encode_scan');
        
        first_decode_x = T.zeros((minibatch_size, data_dim), dtype=theano.config.floatX);
        T.set_subtensor(first_decode_x[:,GO_symbol_index],T.ones((minibatch_size), dtype=theano.config.floatX));
        (output, _), _ = theano.scan(fn=self.lstm_predict_single,
                                     # Input a zero hidden layer
                                     outputs_info=({'initial': first_decode_x, 'taps': [-1]}, {'initial': hiddens_encode[-1], 'taps': [-1]}),
                                     non_sequences=decode_params,
                                     name='decode_scan',
                                     n_steps=n_max_digits);
        
#         error = T.mean(T.nnet.categorical_crossentropy(output, data));
        
        cat_cross = -T.sum(data * T.log(output), axis=output.ndim-1);
        mean_cross_per_sample = T.sum(cat_cross, axis=0) / float(self.n_max_digits);
        error = T.mean(mean_cross_per_sample);
        
        predictions = T.argmax(output, axis=2);
        digits_correct = T.eq(predictions, T.argmax(data, axis=2));
        
        # SGD
        self.var_list = encode_params + decode_params;
        updates = lasagne.updates.nesterov_momentum(error,self.var_list,learning_rate=learning_rate).items();
        
        # Defining functions
        self._sgd = theano.function([data], [error, digits_correct],
                                    updates=updates);
        self._predict = theano.function([data], [output, predictions, digits_correct, error]);
        self._encode = theano.function([data], [hiddens_encode[-1]]);
        
        # Constructing graph for decoding method
        code = T.fmatrix('code');
        first_decode_x = T.zeros((data_dim), dtype=theano.config.floatX);
        T.set_subtensor(first_decode_x[GO_symbol_index],1.);
        (output, _), _ = theano.scan(fn=self.lstm_predict_single,
                                     # Input a zero hidden layer
                                     outputs_info=({'initial': first_decode_x, 'taps': [-1]}, {'initial': code, 'taps': [-1]}),
                                     non_sequences=decode_params,
                                     name='decode_code_scan',
                                     n_steps=n_max_digits);
        code_prediction = T.argmax(output, axis=1);
        self._decode = theano.function([code], [code_prediction]);
        
    def lstm_predict_single(self, previous_output, previous_hidden, hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, hWY, hbY):
        forget_gate = T.nnet.sigmoid(previous_hidden.dot(hWf) + previous_output.dot(XWf));
        input_gate = T.nnet.sigmoid(previous_hidden.dot(hWi) + previous_output.dot(XWi));
        candidate_cell = T.tanh(previous_hidden.dot(hWc) + previous_output.dot(XWc));
        cell = forget_gate * previous_hidden + input_gate * candidate_cell;
        output_gate = T.nnet.sigmoid(previous_hidden.dot(hWo) + previous_output.dot(XWo));
        hidden = output_gate * cell;
        
        Y_output = T.nnet.softmax(hidden.dot(hWY) + hbY);
        
        return Y_output, hidden;
    
    def lstm_predict_single_no_output(self, current_X, previous_hidden, hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo):
        forget_gate = T.nnet.sigmoid(previous_hidden.dot(hWf) + current_X.dot(XWf));
        input_gate = T.nnet.sigmoid(previous_hidden.dot(hWi) + current_X.dot(XWi));
        candidate_cell = T.tanh(previous_hidden.dot(hWc) + current_X.dot(XWc));
        cell = forget_gate * previous_hidden + input_gate * candidate_cell;
        output_gate = T.nnet.sigmoid(previous_hidden.dot(hWo) + current_X.dot(XWo));
        hidden = output_gate * cell;
        
        return hidden;
    
    def sgd(self, data):
        data = np.swapaxes(data, 0, 1);
#         error, _, updates = self._sgd(data);
#         print(str(updates));
        error, _ = self._sgd(data);
        
        return error;
    
    def predict(self, data):
        swapped_data = np.swapaxes(data, 0, 1);
        output, predictions, digits_correct, error = self._predict(swapped_data);
        output = np.swapaxes(output, 0, 1);
        digits_correct = np.swapaxes(digits_correct, 0, 1);
        
        if (self.only_cause_expression is False):
            predictions = [np.argmax(output[:,:,:self.data_dim/2], axis=2),
                           np.argmax(output[:,:,self.data_dim/2:], axis=2)];
            digits_correct_1 = np.equal(np.argmax(data[:,:,:self.data_dim/2], axis=2),predictions[0]);
            digits_correct_2 = np.equal(np.argmax(data[:,:,self.data_dim/2:], axis=2),predictions[1]);
            precision_1, digit_precision_1 = self.precision_from_digits_correct(data, digits_correct_1);
            precision_2, digit_precision_2 = self.precision_from_digits_correct(data, digits_correct_2);
            return predictions, (precision_1 + precision_2) / 2., (digit_precision_1 + digit_precision_2) / 2., error;
        else:
            precision, digit_precision = self.precision_from_digits_correct(data, digits_correct);
            return predictions, precision, digit_precision, error;
    
    def encode(self, data):
        data = np.swapaxes(data, 0, 1);
        code = self._encode(data);
        code = np.swapaxes(code, 0, 1);
        return code;
    
    def decode(self, code):
        code = np.swapaxes(code, 0, 1);
        data = self._decode(code);
        data = np.swapaxes(data, 0, 1);
        return data;
    
    def precision_from_digits_correct(self, data, digits_correct):
        correct = 0;
        d_correct = 0;
        digits_total = 0;
        for i in range(data.shape[0]):
            for j in range(data[i].shape[0]):
                if (digits_correct[i,j] == 0.):
                    # Stop if we encounter a wrong digit
                    break;
                if (np.argmax(data[i,j]) == self.EOS_symbol_index):
                    # Stop if we encounter EOS and count as correct
                    correct += 1;
                    break;
                d_correct += 1;
            digits_total += j+1;
        return correct / float(i+1), d_correct / float(digits_total);
    
    def getVars(self):
        return self.vars.items();

    def randomWalk(self, nrSamples=100):
        current = np.random.uniform(0.0, 1.0, (self.hidden_dim));
        predictions = [self.decode(current)];
        failLimit = 10000;
        while (len(predictions) < nrSamples):
            changeIndex = np.random.randint(0,self.hidden_dim);
            current[changeIndex] = np.random.random();
            newPrediction = self.decode(current);
            if (newPrediction != predictions[-1]):
                predictions.append(newPrediction);
            else:
                failLimit += 1;
                if (failLimit >= failLimit):
                    return [];
        
        return predictions; 
