'''
Created on 22 feb. 2016

@author: Robert-Jan
'''

import theano;
import theano.tensor as T;
import numpy as np;
from models.RecurrentModel import RecurrentModel
from models.GeneratedExpressionDataset import ExpressionNode

class TheanoRecurrentNeuralNetwork(RecurrentModel):
    '''
    Recurrent neural network model with one hidden layer. Models single class 
    prediction based on regular recurrent model or LSTM model. 
    '''


    def __init__(self, data_dim, hidden_dim, output_dim, minibatch_size, 
                 lstm=True, weight_values={}, single_digit=False, 
                 EOS_symbol_index=None, GO_symbol_index=None, n_max_digits=24, 
                 decoder=False, verboseOutputter=None, finishExpressions=False):
        '''
        Initialize all Theano models.
        '''
        # Store settings in self since the initializing functions will need them
        self.single_digit = single_digit;
        self.minibatch_size = minibatch_size;
        self.n_max_digits = n_max_digits;
        self.lstm = lstm;
        self.single_digit = single_digit;
        self.decoder = decoder;
        self.verboseOutputter = verboseOutputter;
        self.finishExpressions = finishExpressions;
        
        if (not self.lstm):
            raise ValueError("Feature LSTM = False is no longer supported!");
                
        self.EOS_symbol_index = EOS_symbol_index;
        self.GO_symbol_index = GO_symbol_index;
        
        # Set dimensions
        self.data_dim = data_dim;
        self.hidden_dim = hidden_dim;
        self.decoding_output_dim = output_dim;
        self.prediction_output_dim = output_dim;
        
        # Set up settings for fake minibatching
        self.fake_minibatch = False;
        if (self.minibatch_size == 1):
            self.fake_minibatch = True;
            self.minibatch_size = 2;
        
        
        
        # Set up shared variables
        varSettings = [];
        varSettings.append(('hWf',self.hidden_dim,self.hidden_dim));
        varSettings.append(('XWf',self.data_dim,self.hidden_dim));
        varSettings.append(('hWi',self.hidden_dim,self.hidden_dim));
        varSettings.append(('XWi',self.data_dim,self.hidden_dim));
        varSettings.append(('hWc',self.hidden_dim,self.hidden_dim));
        varSettings.append(('XWc',self.data_dim,self.hidden_dim));
        varSettings.append(('hWo',self.hidden_dim,self.hidden_dim));
        varSettings.append(('XWo',self.data_dim,self.hidden_dim));
        if (self.decoder):
            # Add variables for the decoding phase
            # All these variables begin with 'D' so they can be 
            # automatically filtered to be used as parameters
            varSettings.append(('DhWf',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DXWf',self.data_dim,self.hidden_dim));
            varSettings.append(('DhWi',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DXWi',self.data_dim,self.hidden_dim));
            varSettings.append(('DhWc',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DXWc',self.data_dim,self.hidden_dim));
            varSettings.append(('DhWo',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DXWo',self.data_dim,self.hidden_dim));
            varSettings.append(('DhWY',self.hidden_dim,self.decoding_output_dim));
        else:
            varSettings.append(('hWY',self.hidden_dim,self.prediction_output_dim));
        
        # Contruct variables
        self.vars = {};
        for (varName,dim1,dim2) in varSettings:
            # Get value for shared variable from constructor if present
            value = np.random.uniform(-np.sqrt(1.0/dim1),np.sqrt(1.0/dim1),(dim1,dim2));
            if (varName in weight_values):
                value = weight_values[varName];
            self.vars[varName] = theano.shared(name=varName, value=value);
        
        
        
        # Set up inputs to prediction and SGD
        # X is 3-dimensional: 1) index in sentence, 2) datapoint, 3) dimensionality of data
        X = T.dtensor3('X');
        if (self.single_digit):
            # targets is 2-dimensional: 1) datapoint, 2) label for each answer
            label = T.dmatrix('label');
        else:
            # targets is 3-dimensional: 1) index in sentence, 2) datapoint, 3) encodings
            label = T.dtensor3('label');
        if (self.finishExpressions):
            intervention_location = T.iscalar('intervention_location');
        
        first_hidden = T.zeros((self.minibatch_size,self.hidden_dim), dtype='float64');
        if (self.single_digit):
            [outputs, hidden], _ = theano.scan(fn=self.lstm_predict_single,
                                               sequences=X,
                                               # Input a zero hidden layer
                                               outputs_info=(None, {'initial': first_hidden, 'taps': [-1]}),
                                               non_sequences=[self.vars[k[0]] for k in filter(lambda name: name[0][0] != 'D', varSettings)])
            
            # We only predict on the final Y because for now we only predict the final digit in the expression
            # Takes the argmax over the last dimension, resulting in a vector representing one predicted symbol
            prediction = T.argmax(outputs[-1], 1);
            right_hand = outputs[-1];
            # Perform crossentropy calculation for each datapoint and take the mean
            error = T.mean(T.nnet.categorical_crossentropy(outputs[-1], label));
        else:
            # Set scan functions and arguments
            encode_function = self.lstm_predict_single_no_output;
            # Set the prediction parameters to be either the prediction 
            # weights or the decoding weights depending on the setting 
            encode_parameters = [self.vars[k[0]] for k in filter(lambda name: name[0][0] != 'D' and name[0] != 'hWY', varSettings)];
            if (self.decoder):
                decode_parameters = [self.vars[k[0]] for k in filter(lambda name: name[0][0] == 'D', varSettings)];
                decode_function = self.lstm_predict_single;
            else:
                decode_parameters = encode_parameters + [self.vars['hWY']];
                decode_function = self.lstm_predict_single;
            
            hidden, _ = theano.scan(fn=encode_function,
                                               sequences=X,
                                               # Input a zero hidden layer
                                               outputs_info=({'initial': first_hidden, 'taps': [-1]}),
                                               non_sequences=encode_parameters)
        
            if (self.GO_symbol_index is None):
                raise ValueError("GO symbol index not set!");
            
            if (self.finishExpressions):
                init_values = (None, {'initial': hidden[-1], 'taps': [-1]});
                [_, right_hand_hiddens], _ = theano.scan(fn=decode_function,
                                                                     sequences=(label[:intervention_location+1]),
                                                                     outputs_info=init_values,
                                                                     non_sequences=decode_parameters)
                right_hand_1 = label[:intervention_location+1];
                init_values = ({'initial': right_hand_1[-1], 'taps': [-1]},
                               {'initial': right_hand_hiddens[-1], 'taps': [-1]});
                [right_hand_2, _], _ = theano.scan(fn=decode_function,
                                                      outputs_info=init_values,
                                                      non_sequences=decode_parameters,
                                                      n_steps=self.n_max_digits-(intervention_location+1))
                right_hand = T.join(0, right_hand_1, right_hand_2);
            else:
                init_X = T.zeros((self.prediction_output_dim,self.minibatch_size), dtype='float64');
                init_X = T.set_subtensor(init_X[self.GO_symbol_index],T.ones(self.minibatch_size));
                init_X = T.swapaxes(init_X, 0, 1);
                
                init_values = ({'initial': init_X, 'taps': [-1]},
                               {'initial': hidden[-1], 'taps': [-1]});
                [right_hand, _], _ = theano.scan(fn=decode_function,
                                         # Inputs the last hidden layer and the last predicted symbol
                                         outputs_info=init_values,
                                         non_sequences=decode_parameters,
                                         n_steps=self.n_max_digits)
            
            # We predict the final n symbols (all symbols predicted as output from input '=')
            prediction = T.argmax(right_hand, axis=2);
            padded_label = T.join(0, label, T.zeros((self.n_max_digits - label.shape[0],self.minibatch_size,self.decoding_output_dim)));
            
            error = T.mean(T.nnet.categorical_crossentropy(right_hand,padded_label));
            
          
        
        
        # Automatic backward pass for all models: gradients
        variables = self.vars.keys();
        derivatives = T.grad(error, map(lambda var: self.vars[var], variables));
           
        # Defining prediction
        if (self.finishExpressions):
            self._predict = theano.function([X, label, intervention_location], [prediction,
                                                                                right_hand]);
        else:
            self._predict = theano.function([X], [prediction,
                                                  right_hand]);
        
        # Defining stochastic gradient descent
        learning_rate = T.dscalar('learning_rate');
        updates = [(var,var-learning_rate*der) for (var,der) in zip(map(lambda var: self.vars[var], variables),derivatives)];
        if (self.finishExpressions):
            self._sgd = theano.function([X, label, intervention_location, learning_rate], [error], 
                                        updates=updates,
                                        allow_input_downcast=True)
        else:
            self._sgd = theano.function([X, label, learning_rate], [error], 
                                        updates=updates,
                                        allow_input_downcast=True)
        
        super(TheanoRecurrentNeuralNetwork, self).__init__();
    
    def loadVars(self, variables):
        """
        Provide vars as a dictionary matching the self.vars structure.
        """
        for key in variables:
            if (key not in self.vars):
                return False;
            self.vars[key].set_value(variables[key].get_value());
        return True;
    
    # PREDICTION FUNCTIONS
    
    def lstm_predict_single(self, current_X, previous_hidden, hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo, hWY):
        forget_gate = T.nnet.sigmoid(previous_hidden.dot(hWf) + current_X.dot(XWf));
        input_gate = T.nnet.sigmoid(previous_hidden.dot(hWi) + current_X.dot(XWi));
        candidate_cell = T.tanh(previous_hidden.dot(hWc) + current_X.dot(XWc));
        cell = forget_gate * previous_hidden + input_gate * candidate_cell;
        output_gate = T.nnet.sigmoid(previous_hidden.dot(hWo) + current_X.dot(XWo));
        hidden = output_gate * cell;
        Y_output = T.nnet.softmax(hidden.dot(hWY));
        return Y_output, hidden;
    
    def lstm_predict_single_no_output(self, current_X, previous_hidden, hWf, XWf, hWi, XWi, hWc, XWc, hWo, XWo):
        forget_gate = T.nnet.sigmoid(previous_hidden.dot(hWf) + current_X.dot(XWf));
        input_gate = T.nnet.sigmoid(previous_hidden.dot(hWi) + current_X.dot(XWi));
        candidate_cell = T.tanh(previous_hidden.dot(hWc) + current_X.dot(XWc));
        cell = forget_gate * previous_hidden + input_gate * candidate_cell;
        output_gate = T.nnet.sigmoid(previous_hidden.dot(hWo) + current_X.dot(XWo));
        hidden = output_gate * cell;
        return [hidden];
    
    # END OF INITIALIZATION
    
    def sanityChecks(self, training_data, training_labels):
        """
        Sanity checks to be called before training a batch. Throws exceptions 
        if things are not right.
        """
        if (not self.single_digit and training_labels.shape[1] > (self.n_max_digits+1)):
            raise ValueError("n_max_digits too small! Increase to %d" % training_labels.shape[1]);
    
    def sgd(self, dataset, data, label, learning_rate, emptySamples=None, expressions=None, intervention_expressions=None, interventionLocation=None):
        """
        The intervention location for finish expressions must be the same for 
        all samples in this batch.
        """
        if (self.finishExpressions):
            return self.sgd_finish_expression(dataset, data, expressions, label, intervention_expressions, interventionLocation, emptySamples, learning_rate);
        else:
            return self._sgd(data, label, learning_rate), 0;
        
    def predict(self, data, label=None, interventionLocation=None, alreadySwapped=False):
        """
        Perform necessary models-specific transformations and call the actual 
        prediction function of the model.
        The intervention location for finish expressions must be the same for 
        all samples in this batch.
        """
        # Swap axes of index in sentence and datapoint for Theano purposes
        if (not alreadySwapped):
            data = np.swapaxes(data, 0, 1);
            if (label is not None):
                label = np.swapaxes(label, 0, 1);
        
        if (self.finishExpressions):
            prediction, right_hand = \
                self._predict(data, label, interventionLocation);
        else:
            prediction, right_hand = \
                self._predict(data);
        
        # Swap sentence index and datapoints back
        if (not self.single_digit):
            prediction = np.swapaxes(prediction, 0, 1);
        
        return prediction, {'right_hand': right_hand};
    
    def sgd_finish_expression(self, dataset, data, data_expression, data_with_intervention, intervention_expressions, intervention_location, emptySamples, learning_rate):
        unswapped_data = np.swapaxes(data, 0, 1);
        predictions, other = self.predict(data, data_with_intervention, intervention_location, alreadySwapped=True);
        right_hand = other['right_hand'];
        
        target = np.zeros((unswapped_data.shape[0],self.n_max_digits,self.decoding_output_dim));
        unused = 0;
        for i, prediction in enumerate(predictions):
            if (i in emptySamples):
                # Skip empty samples caused by the intervention generation process
                unused += 1;
                continue;
            
            # Find the string representation of the prediction
            string_prediction = [];
            for index in prediction:
                if (index >= dataset.EOS_symbol_index):
                    break;
                string_prediction.append(dataset.findSymbol[index]);
            
            # Get all valid predictions for this data sample including intervention
            valid_predictions = dataset.expressionsByPrefix.get(intervention_expressions[i][:intervention_location+1],intervention_location+1);
            if (len(valid_predictions) == 0):
                # Invalid example because the intervention has no corrected examples
                # We don't correct by looking at expression structure here because 
                # that is not a realistic usage of a dataset
                unused += 1;
                unswapped_data[i] = np.zeros((unswapped_data.shape[1], unswapped_data.shape[2]));
                target[i] = np.zeros((target.shape[1],target.shape[2]));
            elif (string_prediction in valid_predictions):
                # If our prediction is valid we set this part of the target to the prediction
                target[i] = np.swapaxes(right_hand[i], 0, 1);
            else:
                # Find the nearest expression to our prediction
                pred_length = len(string_prediction)
                nearest = '';
                nearestScore = 100000;
                for neighbourExpr in valid_predictions:
                    # Compute string difference
                    score = 0;
                    for k,s in enumerate(neighbourExpr[intervention_location+1:]):
                        j = k + intervention_location + 1;
                        if (pred_length <= j):
                            score += 1;
                        elif (s != string_prediction[j]):
                            score += 1;
                    score += max(0,pred_length - (j+1));
                    
                    if (score < nearestScore):
                        nearest = neighbourExpr;
                        nearestScore = score;
                
                if (nearest == ''):
                    a = 1;
                    pass
                
                target[i] = dataset.encodeExpression(nearest, self.n_max_digits);
        
        data = np.swapaxes(unswapped_data, 0, 1);
        target = np.swapaxes(target, 0, 1);
        
        return self._sgd(data, target, intervention_location, learning_rate), unused;
    
    def getVars(self):
        return self.vars.items();
    
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
