'''
Created on 22 feb. 2016

@author: Robert-Jan
'''

import theano;
import theano.tensor as T;
import numpy as np;
from models.RecurrentModel import RecurrentModel
#from theano.compile.nanguardmode import NanGuardMode
import lasagne;

class TheanoRecurrentNeuralNetwork(RecurrentModel):
    '''
    Recurrent neural network model with one hidden layer. Models single class 
    prediction based on regular recurrent model or LSTM model. 
    '''
    
    SGD_OPTIMIZER = 0;
    ADAM_OPTIMIZER = 1;

    def __init__(self, data_dim, hidden_dim, output_dim, minibatch_size, 
                 lstm=True, weight_values={}, single_digit=False, 
                 EOS_symbol_index=None, GO_symbol_index=None, n_max_digits=24, 
                 decoder=False, verboseOutputter=None, finishExpressions=False,
                 optimizer=0, learning_rate=0.01,
                 operators=4, digits=10, only_cause_expression=False, seq2ndmarkov=False):
        '''
        Initialize all Theano models.
        '''
        # Store settings in self since the initializing functions will need them
        self.single_digit = single_digit;
        self.minibatch_size = minibatch_size;
        self.n_max_digits = n_max_digits;
        self.operators = operators;
        self.digits = digits;
        self.learning_rate = learning_rate;
        self.lstm = lstm;
        self.single_digit = single_digit;
        self.decoder = decoder;
        self.only_cause_expression = only_cause_expression;
        self.seq2ndmarkov = seq2ndmarkov;
        self.optimizer = optimizer;
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
        
        actual_data_dim = self.data_dim * 2;
        actual_decoding_output_dim = self.decoding_output_dim * 2;
        actual_prediction_output_dim = self.prediction_output_dim * 2;
        if (self.only_cause_expression):
            actual_data_dim = self.data_dim;
            actual_decoding_output_dim = self.decoding_output_dim;
            actual_prediction_output_dim = self.prediction_output_dim;
        
        # Set up shared variables
        varSettings = [];
        varSettings.append(('hWf',self.hidden_dim,self.hidden_dim));
        varSettings.append(('XWf',actual_data_dim,self.hidden_dim));
        varSettings.append(('hWi',self.hidden_dim,self.hidden_dim));
        varSettings.append(('XWi',actual_data_dim,self.hidden_dim));
        varSettings.append(('hWc',self.hidden_dim,self.hidden_dim));
        varSettings.append(('XWc',actual_data_dim,self.hidden_dim));
        varSettings.append(('hWo',self.hidden_dim,self.hidden_dim));
        varSettings.append(('XWo',actual_data_dim,self.hidden_dim));
        if (self.decoder):
            # Add variables for the decoding phase
            # All these variables begin with 'D' so they can be 
            # automatically filtered to be used as parameters
            varSettings.append(('DhWf',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DXWf',actual_data_dim,self.hidden_dim));
            varSettings.append(('DhWi',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DXWi',actual_data_dim,self.hidden_dim));
            varSettings.append(('DhWc',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DXWc',actual_data_dim,self.hidden_dim));
            varSettings.append(('DhWo',self.hidden_dim,self.hidden_dim));
            varSettings.append(('DXWo',actual_data_dim,self.hidden_dim));
            varSettings.append(('DhWY',self.hidden_dim,actual_decoding_output_dim));
        else:
            varSettings.append(('hWY',self.hidden_dim,actual_prediction_output_dim));
        
        # Contruct variables
        self.vars = {};
        for (varName,dim1,dim2) in varSettings:
            # Get value for shared variable from constructor if present
            value = np.random.uniform(-np.sqrt(1.0/dim1),np.sqrt(1.0/dim1),(dim1,dim2)).astype('float32');
            if (varName in weight_values):
                value = weight_values[varName];
            self.vars[varName] = theano.shared(value, varName);
        
        
        
        # Set up inputs to prediction and SGD
        # X is 3-dimensional: 1) index in sentence, 2) datapoint, 3) dimensionality of data
        X = T.ftensor3('X');
        if (self.single_digit):
            # targets is 2-dimensional: 1) datapoint, 2) label for each answer
            label = T.fmatrix('label');
        else:
            # targets is 3-dimensional: 1) index in sentence, 2) datapoint, 3) encodings
            label = T.ftensor3('label');
        intervention_location = T.iscalar('intervention_location');
        
        # Set the prediction parameters to be either the prediction 
        # weights or the decoding weights depending on the setting 
        encode_parameters = [self.vars[k[0]] for k in filter(lambda name: name[0][0] != 'D' and name[0] != 'hWY', varSettings)];
        if (self.decoder):
            decode_parameters = [self.vars[k[0]] for k in filter(lambda name: name[0][0] == 'D', varSettings)];
            decode_function = self.lstm_predict_single;
        else:
            decode_parameters = encode_parameters + [self.vars['hWY']];
            decode_function = self.lstm_predict_single;
        
        first_hidden = T.zeros((self.minibatch_size,self.hidden_dim));        
        hidden, _ = theano.scan(fn=self.lstm_predict_single_no_output,
                                sequences=X,
                                # Input a zero hidden layer
                                outputs_info=({'initial': first_hidden, 'taps': [-1]}),
                                non_sequences=encode_parameters);
    
        if (self.GO_symbol_index is None):
            raise ValueError("GO symbol index not set!");

        init_values = (None, {'initial': hidden[-1], 'taps': [-1]});
        [right_hand_1, right_hand_hiddens], _ = theano.scan(fn=decode_function,
                                            sequences=(label[:intervention_location+1]),
                                            outputs_info=init_values,
                                            non_sequences=decode_parameters)
        init_values_2 = ({'initial': right_hand_1[-1], 'taps': [-1]},
                         {'initial': right_hand_hiddens[-1], 'taps': [-1]});
        [right_hand_2, _], _ = theano.scan(fn=decode_function,
                                           outputs_info=init_values_2,
                                           non_sequences=decode_parameters,
                                           n_steps=self.n_max_digits-(intervention_location+1))
        right_hand_with_zeros = T.join(0, right_hand_1, right_hand_2);
        right_hand_near_zeros = T.ones_like(right_hand_with_zeros) * 1e-15;
        right_hand = T.maximum(right_hand_with_zeros, right_hand_near_zeros);
        
        # We predict the final n symbols (all symbols predicted as output from input '=')
        prediction_1 = T.argmax(right_hand[:,:,:self.data_dim], axis=2);
        if (not self.only_cause_expression):
            prediction_2 = T.argmax(right_hand[:,:,self.data_dim:], axis=2);
        #padded_label = T.join(0, label, T.zeros((self.n_max_digits - label.shape[0],self.minibatch_size,self.decoding_output_dim*2), dtype=theano.config.floatX));
        
        cat_cross = T.nnet.categorical_crossentropy(right_hand[:label.shape[0]],label);
        error = T.mean(cat_cross);
        
        # Defining prediction
        if (not self.only_cause_expression):
            self._predict = theano.function([X, label, intervention_location], [prediction_1,
                                                                                prediction_2,
                                                                                right_hand]);
        else:
            self._predict = theano.function([X, label, intervention_location], [prediction_1,
                                                                                right_hand]);
        
        # Defining stochastic gradient descent
        variables = self.vars.keys();
        var_list = map(lambda var: self.vars[var], variables)
        if (self.optimizer == self.SGD_OPTIMIZER):
            # Automatic backward pass for all models: gradients
            derivatives = T.grad(error, var_list);
            learning_rate = T.fscalar('learning_rate');
            updates = [(var,var-self.learning_rate*der) for (var,der) in zip(var_list,derivatives)];
        else:
            #updates, derivatives = self.adam(error, map(lambda var: self.vars[var], variables), learning_rate);
            derivatives = T.grad(error, var_list);
            updates = lasagne.updates.nesterov_momentum(derivatives,var_list,learning_rate=self.learning_rate).items();
        self._sgd = theano.function([X, label, intervention_location],
                                        [error, 
                                         cat_cross, 
                                         right_hand,
                                         label] + decode_parameters + derivatives, 
                                    updates=updates,
                                    allow_input_downcast=True)
        
        super(TheanoRecurrentNeuralNetwork, self).__init__();
    
    def loadVars(self, variables, floats=False):
        """
        Provide vars as a dictionary matching the self.vars structure.
        """
        for key in variables:
            if (key not in self.vars):
                return False;
            if (floats):
                self.vars[key].set_value(variables[key].astype('float32'));
            else:
                self.vars[key].set_value(variables[key].get_value().astype('float32'));
        return True;
    
    def loadPartialDataDimVars(self, variables, offset, size):
        """
        Provide vars as a dictionary matching the self.vars structure.
        Data will be loaded into var[offset:offset+size].
        Only vars that have a data dimension will be overwritten.
        """
        for key in variables:
            if (key not in self.vars):
                return False;
            if (key[0] == "X" or key[:2] == "DX"):
                value = self.vars[key].get_value().astype('float32');
                new_part = variables[key].get_value().astype('float32');
                value[offset:offset+size,:] = new_part;
                self.vars[key].set_value(value);
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
        return hidden;
    
    # END OF INITIALIZATION
    
    def sanityChecks(self, training_data, training_labels):
        """
        Sanity checks to be called before training a batch. Throws exceptions 
        if things are not right.
        """
        if (not self.single_digit and training_labels.shape[1] > (self.n_max_digits+1)):
            raise ValueError("n_max_digits too small! Increase to %d" % training_labels.shape[1]);
    
    def sgd(self, dataset, data, label, learning_rate, emptySamples=None, 
            expressions=None, intervention=False, intervention_expressions=None, 
            interventionLocation=None, fixedDecoderInputs=False, topcause=True):
        """
        The intervention location for finish expressions must be the same for 
        all samples in this batch.
        """
        if (self.finishExpressions):
            return self.sgd_finish_expression(dataset, data, label, 
                                              intervention_expressions, interventionLocation, 
                                              learning_rate, emptySamples, 
                                              intervention=intervention, 
                                              fixedDecoderInputs=fixedDecoderInputs,
                                              topcause=topcause);
        else:
            return self._sgd(data, label, learning_rate), 0;
        
    def predict(self, data, label=None, interventionLocation=None, alreadySwapped=False, 
                intervention=True, fixedDecoderInputs=True):
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
        
        if (not intervention):
            if (fixedDecoderInputs):
                interventionLocation = self.n_max_digits-1;
            else:
                raise ValueError("Unfixed decoder inputs not implemented yet!");
        
        if (not self.only_cause_expression):
            prediction_1, prediction_2, right_hand = \
                    self._predict(data, label, interventionLocation);
        else:
            prediction_1, right_hand = \
                    self._predict(data, label, interventionLocation);
        
        # Swap sentence index and datapoints back
        prediction_1 = np.swapaxes(prediction_1, 0, 1);
        if (not self.only_cause_expression):
            prediction_2 = np.swapaxes(prediction_2, 0, 1);
        right_hand = np.swapaxes(right_hand, 0, 1);
        
        if (not self.only_cause_expression):
            return [prediction_1, prediction_2], {'right_hand': right_hand};
        else:
            return prediction_1, {'right_hand': right_hand};
    
    def sgd_finish_expression(self, dataset, encoded_expressions, 
                              encoded_expressions_with_intervention, expressions_with_intervention,
                              intervention_location, learning_rate, emptySamples, 
                              intervention=True, fixedDecoderInputs=True, topcause=True):
        if (not self.only_cause_expression):
            [predictions_1, predictions_2], other = self.predict(encoded_expressions, encoded_expressions_with_intervention, 
                                                                 intervention_location, intervention=intervention,
                                                                 fixedDecoderInputs=fixedDecoderInputs);
        else:
            predictions_1, other = self.predict(encoded_expressions, encoded_expressions_with_intervention, 
                                                intervention_location, intervention=intervention,
                                                fixedDecoderInputs=fixedDecoderInputs);
        right_hand_1 = other['right_hand'][:,:,:self.data_dim];
        if (not self.only_cause_expression):
            right_hand_2 = other['right_hand'][:,:,self.data_dim:];
        
        # Set which expression is cause and which is effect
        if (topcause):
            # Assume all samples in the batch have the same setting for
            # which of the expressions is the cause and which is the effect
            causeExpressionPredictions = predictions_1;
            causeExpressionRightHand = right_hand_1;
            if (not self.only_cause_expression):
                effectExpressionPredictions = predictions_2;
                effectExpressionRightHand = right_hand_2;
            else:
                effectExpressionPredictions = None;
                effectExpressionRightHand = None;
                
            # Unzip the tuples of expressions into two lists
            causeExpressions, _ = zip(*expressions_with_intervention);
        else:
            if (self.only_cause_expression):
                causeExpressionPredictions = predictions_1;
                causeExpressionRightHand = right_hand_1;
                effectExpressionPredictions = None;
                effectExpressionRightHand = None;
            else:
                causeExpressionPredictions = predictions_2;
                causeExpressionRightHand = right_hand_2;
                effectExpressionPredictions = predictions_1;
                effectExpressionRightHand = right_hand_1;
                
            # Unzip the tuples of expressions into two lists
            _, causeExpressions = zip(*expressions_with_intervention);
        
        if (intervention):
            # Change the target of the SGD to the nearest valid expression-subsystem
            encoded_expressions_with_intervention, labels_to_use = \
                self.finish_expression_find_labels(causeExpressionPredictions, effectExpressionPredictions,
                                                   dataset, 
                                                   causeExpressions, 
                                                   intervention_location,
                                                   updateTargets=True, updateLabels=True,
                                                   encoded_causeExpression=causeExpressionRightHand,
                                                   encoded_effectExpression=effectExpressionRightHand,
                                                   emptySamples=emptySamples, topcause=topcause);
        else:
            if (fixedDecoderInputs):
                intervention_location = self.n_max_digits - 1;
            else:
                raise ValueError("fixedDecoderInputs = false not implemented yet!");
        
        # Swap axes of index in sentence and datapoint for Theano purposes
        encoded_expressions = np.swapaxes(encoded_expressions, 0, 1);
        encoded_expressions_with_intervention = np.swapaxes(encoded_expressions_with_intervention, 0, 1);
        
        if (not self.only_cause_expression):
            return self._sgd(encoded_expressions, encoded_expressions_with_intervention, 
                             intervention_location), [predictions_1, predictions_2], labels_to_use;
        else:
            return self._sgd(encoded_expressions, encoded_expressions_with_intervention, 
                             intervention_location), predictions_1, labels_to_use;
    
    def finish_expression_find_labels(self, causeExpressionPredictions, effectExpressionPredictions,
                                       dataset,
                                       causeExpressions, 
                                       intervention_location,
                                       updateTargets=False, updateLabels=False,
                                       encoded_causeExpression=False, encoded_effectExpression=False,
                                       emptySamples=False, test_n=False, useTestStorage=False,
                                       topcause=True):
        if (not self.only_cause_expression):
            target = np.zeros((self.minibatch_size,self.n_max_digits,self.decoding_output_dim*2));
        else:
            target = np.zeros((self.minibatch_size,self.n_max_digits,self.decoding_output_dim));
        label_expressions = [];
        if (emptySamples is False):
            emptySamples = [];
        if (test_n is False):
            test_n = len(causeExpressionPredictions);
        
        # Set the storage and helper methods
        storage = dataset.expressionsByPrefix;
        appendToLabels = lambda cause, effect: (cause, effect);
        def setTarget(j,cause,value):
            if (cause):
                target[j,:,:self.data_dim] = value;
            else:
                target[j,:,self.data_dim:] = value;
        if (self.seq2ndmarkov and not topcause):
            storage = dataset.expressionsByPrefixBot;
            appendToLabels = lambda cause, effect: (effect, cause);
            def setTarget(j,cause,value):
                if (cause):
                    target[j,:,self.data_dim:] = value;
                else:
                    target[j,:,:self.data_dim] = value;
        
        # Set the storage and helper methods for testing
        if (useTestStorage):
            storage = dataset.testExpressionsByPrefix;
            if (self.seq2ndmarkov and not topcause):
                storage = dataset.testExpressionsByPrefixBot;
        
        if (self.only_cause_expression is not False):
            def setTarget(j,cause,value):
                target[j,:,:] = value;
        
        for i, prediction in enumerate(causeExpressionPredictions[:test_n]):
            if (i in emptySamples):
                # Skip empty samples caused by the intervention generation process
                continue;
            
            # Find the string representation of the prediction
            string_prediction = "";
            for index in prediction:
                if (index >= dataset.EOS_symbol_index):
                    break;
                string_prediction += dataset.findSymbol[index];
            
            # Get all valid predictions for this data sample including intervention
            # Note: the prediction might deviate from the label even before the intervention
            # We still use the label expressions provided (even though we know that our
            # prediction will not be in valid_prediction) because we do want to use a label 
            # that does the intervention right so the model can learn from this mistake
            _, _, valid_predictions, validPredictionEffectExpressions = storage.get(causeExpressions[i][:intervention_location+1]);
            if (len(valid_predictions) == 0):
                # Invalid example because the intervention has no corrected examples
                # We don't correct by looking at expression structure here because 
                # that is not a realistic usage of a dataset
                raise ValueError("No valid predictions available! This should not happen at all...");
                if (updateLabels):
                    label_expressions.append(appendToLabels(string_prediction,""));
            elif (string_prediction in valid_predictions):
                # The prediction of the cause expression is right, so we
                # use the right_hand predicted for this part
                if (updateTargets):
                    setTarget(i,True,encoded_causeExpression[i]);
                
                # If our prediction is valid we check if the other expression matches 
                # the predicted expression
                other_string_prediction = "";
                if (not self.only_cause_expression):
                    for index in effectExpressionPredictions[i]:
                        if (index >= dataset.EOS_symbol_index):
                            break;
                        other_string_prediction += dataset.findSymbol[index];
                    prediction_index = valid_predictions.index(string_prediction);
                    if (other_string_prediction == validPredictionEffectExpressions[prediction_index]):
                        # If the effect expression predicted matches the 
                        # expression attached to the cause expression,
                        # this prediction is right, so we use it as right_hand
                        if (updateTargets):
                            setTarget(i,False,encoded_effectExpression[i]);
                    else:
                        # If the cause expression was predicted right but the
                        # effect expression is wrong we need to run SGD with
                        # as target for the effect expression the effect
                        # expression stored with the cause expression
                        if (updateTargets):
                            other_expression_target_expression = validPredictionEffectExpressions[prediction_index];
                            setTarget(i,False,dataset.encodeExpression(other_expression_target_expression, \
                                                                       self.n_max_digits));
                if (updateLabels):
                    # Regardless of whether the effect/other prediction is right we know
                    # what labels to supply: the set of correct cause prediction and its
                    # corresponding effect prediction
                    label_expressions.append(appendToLabels(string_prediction,other_string_prediction));
            else:
                # Find the nearest expression to our prediction
                pred_length = len(string_prediction)
                nearest = 0;
                nearestScore = 100000;
                for i_near, neighbourExpr in enumerate(valid_predictions):
                    # Compute string difference
                    score = 0;
                    j = intervention_location + 1
                    for k,s in enumerate(neighbourExpr[intervention_location+1:]):
                        j = k + intervention_location + 1;
                        if (pred_length <= j):
                            score += 1;
                        elif (s != string_prediction[j]):
                            score += 1;
                    score += max(0,pred_length - (j+1));
                    
                    if (score < nearestScore):
                        nearest = i_near;
                        nearestScore = score;
                
                # Use as targets the found cause expression and its 
                # accompanying effect expression
                if (updateTargets):
                    setTarget(i,True,dataset.encodeExpression(valid_predictions[nearest], self.n_max_digits));
                    if (not self.only_cause_expression):
                        setTarget(i,False,dataset.encodeExpression(validPredictionEffectExpressions[nearest],
                                                                   self.n_max_digits));
                if (updateLabels):
                    label_expressions.append(appendToLabels(valid_predictions[nearest],validPredictionEffectExpressions[nearest]));
        
        return target, label_expressions;
    
    def getVars(self):
        return self.vars.items();
    
    def batch_statistics(self, stats, prediction, 
                         expressions_with_interventions, intervention_location,
                         other, test_n, dataset,
                         excludeStats=None, no_print_progress=False,
                         eos_symbol_index=None, print_sample=False,
                         emptySamples=None, labels_to_use=False,
                         training=False, topcause=True):
        """
        Overriding for finish-expressions.
        expressions_with_interventions contains the label-expressions (in 
        strings) that should be used to lookup the candidate labels for SGD (in 
        finish_expression_find_labels)
        """
        causeExpressions, _ = zip(*expressions_with_interventions);
        if (not topcause):
            _, causeExpressions = zip(*expressions_with_interventions);
        
        dont_switch = False;
        if (len(prediction) <= 1):
            # If we only have one prediction (only_cause_expression) we pad 
            # the prediction with an empty one 
            prediction.append([]);
            dont_switch = True;
        
        if (labels_to_use is False):
            # Set cause and effect predictions
            cause = prediction[0];
            effect = prediction[1];
            if (not topcause and not dont_switch):
                cause = prediction[1];
                effect = prediction[0];
            
            _, labels_to_use = self.finish_expression_find_labels(cause, effect,
                                                                   dataset, 
                                                                   causeExpressions, 
                                                                   intervention_location, emptySamples,
                                                                   updateLabels=True, test_n=test_n,
                                                                   useTestStorage=not training,
                                                                   topcause=topcause);
        
        # Statistics
        causeIndex = 0;
        effectIndex = 1;
        if (not topcause and not dont_switch):
            causeIndex = 1;
            effectIndex = 0;
        if (self.only_cause_expression is not False):
            causeIndex = 0;
        
        for j in range(0,test_n):
            if (emptySamples is not None and j in emptySamples):
                continue;
            
            # Taking argmax over symbols for each sentence returns 
            # the location of the highest index, which is the first 
            # EOS symbol
            eos_location = np.argmax(prediction[causeIndex][j]);
            # Check for edge case where no EOS was found and zero was returned
            if (eos_symbol_index is None):
                eos_symbol_index = dataset.EOS_symbol_index;
            if (prediction[causeIndex][j,eos_location] != eos_symbol_index):
                eos_location = prediction[causeIndex][j].shape[0];
            
            # Convert prediction to string expression
            causeExpressionPrediction = dataset.indicesToStr(prediction[causeIndex][j][:eos_location]);
            if (not self.only_cause_expression):
                effectExpressionPrediction = dataset.indicesToStr(prediction[effectIndex][j][:eos_location]);
            
            if (self.seq2ndmarkov):
                # Check if cause sequence prediction is in dataset
                causeInDataset = False;
                if (causeExpressionPrediction == labels_to_use[j][causeIndex]):
                    stats['structureCorrectCause'] += 1.0;
                    if (topcause):
                        stats['structureCorrectTop'] += 1.0;
                    else:
                        stats['structureCorrectBot'] += 1.0;
                    causeInDataset = True;
                
                # Check if cause sequence prediction is valid
                causeValid = False;
                if (dataset.valid_seq2ndmarkov(prediction[causeIndex][j][:eos_location],dataset.digits,dataset.operators)):
                    causeValid = True;
                    stats['structureValidCause'] += 1.0;
                    if (topcause):
                        stats['structureValidTop'] += 1.0;
                    else:
                        stats['structureValidBot'] += 1.0;
                
                effectInDataset = False;
                effectValid = False;
                if (not self.only_cause_expression):
                    # Check if effect sequence prediction is in dataset
                    if (effectExpressionPrediction == labels_to_use[j][effectIndex]):
                        stats['structureCorrectEffect'] += 1.0;
                        if (topcause):
                            stats['structureCorrectBot'] += 1.0;
                        else:
                            stats['structureCorrectTop'] += 1.0;
                        effectInDataset = True;
                    
                    # Check if effect sequence prediction is valid
                    if (dataset.valid_seq2ndmarkov(prediction[effectIndex][j][:eos_location],dataset.digits,dataset.operators)):
                        effectValid = True;
                        stats['structureValidEffect'] += 1.0;
                        if (topcause):
                            stats['structureValidBot'] += 1.0;
                        else:
                            stats['structureValidTop'] += 1.0;
                
                # Check if effect prediction is valid
                effectMatch = False;
                if (not self.only_cause_expression):
                    effect = dataset.effect_matcher(prediction[causeIndex][j][:eos_location],
                                                    prediction[effectIndex][j][:eos_location],
                                                    self.digits,self.operators,topcause);
                    if (effect == 1 or effect == 2):
                        stats['effectCorrect'] += 1.0;
                        if (effect == 2):
                            stats['noEffect'] += 1.0;
                        #print("".join(map(lambda x: dataset.findSymbol[x], prediction[causeIndex][j][:eos_location])) + '/'\
                        #      + "".join(map(lambda x: dataset.findSymbol[x], prediction[effectIndex][j][:eos_location])));
                        effectMatch = True;
                
                # Determine validity of sample
                if ((causeValid and self.only_cause_expression is not False) or (causeValid and effectValid and effectMatch)):
                    stats['valid'] += 1.0;
                
                # Determine sample in dataset
                if ((causeInDataset and self.only_cause_expression is not False) or (causeInDataset and effectInDataset)):
                    stats['structureCorrect'] += 1.0;
                if ((causeInDataset and self.only_cause_expression is not False) or (causeInDataset and effectInDataset and effectMatch)):
                    stats['correct'] += 1.0;
                else:
                    difference1 = TheanoRecurrentNeuralNetwork.string_difference(causeExpressionPrediction, labels_to_use[j][causeIndex]);
                    if (not self.only_cause_expression):
                        difference2 = TheanoRecurrentNeuralNetwork.string_difference(effectExpressionPrediction, labels_to_use[j][effectIndex]);
                    else:
                        difference2 = 0;
                    stats['error_histogram'][difference1 + difference2] += 1;
            else:
                if (causeExpressionPrediction == labels_to_use[j][causeIndex]):
                    stats['structureCorrect'] += 1.0;
                    stats['structureValidCause'] += 1.0;
                    if (self.only_cause_expression is not False):
                        stats['correct'] += 1.0;
                        stats['valid'] += 1.0;
                    elif (effectExpressionPrediction == labels_to_use[j][effectIndex]):
                        stats['correct'] += 1.0;
                        stats['valid'] += 1.0;
                        stats['effectCorrect'] += 1.0;
                        stats['structureValidEffect'] += 1.0;
                else:
                    difference1 = TheanoRecurrentNeuralNetwork.string_difference(causeExpressionPrediction, labels_to_use[j][causeIndex]);
                    if (not self.only_cause_expression):
                        difference2 = TheanoRecurrentNeuralNetwork.string_difference(effectExpressionPrediction, labels_to_use[j][effectIndex]);
                    else:
                        difference2 = 0;
                    stats['error_histogram'][difference1 + difference2] += 1;
                    # Defer matching of effect prediction and what it should be to dataset effect matcher (dependent on dataset)
                    if (not self.only_cause_expression and dataset.effect_matcher(prediction[causeIndex][j][:eos_location],prediction[effectIndex][j][:eos_location],self.digits,self.operators,topcause)):
                        stats['effectCorrect'] += 1.0;
                    
                    # Check for validity - check the test storage because all 
                    # expressions are contained in the combination of training 
                    # and test storage
                    testStorageToCheck = dataset.testExpressionsByPrefix;
                    if (self.only_cause_expression is False):
                        testStorageToCheckEffect = dataset.testExpressionsByPrefixBot;
                    if (not topcause):
                        testStorageToCheck = dataset.testExpressionsByPrefixBot;
                        if (self.only_cause_expression is False):
                            testStorageToCheckEffect = dataset.testExpressionsByPrefix;
                    validPrediction, validEffectPrediction, _, _ = testStorageToCheck.get(causeExpressionPrediction);
                    causeValid = False;
                    effectValid = False;
                    if (validPrediction is not False):
                        stats['structureValidCause'] += 1.0;
                        causeValid = True;
                    if (validEffectPrediction is not False):
                        # If we found a match in the test storage we can check 
                        # the corresponding prime expression
                        if (validEffectPrediction == effectExpressionPrediction):
                            effectValid = True;
                    elif (self.only_cause_expression is False):
                        # Else we can check the effect expression storage if that exists
                        validEffectPrediction, _, _, _ = testStorageToCheckEffect.get(causeExpressionPrediction);
                        if (validEffectPrediction is not False):
                            effectValid = True;
                    if (effectValid):
                        stats['structureValidEffect'] += 1.0;
                        if (causeValid):
                            stats['valid'] += 1.0;
            
            # Digit precision and prediction size computation
            i = 0;
            for i in range(min(len(causeExpressionPrediction),len(labels_to_use[j][causeIndex]))):
                if (causeExpressionPrediction[i] == labels_to_use[j][causeIndex][i]):
                    stats['digit_1_correct'] += 1.0;
            stats['digit_1_prediction_size'] += len(causeExpressionPrediction);
            
            if (not self.only_cause_expression):
                i = 0;
                for i in range(min(len(effectExpressionPrediction),len(labels_to_use[j][effectIndex]))):
                    if (effectExpressionPrediction[i] == labels_to_use[j][effectIndex][i]):
                        stats['digit_2_correct'] += 1.0;
                stats['digit_2_prediction_size'] += len(effectExpressionPrediction);      

       
            stats['prediction_1_size_histogram'][int(eos_location)] += 1;
            for digit_prediction in prediction[causeIndex][j][:len(causeExpressionPrediction)]:
                stats['prediction_1_histogram'][int(digit_prediction)] += 1;
            
            if (not self.only_cause_expression):
                stats['prediction_2_size_histogram'][int(eos_location)] += 1;
                for digit_prediction in prediction[effectIndex][j][:len(effectExpressionPrediction)]:
                    stats['prediction_2_histogram'][int(digit_prediction)] += 1;
            
            stats['prediction_size'] += 1;
        
        return stats;
    
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
    
    @staticmethod
    def string_difference(string1, string2):
        # Compute string difference
        score = 0;
        for k,s in enumerate(string2):
            if (len(string1) <= k):
                score += 1;
            elif (s != string1[k]):
                score += 1;
        score += max(0,len(string1) - (k+1));
        return score;
