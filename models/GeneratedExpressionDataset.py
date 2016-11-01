'''
Created on 22 feb. 2016

@author: Robert-Jan
'''

import numpy as np;
import json, os;

from models.Dataset import Dataset
from models.SequencesByPrefix import SequencesByPrefix

from collections import Counter;

class GeneratedExpressionDataset(Dataset):
    
    DATASET_EXPRESSIONS = 0;
    DATASET_SEQ2NDMARKOV = 1;
    
    def __init__(self, trainSource, testSource, configSource, 
                 preload=True,
                 test_batch_size=10000, train_batch_size=10000,
                 max_training_size=False, max_testing_size=False,
                 sample_testing_size=False, 
                 use_GO_symbol=False, finishExpressions=False,
                 reverse=False, copyMultipleExpressions=False,
                 operators=4, digits=10, only_cause_expression=False,
                 dataset_type=0, bothcause=False, debug=False):
        self.sources = [trainSource, testSource];
        self.test_batch_size = test_batch_size;
        self.train_batch_size = train_batch_size;
        self.max_training_size = max_training_size;
        self.max_testing_size = max_testing_size;
        self.sample_testing_size = sample_testing_size;
        self.only_cause_expression = only_cause_expression;
        self.dataset_type = dataset_type;
        self.bothcause = bothcause;
        self.debug = debug;
        
        self.operators = operators;
        self.digits = digits;
        
        self.finishExpressions = finishExpressions;
        self.reverse = reverse;
        self.copyMultipleExpressions = copyMultipleExpressions;
        
        # Set the method that should process the lines of the dataset
        self.processor = self.processSample;
        if (self.dataset_type == GeneratedExpressionDataset.DATASET_SEQ2NDMARKOV):
            self.processor = self.processSeq2ndMarkov;
        elif (self.copyMultipleExpressions):
            self.processor = self.processSampleCopyMultipleInputs;
        elif (self.finishExpressions):
            self.processor = self.processSampleCopyInput;
        else:
            self.processor = self.processSampleMultiDigit;
        
        # Set the method that matches an effect prediction against an effect 
        # expression generated from the cause prediction
        self.effect_matcher = self.effect_matcher_expressions_simple;
        if (dataset_type == GeneratedExpressionDataset.DATASET_SEQ2NDMARKOV):
            self.effect_matcher = self.effect_matcher_seq2ndmarkov;
        
        # Read config to overwrite settings
        if (os.path.exists(configSource)):
            config_f = open(configSource, 'r');
            self.config = json.load(config_f);
            config_f.close();
            for key in self.config:
                if (key == 'effect_matcher'):
                    if (self.config[key] == 'seq2ndmarkov_0'):
                        self.effect_matcher = self.effect_matcher_seq2ndmarkov;
                    elif (self.config[key] == 'seq2ndmarkov_2'):
                        self.effect_matcher = self.effect_matcher_seq2ndmarkov_2;
                    elif (self.config[key] == 'seq2ndmarkov_both'):
                        self.effect_matcher = self.effect_matcher_seq2ndmarkov_both;
        
        # Digits are pre-assigned 0-self.digits
        self.oneHot = {};
        for digit in range(self.digits):
            self.oneHot[str(digit)] = digit;
        symbols = ['+','-','*','/'][:self.operators] + ['(',')','=','_','G'];
        i = max(self.oneHot.values())+1;
        for sym in symbols:
            self.oneHot[sym] = i;
            i += 1;
        
        self.findSymbol = {v: k for (k,v) in self.oneHot.items()};
        self.key_indices = {k: i for (i,k) in enumerate(self.oneHot.keys())};
        
        # Data dimension = number of symbols
        self.data_dim = self.digits + len(symbols);
        self.EOS_symbol_index = self.data_dim-2;
        self.GO_symbol_index = self.data_dim-1;
        # We predict the same symbols as we have as input, so input and data
        # dimension are equal
        self.output_dim = self.data_dim;
        
        # Store locations and sizes for both train and testing
        self.locations = [0, 0];
        train_length, train_data_length, train_target_length = self.filemeta(self.sources[self.TRAIN], self.max_training_size);
        test_length, test_data_length, test_target_length = self.filemeta(self.sources[self.TEST], self.max_testing_size);
        self.data_length = max(train_data_length, test_data_length);
        self.target_length = max(train_target_length, test_target_length);
        self.lengths = [train_length, test_length];
        if (self.max_training_size is not False):
            self.lengths[self.TRAIN] = self.max_training_size;
        if (self.max_testing_size is not False):
            self.lengths[self.TEST] = self.max_testing_size;
        # Set test batch settings
        self.train_done = False;
        self.test_done = False;
        
        if (self.finishExpressions or self.copyMultipleExpressions):
            # We need to save all expressions by answer for the fast lookup of
            # nearest labels
            self.preload(onlyStoreByPrefix=True);
        else:
            self.outputByInput = None;
        
        if (preload):
            self.preloaded = self.preload();
            if (not self.preloaded):
                print("WARNING! PRELOADING DATASET WAS ATTEMPTED BUT FAILED!");
        else:
            self.preloaded = False;
    
    def preload(self, onlyStoreByPrefix=False):
        """
        Preloads the dataset into memory. If onlyStoreLookupByInput is True,
        the data is not stored but only saved into the dictionary outputByInput
        which can be used to lookup all outputs for a given unique input. In
        this case only the training data is stored.
        """
        if (onlyStoreByPrefix):
            self.expressionLengths = Counter();
            self.expressionsByPrefix = SequencesByPrefix();
            if (not self.only_cause_expression): 
                self.expressionsByPrefixBot = SequencesByPrefix();
            f = open(self.sources[self.TRAIN],'r');
            line = f.readline().strip();
            n = 0;
            
            # Debug code that can make the model load the entire dataset 
            # instead of just the 1000 samples usually used in debugging
#             if (self.debug):
#                 self.lengths = [900000,100000];
            
            # Check for n is to make the code work with max_training_size
            while (line != "" and n < self.lengths[self.TRAIN]):
                result = line.split(";");
                if (self.dataset_type == GeneratedExpressionDataset.DATASET_SEQ2NDMARKOV and not self.bothcause):
                    expression, expression_prime, topcause = result;
                else:
                    expression, expression_prime = result;
                    topcause = '1';
                
                if (self.only_cause_expression == 1):
                    expression_prime = "";
                elif (self.only_cause_expression == 2):
                    expression = expression_prime;
                    expression_prime = "";
                
                if (not self.only_cause_expression and \
                        (topcause == '0' or \
                        self.dataset_type == GeneratedExpressionDataset.DATASET_EXPRESSIONS or \
                        self.bothcause)):
                    self.expressionsByPrefixBot.add(expression_prime, expression);
                if (topcause == '1'):
                    self.expressionsByPrefix.add(expression, expression_prime);
                self.expressionLengths[len(expression)] += 1;
                line = f.readline().strip();
                n += 1;
            f.close();
            
            self.testExpressionLengths = Counter();
            self.testExpressionsByPrefix = SequencesByPrefix();
            if (not self.only_cause_expression): 
                self.testExpressionsByPrefixBot = SequencesByPrefix();
            f = open(self.sources[self.TEST],'r');
            line = f.readline().strip();
            n = 0;
            while (line != "" and n < self.lengths[self.TEST]):
                result = line.split(";");
                if (self.dataset_type == GeneratedExpressionDataset.DATASET_SEQ2NDMARKOV and not self.bothcause):
                    expression, expression_prime, topcause = result;
                else:
                    expression, expression_prime = result;
                    topcause = '1';
                
                if (self.only_cause_expression == 1):
                    expression_prime = "";
                elif (self.only_cause_expression == 2):
                    expression = expression_prime;
                    expression_prime = "";
                
                if (not self.only_cause_expression and \
                        (topcause == '0' or \
                        self.dataset_type == GeneratedExpressionDataset.DATASET_EXPRESSIONS or \
                        self.bothcause)):
                    self.testExpressionsByPrefixBot.add(expression_prime, expression);
                if (topcause == '1'):
                    self.testExpressionsByPrefix.add(expression, expression_prime);
                self.testExpressionLengths[len(expression)] += 1;
                line = f.readline().strip();
            f.close();
        else:
            train, train_targets, train_labels, train_expressions = \
                self.loadFile(self.sources[self.TRAIN], 
                              location_index=self.TRAIN, 
                              file_length=self.lengths[self.TRAIN]);
            test, test_targets, test_labels, test_expressions = \
                self.loadFile(self.sources[self.TEST], 
                              location_index=self.TEST, 
                              file_length=self.lengths[self.TEST]);
            
            self.train, self.train_targets, self.train_labels, self.train_expressions =\
                train, train_targets, train_labels, train_expressions;
            self.test, self.test_targets, self.test_labels, self.test_expressions =\
                test, test_targets, test_labels, test_expressions;
        
        
        return True;
    
    def filemeta(self, source, max_length=False):
        f = open(source, 'r');
        length = 0;
        data_length = 0;
        target_length = 0;
        line = f.readline();
        while (line.strip() != "" and (max_length is False or length+1 <= max_length)):
            length += 1;
#             try:
            data, target, _, _, _ = self.processor(line.strip(), [], [], [], []);
#             except Exception:
#                 print(line);
            if (data[0].shape[0] > data_length):
                data_length = data[0].shape[0];
            if (target[0].shape[0] > target_length):
                target_length = target[0].shape[0];
            line = f.readline();
        
        return length, data_length, target_length;
    
    def load(self, source, size, location):
        f = open(source,'r');
        
        # Prepare data storage 
        data = [];
        targets = [];
        labels = [];
        expressions = [];
        
        # Skip all lines until start of batch
        for j in range(location+1):
            line = f.readline();
        
        # Read in lines until we match the size asked for
        i = 0;
        # Keep track of how many lines we process to update the location
        line_number = j;
        while i < size:
            data, targets, labels, expressions, count = self.processor(line, data, targets, labels, expressions);
            i += count;
            
            line = f.readline();
            line_number += 1;
            # Skip empty lines and restart file at the end (if the end of file
            # is not also end of reading
            if (line.strip() == ""):
                # http://stackoverflow.com/questions/3906137/why-cant-i-call-read-twice-on-an-open-file
                f.seek(0);
                line = f.readline();
                line_number = 0;
        
        f.close();
        
        # Convert list of ndarrays to a proper ndarray so minibatching will work later
        data = self.fill_ndarray(data, 1);
        targets = self.fill_ndarray(targets, 1);
        
        # Return (data, new location)
        return (data, targets, np.array(labels), np.array(expressions)), line_number;
    
    def fill_ndarray(self, data, axis, fixed_length=None):
        if (axis <= 0):
            raise ValueError("Max length axis cannot be the first axis!");
        if (len(data) == 0):
            raise ValueError("Data is empty!");
        if (fixed_length is None):
            max_length = max(map(lambda a: a.shape[axis-1], data));
        else:
            max_length = fixed_length;
        if (not self.only_cause_expression):
            nd_data = np.zeros((len(data), max_length, self.data_dim*2), dtype='float32');
        else:
            nd_data = np.zeros((len(data), max_length, self.data_dim), dtype='float32');
        for i,datapoint in enumerate(data):
            if (datapoint.shape[0] > max_length):
                raise ValueError("n_max_digits too small! Increase from %d to %d" % (max_length, datapoint.shape[0]));
            nd_data[i,:datapoint.shape[0]] = datapoint;
        return nd_data;
    
    def loadFile(self, source, location_index=0, file_length=None):
        if (file_length is None):
            file_length = self.filelength(source);
        data, loc = self.load(source, file_length, self.locations[location_index]);
        self.locations[location_index] = loc;
        return data;
    
    def get_train_batch(self, size):
        """
        Loads and returns the next training batch based on the size indicated 
        by the caller of the method. On overflow the dataset wraps around.
        """
        if (self.train_done):
            self.train_done = False;
            self.locations[self.TRAIN] = 0;
            return False;
        if (self.preloaded):
            # Determine the range of dataset to use for this iteration
            if (self.locations[self.TRAIN] + size > self.lengths[self.TRAIN]):
                # If this iteration will overflow make it resume from the beginning of the dataset
                indices = range(self.locations[self.TRAIN],self.lengths[self.TRAIN]) + range(0,size - (self.lengths[self.TRAIN] - self.locations[self.TRAIN]));
            else:
                indices = range(self.locations[self.TRAIN],self.locations[self.TRAIN]+size);
            # Update the location of the pointer of the dataset
            self.locations[self.TRAIN] = (self.locations[self.TRAIN] + size) % self.lengths[self.TRAIN];
            self.train_done = True;
            return self.train[indices], self.train_targets[indices], self.train_labels[indices], self.train_expressions[indices];
        else:
            # Truncate the batch to be maximally the remaining part of the repetition
            if (self.locations[self.TRAIN] + size > self.lengths[self.TRAIN]):
                size -= (self.locations[self.TRAIN] + size) - self.lengths[self.TRAIN];
            
            data, loc = self.load(self.sources[self.TRAIN], size, self.locations[self.TRAIN]);
            self.locations[self.TRAIN] = loc;
            if (self.locations[self.TRAIN] >= self.lengths[self.TRAIN] or self.locations[self.TRAIN] == 0):
                self.train_done = True;
            return data;
    
    def get_test_batch(self, no_sampling=False):
        """
        Computes, loads and returns the next training batch. On overflow the
        dataset returns the final batch which may be smaller than the regular
        batch size. After this, calling the method returns False and the 
        dataset prepares itself for a new testing iteration.
        """
        length = self.lengths[self.TEST];
        
        # Terminate this batching iteration if the test is marked as done
        if (self.test_done):
            # If we have marked this test iteration as being completed, return
            # False and reset internal pointers for test batching
            self.test_done = False;
            self.locations[self.TEST] = 0;
            return False;
        
        # Set up batching range
        if (not no_sampling and self.sample_testing_size is not False):
            batch_size = self.sample_testing_size;
            if (length - batch_size <= 0):
                batch_size = length;
                startingIndex = 0;
            else:
                startingIndex = np.random.randint(0,length-batch_size);            
            testingRange = (startingIndex,startingIndex+batch_size);
        else:
            testingRange = (0,length);
         
        # Load batch
        if (self.preloaded):
            # If we have preloaded the test data, return the test data and mark
            # this iteration immediately as done
            self.test_done = True;
            return self.test[testingRange[0]:testingRange[1]], \
                    self.test_targets[testingRange[0]:testingRange[1]], \
                    self.test_labels[testingRange[0]:testingRange[1]], \
                    self.test_expressions[testingRange[0]:testingRange[1]];
        else:
            if (not no_sampling and self.sample_testing_size is not False):
                # Load in the relevant part and return
                results, _ = self.load(self.sources[self.TEST], self.sample_testing_size, testingRange[0]);
                self.test_done = True;
            else:
                # Else, compute the batch size. Truncate the batch to be maximally 
                # the remaining part of the dataset  
                batch_size = self.test_batch_size;
                if (self.locations[self.TEST]+batch_size > length):
                    batch_size = length % self.test_batch_size;
                
                # Load the relevant part of the dataset
                results, loc = self.load(self.sources[self.TEST], batch_size, self.locations[self.TEST]);
                self.locations[self.TEST] = loc;
                
                # Updating location manually is not necessary as that is already 
                # done by load()
                if (self.locations[self.TEST] >= length or self.locations[self.TEST] == 0):
                    # If the location of the pointer is end the end of file or at 
                    # the beginning (indicating overflow has taken place) mark this
                    # test as done
                    self.test_done = True;
            
            return results;
    
    def get_train_all(self):
        """
        Returns all preloaded test data.
        """
        return self.train, self.train_targets, self.train_labels, self.train_expressions;
    
    def processSample(self, line, data, targets, labels, expressions):
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
        labels.append(np.array(self.oneHot[right_hand]));
        expressions.append(expression);
        
        return data, targets, labels, expressions, 1;
    
    def processSampleCopyInput(self, line, data, targets, labels, expressions, reverse=True):
        expression = line.strip();
        
        if (self.reverse and reverse):
            expression = expression[::-1];
        
        # Old expression = data
        expression_embeddings = np.zeros((len(expression)+1,self.data_dim));
        for i, literal in enumerate(expression):
            expression_embeddings[i,self.oneHot[literal]] = 1.0;
        
        # Add EOS's
        expression_embeddings[-1,self.EOS_symbol_index] = 1.0;
        
        # Append data
        data.append(expression_embeddings);
        labels.append(np.argmax(expression_embeddings, axis=1));
        targets.append(expression_embeddings);
        expressions.append(expression);
        
        return data, targets, labels, expressions, 1;
    
    def processSampleCopyMultipleInputs(self, line, data, targets, labels, expressions):
        expressions_line = line.strip();
        expression, expression_prime = expressions_line.split(";");
        
        # We concatenate the expressions on the data_dim axis
        # Both expressions are of the same length, so no checks needed here
        if (not self.only_cause_expression):
            expression_embeddings = np.zeros((max(len(expression),len(expression_prime))+1,self.data_dim*2), dtype='float32');
        else:
            expression_embeddings = np.zeros((max(len(expression),len(expression_prime))+1,self.data_dim), dtype='float32');
            
        for i, literal in enumerate(expression):
            expression_embeddings[i,self.oneHot[literal]] = 1.0;
        if (not self.only_cause_expression):
            for j, literal in enumerate(expression_prime):
                expression_embeddings[j,self.data_dim + self.oneHot[literal]] = 1.0;
        
        # Add EOS's
        expression_embeddings[-1,self.EOS_symbol_index] = 1.0;
        if (not self.only_cause_expression):
            expression_embeddings[-1,self.data_dim + self.EOS_symbol_index] = 1.0;
        
        # Append data
        data.append(expression_embeddings);
        labels.append(np.argmax(expression_embeddings, axis=1));
        targets.append(expression_embeddings);
        if (not self.only_cause_expression):
            expressions.append((expression, expression_prime));
        else:
            expressions.append((expression, ""));
        
        return data, targets, labels, expressions, 1;
    
    def processSampleMultiDigit(self, line, data, targets, labels, expressions):
        expression = line.strip();
        
        expression_embeddings = np.zeros((len(expression)+1,self.data_dim));
        right_hand_start = expression.find('=')+1;
        right_hand_digits = [];
        for i, literal in enumerate(expression):
            expression_embeddings[i,self.oneHot[literal]] = 1.0;
            if (i >= right_hand_start):
                right_hand_digits.append(literal);
        # Add EOS
        expression_embeddings[-1,self.EOS_symbol_index] = 1.0;
        right_hand_digits.append('<EOS>');
        # The targets are simple the right hand part of the expression
        X = expression_embeddings[:right_hand_start];
        target = expression_embeddings[right_hand_start:]
        
        # Append data
        data.append(X);
        labels.append(np.array(right_hand_digits));
        targets.append(target);
        expressions.append(expression);
        
        return data, targets, labels, expressions, 1;
    
    def processSeq2ndMarkov(self, line, data, targets, labels, expressions):
        expressions_line = line.strip();
        if (not self.bothcause):
            expression, expression_prime, _ = expressions_line.split(";");
        else:
            expression, expression_prime = expressions_line.split(";");
        
        if (self.only_cause_expression == 2):
            expression = expression_prime;
            expression_prime = "";
        
        # We concatenate the expressions on the data_dim axis
        # Both expressions are of the same length, so no checks needed here
        if (not self.only_cause_expression):
            expression_embeddings = np.zeros((max(len(expression),len(expression_prime))+1,self.data_dim*2), dtype='float32');
        else:
            expression_embeddings = np.zeros((max(len(expression),len(expression_prime))+1,self.data_dim), dtype='float32');
            
        for i, literal in enumerate(expression):
            expression_embeddings[i,self.oneHot[literal]] = 1.0;
        if (not self.only_cause_expression):
            for j, literal in enumerate(expression_prime):
                expression_embeddings[j,self.data_dim + self.oneHot[literal]] = 1.0;
        
        # Add EOS's
        expression_embeddings[-1,self.EOS_symbol_index] = 1.0;
        if (not self.only_cause_expression):
            expression_embeddings[-1,self.data_dim + self.EOS_symbol_index] = 1.0;
        
        # Append data
        data.append(expression_embeddings);
        labels.append(np.argmax(expression_embeddings, axis=1));
        targets.append(expression_embeddings);
        if (not self.only_cause_expression):
            expressions.append((expression, expression_prime));
        else:
            expressions.append((expression, ""));
        
        return data, targets, labels, expressions, 1;
    
    def insertInterventions(self, targets, target_expressions, topcause, interventionLocations, possibleInterventions):
        # Apply interventions to targets samples in this batch
        for i in range(targets.shape[0]):
            if (topcause):
                currentSymbol = np.argmax(targets[i,interventionLocations[i],:self.data_dim]);
            else:
                currentSymbol = np.argmax(targets[i,interventionLocations[i],self.data_dim:]);
            
            # Pick a new symbol
            newSymbol = possibleInterventions[i][np.random.randint(0,len(possibleInterventions[i]))];
            while (newSymbol == currentSymbol):
                newSymbol = possibleInterventions[i][np.random.randint(0,len(possibleInterventions[i]))];
            
            offset = 0;
            expression_index = 0;
            if (not topcause):
                offset += self.data_dim;
                expression_index = 1;
            
            targets[i,interventionLocations[i],offset+currentSymbol] = 0.0;
            targets[i,interventionLocations[i],offset+newSymbol] = 1.0;
            new_target_cause_expression = target_expressions[i][expression_index][:interventionLocations[i]] + \
                                          self.findSymbol[newSymbol] + \
                                          target_expressions[i][expression_index][interventionLocations[i]+1:];
            if (topcause):
                target_expressions[i] = (new_target_cause_expression,target_expressions[i][1]);
            else:
                target_expressions[i] = (target_expressions[i][0],new_target_cause_expression);
        
        return targets, target_expressions, interventionLocations;
    
    def effect_matcher_expressions_simple(self, cause_expression_encoded, predicted_effect_expression_encoded, nr_digits, nr_operators, topcause):
        new_expression_encoded = [];
        for x in cause_expression_encoded:
            if (x < nr_digits):
                x = (x+1) % nr_digits;
            elif (x < nr_digits + nr_operators):
                x += 1;
                if (x == nr_digits + nr_operators):
                    x = nr_digits;
            new_expression_encoded.append(x);
        return int(np.array_equal(new_expression_encoded, predicted_effect_expression_encoded));
    
    def effect_matcher_seq2ndmarkov(self, cause_expression_encoded, predicted_effect_expression_encoded, nr_digits, nr_operators, topcause):
        """
        Success = 0 (no match), 1 (match), 2 (no effect)
        """
        success = 2;
        if (topcause):
            for i, symbolIndex in enumerate(cause_expression_encoded):
                if (i % 3 == 2):
                    if (symbolIndex == 8):
                        success = int(predicted_effect_expression_encoded[i] == 8);
                    if (symbolIndex == 7):
                        success = int(predicted_effect_expression_encoded[i] == 6);
                if (success == 0):
                    return success;
        else:
            for i, symbolIndex in enumerate(cause_expression_encoded):
                if (i % 3 == 2):
                    if (symbolIndex == 5):
                        success = int(predicted_effect_expression_encoded[i] == 5);
                    if (symbolIndex == 4):
                        success = int(predicted_effect_expression_encoded[i] == 3);
                if (success == 0):
                    return success;
        
        return success;
    
    def effect_matcher_seq2ndmarkov_2(self, cause_expression_encoded, predicted_effect_expression_encoded, nr_digits, nr_operators, topcause):
        """
        Success = 0 (no match), 1 (match), 2 (no effect)
        """
        success = 2;
        if (topcause):
            for i in range(2,len(cause_expression_encoded),3):
                symbolIndex = cause_expression_encoded[i];
                if (symbolIndex == 6):
                    success = int(predicted_effect_expression_encoded[i] == 6);
                if (symbolIndex == 1):
                    success = int(predicted_effect_expression_encoded[i] == 1);
                if (symbolIndex == 3):
                    success = int(predicted_effect_expression_encoded[i] == 5);
                if (symbolIndex == 0):
                    success = int(predicted_effect_expression_encoded[i] == 3);
                if (symbolIndex == 2):
                    success = int(predicted_effect_expression_encoded[i] == 0);
                if (success == 0):
                    return success;
        else:
            for i in range(2,len(cause_expression_encoded),3):
                symbolIndex = cause_expression_encoded[i];
                if (symbolIndex == 7):
                    success = int(predicted_effect_expression_encoded[i] == 7);
                if (symbolIndex == 2):
                    success = int(predicted_effect_expression_encoded[i] == 2);
                if (symbolIndex == 4):
                    success = int(predicted_effect_expression_encoded[i] == 3);
                if (symbolIndex == 1):
                    success = int(predicted_effect_expression_encoded[i] == 7);
                if (symbolIndex == 0):
                    success = int(predicted_effect_expression_encoded[i] == 4);
                if (success == 0):
                    return success;
        
        return success;

    def effect_matcher_seq2ndmarkov_both(self, top_expression_encoded, bot_expression_encoded, nr_digits, nr_operators, topcause):
        """
        Success = 0 (no match), 1 (match), 2 (no effect)
        """
        success = 2;
        for i in range(2,min(len(top_expression_encoded),len(bot_expression_encoded)),3):
            symbolIndex = bot_expression_encoded[i];
            effectHere = False;
            if (symbolIndex == 7):
                success = int(top_expression_encoded[i] == 7);
                effectHere = True;
            if (symbolIndex == 2):
                success = int(top_expression_encoded[i] == 2);
                effectHere = True;
            if (symbolIndex == 4):
                success = int(top_expression_encoded[i] == 3);
                effectHere = True;
            if (symbolIndex == 1):
                success = int(top_expression_encoded[i] == 7);
                effectHere = True;
            if (symbolIndex == 0):
                success = int(top_expression_encoded[i] == 4);
                effectHere = True;
                
            if (success == 0):
                return success;
            
            if (effectHere == False):
                # If no effect was found for bot to top, continue looking for
                # an effect from top to bottom
                # Stop otherwise because any relationship from top to bot
                # might have been overwritten by a bot to top relationship
                symbolIndex = top_expression_encoded[i];
                if (symbolIndex == 6):
                    success = int(bot_expression_encoded[i] == 6);
                if (symbolIndex == 1):
                    success = int(bot_expression_encoded[i] == 1);
                if (symbolIndex == 3):
                    success = int(bot_expression_encoded[i] == 5);
                if (symbolIndex == 0):
                    success = int(bot_expression_encoded[i] == 3);
                if (symbolIndex == 2):
                    success = int(bot_expression_encoded[i] == 0);
                if (success == 0):
                    return success;
        
        return success;
    
    def valid_seq2ndmarkov(self, expression_encoded, nr_digits, nr_operators):
        OPERATORS = [lambda x, y, max: (x+y) % max,
                     lambda x, y, max: (x-y) % max,
                     lambda x, y, max: (x*y) % max];
        
        if (len(expression_encoded) < 4):
            return False;
        if (expression_encoded[0] >= nr_digits):
            return False;
        result = expression_encoded[0];
        i = 1;
        while ((i+2) < len(expression_encoded)):
            if (expression_encoded[i] - nr_digits >= nr_operators or \
                expression_encoded[i] - nr_digits < 0):
                if (self.findSymbol[expression_encoded[i]] == "_" and i > 1):
                    return True;
                return False;
            op = OPERATORS[expression_encoded[i] - nr_digits];
            arg2 = expression_encoded[i+1];
            result = op(result,arg2,nr_digits);
            if (result != expression_encoded[i+2]):
                return False;
            i += 3;
        return True;
    
    def findAnswer(self, onehot_encodings):
        answer_allzeros = map(lambda d: d.sum() == 0.0, onehot_encodings);
        try:
            answer_length = answer_allzeros.index(True) - 1;
        except ValueError:
            answer_length = onehot_encodings.shape[0];
        answer_onehot = np.argmax(onehot_encodings[:answer_length],1);
        
        answer = '';
        for i in range(answer_onehot.shape[0]):
            if (answer_onehot[i] < self.EOS_symbol_index):
                answer += self.findSymbol[answer_onehot[i]];
        
        return answer;
    
    def findExpressionStructure(self, expression):
        """
        Give a list of characters representing integers and operators.
        Returns a tree 
        """
        return ExpressionNode.fromStr(expression);
    
    def indicesToStr(self, prediction):
        expression = "";
        for index in prediction:
            if (index == self.EOS_symbol_index):
                break;
            expression += self.findSymbol[index];
        return expression;
    
    def encodeExpression(self, structure, max_length):
        str_repr = str(structure);
        data = np.zeros((max_length,self.data_dim));
        add_eos = True;
        for i,symbol in enumerate(str_repr):
            if (i >= max_length-1):
                add_eos = False;
            if (i >= max_length):
                break;
            data[i,self.oneHot[symbol]] = 1.0;
        
        if (add_eos):
            data[i+1,self.EOS_symbol_index] = 1.0;
            
        return data;
        
        
class ExpressionNode(object):
    
    TYPE_DIGIT = 0;
    TYPE_OPERATOR = 1;
    
    OP_PLUS = 0;
    OP_MINUS = 1;
    OP_MULTIPLY = 2;
    OP_DIVIDE = 3;
    OPERATOR_SIZE = 4;
    
    OP_FUNCTIONS = [lambda x, y: x + y,
                    lambda x, y: x - y,
                    lambda x, y: x * y,
                    lambda x, y: x / y];
    
    operators = range(OPERATOR_SIZE);
    
    def __init__(self, nodeType, value):
        self.nodeType = nodeType;
        self.value = value;
    
    @staticmethod
    def createRandomDigit(maxIntValue):
        # Create terminal child (digit)
        return ExpressionNode(ExpressionNode.TYPE_DIGIT, np.random.randint(maxIntValue));
    
    @staticmethod
    def createDigit(value):
        # Create terminal child (digit)
        return ExpressionNode(ExpressionNode.TYPE_DIGIT, value);
    
    @staticmethod
    def createOperator(operator, left, right):
        node = ExpressionNode(ExpressionNode.TYPE_OPERATOR, operator);
        node.left = left;
        node.right = right;
        return node;
    
    def getValue(self):
        if (self.nodeType == self.TYPE_DIGIT):
            return self.value;
        else:
            if (self.value == self.OP_PLUS):
                return self.left.getValue() + self.right.getValue();
            elif (self.value == self.OP_MINUS):
                return self.left.getValue() - self.right.getValue();
            elif (self.value == self.OP_MULTIPLY):
                return self.left.getValue() * self.right.getValue();
            elif (self.value == self.OP_DIVIDE):
                return self.left.getValue() / float(self.right.getValue());
            else:
                raise ValueError("Invalid operator type");
    
    def solveAll(self, targetValue):
        """
        We need a cheap way to find expressions that are near to the current 
        expression. While the digits have to be low the problem is that target 
        values can be very large and multiplication and division are the cause of
        this. We need a way to control the complexity of the solver so we can
        run the least complex one before we use complexer stuff. 
        """
        ownValue = self.getValue();
        difference = targetValue - ownValue;
        
        if (self.nodeType == self.TYPE_DIGIT):
            if (difference == 0):
                return [self];
            if (targetValue >= 10 or targetValue < 0):
                return [];
            newMe = ExpressionNode.createDigit(targetValue)
            return [newMe];
        
        answers = [];
        if (difference == 0):
            answers.append(self);
        
        leftValue = self.left.getValue();
        rightValue = self.right.getValue();
        
        if (self.value == self.OP_PLUS):
            combinations = [];
            for i in range(10):
                for j in range(10):
                    if (i + j == targetValue and (i != leftValue or j != rightValue)):
                        combinations.append((i,j));
        elif (self.value == self.OP_MINUS):
            combinations = [];
            for i in range(10):
                for j in range(10):
                    if (i - j == targetValue):
                        combinations.append((i,j));
            
#             if (difference > 0):
#                 for i in range(0,difference+1):
#                     if (leftValue + i < 10 and rightValue - (difference - i) >= 0):
#                         combinations.append((leftValue+i,rightValue-(difference-i)));
#             elif (difference < 0):
#                 for i in range(difference+1,0):
#                     if (leftValue + i >= 0 and rightValue - (difference - i) < 10):
#                         combinations.append((leftValue+i,rightValue-(difference-i)));
#             else:
#                 combinations.extend([(i, i) for i in range(10)]);
        elif (self.value == self.OP_MULTIPLY):
            combinations = [(1,targetValue),(targetValue,1)];
            if (targetValue == 0):
                combinations.append((0,0));
                for j in range(2,10):
                    combinations.append((0,j));
                    combinations.append((j,0));
            else:
                for i in range(2,10):
                    if (i == targetValue):
                        continue;
                    if ((targetValue / float(i)) % 1.0 == 0.0):
                        combinations.append((i,targetValue/i));
                        combinations.append((targetValue/i,i));
        elif (self.value == self.OP_DIVIDE):
            combinations = [(targetValue,1)];
            if (targetValue == 0):
                for j in range(2,10):
                    combinations.append((0,j));
            else:
                for i in range(2,10):
                    if ((targetValue * i) < 10):
                        combinations.append((targetValue*i,i));
            
        for (leftVal, rightVal) in combinations:
            lefts = self.left.solveAll(leftVal);
            rights = self.right.solveAll(rightVal);
            for left in lefts:
                for right in rights:
                    answers.append(ExpressionNode.createOperator(self.value,left,right));
    
        return answers;
    
    def __str__(self):
        return self.getStr(True);
    
    @staticmethod
    def fromStr(expression):
        if (len(expression) == 1):
            return ExpressionNode.createDigit(int(expression));
        
        i = 0;
        if (expression[i] == '('):
            subclause = ExpressionNode.getClause(expression[i:]);
            left = ExpressionNode.fromStr(subclause);
            j = i + len(subclause) + 2;
        else:
            left = ExpressionNode(ExpressionNode.TYPE_DIGIT, int(expression[i]));
            j = i+1;
        op_type = ExpressionNode.getOperator(expression[j]);
        operator = ExpressionNode(ExpressionNode.TYPE_OPERATOR, op_type);
        j = j+1;
        if (expression[j] == '('):
            subclause = ExpressionNode.getClause(expression[j:]);
            right = ExpressionNode.fromStr(subclause);
        else:
            right = ExpressionNode(ExpressionNode.TYPE_DIGIT, int(expression[j]));
        
        operator.left = left;
        operator.right = right;
        
        return operator;
    
    @staticmethod
    def getClause(expr):
        openedBrackets = 0;
        for j_sub, s in enumerate(expr[1:]):
            if (s == '('):
                openedBrackets += 1;
            if (s == ')'):
                openedBrackets -= 1;
            if (openedBrackets == -1):
                return expr[1:j_sub+1];
    
    def getStr(self, upperLevel=False):
        if (self.nodeType == self.TYPE_DIGIT):
            return str(self.value);
        else:
            output = "";
            if (not upperLevel):
                output += "(";
            
            if (self.value == self.OP_PLUS):
                output += self.left.getStr() + "+" + self.right.getStr();
            elif (self.value == self.OP_MINUS):
                output += self.left.getStr() + "-" + self.right.getStr();
            elif (self.value == self.OP_MULTIPLY):
                output += self.left.getStr() + "*" + self.right.getStr();
            elif (self.value == self.OP_DIVIDE):
                output += self.left.getStr() + "/" + self.right.getStr();
            else:
                raise ValueError("Invalid operator type");
            
            if (not upperLevel):
                output += ")";
            
            return output;
    
    @staticmethod
    def getOperator(op_str):
        if (op_str == "+"):
            return ExpressionNode.OP_PLUS;
        elif (op_str == "-"):
            return ExpressionNode.OP_MINUS;
        elif (op_str == "*"):
            return ExpressionNode.OP_MULTIPLY;
        elif (op_str == "/"):
            return ExpressionNode.OP_DIVIDE;
