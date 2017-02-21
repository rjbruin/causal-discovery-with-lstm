'''
Created on 22 feb. 2016

@author: Robert-Jan
'''

import numpy as np;
import json, os;

from models.SequencesByPrefix import SequencesByPrefix

from collections import Counter;

class GeneratedExpressionDataset(object):
    
    DATASET_EXPRESSIONS = 0;
    DATASET_SEQ2NDMARKOV = 1;
    DATASET_DOUBLEOPERATOR = 2;
    DATASET_DISCRETEPROCESS = 3;
    
    TRAIN = 0;
    TEST = 1;
    
    def __init__(self, source, configSource, 
                 preload=True,
                 test_batch_size=10000, train_batch_size=10000,
                 max_training_size=False, max_testing_size=False,
                 sample_testing_size=False, 
                 use_GO_symbol=False, 
                 finishExpressions=False, repairExpressions=False, find_x=False,
                 reverse=False, copyMultipleExpressions=False,
                 operators=4, digits=10, only_cause_expression=False,
                 dataset_type=0, bothcause=False, debug=False,
                 test_size=0.1, test_offset=0., linearProcess=False):
        self.source = source;
        self.test_batch_size = test_batch_size;
        self.train_batch_size = train_batch_size;
        self.max_training_size = max_training_size;
        self.max_testing_size = max_testing_size;
        self.sample_testing_size = sample_testing_size;
        self.only_cause_expression = only_cause_expression;
        self.dataset_type = dataset_type;
        self.bothcause = bothcause;
        self.debug = debug;
        self.linearProcess = linearProcess;
        
        self.operators = operators;
        self.digits = digits;
        
        self.finishExpressions = finishExpressions;
        self.repairExpressions = repairExpressions;
        self.find_x = find_x;
        self.reverse = reverse;
        self.copyMultipleExpressions = copyMultipleExpressions;
        
        # Set the method that should process the lines of the dataset
        self.processor = self.processSample;
        if (self.linearProcess):
            self.processor = self.processSampleLinearProcess;
        elif (self.dataset_type == GeneratedExpressionDataset.DATASET_DISCRETEPROCESS):
            self.processor = self.processSampleDiscreteProcess;
        elif (self.dataset_type == GeneratedExpressionDataset.DATASET_SEQ2NDMARKOV):
            self.processor = self.processSeq2ndMarkov;
        elif (self.copyMultipleExpressions):
            self.processor = self.processSampleCopyMultipleInputs;
        elif (self.repairExpressions):
            self.processor = self.processSampleRepairing;
        elif (self.find_x):
            self.processor = self.processSampleFindX;
        elif (self.finishExpressions):
            self.processor = self.processSampleCopyInput;
        else:
            self.processor = self.processSampleMultiDigit;
        
        # Set the method that matches an effect prediction against an effect 
        # expression generated from the cause prediction
        self.effect_matcher = self.effect_matcher_expressions_simple;
        self.valid_checker = self.valid_expression;
        if (dataset_type == GeneratedExpressionDataset.DATASET_SEQ2NDMARKOV):
            self.effect_matcher = self.effect_matcher_seq2ndmarkov;
            self.valid_checker = self.valid_seq2ndmarkov
        elif (dataset_type == GeneratedExpressionDataset.DATASET_DOUBLEOPERATOR):
            self.effect_matcher = self.effect_matcher_doubleoperator;
            self.valid_checker = self.valid_doubleoperator;
        
        # Read config to overwrite settings
        if (os.path.exists(configSource)):
            config_f = open(configSource, 'r');
            self.config = json.load(config_f);
            config_f.close();
            for key in self.config:
                if (key == 'effect_matcher'):
                    if (self.config[key] == 'expressions_simple'):
                        self.effect_matcher = self.effect_matcher_expressions_simple;
                    elif (self.config[key] == 'seq2ndmarkov_0'):
                        self.effect_matcher = self.effect_matcher_seq2ndmarkov;
                    elif (self.config[key] == 'seq2ndmarkov_2'):
                        self.effect_matcher = self.effect_matcher_seq2ndmarkov_2;
                    elif (self.config[key] == 'seq2ndmarkov_both'):
                        self.effect_matcher = self.effect_matcher_seq2ndmarkov_both;
                    elif (self.config[key] == 'seq2ndmarkov_both_2'):
                        self.effect_matcher = self.effect_matcher_seq2ndmarkov_both_2;
                elif (key == 'valid_checker'):
                    if (self.config[key] == 'expressions_simple'):
                        self.valid_checker = self.valid_expressions_simple;
                    elif (self.config[key] == 'seq2ndmarkov'):
                        self.valid_checker = self.valid_seq2ndmarkov;
                    elif (self.config[key] == 'doubleoperator'):
                        self.valid_checker = self.valid_doubleoperator;
        
        # Digits are pre-assigned 0-self.digits
        self.oneHot = {};
        for digit in range(self.digits):
            self.oneHot[str(digit)] = digit;
        symbols = [];
        if (self.dataset_type is not GeneratedExpressionDataset.DATASET_DISCRETEPROCESS):
            symbols = ['+','-','*','/'][:self.operators] + ['(',')','='];
            if (self.repairExpressions or self.find_x):
                symbols.append('x');
        symbols.extend(['_','G']);
        i = max(self.oneHot.values())+1;
        for sym in symbols:
            self.oneHot[sym] = i;
            i += 1;
        
        self.findSymbol = {v: k for (k,v) in self.oneHot.items()};
        self.key_indices = {k: i for (i,k) in enumerate(self.oneHot.keys())};
        
        # Data dimension = number of symbols
        self.data_dim = len(self.oneHot.keys());
        self.EOS_symbol_index = self.data_dim-2;
        self.GO_symbol_index = self.data_dim-1;
        # We predict the same symbols as we have as input, so input and data
        # dimension are equal
        self.output_dim = self.data_dim;
        
        data_length, _, _ = self.filemeta(self.source, self.max_training_size);
        self.lengths = [data_length * (1. - test_size), data_length * test_size];
        if (self.max_training_size is not False):
            self.lengths[self.TRAIN] = self.max_training_size;
        if (self.max_testing_size is not False):
            self.lengths[self.TEST] = self.max_testing_size;
        # Set test batch settings
        self.train_done = False;
        self.test_done = False;
        
        self.expressionLengths = Counter();
        self.expressionsByPrefix = SequencesByPrefix();
        if (not self.only_cause_expression): 
            self.expressionsByPrefixBot = SequencesByPrefix();
        self.testExpressionLengths = Counter();
        self.testExpressionsByPrefix = SequencesByPrefix();
        if (not self.only_cause_expression): 
            self.testExpressionsByPrefixBot = SequencesByPrefix();
        
        if (preload):    
            f = open(self.source,'r');
            line = f.readline().strip();
            n = 0;
            
            append_to_train = True;
            test_set_done = False;
            if (test_offset == 0.):
                append_to_train = False;
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
                
                if (append_to_train):
                    if (not self.only_cause_expression and \
                            (topcause == '0' or \
                            self.dataset_type == GeneratedExpressionDataset.DATASET_EXPRESSIONS or \
                            self.bothcause)):
                        self.expressionsByPrefixBot.add(expression_prime, expression);
                    if (topcause == '1'):
                        self.expressionsByPrefix.add(expression, expression_prime);
                    self.expressionLengths[len(expression)] += 1;
                else:
                    if (not self.only_cause_expression and \
                            (topcause == '0' or \
                            self.dataset_type == GeneratedExpressionDataset.DATASET_EXPRESSIONS or \
                            self.bothcause)):
                        self.testExpressionsByPrefixBot.add(expression_prime, expression);
                    if (topcause == '1'):
                        self.testExpressionsByPrefix.add(expression, expression_prime);
                    self.testExpressionLengths[len(expression)] += 1;
                
                line = f.readline().strip();
                n += 1;
                
                # Reassess whether to switch target dataset part
                if (not test_set_done):
                    if (append_to_train):
                        if (n / float(self.lengths[self.TRAIN]) >= test_offset and test_size > 0.):
                            append_to_train = False;
                    else:
                        if (n / float(self.lengths[self.TRAIN]) >= test_offset + test_size):
                            test_set_done = True;
                            append_to_train = True;
            f.close();
    
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
    
    def fill_ndarray(self, data, axis, fixed_length=None):
        if (axis <= 0):
            raise ValueError("Max length axis cannot be the first axis!");
        if (len(data) == 0):
            raise ValueError("Data is empty!");
        if (fixed_length is None):
            max_length = max(map(lambda a: a.shape[axis-1], data));
        else:
            max_length = fixed_length;
        if (not self.only_cause_expression and not self.repairExpressions and not self.find_x):
            nd_data = np.zeros((len(data), max_length, self.data_dim*2), dtype='float32');
        else:
            nd_data = np.zeros((len(data), max_length, self.data_dim), dtype='float32');
        for i,datapoint in enumerate(data):
            if (datapoint.shape[0] > max_length):
                raise ValueError("n_max_digits too small! Increase from %d to %d" % (max_length, datapoint.shape[0]));
            nd_data[i,:datapoint.shape[0]] = datapoint;
        return nd_data;
    
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
    
    def processSampleRepairing(self, line, data, targets, labels, expressions):
        # Get expression from line
        expression, expression_prime = line.strip().split(";");
        
        # Generate encodings for data and target
        X = np.zeros((len(expression)+1,self.data_dim));
        for i, literal in enumerate(expression):
            X[i,self.oneHot[literal]] = 1.0;
        X[i+1,self.EOS_symbol_index] = 1.0;
        
        target = np.zeros((len(expression_prime)+1,self.data_dim));
        for i, literal in enumerate(expression_prime):
            target[i,self.oneHot[literal]] = 1.0;
        target[i+1,self.EOS_symbol_index] = 1.0;
        
        # Set training variables
        data.append(np.array(X));
        targets.append(np.array(target));
        labels.append(np.argmax(expression_prime));
        expressions.append(expression_prime);
        
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
            expression_embeddings = np.zeros((max(len(expression),len(expression_prime)),self.data_dim*2), dtype='float32');
        else:
            expression_embeddings = np.zeros((max(len(expression),len(expression_prime)),self.data_dim), dtype='float32');
            
        for i, literal in enumerate(expression):
            expression_embeddings[i,self.oneHot[literal]] = 1.0;
        if (not self.only_cause_expression):
            for j, literal in enumerate(expression_prime):
                expression_embeddings[j,self.data_dim + self.oneHot[literal]] = 1.0;
        
        # Add EOS's
        #expression_embeddings[-1,self.EOS_symbol_index] = 1.0;
        #if (not self.only_cause_expression):
        #    expression_embeddings[-1,self.data_dim + self.EOS_symbol_index] = 1.0;
        
        # Append data
        data.append(expression_embeddings);
        labels.append(np.argmax(expression_embeddings, axis=1));
        targets.append(expression_embeddings);
        if (not self.only_cause_expression):
            expressions.append((expression, expression_prime));
        else:
            expressions.append((expression, ""));
        
        return data, targets, labels, expressions, 1;
    
    def processSampleLinearProcess(self, line, data, targets, labels, expressions):
        """
        Data is ndarray of size (nr lines, sequence length, nr input vars).
        Targets is same as data.
        Labels is same as data.
        Expressions is string representation.
        """
        
        _, samplesStr = line.split("|");
        samples = samplesStr.split(";");
        encoding = np.zeros((len(samples), len(samples[0].split(","))), dtype='float32');
        
        for i in range(len(samples)):
            vals = samples[i].split(",");
            for j in range(len(vals)):
                encoding[i,j] = float(vals[j]);
        data.append(encoding);
        targets.append(encoding);
        labels.append(encoding);
        expressions.append(line);
        
        return data, targets, labels, expressions, 1;
    
    def processSampleDiscreteProcess(self, line, data, targets, labels, expressions):
        """
        Data is ndarray of size (nr lines, sequence length, nr input vars).
        Targets is same as data.
        Labels is same as data.
        Expressions is string representation.
        """
        sample1, sample2 = line.split(";");
        encoding = np.zeros((len(sample1), self.data_dim*2), dtype='float32');
        
        for i in range(len(sample1)):
            encoding[i,self.oneHot[sample1[i]]] = 1.0;
        
        for i in range(len(sample2)):
            encoding[i,self.oneHot[sample2[i]]+self.data_dim] = 1.0;
        
        data.append(encoding);
        targets.append(encoding);
        labels.append(encoding);
        expressions.append((sample1, sample2));
        
        return data, targets, labels, expressions, 1;
    
    def processSampleFindX(self, line, data, targets, labels, expressions):
        expression, _ = line.strip().split(";");
        
        x_position = np.random.randint(0,len(expression));
        x = expression[x_position];
        expression = expression[:x_position] + 'x' + expression[x_position+1:];
        
        expression_embeddings = np.zeros((len(expression)+1,self.data_dim));
        for i, literal in enumerate(expression):
            expression_embeddings[i,self.oneHot[literal]] = 1.0;
        expression_embeddings[i+1,self.EOS_symbol_index] = 1.0;
        
        target_embedding = np.zeros((self.data_dim));
        target_embedding[self.oneHot[x]] = 1.;
        
        # Append data
        data.append(expression_embeddings);
        labels.append(self.oneHot[x]);
        targets.append(target_embedding);
        expressions.append(expression);
        
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
    
    def effect_matcher_seq2ndmarkov_both_2(self, top_expression_encoded, bot_expression_encoded, nr_digits, nr_operators, topcause):
        """
        Success = 0 (no match), 1 (match), 2 (no effect)
        """
        success = 2;
        for i in range(2,min(len(top_expression_encoded),len(bot_expression_encoded)),3):
            symbolIndex = bot_expression_encoded[i-2];
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
                symbolIndex = top_expression_encoded[i-2];
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
    
    def effect_matcher_doubleoperator(self, cause_expression_encoded, predicted_effect_expression_encoded, nr_digits, nr_operators, topcause):
        OPERATORS = [lambda x, y, max: (x+y) % max,
                     lambda x, y, max: (x-y) % max,
                     lambda x, y, max: (x*y) % max];
        
        for i in range(2,len(cause_expression_encoded),2):
            digit_top = cause_expression_encoded[i-2];
            digit_bot = predicted_effect_expression_encoded[i-2];
            op_top = cause_expression_encoded[i-1] - nr_digits;
            op_bot = predicted_effect_expression_encoded[i-1] - nr_digits;
            result_top = cause_expression_encoded[i];
            result_bot = predicted_effect_expression_encoded[i];
            if (op_top < 0 or op_top >= nr_operators or OPERATORS[op_top](digit_top, digit_bot, nr_digits) != result_top):
                return 0;
            if (op_bot < 0 or op_bot >= nr_operators or OPERATORS[op_bot](digit_top, digit_bot, nr_digits) != result_bot):
                return 0;
        return 1;
    
    def valid_seq2ndmarkov(self, expression_encoded, nr_digits, nr_operators):
        OPERATORS = [lambda x, y, max: (x+y) % max,
                     lambda x, y, max: (x-y) % max,
                     lambda x, y, max: (x*y) % max];
        
        if (len(expression_encoded) < 4):
            return False;
        if (expression_encoded[0] >= nr_digits):
            return False;
        result = expression_encoded[0];
        i = 0;
        while ((i+2) < len(expression_encoded)):
            if (nr_digits - expression_encoded[i] >= nr_operators or \
                nr_digits - expression_encoded[i] < 0):
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
    
    def valid_expression(self, expression, nr_digits, nr_operators):
        """
        Valid = syntactically valid.
        """
        expression = self.indicesToStr(expression, ignoreEOS=False)
        try:
            equals_index = expression.index("=");
            _ = ExpressionNode.fromStr(expression[:equals_index]);
            # Check if right hand side is not empty
            if (len(expression[equals_index+1:]) == 0):
                return False;
            # Check if right hand side contains only digits
            for sym in expression[equals_index+1:]:
                if (self.oneHot[sym] >= self.digits):
                    return False;
            return True;
        except Exception:
            return False;
    
    def valid_correct_expression(self, expression, nr_digits, nr_operators):
        """
        Check syntactic validity (valid), semantic validity (correct) for 
        entire expression and separate sides
        """
        valid = False;
        correct = False;
        left_hand_side_valid = False;
        right_hand_side_valid = False;
        
        expression = self.indicesToStr(expression, ignoreEOS=False)
        try:
            equals_index = expression.index("=");
            node = ExpressionNode.fromStr(expression[:equals_index]);
            left_hand_side_valid = True;
        except Exception:
            pass
        try:    
            # Check if right hand side is not empty
            if (len(expression[equals_index+1:]) == 0):
                return False;
            # Check if right hand side contains only digits
            for sym in expression[equals_index+1:]:
                if (self.oneHot[sym] >= self.digits):
                    raise ValueError();
            right_hand_side_valid = True;
        except Exception:
            pass
        
        if (left_hand_side_valid and right_hand_side_valid):
            valid = True;
            try:
                correct = node.getValue() == int(expression[equals_index+1:]);
            except Exception:
                pass
        
        return valid, correct, left_hand_side_valid, right_hand_side_valid;
    
    def valid_doubleoperator(self, expression_encoded, nr_digits, nr_operators):
        for i in range(1,len(expression_encoded),2):
            if (not (expression_encoded[i-1] < nr_digits and \
                expression_encoded[i] >= nr_digits and \
                expression_encoded[i] < nr_digits + nr_operators)):
                return 0;
        return 1;
    
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
    
    def indicesToStr(self, prediction, ignoreEOS=False):
        expression = "";
        for index in prediction:
            if (index == self.EOS_symbol_index and not ignoreEOS):
                break;
            expression += self.findSymbol[index];
        return expression;
    
    def strToIndices(self, prediction):
        expression = [];
        for symbol in prediction:
            if (symbol == "_"):
                break;
            expression.append(self.oneHot[symbol]);
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
    
    def abstractExpression(self, expression):
        """
        value bins        0-10, 10-50, 50-100, 100+
        nr of unique operators    0, 1, 2, 3, 4
        depth of expression    0, 1, 2, 3
        """
        value = int(expression[expression.index("=")+1:]);
        valueBins = [0, 0, 0, 0];
        if (value < 10):
            valueBins[0] = 1;
        elif (value < 50):
            valueBins[1] = 1;
        elif (value < 100):
            valueBins[2] = 1;
        else:
            valueBins[3] = 1;
        
        uniqueOperators = [];
        for s in expression:
            if (s in ['+','-','*','/']):
                if (s not in uniqueOperators):
                    uniqueOperators.append(s);
        
        operators = [0, 0, 0, 0, 0];
        operators[len(uniqueOperators)] = 1;
        
        expressionDepth = [0, 0, 0, 0];
        if (len(expression) <= 3):
            expressionDepth[0] = 1;
        elif (len(expression) <= 5):
            expressionDepth[1] = 1;
        elif (len(expression) <= 9):
            expressionDepth[2] = 1;
        else:
            expressionDepth[3] = 1;
        
        return np.array(valueBins + operators + expressionDepth, dtype='float32');
            
        
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
        if (nodeType == self.TYPE_DIGIT and value >= 10):
            raise ValueError("Illegal digit value!");
        elif (nodeType == self.TYPE_OPERATOR and value >= self.OPERATOR_SIZE):
            raise ValueError("Illegal operator value!");
        
        self.nodeType = nodeType;
        self.value = value;
    
    @staticmethod
    def createRandomDigit(maxIntValue):
        # Create terminal child (digit)
        return ExpressionNode(ExpressionNode.TYPE_DIGIT, np.random.randint(maxIntValue));
    
    @staticmethod
    def createDigit(value):
        # Create terminal child (digit)
        if (value > 10):
            raise ValueError("Illegal digit value!");
        return ExpressionNode(ExpressionNode.TYPE_DIGIT, value);
    
    @staticmethod
    def createOperator(operator, left, right):
        if (operator >= ExpressionNode.OPERATOR_SIZE):
            raise ValueError("Illegal operator value!");
        
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
            j += len(subclause) + 2;
        else:
            right = ExpressionNode(ExpressionNode.TYPE_DIGIT, int(expression[j]));
            j += 1;
        
        operator.left = left;
        operator.right = right;
        
        if (len(expression) > j):
            raise ValueError("Incorrect syntax: characters present outside clause!");
        
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
