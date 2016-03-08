'''
Created on 22 feb. 2016

@author: Robert-Jan
'''

import numpy as np;

class GeneratedExpressionDataset(object):
    
    def __init__(self, sourceFolder, preload=True, add_x=False, single_digit=False, single_class=False, balanced=False):
        self.train_source = sourceFolder + '/train.txt';
        self.test_source = sourceFolder + '/test.txt';
        
        self.processor = self.processSample;
        if (add_x):
            self.processor = self.processSampleWithX;
        elif (single_class):
            self.processor = self.processSampleSingleClass;
        elif (not single_digit):
            self.processor = self.processSampleMultiDigit;
        elif (balanced):
            self.processor = self.processSampleBalanced;
        
        # Setting one-hot encoding
        self.digits_range = 10;
        if (single_class is not False):
            self.digits_range = single_class;
        
        # Digits are pre-assigned 0-9
        self.oneHot = {};
        for digit in range(self.digits_range):
            self.oneHot[str(digit)] = digit;
        symbols = ['+','-','*','/','(',')','='];
        if (add_x):
            symbols.append('x');
        i = max(self.oneHot.values())+1;
        for sym in symbols:
            self.oneHot[sym] = i;
            i += 1;
        
        self.operators = self.oneHot.keys();
        self.findSymbol = {v: k for (k,v) in self.oneHot.items()};
        
        # Data dimension = number of symbols + optional EOS
        self.data_dim = self.digits_range + len(symbols);
        if (not single_digit):
            self.data_dim += 1;
        self.output_dim = self.data_dim;
        self.EOS_symbol_index = self.data_dim-1;
        
        if (preload):
            self.preloaded = self.load();
            if (not self.preloaded):
                print("WARNING! PRELOADING DATASET WAS ATTEMPTED BUT FAILED!");
        else:
            self.preloaded = False;
    
    def load(self):
        try:
            self.train, self.train_targets, self.train_labels, self.train_expressions = self.loadFile(self.train_source, self.processor);
            self.test, self.test_targets, self.test_labels, self.test_expressions = self.loadFile(self.test_source, self.processor);
        except Exception:
            return False;
        return True;
    
    def loadFile(self, source, processor):
        # Importing data
        f_data = open(source,'r');
        data = [];
        targets = [];
        labels = [];
        expressions = [];
        for line in f_data:
            data, targets, labels, expressions = processor(line, data, targets, labels, expressions);
        
        return np.array(data), np.array(targets), np.array(labels), expressions;
    
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
        labels.append(self.oneHot[right_hand]);
        expressions.append(expression);
        
        return data, targets, labels, expressions;
    
    def processSampleSingleClass(self, line, data, targets, labels, expressions):
        # Get expression from line
        expression = line.strip();
        right_hand_start = expression.find('=')+1;
        left_hand = expression[:right_hand_start];
        right_hand = int(expression[right_hand_start:]);
        if (right_hand >= self.digits_range):
            # If the right hand size has a value that we cannot encode, skip 
            # the sample
            return data, targets, labels, expressions;
        # Generate encodings for data and target
        X = np.zeros((len(left_hand),self.data_dim));
        for i, literal in enumerate(left_hand):
            X[i,self.oneHot[literal]] = 1.0;
        target = np.zeros(self.data_dim);
        target[self.oneHot[str(right_hand)]] = 1.0;
        
        # Set training variables
        data.append(X);
        targets.append(np.array([target]));
        labels.append(self.oneHot[str(right_hand)]);
        expressions.append(expression);
        
        return data, targets, labels, expressions;
    
    def processSampleWithX(self, line, data, targets, labels, expressions):
        # Get expression from line
        expression = line.strip();
        # Generate encodings for data and target for each index in expression
        for i in range(len(expression)):
            X = np.zeros((len(expression),self.data_dim));
            for j, literal in enumerate(expression):
                if (i != j):
                    X[j,self.oneHot[literal]] = 1.0;
            X[i,self.oneHot['x']] = 1.0;
            target = np.zeros(self.data_dim);
            target[self.oneHot[expression[i]]] = 1.0;
            
            # Set training variables
            data.append(X);
            targets.append(np.array([target]));
            labels.append(self.oneHot[expression[i]]);
            expressions.append(expression);
        
        return data, targets, labels, expressions;
    
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
        labels.append(right_hand_digits);
        targets.append(np.array(target));
        expressions.append(expression);
        
        return data, targets, labels, expressions;
    
    def processSampleBalanced(self, line, data, targets, labels, expressions):
        expression = line.strip();
        
        # TODO: implement
        
        # Append data
#         data.append(X);
#         labels.append(right_hand_digits);
#         targets.append(np.array(target));
#         expressions.append(expression);
        
        return data, targets, labels, expressions;
    
#     def batch(self):
#         
    
    @staticmethod
    def operator_scores(expression, correct, operators, key_indices, op_scores):
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