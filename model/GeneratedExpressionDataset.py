'''
Created on 22 feb. 2016

@author: Robert-Jan
'''

import numpy as np;

class GeneratedExpressionDataset(object):
    
    def __init__(self, sourceFolder, add_x=False, single_digit=False):
        self.train_source = sourceFolder + '/train.txt';
        self.test_source = sourceFolder + '/test.txt';
        
        self.processor = self.processSample;
        if (add_x):
            self.processor = self.processSampleWithX;
        if (not single_digit):
            self.processor = self.processSampleMultiDigit;
        
        # Setting one-hot encoding
        self.oneHot = {'+': 10, '-': 11, '*': 12, '/': 13, '(': 14, ')': 15, '=': 16};
        if (add_x):
            self.oneHot['x'] = 17;
        self.operators = self.oneHot.keys();
        # Digits are pre-assigned 0-9
        for digit in range(10):
            self.oneHot[str(digit)] = digit;
        # Data dimension = number of symbols + optional EOS
        self.data_dim = max(self.oneHot.values())+1;
        if (not single_digit):
            self.data_dim += 1;
        self.output_dim = self.data_dim;
        self.EOS_symbol_index = self.output_dim-1;
        
        self.load();
    
    def load(self):
        self.train, self.train_targets, self.train_labels, self.train_expressions = self.loadFile(self.train_source, self.processor);
        self.test, self.test_targets, self.test_labels, self.test_expressions = self.loadFile(self.test_source, self.processor);
    
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
        expression_embeddings[-1,self.data_dim-1] = 1.0;
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