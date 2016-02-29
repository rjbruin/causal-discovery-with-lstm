'''
Created on 22 feb. 2016

@author: Robert-Jan
'''

import numpy as np;

class GeneratedExpressionDataset(object):
    
    def __init__(self, sourceFolder, add_x=False):
        self.train_source = sourceFolder + '/train.txt';
        self.test_source = sourceFolder + '/test.txt';
        
        self.processor = self.processSample;
        if (add_x):
            self.processor = self.processSampleWithX;
        
        # Setting one-hot encoding
        self.oneHot = {'+': 10, '-': 11, '*': 12, '/': 13, '(': 14, ')': 15, '=': 16};
        if (add_x):
            self.oneHot['x'] = 17;
        self.operators = self.oneHot.keys();
        # Digits are pre-assigned 0-9
        for digit in range(10):
            self.oneHot[str(digit)] = digit;
        # Data dimension = number of symbols + 1
        self.data_dim = max(self.oneHot.values()) + 1;
        self.output_dim = self.data_dim;
        
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