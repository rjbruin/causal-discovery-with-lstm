'''
Created on 22 feb. 2016

@author: Robert-Jan
'''

import numpy as np;

class GeneratedExpressionDataset(object):
    
    def __init__(self, sourceFolder):
        self.train_source = sourceFolder + '/train.txt';
        self.test_source = sourceFolder + '/test.txt';
        
        # Setting one-hot encoding
        self.oneHot = {'+': 10, '-': 11, '*': 12, '/': 13, '(': 14, ')': 15, '=': 16};
        self.operators = self.oneHot.keys();
        # Digits are pre-assigned 0-9
        for digit in range(10):
            self.oneHot[str(digit)] = digit;
        # Data dimension = number of symbols + 1
        self.data_dim = max(self.oneHot.values()) + 1;
        self.output_dim = self.data_dim;
        
        self.load();
    
    def load(self):
        self.train, self.train_targets, self.train_labels, self.train_expressions = self.loadFile(self.train_source);
        self.test, self.test_targets, self.test_labels, self.test_expressions = self.loadFile(self.test_source);
    
    def loadFile(self, source):
        # Importing data
        f_data = open(source,'r');
        data = [];
        targets = [];
        labels = [];
        expressions = [];
        for line in f_data:
            # Get expression from line
            expression = line.strip();
            expressions.append(expression);
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
        
        return data, targets, np.array(labels), expressions;
    
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