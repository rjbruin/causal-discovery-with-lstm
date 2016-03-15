'''
Created on 22 feb. 2016

@author: Robert-Jan
'''

import numpy as np;

class GeneratedExpressionDataset(object):
    
    def __init__(self, sourceFolder, preload=True, add_x=False, single_digit=False, single_class=False, balanced=False, test_batch_size=10000):
        self.train_source = sourceFolder + '/train.txt';
        self.test_source = sourceFolder + '/test.txt';
        self.test_batch_size = test_batch_size;
        
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
            self.preloaded = self.preload();
            if (not self.preloaded):
                print("WARNING! PRELOADING DATASET WAS ATTEMPTED BUT FAILED!");
        else:
            self.preloaded = False;
            
        # Set test batch settings
        self.test_location = 0;
        self.test_done = False;
        # Get basic statistics from dataset
        self.train_length = self.filelength(self.train_source);
        self.test_length = self.filelength(self.test_source);
    
    def preload(self):
        try:
            self.train, self.train_targets, self.train_labels, self.train_expressions = self.loadFile(self.train_source);
            self.test, self.test_targets, self.test_labels, self.test_expressions = self.loadFile(self.test_source);
        except Exception:
            return False;
        return True;
    
    def filelength(self, source):
        f = open(source, 'r');
        length = 0;
        line = f.readline();
        while (line != ""):
            length += 1;
            line = f.readline();
        
        return length;
    
    def load(self, source, start, end):
        f = open(source,'r');
        
        # Prepare data storage 
        data = [];
        targets = [];
        labels = [];
        expressions = [];
        
        # Skip all lines until start of batch
        i = 0;
        for i in range(start+1):
            line = f.readline();
        
        # Read in lines until end
        while i != end:
            data, targets, labels, expressions, count = self.processor(line, data, targets, labels, expressions);
            i += 1;
            
            line = f.readline();
            # Skip empty lines and restart file at the end (if the end of file
            # is not also end of reading
            if (line == "" and i != end):
                # http://stackoverflow.com/questions/3906137/why-cant-i-call-read-twice-on-an-open-file
                f.seek(0);
                i = 0;
                line = f.readline();
        
        f.close();
        return np.array(data), np.array(targets), np.array(labels), np.array(expressions);
    
    def loadFile(self, source):
        file_length = self.filelength(source);
        return self.load(source, 0, file_length);
    
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
    
    def processSampleSingleClass(self, line, data, targets, labels, expressions):
        # Get expression from line
        expression = line.strip();
        right_hand_start = expression.find('=')+1;
        left_hand = expression[:right_hand_start];
        right_hand = int(expression[right_hand_start:]);
        if (right_hand >= self.digits_range):
            # If the right hand size has a value that we cannot encode, skip 
            # the sample
            return data, targets, labels, expressions, 0;
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
        
        return data, targets, labels, expressions, 1;
    
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
        
        return data, targets, labels, expressions, i+1;
    
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
    
    def processSampleBalanced(self, line, data, targets, labels, expressions):
        expression = line.strip();
        
        # TODO: implement
        
        # Append data
#         data.append(X);
#         labels.append(right_hand_digits);
#         targets.append(np.array(target));
#         expressions.append(expression);
        
        return data, targets, labels, expressions, 1;
    
    def batch(self, indices):
        if (self.preloaded):
            return self.train[indices], self.train_labels[indices], self.train_targets[indices], self.train_expressions[indices];
        else:
            start = indices[0];
            end = indices[-1]+1;
            return self.load(self.train_source, start, end);
    
    def get_test_batch(self):
        """
        Query this method for any remaining test batches.
        """
        if (self.test_done):
            self.test_done = False;
            self.test_location = 0;
            return False;
        if (self.preloaded):
            self.test_done = True;
            return self.test, self.test_labels, self.test_targets, self.test_expressions;
        else:
            self.test_location += self.test_batch_size;
            batch_size = self.test_batch_size;
            if (self.test_location >= self.test_length):
                batch_size = self.test_length % self.test_batch_size;
                self.test_location = self.test_length;
                self.test_done = True;
            return self.load(self.test_source, self.test_location - batch_size, self.test_location);
    
    def all(self):
        return self.train, self.train_labels, self.train_targets;
    
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