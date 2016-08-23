'''
Created on 22 feb. 2016

@author: Robert-Jan
'''

import numpy as np;
from models.Dataset import Dataset

class GeneratedExpressionDataset(Dataset):
    
    def __init__(self, trainSource, testSource, preload=True, add_x=False, 
                 single_digit=False, single_class=False, balanced=False,
                 correction=False, 
                 test_batch_size=10000, train_batch_size=10000,
                 max_training_size=False, max_testing_size=False,
                 sample_testing_size=False, predictExpressions=False,
                 fillX=False):
        self.sources = [trainSource, testSource];
        self.test_batch_size = test_batch_size;
        self.train_batch_size = train_batch_size;
        self.max_training_size = max_training_size;
        self.max_testing_size = max_testing_size;
        self.sample_testing_size = sample_testing_size;
        
        self.fill_x = fillX;
        
        # Set the method that should process the lines of the dataset
        self.processor = self.processSample;
        if (add_x):
            self.processor = self.processSampleWithX;
        elif (single_class):
            self.processor = self.processSampleSingleClass;
        elif (predictExpressions):
            self.processor = self.processSamplePredictExpression;
        elif (fillX):
            self.processor = self.processSampleFillX;
        elif (correction):
            self.processor = self.processSampleCorrection;
        else:
            self.processor = self.processSampleMultiDigit;
        
        # Setting one-hot encoding
        self.digits_range = 10;
        if (single_class is not False):
            self.digits_range = single_class;
        
        # Digits are pre-assigned 0-9
        self.oneHot = {};
        for digit in range(self.digits_range):
            self.oneHot[str(digit)] = digit;
        symbols = ['+','-','*','/','(',')','='];
        if (add_x or fillX):
            symbols.append('x');
        i = max(self.oneHot.values())+1;
        for sym in symbols:
            self.oneHot[sym] = i;
            i += 1;
        # EOS
        self.oneHot['_'] = i;
        
        self.operators = self.oneHot.keys();
        self.findSymbol = {v: k for (k,v) in self.oneHot.items()};
        self.key_indices = {k: i for (i,k) in enumerate(self.operators)};
        
        # Data dimension = number of symbols (including EOS)
        self.data_dim = self.digits_range + len(symbols) + 1;
        if (single_digit):
            # Remove EOS
            self.data_dim -= 1;
        # We predict the same symbols as we have as input, so input and data
        # dimension are equal
        self.output_dim = self.data_dim;
        self.EOS_symbol_index = self.data_dim-1;
        
        # Store locations and sizes for both train and testing
        self.locations = [0, 0];
        self.lengths = [self.filelength(self.sources[self.TRAIN]), self.filelength(self.sources[self.TEST])];
        if (self.max_training_size is not False):
            self.lengths[self.TRAIN] = self.max_training_size;
        if (self.max_testing_size is not False):
            self.lengths[self.TEST] = self.max_testing_size;
        # Set test batch settings
        self.train_done = False;
        self.test_done = False;
        
        if (predictExpressions):
            # We need to save all expressions by answer for the fast lookup of
            # nearest labels
            self.preload(onlyStoreLookupByInput=True);
        else:
            self.outputByInput = None;
        
        if (preload):
            self.preloaded = self.preload();
            if (not self.preloaded):
                print("WARNING! PRELOADING DATASET WAS ATTEMPTED BUT FAILED!");
        else:
            self.preloaded = False;
    
    def preload(self, onlyStoreLookupByInput=False):
        """
        Preloads the dataset into memory. If onlyStoreLookupByInput is True,
        the data is not stored but only saved into the dictionary outputByInput
        which can be used to lookup all outputs for a given unique input. In
        this case only the training data is stored.
        """
        train, train_targets, train_labels, train_expressions = \
            self.loadFile(self.sources[self.TRAIN], 
                          location_index=self.TRAIN, 
                          file_length=self.lengths[self.TRAIN]);
        
        if (onlyStoreLookupByInput):
            self.outputByInput = {};
            for expression in train_expressions:
                left_hand_expression, right_hand_expression = expression.split("=");
                if (left_hand_expression not in self.outputByInput):
                    self.outputByInput[left_hand_expression] = [];
                self.outputByInput[left_hand_expression].append(right_hand_expression);
        else:
            test, test_targets, test_labels, test_expressions = \
            self.loadFile(self.sources[self.TEST], 
                          location_index=self.TEST, 
                          file_length=self.lengths[self.TEST]);
            self.train, self.train_targets, self.train_labels, self.train_expressions =\
                train, train_targets, train_labels, train_expressions;
            self.test, self.test_targets, self.test_labels, self.test_expressions =\
                test, test_targets, test_labels, test_expressions;
        
        
        return True;
    
    def filelength(self, source):
        f = open(source, 'r');
        length = 0;
        line = f.readline();
        while (line != ""):
            length += 1;
            line = f.readline();
        
        return length;
    
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
            if (line == ""):
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
    
    def fill_ndarray(self, data, axis):
        if (axis <= 0):
            raise ValueError("Max length axis cannot be the first axis!");
        if (len(data) == 0):
            raise ValueError("Data is empty!");
        max_length = max(map(lambda a: a.shape[axis-1], data));
        nd_data = np.zeros((len(data), max_length, self.data_dim));
        for i,datapoint in enumerate(data):
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
    
    def get_test_batch(self):
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
        if (self.sample_testing_size is not False):
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
            if (self.sample_testing_size is not False):
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
        i = np.random.randint(0,len(expression));
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
    
    def processSamplePredictExpression(self, line, data, targets, labels, expressions):
        line = line.strip();
        right_hand_start = line.find('=')+1;
        
        # Simply swap left and right hand side and use the multi-digit processor
        line = line[right_hand_start:] + "=" + line[:right_hand_start-1];
        
        return self.processSampleMultiDigit(line, data, targets, labels, expressions);
    
    def processSampleFillX(self, line, data, targets, labels, expressions):
        line = line.strip();
        left, right = line.split(";");
        return self.processSampleCorrection(right + ";" + left, data, targets, labels, expressions);
    
    def processSampleCorrection(self, line, data, targets, labels, expressions):
        line = line.strip();
        old_expression, new_expression = line.split(";");
        
        # Old expression = data
        old_expression_embeddings = np.zeros((len(old_expression)+1,self.data_dim));
        for i, literal in enumerate(old_expression):
            old_expression_embeddings[i,self.oneHot[literal]] = 1.0;
        
        # New expression = label/target
        new_expression_embeddings = np.zeros((len(new_expression)+1,self.data_dim));
        for i, literal in enumerate(new_expression):
            new_expression_embeddings[i,self.oneHot[literal]] = 1.0;
        
        # Add EOS's
        old_expression_embeddings[-1,self.EOS_symbol_index] = 1.0;
        new_expression_embeddings[-1,self.EOS_symbol_index] = 1.0;
        
        # Append data
        data.append(new_expression_embeddings);
        labels.append(np.argmax(old_expression_embeddings, axis=1));
        targets.append(old_expression_embeddings);
        expressions.append(new_expression);
        
        return data, targets, labels, expressions, 1;
    
    def findAnswer(self, onehot_encodings):
        answer_allzeros = map(lambda d: d.sum() == 0.0, onehot_encodings);
        if (all(map(lambda d: not d,answer_allzeros))):
            answer_length = onehot_encodings.shape[0];
        else:
            answer_length = answer_allzeros.index(True) - 1;
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
    
    def encodeExpression(self, structure):
        str_repr = str(structure);
        data = np.zeros((len(str_repr),self.data_dim));
        for i,symbol in enumerate(str_repr):
            data[i,self.oneHot[symbol]] = 1.0;
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
    
    def createDigit(self, maxIntValue):
        # Create terminal child (digit)
        self.nodeType = self.TYPE_DIGIT;
        self.value = np.random.randint(maxIntValue);
    
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
