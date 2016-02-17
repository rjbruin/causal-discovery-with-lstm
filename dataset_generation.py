'''
Created on 15 feb. 2016

@author: Robert-Jan
'''

import os;
import numpy as np;

class ExpressionNode(object):
    
    TYPE_DIGIT = 0;
    TYPE_OPERATOR = 1;
    
    OP_PLUS = 0;
    OP_MINUS = 1;
    OP_MULTIPLY = 2;
    OP_DIVIDE = 3;
    OPERATOR_SIZE = 4;
    
    def __init__(self, currentRecursionDepth=0, minRecursionDepth=1, maxRecursionDepth=10, terminalProb=0.5, maxIntValue=10):
        if (currentRecursionDepth < maxRecursionDepth):
            if (currentRecursionDepth >= minRecursionDepth and np.random.random() < terminalProb):
                # Create terminal child (digit) with probability 'terminalProb'
                self.createDigit(maxIntValue);
            else:
                # Create operator
                self.nodeType = self.TYPE_OPERATOR;
                self.value = np.random.randint(self.OPERATOR_SIZE);
                # Create children
                self.left = ExpressionNode(currentRecursionDepth+1,minRecursionDepth,maxRecursionDepth,terminalProb,maxIntValue);
                self.right = ExpressionNode(currentRecursionDepth+1,minRecursionDepth,maxRecursionDepth,terminalProb,maxIntValue);
                while (self.value == self.OP_DIVIDE and self.right.getValue() == 0.0):
                    # Replace right-hand while it is zero
                    self.right = ExpressionNode(currentRecursionDepth+1,minRecursionDepth,maxRecursionDepth,terminalProb,maxIntValue);
        else:
            # Create terminal child (digit) if we are at maximum expression depth
            self.createDigit(maxIntValue);
    
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

def generateExpressions(baseFilePath, n, test_percentage, filters, currentRecursionDepth=0, minRecursionDepth=1, maxRecursionDepth=2, terminalProb=0.5, verbose=False):
    savedExpressions = {};
    sequential_fails = 0;
    fail_limit = 100000;
    while len(savedExpressions) < n and sequential_fails < fail_limit:
        expression = ExpressionNode(currentRecursionDepth, minRecursionDepth, maxRecursionDepth, terminalProb);
        full_expression = str(expression) + "=" + str(int(expression.getValue()));
        # Check if expression already exists
        if (full_expression in savedExpressions):
            sequential_fails += 1;
            continue;
        if (verbose):
            print(str(expression) + " = " + str(expression.getValue()));
        fail = False;
        for f_i, filterFunction in enumerate(filters):
            if (not filterFunction(expression)):
                if (verbose):
                    print("\tFails filter %d" % f_i);
                fail = True;
                break;
        if (not fail):
            savedExpressions[full_expression] = True;    
    
    # Define train/test split
    train_n = int(len(savedExpressions) - (len(savedExpressions) * test_percentage));
    
    # Generate training file
    f = open(baseFilePath + '/train.txt','w');
    f.write("\n".join(savedExpressions.keys()[:train_n]));
    f.close();
    
    # Generate training file
    f = open(baseFilePath + '/test.txt','w');
    f.write("\n".join(savedExpressions.keys()[train_n:]));
    f.close();

if __name__ == '__main__':
    # Settings
    folder = 'data/expressions_one_digit_answer';
    train_size = 100000; # Hundred thousand
    test_size = 0.10; # Twenty thousand
    currentRecursionDepth = 0;
    minRecursionDepth = 1;
    maxRecursionDepth = 2;
    terminalProb = 0.5;
    filters = [lambda x: x.getValue() % 1.0 == 0, # Value must be integer
               lambda x: x.getValue() < 10, # Value must be single-digit
               lambda x: x.getValue() >= 0
               ];
    
    # Generate other variables
    trainFilePath = folder + '/train.txt';
    testFilePath = folder + '/test.txt'
    
    # http://stackoverflow.com/questions/273192/in-python-check-if-a-directory-exists-and-create-it-if-necessary
    if (not os.path.exists(folder)):
        os.makedirs(folder);
    if (os.path.exists(trainFilePath)):
        raise ValueError("Train part of dataset already present");
    if (os.path.exists(testFilePath)):
        raise ValueError("Test part of dataset already present");
    
    generateExpressions(folder,train_size,test_size,filters,currentRecursionDepth,
                 minRecursionDepth, maxRecursionDepth, terminalProb);
