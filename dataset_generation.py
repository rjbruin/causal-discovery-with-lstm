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

if __name__ == '__main__':
    # http://stackoverflow.com/questions/273192/in-python-check-if-a-directory-exists-and-create-it-if-necessary
    if (not os.path.exists('data/expressions-one_digit_answer')):
        #raise ValueError("Dataset already present");
        os.makedirs('data/expressions-one_digit_answer');
    if (os.path.exists('data/expressions-one_digit_answer/test.txt')):
        raise ValueError("Dataset already present");
    
    n = 20000; # Amount of expressions to generate
    
    filters = [lambda x: x.getValue() % 1.0 == 0, # Value must be integer
               lambda x: x.getValue() < 10, # Value must be single-digit
               lambda x: x.getValue() >= 0
               ];
    
    savedExpressions = [];
    while len(savedExpressions) < n:
        expression = ExpressionNode(0, 1, 2);
        #print(str(expression) + " = " + str(expression.getValue()));
        fail = False;
        for f_i, filterFunction in enumerate(filters):
            if (not filterFunction(expression)):
                #print("\tFails filter %d" % f_i);
                fail = True;
                break;
        if (not fail):
            savedExpressions.append(str(expression) + "=" + str(int(expression.getValue())));    
    
    f = open('data/expressions-one_digit_answer/test.txt','w');
    f.write("\n".join(savedExpressions));
    f.close();
