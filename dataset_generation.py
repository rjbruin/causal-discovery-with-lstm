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
    
    operators = range(OPERATOR_SIZE);
    
    def __init__(self, nodeType, value):
        self.nodeType = nodeType;
        self.value = value;
    
    @staticmethod
    def allExpressions(currentRecursionDepth=0, minRecursionDepth=1, maxRecursionDepth=10, maxIntValue=10):
        expressions = [];
        
        nodes = True;
        if (currentRecursionDepth >= maxRecursionDepth):
            nodes = False;
        
        leafs = True;
        if (currentRecursionDepth < minRecursionDepth):
            leafs = False;
        
        # Generate all expressions that end here (leaf)
        if (leafs):
            for i in range(maxIntValue):
                leaf = ExpressionNode(ExpressionNode.TYPE_DIGIT, i);
                expressions.append(leaf);
        
        if (nodes):
            # Generate all expressions that continue (node)
            possible_children = ExpressionNode.allExpressions(currentRecursionDepth+1, minRecursionDepth, maxRecursionDepth, maxIntValue);
            for i in range(ExpressionNode.OPERATOR_SIZE):
                for left_child in possible_children:
                    for right_child in possible_children:
                        # Create no expressions that divide by zero 
                        if (i == ExpressionNode.OP_DIVIDE and right_child.getValue() == 0.0):
                            continue;
                        node = ExpressionNode(ExpressionNode.TYPE_OPERATOR, i);
                        node.left = left_child;
                        node.right = right_child;
                        expressions.append(node);
        
        return expressions;
    
    @staticmethod
    def randomExpression(currentRecursionDepth=0, minRecursionDepth=1, maxRecursionDepth=10, terminalProb=0.5, maxIntValue=10):
        """
        Alternative constructor that generates a random expression.
        """
        node = ExpressionNode(0,0);
        
        if (currentRecursionDepth < maxRecursionDepth):
            if (currentRecursionDepth >= minRecursionDepth and np.random.random() < terminalProb):
                # Create terminal child (digit) with probability 'terminalProb'
                node.createDigit(maxIntValue);
            else:
                # Create operator
                node.nodeType = ExpressionNode.TYPE_OPERATOR;
                node.value = np.random.randint(ExpressionNode.OPERATOR_SIZE);
                # Create children
                node.left = ExpressionNode.randomExpression(currentRecursionDepth+1,minRecursionDepth,maxRecursionDepth,terminalProb,maxIntValue);
                node.right = ExpressionNode.randomExpression(currentRecursionDepth+1,minRecursionDepth,maxRecursionDepth,terminalProb,maxIntValue);
                while (node.value == ExpressionNode.OP_DIVIDE and node.right.getValue() == 0.0):
                    # Replace right-hand while it is zero
                    node.right = ExpressionNode.randomExpression(currentRecursionDepth+1,minRecursionDepth,maxRecursionDepth,terminalProb,maxIntValue);
        else:
            # Create terminal child (digit) if we are at maximum expression depth
            node.createDigit(maxIntValue);
        
        return node;
    
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

def generateExpressions(baseFilePath, n, test_percentage, filters, minRecursionDepth=1, maxRecursionDepth=2, terminalProb=0.5, verbose=False):
    savedExpressions = {};
    sequential_fails = 0;
    fail_limit = 100000;
    while len(savedExpressions) < n and sequential_fails < fail_limit:
        expression = ExpressionNode.randomExpression(0, minRecursionDepth, maxRecursionDepth, terminalProb);
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
    
    writeToFiles(savedExpressions, baseFilePath, test_percentage);
    
def generateAllExpressions(baseFilePath, test_percentage, filters, minRecursionDepth=1, maxRecursionDepth=2, maxIntValue=10):
    print("Generating expressions for level 2 and deeper...");
    expressions_1 = ExpressionNode.allExpressions(1, minRecursionDepth=minRecursionDepth, maxRecursionDepth=maxRecursionDepth, maxIntValue=maxIntValue);
    
    print("Generating expressions for level 1");
    expressions = [];
    for op in ExpressionNode.operators:
        for right_child in expressions_1:
            if (right_child.getValue() == 0.0 and op == ExpressionNode.OP_DIVIDE):
                continue;
            for left_child in expressions_1:
                exp = ExpressionNode(ExpressionNode.TYPE_OPERATOR, op);
                exp.left = left_child;
                exp.right = right_child;
                expressions.append(exp);
    
    print("Filtering expressions...");
    filteredExpressions = []
    for expression in expressions:
        discard = False;
        for filter in filters:
            discard = not filter(expression);
            if (discard):
                break;
        if (not discard):
            full_expression = str(expression) + "=" + str(int(expression.getValue()));
            filteredExpressions.append(full_expression);
    
    print("Shuffling expressions...");
    
    # Shuffle expressions
    np.random.shuffle(filteredExpressions);
    
    print("Writing to files...");
    
    writeToFiles(filteredExpressions, baseFilePath, test_percentage, isList=True);

def writeToFiles(expressions,baseFilePath,test_percentage,isList=False):
    # Define train/test split
    train_n = int(len(expressions) - (len(expressions) * test_percentage));
    
    if (not isList):
        expressions = expressions.keys();
    
    # Generate training file
    f = open(baseFilePath + '/train.txt','w');
    f.write("\n".join(expressions[:train_n]));
    f.close();
    
    # Generate training file
    f = open(baseFilePath + '/test.txt','w');
    f.write("\n".join(expressions[train_n:]));
    f.close();

if __name__ == '__main__':
    # Settings
    folder = 'data/test';
    test_size = 0.10;
    currentRecursionDepth = 0;
    minRecursionDepth = 1;
    maxRecursionDepth = 2;
    maxIntValue = 10;
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
    
    generateAllExpressions(folder, test_size, filters, minRecursionDepth, maxRecursionDepth, maxIntValue);
