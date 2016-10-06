'''
Created on 24 jun. 2016

@author: Robert-Jan
'''

import sys;
import numpy as np;

# str_digits = [str(i) for i in range(10)];
# operators = ['+','-','*','/'];
str_digits = [str(i) for i in range(10)];
operators = ['+','-','*'];

def changeSimpleBracket(symbol):
    # Don't change brackets
    return symbol;

def changeSimpleOperator(symbol):
    # Change operator to next operator in list
    currentSymbolIndex = operators.index(symbol);
    newSymbolIndex = (currentSymbolIndex + 1) % len(operators);
    return operators[newSymbolIndex];

def changeSimpleDigit(symbol):
    # Change digit to next digit in list
    currentSymbolIndex = str_digits.index(symbol);
    newSymbolIndex = (currentSymbolIndex + 1) % len(str_digits);
    return str_digits[newSymbolIndex];

def changeSimple(expression):
    newExpression = "";
    for symbol in expression:
        if (symbol in str_digits):
            newExpression += changeSimpleDigit(symbol);
        elif (symbol in operators):
            newExpression += changeSimpleOperator(symbol);
        else:
            newExpression += changeSimpleBracket(symbol);
    return newExpression;

if __name__ == '__main__':
    # Set default parameters
    copies_per_expression = 1;
    seed = 1;

    # Read parameters
    source_dataset = sys.argv[1];
    destination_dataset = sys.argv[2];
    if (len(sys.argv) > 3):
        copies_per_expression = int(sys.argv[3]);

    # Get and set seed
    seed = int(raw_input("Please input an integer seed for the random number generator:"));
    np.random.seed(seed);

    for filename in ['train.txt','test.txt']:
        f = open(source_dataset + '/' + filename, 'r');
        f_out = open(destination_dataset + '/' + filename, 'w');

        # Read expressions and save new expressions
        new_expressions = [];
        line = f.readline();
        while (line != ""):
            # Read expression
            expression = line;

            for i in range(copies_per_expression):
                #topcause = np.random.randint(2) == 1;
                topcause = True;
                if (topcause):
                    expression = expression.strip();
                    expression_prime = changeSimple(expression);
                else:
                    # TODO
                    expression_prime = expression.strip();
                    expression = changeSimple(expression_prime);

                # Write expression
                new_expressions.append((expression,expression_prime));

            # Read next line
            line = f.readline();

        # Shuffle and write new expressions
        shuffle_order = range(len(new_expressions))
        np.random.shuffle(shuffle_order);
        for i in shuffle_order:
            old, new = new_expressions[i]
            f_out.write("%s;%s\n" % (old, new));

        f.close();
        f_out.close();
