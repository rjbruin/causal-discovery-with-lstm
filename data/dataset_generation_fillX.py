'''
Created on 24 jun. 2016

@author: Robert-Jan
'''

import sys;
import numpy as np;

if __name__ == '__main__':
    # Set default parameters
    copies_per_expression = 2;
    only_change_digits = False;
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
            expression = line.strip();
            left_hand_size = expression.index('=');
            old_expression = expression;
            
            for i in range(copies_per_expression):
                # Find symbol to change
                index_to_change = np.random.randint(0,left_hand_size);
                symbol_to_change = old_expression[index_to_change];
                while (only_change_digits and symbol_to_change not in map(str,range(10))):
                    index_to_change = np.random.randint(0,left_hand_size);
                    symbol_to_change = old_expression[index_to_change];
                
                # Write new symbol
                new_symbol = 'x';
                expression = old_expression[:index_to_change] + new_symbol + old_expression[index_to_change+1:];
            
                # Write expression
                new_expressions.append((expression,old_expression));
            
            # Read next line
            line = f.readline();
        
        # Shuffle and write new expressions
        shuffle_order = range(len(new_expressions))
        np.random.shuffle(shuffle_order);
        for i in shuffle_order:
            withX, fixed = new_expressions[i]
            f_out.write("%s;%s\n" % (withX, fixed));
        
        f.close();
        f_out.close();