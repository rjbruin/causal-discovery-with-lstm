'''
Created on 24 jun. 2016

@author: Robert-Jan
'''

import sys;
import numpy as np;

if __name__ == '__main__':
    # Set internal constants
    new_symbols = map(str,range(10)) + ['+','-','*','/','(',')'];
    
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
            left_hand_size = expression.index('=');
            
            for i in range(copies_per_expression):
                old_expression = expression.strip();
                
                # Find symbol to change
                index_to_change = np.random.randint(0,left_hand_size);
                symbol_to_change = expression[index_to_change];
                
                # Choose new symbol
                new_symbol = new_symbols[np.random.randint(0,len(new_symbols))];
                while (new_symbol == symbol_to_change):
                    new_symbol = new_symbols[np.random.randint(0,len(new_symbols))];
                
                # Write new symbol
                expression = expression[:index_to_change] + new_symbol + expression[index_to_change+1:];
            
                # Write expression
                new_expressions.append((old_expression,expression));
            
            # Read next line
            line = f.readline();
        
        # Shuffle and write new expressions
        shuffle_order = range(len(new_expressions))
        np.random.shuffle(shuffle_order);
        for i in shuffle_order:
            old, new = new_expressions[i]
            f_out.write("%s;%s" % (old, new));
        
        f.close();
        f_out.close();