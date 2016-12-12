'''
Created on 2 nov. 2016

@author: Robert-Jan
'''

import os;
import numpy as np;

OPERATOR_SYMBOLS = ['+','-','*'];
SYMBOLS = [str(i) for i in range(10)] + OPERATOR_SYMBOLS;

OPERATORS = [lambda x, y, max: (x+y) % max,
             lambda x, y, max: (x-y) % max,
             lambda x, y, max: (x*y) % max];

TO_VALUE = lambda x: int(x) if ord(x) < 65 else ((ord(x) - 65) + 10);

def createSample():
    # Create top and bottom sequences apart from each other
    length = np.random.randint(min_length, max_length+1);
    
    # Choose first digit arguments
    arg_top = np.random.randint(max_digits);
    arg_bot = np.random.randint(max_digits);
    while (arg_top == arg_bot):
        arg_bot = np.random.randint(max_digits);
    
    sample_top = SYMBOLS[arg_top];
    sample_bot = SYMBOLS[arg_bot];
    while (len(sample_top) < length):
        # Choose operators
        op_top = np.random.randint(max_ops);
        op_bot = np.random.randint(max_ops);
        # Compute results
        result_top = OPERATORS[op_top](arg_top, arg_bot, max_digits);
        result_bot = OPERATORS[op_bot](arg_top, arg_bot, max_digits);
        # Add to sequences
        sample_top += SYMBOLS[max_digits + op_top] + SYMBOLS[result_top];
        sample_bot += SYMBOLS[max_digits + op_bot] + SYMBOLS[result_bot];
        # Reset for next iteration
        arg_top = result_top;
        arg_bot = result_bot;
    
    return sample_top, sample_bot;

def generateSequences(baseFilePath, n, test_percentage):
    savedSequences = {};
    sequential_fails = 0;
    fail_limit = 1000000000;

    print("Generating expressions...");
    while len(savedSequences) < n and sequential_fails < fail_limit:
        top, bot = createSample();
        full_seq = top + ";" + bot;
        # Check if sequence already exists
        if (full_seq in savedSequences):
            sequential_fails += 1;
            continue;
        if (verbose):
            print(full_seq);
        savedSequences[full_seq] = True;

        if len(savedSequences) % (n/100) == 0:
            print("%.0f percent generated" % (len(savedSequences)*100/float(n)));

    writeToFiles(savedSequences, baseFilePath, test_percentage);

def writeToFiles(sequences,baseFilePath,test_percentage,isList=False):
    # Define train/test split
    train_n = int(len(sequences) - (len(sequences) * test_percentage));

    if (not isList):
        sequences = sequences.keys();

    # Generate training file
    f = open(baseFilePath + '/train.txt','w');
    f.write("\n".join(sequences[:train_n]));
    f.close();

    # Generate training file
    f = open(baseFilePath + '/test.txt','w');
    f.write("\n".join(sequences[train_n:]));
    f.close();

if __name__ == '__main__':
    # Settings
    test_size = 0.10;

    folder = 'doubleoperator';
    n = 1000000;
    max_digits = 10;
    max_ops = 3;
    min_length = 11;
    max_length = 16;
    verbose = False;

    # Generate other variables
    trainFilePath = folder + '/train.txt';
    testFilePath = folder + '/test.txt'

    # http://stackoverflow.com/questions/273192/in-python-check-if-a-directory-
    # exists-and-create-it-if-necessary
    if (not os.path.exists(folder)):
        os.makedirs(folder);

    stop = False;
    if (os.path.exists(trainFilePath)):
        inp = raw_input("Train part of dataset already present! Continue? ([y]/n) ");
        if inp != 'y':
            print("Terminated.");
            stop = True;
    if (os.path.exists(testFilePath)):
        inp = raw_input("Test part of dataset already present! Continue? ([y]/n) ");
        if inp != 'y':
            print("Terminated.");
            stop = True;

    if (not stop):
        generateSequences(folder, n, test_size);
