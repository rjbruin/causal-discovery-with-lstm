'''
Created on 5 oct. 2016

@author: Robert-Jan
'''

import os;
import numpy as np;

SYMBOLS = [str(i) for i in range(10)] + ['A','B','C','D','E','F'];

OPERATOR_SYMBOLS = ['+','-','*'];
OPERATORS = [lambda x, y, max: (x+y) % max,
             lambda x, y, max: (x-y) % max,
             lambda x, y, max: (x*y) % max];

TO_VALUE = lambda x: int(x) if ord(x) < 65 else ((ord(x) - 65) + 10);

def _mutator_digit_copy(digit_in, top, bot):
    """
    Only mutates in odd (AKA legal) positions, which is the position of the
    randomly generated argument to the operator.
    """
    for i in range(2,len(top),3):
        symbol = top[i];
        if (symbol == SYMBOLS[digit_in]):
            argument1 = TO_VALUE(bot[i-2]);
            op = OPERATOR_SYMBOLS.index(bot[i-1]);
            argument2 = TO_VALUE(top[i]);
            result = OPERATORS[op](argument1,argument2,max_digits);
            bot = finishSeq(bot[:i] + symbol + SYMBOLS[result], len(bot), result);
    return bot;

def _mutator_digit_change(digit_in, digit_out, top, bot):
    """
    Only mutates in odd (AKA legal) positions, which is the position of the
    randomly generated argument to the operator.
    """
    for i in range(2,len(top),3):
        symbol = top[i];
        if (symbol == SYMBOLS[digit_in]):
            argument1 = TO_VALUE(bot[i-2]);
            op = OPERATOR_SYMBOLS.index(bot[i-1]);
            argument2 = TO_VALUE(SYMBOLS[digit_out]);
            result = OPERATORS[op](argument1,argument2,max_digits);
            bot = finishSeq(bot[:i] + symbol + SYMBOLS[result], len(bot), result);
    return bot;

def finishSeq(seq, length, result):
    if (len(seq) < length):
        # Draw operators
        op = np.random.randint(max_ops);
        seq += OPERATOR_SYMBOLS[op];
        # Draw argument
        arg_digit = np.random.randint(max_digits);
        seq += SYMBOLS[arg_digit];
        # Compute
        result = OPERATORS[op](result,arg_digit,max_digits);
        seq += SYMBOLS[result];
        return finishSeq(seq, length, result);
    else:
        return seq;

def createSeq(length):
    digit_seed = np.random.randint(max_digits);
    result = digit_seed;
    seq = SYMBOLS[digit_seed];
    return finishSeq(seq, length, result);

def createSample(min_length, max_length,
                 top_to_bot_mutators, bot_to_top_mutators):
    # Create top and bottom sequences apart from each other
    length = np.random.randint(min_length, max_length+1);
    top = createSeq(length);
    bot = createSeq(length);

    # Add in causal mutations from cause to effect
    topcause = np.random.randint(2) == 1;
    if (topcause):
        for mutator in top_to_bot_mutators:
            bot = mutator(top, bot);
    else:
        for mutator in bot_to_top_mutators:
            top = mutator(bot, top);
    return top, bot, topcause;

def generateSequences(baseFilePath, n, test_percentage,
                      top_to_bot_mutators, bot_to_top_mutators,
                      min_length=5,
                      max_length=21, verbose=False):
    savedSequences = {};
    sequential_fails = 0;
    fail_limit = 1000000000;

    print("Generating expressions...");
    while len(savedSequences) < n and sequential_fails < fail_limit:
        top, bot, topcause = createSample(min_length, max_length,
                                          top_to_bot_mutators,
                                          bot_to_top_mutators);
        full_seq = top + ";" + bot + ";" + str(int(topcause));
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
    folder = 'seq2ndmarkov';
    test_size = 0.10;
    n = 10000;
    max_digits = 10;
    max_ops = 3;
    top_to_bot_mutators = [lambda top, bot: _mutator_digit_copy(8, top, bot),
                           lambda top, bot: _mutator_digit_change(7, 6, top, bot)];
    bot_to_top_mutators = [lambda bot, top: _mutator_digit_copy(5, bot, top),
                           lambda bot, top: _mutator_digit_change(4, 3, bot, top)];
    min_length = 8;
    max_length = 15;
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
        generateSequences(folder, n, test_size,
                          top_to_bot_mutators, bot_to_top_mutators,
                          min_length=min_length, max_length=max_length,
                          verbose=verbose);
