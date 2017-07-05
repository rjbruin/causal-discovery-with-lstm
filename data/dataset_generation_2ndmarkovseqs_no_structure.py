'''
Created on 14 oct. 2016

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
            bot = bot[:i] + symbol + bot[i+1:];
    return bot;

def _mutator_digit_change(digit_in, digit_out, top, bot):
    """
    Only mutates in odd (AKA legal) positions, which is the position of the
    randomly generated argument to the operator.
    """
    for i in range(2,len(top),3):
        symbol = top[i];
        if (symbol == SYMBOLS[digit_in]):
            bot = bot[:i] + SYMBOLS[digit_out] + bot[i+1:];
    return bot;

def _mutator_digit_change_2ndorder(digit_in_1, digit_in_2, digit_out, top, bot):
    for i in range(2,len(top),3):
        symbol_1 = top[i-1];
        if (symbol_1 == SYMBOLS[digit_in_1]):
            symbol_2 = top[i];
            if (symbol_2 == SYMBOLS[digit_in_2]):
                bot = bot[:i] + SYMBOLS[digit_out] + bot[i+1:];
    return bot;

def finishSeq(seq, length, result):
    if (len(seq) < length):
        # Draw new digit
        arg_digit = np.random.randint(max_digits);
        seq += SYMBOLS[arg_digit];
        return finishSeq(seq, length, result);
    else:
        return seq;

def createSeq(length):
    digit_seed = np.random.randint(max_digits);
    result = digit_seed;
    seq = SYMBOLS[digit_seed];
    return finishSeq(seq, length, result);

def createSample(min_length, max_length,
                 top_to_bot_mutators, bot_to_top_mutators, bothways=False):
    # Create top and bottom sequences apart from each other
    length = np.random.randint(min_length, max_length+1);
    top = createSeq(length);
    bot = createSeq(length);

    # Add in causal mutations from cause to effect
    topcause = np.random.randint(2) == 1;
    for i in range(len(top)):
        if (topcause or bothways):
            symbolVal = TO_VALUE(top[i])
            if (symbolVal in top_to_bot_mutators):
                mutator = top_to_bot_mutators[symbolVal];
                bot = mutator(top, bot);
        if (not topcause or bothways):
            symbolVal = TO_VALUE(bot[i]);
            if (symbolVal in bot_to_top_mutators):
                mutator = bot_to_top_mutators[symbolVal];
                top = mutator(bot, top);
    return top, bot, topcause;

def generateSequences(baseFilePath, n, test_percentage,
                      top_to_bot_mutators, bot_to_top_mutators,
                      min_length=5,
                      max_length=21, verbose=False,
                      bothways=False):
    savedSequences = {};
    sequential_fails = 0;
    fail_limit = 1000000000;

    print("Generating expressions...");
    while len(savedSequences) < n and sequential_fails < fail_limit:
        top, bot, topcause = createSample(min_length, max_length,
                                          top_to_bot_mutators,
                                          bot_to_top_mutators,
                                          bothways=bothways);
        if (not bothways):
            full_seq = top + ";" + bot + ";" + str(int(topcause));
        else:
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

    folder = 'seq2ndmarkov_nostructure';
    n = 1000000;
    max_digits = 14;
    max_ops = 0;
    top_to_bot_mutators = {6: [lambda top, bot: _mutator_digit_copy(6, top, bot)],
                           1: [lambda top, bot: _mutator_digit_copy(1, top, bot)],
                           3: [lambda top, bot: _mutator_digit_change(3, 5, top, bot)],
                           0: [lambda top, bot: _mutator_digit_change(0, 3, top, bot)],
                           12: [lambda top, bot: _mutator_digit_change(12, 4, top, bot)],
                           13: [lambda top, bot: _mutator_digit_change(13, 0, top, bot)],
                           2: [lambda top, bot: _mutator_digit_change_2ndorder(10, 2, 3, top, bot),
                               lambda top, bot: _mutator_digit_change_2ndorder(11, 2, 1, top, bot)],
                           4: [lambda top, bot: _mutator_digit_change_2ndorder(10, 4, 5, top, bot),
                               lambda top, bot: _mutator_digit_change_2ndorder(11, 4, 3, top, bot)]};
    bot_to_top_mutators = {7: [lambda bot, top: _mutator_digit_copy(7, bot, top)],
                           2: [lambda bot, top: _mutator_digit_copy(2, bot, top)],
                           1: [lambda bot, top: _mutator_digit_change(1, 7, bot, top)],
                           0: [lambda bot, top: _mutator_digit_change(0, 4, bot, top)],
                           10: [lambda top, bot: _mutator_digit_change(10, 4, top, bot)],
                           11: [lambda top, bot: _mutator_digit_change(11, 0, top, bot)],
                           5: [lambda bot, top: _mutator_digit_change_2ndorder(12, 5, 7, bot, top),
                               lambda bot, top: _mutator_digit_change_2ndorder(11, 5, 4, bot, top)],
                           6: [lambda bot, top: _mutator_digit_change_2ndorder(12, 6, 8, bot, top),
                               lambda bot, top: _mutator_digit_change_2ndorder(11, 6, 5, bot, top)]};
    min_length = 11;
    max_length = 16;
    verbose = False;
    bothways = False;

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
                          verbose=verbose, bothways=bothways);
