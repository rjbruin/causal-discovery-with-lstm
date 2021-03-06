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

def _mutator_digit_copy(digit_in, top, bot, start):
    """
    Only mutates in odd (AKA legal) positions, which is the position of the
    randomly generated argument to the operator.
    """
    for i in range(start,len(top),3):
        symbol = top[i];
        if (symbol == SYMBOLS[digit_in]):
            argument1 = TO_VALUE(bot[i-2]);
            op = OPERATOR_SYMBOLS.index(bot[i-1]);
            argument2 = TO_VALUE(top[i]);
            result = OPERATORS[op](argument1,argument2,max_digits);
            bot = finishSeq(bot[:i] + symbol + SYMBOLS[result], len(bot), result);
    return bot;

def _mutator_digit_change(digit_in, digit_out, top, bot, start):
    """
    Only mutates in odd (AKA legal) positions, which is the position of the
    randomly generated argument to the operator.
    """
    for i in range(start,len(top),3):
        symbol = top[i];
        if (symbol == SYMBOLS[digit_in]):
            argument1 = TO_VALUE(bot[i-2]);
            op = OPERATOR_SYMBOLS.index(bot[i-1]);
            argument2 = TO_VALUE(SYMBOLS[digit_out]);
            result = OPERATORS[op](argument1,argument2,max_digits);
            bot = finishSeq(bot[:i] + SYMBOLS[digit_out] + SYMBOLS[result], len(bot), result);
    return bot;

def _mutator_digit_change_next(digit_in, digit_out, top, bot, start):
    """
    Only mutates in odd (AKA legal) positions, which is the position of the
    randomly generated argument to the operator.
    """
    for i in range(start,len(top),3):
        symbol = top[i-2];
        if (symbol == SYMBOLS[digit_in]):
            argument1 = TO_VALUE(bot[i-2]);
            op = OPERATOR_SYMBOLS.index(bot[i-1]);
            argument2 = TO_VALUE(SYMBOLS[digit_out]);
            result = OPERATORS[op](argument1,argument2,max_digits);
            bot = finishSeq(bot[:i] + SYMBOLS[digit_out] + SYMBOLS[result], len(bot), result);
    return bot;

def _mutator_digit_change_2ndorder(digit_in_1, digit_in_2, digit_out, top, bot, start):
    for i in range(start,len(top),3):
        symbol_1 = top[i-1];
        if (symbol_1 == SYMBOLS[digit_in_1]):
            symbol_2 = top[i];
            if (symbol_2 == SYMBOLS[digit_in_2]):
                argument1 = TO_VALUE(bot[i-2]);
                op = OPERATOR_SYMBOLS.index(bot[i-1]);
                argument2 = TO_VALUE(SYMBOLS[digit_out]);
                result = OPERATORS[op](argument1,argument2,max_digits);
                bot = finishSeq(bot[:i] + SYMBOLS[digit_out] + SYMBOLS[result], len(bot), result);
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

def createSample(length, top_to_bot_mutators, bot_to_top_mutators, bothways=False, nextMutators=True):
    # Create top and bottom sequences apart from each other
    top = createSeq(length);
    bot = createSeq(length);

    if (nextMutators):
        mutator_trigger_offset = -2;
    else:
        mutator_trigger_offset = 0;

    # Add in causal mutations from cause to effect
    topcause = np.random.randint(2) == 1;
    for i in range(2,len(top),3):
        if (topcause or bothways):
            symbolVal = TO_VALUE(top[i+mutator_trigger_offset])
            if (symbolVal in top_to_bot_mutators):
                mutator = top_to_bot_mutators[symbolVal];
                bot = mutator(top, bot, i);
        if (not topcause or bothways):
            symbolVal = TO_VALUE(bot[i+mutator_trigger_offset]);
            if (symbolVal in bot_to_top_mutators):
                mutator = bot_to_top_mutators[symbolVal];
                top = mutator(bot, top, i);
    return top, bot, topcause;

def generateSequences(baseFilePath, n, test_percentage,
                      top_to_bot_mutators, bot_to_top_mutators,
                      length=15, verbose=False,
                      bothways=False):
    savedSequences = {};
    sequential_fails = 0;
    fail_limit = 1000000000;

    print("Generating expressions...");
    while len(savedSequences) < n and sequential_fails < fail_limit:
        top, bot, topcause = createSample(length,
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

    # folder = 'seq2ndmarkov_fixed';
    # n = 1000000;
    # max_digits = 10;
    # max_ops = 3;
    # top_to_bot_mutators = [lambda top, bot, start: _mutator_digit_copy(8, top, bot, start),
    #                        lambda top, bot, start: _mutator_digit_change(7, 6, top, bot, start)];
    # bot_to_top_mutators = [lambda bot, top, start: _mutator_digit_copy(5, bot, top, start),
    #                        lambda bot, top, start: _mutator_digit_change(4, 3, bot, top, start)];
    # length = 15;
    # verbose = False;
    # bothways = False;

    # folder = 'seq2ndmarkov_2_fixed';
    # n = 1000000;
    # max_digits = 8;
    # max_ops = 2;
    # top_to_bot_mutators = {6: lambda top, bot, start: _mutator_digit_copy(6, top, bot, start),
    #                        1: lambda top, bot, start: _mutator_digit_copy(1, top, bot, start),
    #                        3: lambda top, bot, start: _mutator_digit_change(3, 5, top, bot, start),
    #                        0: lambda top, bot, start: _mutator_digit_change(0, 3, top, bot, start),
    #                        2: lambda top, bot, start: _mutator_digit_change(2, 0, top, bot, start)};
    # bot_to_top_mutators = {7: lambda bot, top, start: _mutator_digit_copy(7, bot, top, start),
    #                        2: lambda bot, top, start: _mutator_digit_copy(2, bot, top, start),
    #                        4: lambda bot, top, start: _mutator_digit_change(4, 3, bot, top, start),
    #                        1: lambda bot, top, start: _mutator_digit_change(1, 7, bot, top, start),
    #                        0: lambda bot, top, start: _mutator_digit_change(0, 4, bot, top, start)};
    # length = 15;
    # verbose = False;
    # bothways = False;

    # folder = 'seq2ndmarkov_both_fixed';
    # n = 1000000;
    # max_digits = 8;
    # max_ops = 2;
    # top_to_bot_mutators = {6: lambda top, bot, start: _mutator_digit_copy(6, top, bot, start),
    #                        1: lambda top, bot, start: _mutator_digit_copy(1, top, bot, start),
    #                        3: lambda top, bot, start: _mutator_digit_change(3, 5, top, bot, start),
    #                        0: lambda top, bot, start: _mutator_digit_change(0, 3, top, bot, start),
    #                        2: lambda top, bot, start: _mutator_digit_change(2, 0, top, bot, start)};
    # bot_to_top_mutators = {7: lambda bot, top, start: _mutator_digit_copy(7, bot, top, start),
    #                        2: lambda bot, top, start: _mutator_digit_copy(2, bot, top, start),
    #                        4: lambda bot, top, start: _mutator_digit_change(4, 3, bot, top, start),
    #                        1: lambda bot, top, start: _mutator_digit_change(1, 7, bot, top, start),
    #                        0: lambda bot, top, start: _mutator_digit_change(0, 4, bot, top, start)};
    # length = 15;
    # verbose = False;
    # bothways = True;

    folder = 'seq2ndmarkov_both_2';
    n = 1000000;
    max_digits = 8;
    max_ops = 2;
    top_to_bot_mutators = {6: lambda top, bot, start: _mutator_digit_change_next(6, 6, top, bot, start),
                           1: lambda top, bot, start: _mutator_digit_change_next(1, 1, top, bot, start),
                           3: lambda top, bot, start: _mutator_digit_change_next(3, 5, top, bot, start),
                           0: lambda top, bot, start: _mutator_digit_change_next(0, 3, top, bot, start),
                           2: lambda top, bot, start: _mutator_digit_change_next(2, 0, top, bot, start)};
    bot_to_top_mutators = {7: lambda bot, top, start: _mutator_digit_change_next(7, 7, bot, top, start),
                           2: lambda bot, top, start: _mutator_digit_change_next(2, 2, bot, top, start),
                           4: lambda bot, top, start: _mutator_digit_change_next(4, 3, bot, top, start),
                           1: lambda bot, top, start: _mutator_digit_change_next(1, 7, bot, top, start),
                           0: lambda bot, top, start: _mutator_digit_change_next(0, 4, bot, top, start)};
    length = 15;
    verbose = False;
    bothways = True;

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
                          length=length,
                          verbose=verbose, bothways=bothways);
