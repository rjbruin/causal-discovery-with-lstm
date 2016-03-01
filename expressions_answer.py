'''
Created on 16 feb. 2016

@author: Robert-Jan
'''

import time;
import sys, os;
import pickle;
import theano;
import numpy as np;

import model.RecurrentNeuralNetwork as rnn;
import model.GeneratedExpressionDataset as ge_dataset;

from statistic_tools import confusion_matrix;

#theano.config.mode = 'FAST_COMPILE'

def statistics(start, score, prediction_histogram=None, groundtruth_histogram=None, prediction_confusion_matrix=None, digit_score=None):
    output = "\n";

    # Print statistics
    duration = time.clock() - start;
    output += "Duration: %d seconds\n" % duration;
    output += "Score: %.2f percent\n" % (score*100);

    if (digit_score is not None):
        output += "Digit-based score: %.2f percent\n" % (digit_score*100);
    
    if (prediction_histogram is not None):
        output += "Prediction histogram:   %s\n" % (str(prediction_histogram));
        
    if (groundtruth_histogram is not None):
        output += "Ground truth histogram: %s\n" % (str(groundtruth_histogram));
    
    if (prediction_confusion_matrix is not None):
        output += "Confusion matrix:\n";
        output += confusion_matrix(prediction_confusion_matrix);
    
    output += "\n";
    
    return output;
    
def append_to_file(filepath, string):
    f = open(filepath, 'a');
    f.write(string);
    f.close();

if (__name__ == '__main__'):
    
    # Default settings
    dataset_path = './data/expressions_positive_integer_answer_shallow';
    single_digit = False;
    repetitions = 3;
    hidden_dim = 128;
    learning_rate = 0.01;
    lstm = True;
    max_training_size = None;
    test_interval = 100000; # 100,000
    # Default name is time of experiment
    raw_results_folder = './raw_results';
    name = time.strftime("%d-%m-%Y_%H-%M-%S");
    saveModels = True;
    
    # Generated variables
    raw_results_filepath = os.path.join(raw_results_folder,name+'.txt');
    
    # Command-line arguments
    key = None;
    for arg in sys.argv[1:]:
        if (arg[:2] == '--'):
            # Key
            key = arg[2:];
        else:
            val = arg;
            if (key is not None):
                if (key == 'dataset'):
                    dataset_path = val;
                elif (key == 'single_digit'):
                    single_digit = val == 'True';
                elif (key == 'repetitions'):
                    repetitions = int(val);
                elif (key == 'hidden_dim'):
                    hidden_dim = int(val);
                elif (key == 'learning_rate'):
                    learning_rate = float(val);
                elif (key == 'model'):
                    lstm = val == 'lstm';
                elif (key == 'max_training_size'):
                    if (val == "False"):
                        max_training_size = None;
                    else:
                        max_training_size = int(val);
                elif (key == 'testing_interval'):
                    if (val == "False"):
                        test_interval = None;
                    else:
                        test_interval = int(val);
                elif (key == 'name'):
                    name = val;
                elif (key == 'save_models'):
                    saveModels = val == 'False';
                key = None;
                val = None;
                                    
    
    # Debug settings
    if (max_training_size is not None):
        print("WARNING! RUNNING WITH LIMIT ON TRAINING SIZE!");
    
    # Construct models
    dataset = ge_dataset.GeneratedExpressionDataset(dataset_path, single_digit=single_digit);
    rnn = rnn.RecurrentNeuralNetwork(dataset.data_dim, hidden_dim, dataset.output_dim, 
                                     #lstm=lstm, single_digit=False, EOS_symbol_index=dataset.EOS_symbol_index);
                                     lstm=lstm, single_digit=False);
    if (single_digit):
        targets = dataset.train_labels;
    else:
        targets = dataset.train_targets;
    # Set up training indices
    if (max_training_size is not None):
        repetition_size = max_training_size;
    else:
        repetition_size = len(dataset.train);
    indices_to_use = repetition_size * repetitions;
    
    if (test_interval is not None):    
        batches = [];
        i = 0;
        while (indices_to_use - i > 0):
            end = i + test_interval;
            if (indices_to_use - end < 0):
                end = i + indices_to_use;
            batches.append((i,end));
            i = end;
 
    # Start timing
    start = time.clock();
    
    # Set up statistics
    key_indices = {k: i for (i,k) in enumerate(dataset.operators)};
      
    # Train
    current_repetition = 1;
    if (test_interval is not None):
        for b, (begin, end) in enumerate(batches):
            batch = range(begin % repetition_size, end % repetition_size);
            if (end <= begin):
                batch = range(begin,repetition_size) + range(end);
                current_repetition += 1;
            
            print("Batch %d of %d (repetition %d) (samples processed after batch: %d)" % (b+1,len(batches),current_repetition,(current_repetition-1)*repetition_size + end));
            append_to_file(raw_results_filepath, "Batch %d of %d (repetition %d) (samples processed after batch: %d)" % (b+1,len(batches),current_repetition,(current_repetition-1)*repetition_size + end));
            
            predicted_size_histogram = rnn.train(dataset.train[batch], targets[batch], learning_rate);
            print(predicted_size_histogram);
            if (b != len(batches)-1):
                # Intermediate testing if this was not the last iteration of training
                stats = rnn.test(dataset.test, dataset.test_targets, dataset.test_labels, dataset.test_expressions, dataset.operators, key_indices, dataset)
                if (single_digit):
                    score, prediction_histogram, groundtruth_histogram, _, _ = stats;
                    stats_str = statistics(start, score, prediction_histogram, groundtruth_histogram);
                else:
                    score, digit_score = stats;
                    stats_str = statistics(start, score, digit_score=digit_score);
                print(stats_str);
                append_to_file(raw_results_filepath, stats_str)
                # Save weights to pickles
                if (saveModels):
                    saveVars = rnn.vars.items();
                    f = open('saved_models/%s_%d.model' % (name, b),'w');
                    pickle.dump(saveVars,f);
                    f.close();
                    
    else:
        predicted_size_histogram = rnn.train(dataset.train[np.array(range(repetition_size))], targets[np.array(range(repetition_size))], learning_rate);
        print(predicted_size_histogram);
      
    # Final test
    stats = rnn.test(dataset.test, dataset.test_targets, dataset.test_labels, dataset.test_expressions, dataset.operators, key_indices, dataset)
    if (single_digit):
        score, prediction_histogram, groundtruth_histogram, prediction_confusion_matrix, _ = stats;
        stats_str = statistics(start, score, prediction_histogram, groundtruth_histogram, prediction_confusion_matrix);
    else:
        score, digit_score = stats;
        stats_str = statistics(start, score, digit_score);
    print(stats_str);
    append_to_file(raw_results_filepath, stats_str)
    
    # Save weights to pickles
    if (saveModels):
        saveVars = rnn.vars.items();
        f = open('saved_models/%s.model' % name, 'w');
        pickle.dump(saveVars,f);
        f.close();
    
