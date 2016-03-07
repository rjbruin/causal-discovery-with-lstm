'''
Created on 4 mrt. 2016

@author: Robert-Jan
'''

import numpy as np;

from tools.statistics import str_statistics;
from tools.file import append_to_file;
from tools.file import save_to_pickle;

def train(model, dataset, train, train_labels, test, test_targets, test_labels, batches, repetition_size, parameters, raw_results_filepath, key_indices, exp_name, start_time, saveModels=True):
    # Create settings string for saving models
    
    
    current_repetition = 1;
    if (parameters['test_interval'] is not None):
        for b, (begin, end) in enumerate(batches):
            begin = begin % repetition_size;
            end = end % repetition_size;
            batch = range(begin, end);
            if (end <= begin):
                batch = range(begin,repetition_size) + range(end);
                current_repetition += 1;
            
            print("Batch %d of %d (repetition %d) (samples processed after batch: %d)" % (b+1,len(batches),current_repetition,(current_repetition-1)*repetition_size + end));
            append_to_file(raw_results_filepath, "Batch %d of %d (repetition %d) (samples processed after batch: %d)" % (b+1,len(batches),current_repetition,(current_repetition-1)*repetition_size + end));
             
            model.train(train[batch], train_labels[batch], parameters['learning_rate']);
            if (b != len(batches)-1):
                # Intermediate testing if this was not the last iteration of training
                test_and_save(model, dataset, test, test_targets, test_labels, parameters, raw_results_filepath, key_indices, start_time);
                # Save weights to pickles
                if (saveModels):
                    saveVars = model.vars.items();
                    save_to_pickle('saved_models/%s_%d.model' % (exp_name, b), saveVars, settings={'test': 'True'});
                    
    else:
        model.train(train[np.array(range(repetition_size))], train_labels[np.array(range(repetition_size))], parameters['learning_rate']);

def test_and_save(model, dataset, test, test_targets, test_labels, parameters, raw_results_filepath, key_indices, start_time, show_prediction_conf_matrix=False):
    stats = model.test(test, test_targets, test_labels, dataset.test_expressions, dataset.operators, key_indices, dataset)
    if (parameters['single_digit']):
        score, prediction_histogram, groundtruth_histogram, prediction_confusion_matrix, _ = stats;
        if (show_prediction_conf_matrix):
            stats_str = str_statistics(start_time, score, prediction_histogram, groundtruth_histogram, prediction_confusion_matrix);
        else:
            stats_str = str_statistics(start_time, score, prediction_histogram, groundtruth_histogram);
    else:
        score, digit_score, prediction_size_histogram = stats;
        stats_str = str_statistics(start_time, score, digit_score=digit_score, prediction_size_histogram=prediction_size_histogram);
    print(stats_str);
    append_to_file(raw_results_filepath, stats_str);