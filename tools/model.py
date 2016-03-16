'''
Created on 4 mrt. 2016

@author: Robert-Jan
'''

import numpy as np;

from tools.statistics import str_statistics;
from tools.file import append_to_file;
from tools.file import save_to_pickle;
from tools.data import get_batch_statistics

def train(model, dataset, parameters, raw_results_filepath, key_indices, exp_name, start_time, saveModels=True, targets=False):
    # Print settings headers to raw results file
    append_to_file(raw_results_filepath, str(parameters) + "\n\n")
    
    # Compute number of batches
    batch_size, repetition_size, total_iterations_size = get_batch_statistics(dataset, parameters);
    nrBatches = total_iterations_size / batch_size;
    
    current_repetition = 1;
    if (parameters['test_interval'] is not None):
        for b in range(nrBatches):
            # Increase repetition counter if the batch will wrap around the dataset
            if (dataset.locations[dataset.TRAIN] + batch_size > dataset.lengths[dataset.TRAIN]):
                current_repetition += 1;
            
            # Print progress and save to raw results file
            progress = "Batch %d of %d (repetition %d) (samples processed after batch: %d)" % (b+1,nrBatches,current_repetition,(b+1)*batch_size);
            print(progress);
            append_to_file(raw_results_filepath, progress);
            
            # Get the part of the dataset for this batch
            batch_train, batch_train_targets, batch_train_labels, _ = dataset.batch(batch_size);
            if (targets):
                batch_train_labels = batch_train_targets;
            
            # Train 
            model.train(batch_train, batch_train_labels, parameters['learning_rate']);
            if (b != nrBatches-1):
                # Intermediate testing if this was not the last iteration of training
                test_and_save(model, dataset, parameters, raw_results_filepath, key_indices, start_time);
                # Save weights to pickles
                if (saveModels):
                    saveVars = model.vars.items();
                    save_to_pickle('saved_models/%s_%d.model' % (exp_name, b), saveVars, settings={'test': 'True'});
                    
    else:
        batch_train, batch_train_targets, batch_train_labels, _ = dataset.all();
        if (targets):
            batch_train_labels = batch_train_targets;
        model.train(batch_train[np.array(range(repetition_size))], batch_train_labels[np.array(range(repetition_size))], parameters['learning_rate']);

def test_and_save(model, dataset, parameters, raw_results_filepath, key_indices, start_time, show_prediction_conf_matrix=False):
    stats = model.test(dataset, key_indices);
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