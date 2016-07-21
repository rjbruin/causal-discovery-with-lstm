'''
Created on 4 mrt. 2016

@author: Robert-Jan
'''

import numpy as np;

from tools.statistics import str_statistics;
from tools.file import save_to_pickle;
from tools.data import get_batch_statistics

def train(model, dataset, parameters, exp_name, start_time, saveModels=True, targets=False, verboseOutputter=None):
    # Print settings headers to raw results file
    print(str(parameters));
    
    # Compute number of batches
    batch_size, repetition_size, _, nrBatches = get_batch_statistics(dataset, parameters);
    next_testing_threshold = parameters['test_interval'] * repetition_size;
    
    total_datapoints_processed = 0;
    for b in range(nrBatches):
        # Print progress and save to raw results file
        progress = "Batch %d of %d (repetition %d) (samples processed after batch: %d)" % (b+1,nrBatches,int(total_datapoints_processed/repetition_size)+1,total_datapoints_processed+batch_size);
        print(progress);
        if (verboseOutputter is not None):
            verboseOutputter['write'](progress);
        
        # Get the part of the dataset for this batch
        batch_train, batch_train_targets, batch_train_labels, _ = dataset.get_train_batch(batch_size);
        if (targets):
            batch_train_labels = batch_train_targets;
        
        # Train 
        model.train(batch_train, batch_train_labels, parameters['learning_rate']);
        
        # Update stats
        total_datapoints_processed += len(batch_train);
        
        # Intermediate testing if this was not the last iteration of training
        # and we have passed the testing threshold
        if (b != nrBatches-1):
            if (total_datapoints_processed >= next_testing_threshold):
                if (verboseOutputter is not None):
                    for varname in model.vars:
                        varsum = model.vars[varname].get_value().sum();
                        verboseOutputter['write']("summed %s: %.8f" % (varname, varsum));
                        if (varsum == 0.0):
                            verboseOutputter['write']("!!!!! Variable sum value is equal to zero!");
                            verboseOutputter['write']("=> name = %s, value:\n%s" % (varname, str(model.vars[varname].get_value())));
                
                test_and_save(model, dataset, parameters, start_time, verboseOutputter=verboseOutputter);
                # Save weights to pickles
                if (saveModels):
                    saveVars = model.vars.items();
                    save_to_pickle('saved_models/%s_%d.model' % (exp_name, b), saveVars, settings={'test': 'True'});
                next_testing_threshold += parameters['test_interval'] * repetition_size;

def test_and_save(model, dataset, parameters, start_time, show_prediction_conf_matrix=False, verboseOutputter=None):
    # Test
    print("Testing...");
        
    total = dataset.lengths[dataset.TEST];
    printing_interval = 1000;
    if (parameters['max_testing_size'] is not False):
        total = parameters['max_testing_size'];
    if (total < printing_interval * 10):
        printing_interval = total / 10;
    
    # Set up statistics
    stats = set_up_statistics(dataset.output_dim, dataset.key_indices);
    
    # Predict
    batch = dataset.get_test_batch();
    while (batch != False):
        # Get data from batch
        test_data, test_targets, test_labels, test_expressions = batch;
        
        if (verboseOutputter is not None):
            verboseOutputter['write']("%s samples in this test batch" % len(test_data));
            verboseOutputter['write']("example: %s" % (str(test_data[0])));
        
        # Test and retrieve updated stats
        stats = model.test(test_data, test_labels, test_targets, test_expressions, dataset, stats, print_sample=parameters['print_sample']);
        
        # Get new batch
        batch = dataset.get_test_batch();
    
    # Print statistics
    if (parameters['single_digit']):
        if (show_prediction_conf_matrix):
            stats_str = str_statistics(start_time, stats['score'], 
                                       stats['prediction_histogram'], 
                                       stats['groundtruth_histogram'], 
                                       stats['prediction_confusion_matrix']);
        else:
            stats_str = str_statistics(start_time, stats['score'], 
                                       stats['prediction_histogram'], 
                                       stats['groundtruth_histogram']);
    else:
        stats_str = str_statistics(start_time, stats['score'], 
                                   digit_score=stats['digit_score'], 
                                   prediction_size_histogram=\
                                    stats['prediction_size_histogram']);
    print(stats_str);

def set_up_statistics(output_dim, key_indices):
    return {'correct': 0.0, 'prediction_size': 0, 'digit_correct': 0.0, 'digit_prediction_size': 0,
            'prediction_histogram': {k: 0 for k in range(output_dim)},
            'groundtruth_histogram': {k: 0 for k in range(output_dim)},
            # First dimension is actual class, second dimension is predicted dimension
            'prediction_confusion_matrix': np.zeros((output_dim,output_dim)),
            # For each non-digit symbol keep correct and total predictions
            'operator_scores': np.zeros((len(key_indices),2)),
            'prediction_size_histogram': {k: 0 for k in range(60)}};