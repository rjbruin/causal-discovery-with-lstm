'''
Created on 9 sep. 2016

@author: Robert-Jan
'''

import time;
import sys;

from tools.file import save_to_pickle;
from tools.arguments import processCommandLineArguments;
from tools.model import constructModels, set_up_statistics;
from tools.gpu import using_gpu; # @UnresolvedImport
from tools.statistics import str_statistics;

import numpy as np;

def get_train_batch(dataset):
    batch = [];
    while (len(batch) < dataset.minibatch_size):
        batch.extend(dataset.expressionsByPrefix.get_random(dataset.minibatch_size - len(batch)));
    
    data = [];
    targets = [];
    labels = [];
    expressions = [];
    for expression in batch:
        data, targets, labels, expressions, _ = dataset.processor(expression, data, targets, labels, expressions);
    
    return data, targets, labels, expressions;

def get_test_batch(dataset):
    batch = [];
    while (len(batch) < dataset.minibatch_size):
        batch.extend(dataset.testExpressionsByPrefix.get_random(dataset.minibatch_size - len(batch)));
    
    data = [];
    targets = [];
    labels = [];
    expressions = [];
    for expression in batch:
        data, targets, labels, expressions, _ = dataset.processor(expression, data, targets, labels, expressions);
    
    return data, targets, labels, expressions;

def test(model, dataset, parameters, print_samples=False):
    # Test
    print("Testing...");
        
    total = dataset.lengths[dataset.TEST];
    printing_interval = 1000;
    if (parameters['max_testing_size'] is not False):
        total = parameters['max_testing_size'];
    
    # Set up statistics
    stats = set_up_statistics(dataset.output_dim);
    
    # Predict
    printed_samples = False;
    batch_range = range(0,len(total),model.minibatch_size);
    for _ in batch_range:
        # Get data from batch
        test_data, test_targets, test_labels, test_expressions, interventionLocation = get_test_batch(dataset);
        test_n = model.minibatch_size;
        
        test_targets, _, interventionLocation, _ = dataset.insertInterventions(test_targets, test_expressions, parameters['min_intervention_location'], parameters['n_max_digits'], fixedLocation=interventionLocation);
        
        prediction, other = model.predict(test_data, label=test_targets, interventionLocation=interventionLocation);
        
        # Print samples
        if (print_samples and not printed_samples):
            for i in range(prediction.shape[0]):
                print("# Input: %s" % "".join((map(lambda x: dataset.findSymbol[x], np.argmax(data[i],len(test_data.shape)-2)))));
                if (model.single_digit):
                    print("# Label: %s" % "".join(dataset.findSymbol[np.argmax(test_targets[i])]));
                    print("# Output: %s" % "".join(dataset.findSymbol[prediction[i]]));
                else:
                    print("# Label: %s" % "".join((map(lambda x: dataset.findSymbol[x], np.argmax(test_targets[i],len(test_data.shape)-2)))));
                    print("# Output: %s" % "".join(map(lambda x: dataset.findSymbol[x], prediction[i])));
                
            printed_samples = True;
        
        stats = model.batch_statistics(stats, prediction, test_labels, 
                                       test_targets, expressions, 
                                       other,
                                       test_n, dataset, 
                                       eos_symbol_index=dataset.EOS_symbol_index);
    
        if (stats['prediction_size'] % printing_interval == 0):
            print("# %d / %d" % (stats['prediction_size'], total));
    
    stats = model.total_statistics(stats);
    
    # Print statistics
    stats_str = str_statistics(0, stats['score'], 
                               digit_score=stats['digit_score'], 
                               prediction_size_histogram=\
                                stats['prediction_size_histogram']);
    print(stats_str);
    
    return stats;

if __name__ == '__main__':
    np.set_printoptions(precision=3, threshold=10000000);
    
    # Specific settings - default name is time of experiment
    name = time.strftime("%d-%m-%Y_%H-%M-%S");
    saveModels = True;
    
    # Process parameters
    parameters = processCommandLineArguments(sys.argv[1:]);
    
    # Warn for unusual parameters
    if (parameters['max_training_size'] is not False):
        print("WARNING! RUNNING WITH LIMIT ON TRAINING SIZE!");
    if (not using_gpu()):
        print("WARNING! RUNNING WITHOUT GPU USAGE!");
    
    # Construct models
    datasets, model = constructModels(parameters, 0, {});
    
    ### From here the experiment should be the same every time
    
    # Start experiment clock
    start = time.clock();
    
    # Train on all datasets in succession
    # Print settings headers to raw results file
    print(str(parameters));
    
    dataset = datasets[0];
    reps = parameters['repetitions'];
    
    # Compute batching variables
    repetition_size = dataset.lengths[dataset.TRAIN];
    if (parameters['train_batch_size'] is not False):
        batch_size = min(parameters['train_batch_size'],repetition_size);
    else:
        batch_size = min(dataset.lengths[dataset.TRAIN],repetition_size);
    next_testing_threshold = parameters['test_interval'] * repetition_size;
    
    total_datapoints_processed = 0;
    b = 0;
    
    for r in range(reps):
        unused_in_rep = 0;
        total_error = 0.0;
        # Print progress and save to raw results file
        progress = "Batch %d (repetition %d of %d) (samples processed after batch: %d)" % \
            (r+1,r+1,reps,total_datapoints_processed+batch_size);
        print(progress);
        
        # Train model per minibatch
        batch_range = range(0,repetition_size,model.minibatch_size);
        for k in batch_range:
            if (parameters['time_training_batch']):
                start = time.clock();
            
            data, target, _, expressions, interventionLocation = get_train_batch(dataset);
            
            # Perform interventions
            target, target_expressions, interventionLocation, emptySamples = dataset.insertInterventions(target, expressions, parameters['min_intervention_location'], parameters['n_max_digits'], fixedLocation=interventionLocation);
            #differences = map(lambda (d,t): d == t, zip(np.argmax(data, axis=2), np.argmax(target, axis=2)));
            
            # Swap axes of index in sentence and datapoint for Theano purposes
            data = np.swapaxes(data, 0, 1);
            if (not model.single_digit):
                target = np.swapaxes(target, 0, 1);
            # Run training
            outputs, unused = model.sgd(dataset, data, target, parameters['learning_rate'],
                                emptySamples=emptySamples, expressions=expressions,
                                intervention_expressions=target_expressions, 
                                interventionLocation=interventionLocation);
            unused_in_rep += unused;
            total_error += outputs[0];
            
            if (k+model.minibatch_size % 100 == 0):
                print("# %d / %d (%d unused)" % (k, repetition_size, unused_in_rep));
            
        # Update stats
        total_datapoints_processed += len(repetition_size);
        
        # Report on error
        print("Total error: %.2f" % total_error);
        
        # Intermediate testing if this was not the last iteration of training
        # and we have passed the testing threshold
        if (r != repetition_size-1 and total_datapoints_processed >= next_testing_threshold):
            test(model, dataset, parameters, print_samples=parameters['debug']);
            # Save weights to pickles
            if (saveModels):
                saveVars = model.getVars();
                save_to_pickle('saved_models/%s_%d.model' % (name, b), saveVars, settings=parameters);
            next_testing_threshold += parameters['test_interval'] * repetition_size;
    
    print("Training all datasets finished!");
    
    # Final test on last dataset
    test(model, dataset, parameters, print_samples=parameters['debug']);
    
    # Save weights to pickles
    if (saveModels):
        saveVars = model.getVars();
        save_to_pickle('saved_models/%s.model' % name, saveVars);
    