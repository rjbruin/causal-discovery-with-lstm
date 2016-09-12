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

def get_batch(isTrain, dataset, model, intervention_offset, max_length):
    tries = 0;
    limit = 1000;
    fail = False;
    
    batch = [];
    while (len(batch) < model.minibatch_size):
        batch = [];
        interventionLocation = np.random.randint(max_length-intervention_offset-3, 
                                                 max_length-3);
        
        while (not fail and len(batch) < model.minibatch_size):
            if (isTrain):
                candidates = dataset.expressionsByPrefix.get_random(model.minibatch_size -
                                                                    len(batch));
            else:
                candidates = dataset.testExpressionsByPrefix.get_random(model.minibatch_size -
                                                                        len(batch));
            for candidate in candidates:
                if (interventionLocation < len(candidate)):
                    symbolIndex = dataset.oneHot[candidate[interventionLocation]];
                    if (symbolIndex < dataset.EOS_symbol_index-2):
                        batch.append(candidate);
                    else:
                        tries += 1;
                        if (tries >= limit):
                            # Catch where we are stuck with an impossible intervention location
                            fail = True;
    
    data = [];
    targets = [];
    labels = [];
    expressions = [];
    for expression in batch:
        data, targets, labels, expressions, _ = dataset.processor(expression, data,
                                                                  targets, labels,
                                                                  expressions);
    
    return data, targets, labels, expressions, interventionLocation;

def test(model, dataset, parameters, max_length, print_samples=False):
    # Test
    print("Testing...");
        
    total = dataset.lengths[dataset.TEST];
    printing_interval = 100;
    if (parameters['max_testing_size'] is not False):
        total = parameters['max_testing_size'];
    
    # Set up statistics
    stats = set_up_statistics(dataset.output_dim);
    
    # Predict
    printed_samples = False;
    batch_range = range(0,total,model.minibatch_size);
    for _ in batch_range:
        # Get data from batch
        test_data, test_targets, test_labels, test_expressions, \
            interventionLocation = get_batch(False, dataset, model, 5, max_length);
        test_n = model.minibatch_size;
        
        test_targets, _, interventionLocation, _ = \
            dataset.insertInterventions(test_targets, test_expressions, 
                                        parameters['min_intervention_location'],
                                        parameters['n_max_digits'], 
                                        fixedLocation=interventionLocation);
        
        prediction, other = model.predict(test_data, label=test_targets, 
                                          interventionLocation=interventionLocation);
        
        # Print samples
        if (print_samples and not printed_samples):
            for i in range(prediction.shape[0]):
                print("# Input: %s" % "".join((map(lambda x: dataset.findSymbol[x], np.argmax(data[i],len(test_data.shape)-2)))));
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
    
    # Determine the minimum max_length needed to get batches quickly
    min_samples_required = dataset.lengths[dataset.TRAIN] * 0.10;
    max_length = model.n_max_digits;
    samples_available = dataset.expressionLengths[max_length];
    while (samples_available < min_samples_required):
        max_length -= 1;
        samples_available += dataset.expressionLengths[max_length];
    
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
            data, target, _, expressions, interventionLocation = get_batch(True, dataset, model, 5, max_length);
            
            data = dataset.fill_ndarray(data, 1);
            target = dataset.fill_ndarray(data, 1);
            
            # Perform interventions
            target, target_expressions, interventionLocation, emptySamples = dataset.insertInterventions(target, expressions, parameters['min_intervention_location'], parameters['n_max_digits'], fixedLocation=interventionLocation);
            #differences = map(lambda (d,t): d == t, zip(np.argmax(data, axis=2), np.argmax(target, axis=2)));
            
            # Swap axes of index in sentence and datapoint for Theano purposes
            data = np.swapaxes(data, 0, 1);
            target = np.swapaxes(target, 0, 1);
            # Run training
            outputs, unused = model.sgd(dataset, data, target, parameters['learning_rate'],
                                emptySamples=emptySamples, expressions=expressions,
                                intervention_expressions=target_expressions, 
                                interventionLocation=interventionLocation);
            unused_in_rep += unused;
            total_error += outputs[0];
            
            if ((k+model.minibatch_size) % 100 == 0):
                print("# %d / %d (%d unused)" % (k, repetition_size, unused_in_rep));
            
        # Update stats
        total_datapoints_processed += repetition_size;
        
        # Report on error
        print("Total error: %.2f" % total_error);
        
        # Intermediate testing if this was not the last iteration of training
        # and we have passed the testing threshold
        #if (r != repetition_size-1):
        test(model, dataset, parameters, max_length, print_samples=parameters['debug']);
        # Save weights to pickles
        if (saveModels):
            saveVars = model.getVars();
            save_to_pickle('saved_models/%s_%d.model' % (name, b), saveVars, settings=parameters);
    
    print("Training all datasets finished!");
    
    # Final test on last dataset
    #test(model, dataset, parameters, print_samples=parameters['debug']);
    
    # Save weights to pickles
#     if (saveModels):
#         saveVars = model.getVars();
#         save_to_pickle('saved_models/%s.model' % name, saveVars);
    