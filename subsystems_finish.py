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

import numpy as np;
import theano;
import copy;

def get_batch(isTrain, dataset, model, intervention_range, max_length, debug=False, base_offset=12, applyIntervention=True):
    limit = 1000;
    
    # Reseed the random generator to prevent generating identical batches
    np.random.seed();
    
    if (isTrain):
        storage = dataset.expressionsByPrefix;
    else:
        storage = dataset.testExpressionsByPrefix;
    
    batch = [];
    fails = 0;
    while (len(batch) < model.minibatch_size):
        fail = False;
        tries = 0;
        batch = [];
        
        interventionLocation = np.random.randint(max_length-intervention_range-base_offset, 
                                                 max_length-base_offset);
        
        while (not fail and len(batch) < model.minibatch_size):
            branch = storage.get_random_by_length(interventionLocation, getStructure=True);
            validPrefixes = {};
            if (len(branch.prefixedExpressions) >= 2):
                for prefix in branch.prefixedExpressions:
                    symbolIndex = dataset.oneHot[prefix];
                    if (not applyIntervention or symbolIndex < dataset.EOS_symbol_index-4 and len(branch.prefixedExpressions[prefix].fullExpressions) >= 1):
                        # Check for valid intervention symbol: has to be the right
                        # symbol and has to have expressions
                        validPrefixes[prefix] = branch.prefixedExpressions[prefix];
                    
            
            if (not applyIntervention or len(validPrefixes.keys()) >= 2):
                # If there are at least two valid prefixes we will always be
                # able to find an intervention sample for a random sample from
                # this branch 
                randomPrefix = validPrefixes.keys()[np.random.randint(0,len(validPrefixes.keys()))];
                randomCandidate = np.random.randint(0,len(branch.prefixedExpressions[randomPrefix].fullExpressions));
                batch.append((branch.prefixedExpressions[randomPrefix].fullExpressions[randomCandidate],
                              branch.prefixedExpressions[randomPrefix].primedExpressions[randomCandidate],
                              validPrefixes.keys()));
            else:
                tries += 1;
                if (tries >= limit):
                    # Catch where we are stuck with an impossible intervention location
                    fail = True;
                    if (debug):
                        fails += 1;
                        print("#\tBatching failed at iteration %d" % (fails));
    
    data = [];
    targets = [];
    labels = [];
    expressions = [];
    interventionSymbols = [];
    for (expression, expression_prime, possibleInterventions) in batch:
        data, targets, labels, expressions, _ = dataset.processor(";".join([expression, expression_prime]), 
                                                                  data,targets, labels, expressions);
        # Convert symbols to indices
        if (applyIntervention):
            interventionSymbols.append(map(lambda s: dataset.oneHot[s], possibleInterventions));
    
    data = dataset.fill_ndarray(data, 1);
    targets = dataset.fill_ndarray(copy.deepcopy(targets), 1, fixed_length=model.n_max_digits);
    
    if (debug and applyIntervention):
        # Sanity check: interventionSymbols must match each other
        passed = True;
        for indices in interventionSymbols:
            passed = passed and (all(map(lambda i: i < 10, indices)) or all(map(lambda i: i >= 10 and i < 14, indices)));
        
        if (not passed):
            raise ValueError("Illegal intervention symbols! => %s" % str(interventionSymbols));
    
    return data, targets, labels, expressions, interventionSymbols, interventionLocation;

def test(model, dataset, parameters, max_length, print_samples=False, sample_size=False, trimmed_from_max_length=0):
    # Test
    print("Testing...");
        
    total = dataset.lengths[dataset.TEST];
    printing_interval = 1000;
    if (parameters['max_testing_size'] is not False):
        total = parameters['max_testing_size'];
        printing_interval = 100;
    elif (sample_size != False):
        total = sample_size;
    
    # Set up statistics
    stats = set_up_statistics(dataset.output_dim, model.n_max_digits);
    
    # Predict
    printed_samples = False;
    batch_range = range(0,total,model.minibatch_size);
    for _ in batch_range:
        # Get data from batch
        test_data, test_targets, test_labels, test_expressions, \
            possibleInterventions, interventionLocation = get_batch(False, dataset, model, 
                                                                    parameters['intervention_range'], 
                                                                    max_length, debug=parameters['debug'],
                                                                    base_offset=parameters['intervention_base_offset'] - trimmed_from_max_length,
                                                                    applyIntervention=parameters['test_interventions']);
        test_n = model.minibatch_size;
        stats['intervention_locations'][interventionLocation] += 1;
        
        # Interventions are not optional in testing
        if (parameters['test_interventions']):
            test_targets, test_expressions, interventionLocation, _ = \
                dataset.insertInterventions(test_targets, test_expressions, 
                                            0,
                                            interventionLocation, 
                                            possibleInterventions);
        
        predictions, other = model.predict(test_data, label=test_targets, 
                                           interventionLocation=interventionLocation,
                                           intervention=parameters['test_interventions']);
        
        # Print samples
        if (print_samples and not printed_samples):
            for i in range(predictions[0].shape[0]):
                print("# Input 1: %s" % "".join((map(lambda x: dataset.findSymbol[x], 
                                                     np.argmax(test_data[i,:,:model.data_dim],len(test_data.shape)-2)))));
                print("# Label 1: %s" % "".join((map(lambda x: dataset.findSymbol[x], 
                                                   np.argmax(test_targets[i,:,:model.data_dim],len(test_data.shape)-2)))));
                print("# Output 1: %s" % "".join(map(lambda x: dataset.findSymbol[x], predictions[0][i])));
                
                print("# Input 2: %s" % "".join((map(lambda x: dataset.findSymbol[x], 
                                                     np.argmax(test_data[i,:,model.data_dim:],len(test_data.shape)-2)))));
                print("# Label 2: %s" % "".join((map(lambda x: dataset.findSymbol[x], 
                                                   np.argmax(test_targets[i,:,model.data_dim:],len(test_data.shape)-2)))));
                print("# Output 2: %s" % "".join(map(lambda x: dataset.findSymbol[x], predictions[1][i])));
            printed_samples = True;
        
        stats = model.batch_statistics(stats, predictions, 
                                       test_expressions, interventionLocation, 
                                       other, test_n, dataset, 
                                       eos_symbol_index=dataset.EOS_symbol_index);
    
        if (stats['prediction_size'] % printing_interval == 0):
            print("# %d / %d" % (stats['prediction_size'], total));
    
    stats = model.total_statistics(stats);
    
    # Print statistics
    output = "\n";

    # Print statistics
    output += "Score: %.2f percent\n" % (stats['score']*100);
    output += "Cause score: %.2f percent\n" % (stats['causeScore']*100);
    output += "Effect score: %.2f percent\n" % (stats['effectScore']*100);
    output += "Intervention locations:   %s\n" % (str(stats['intervention_locations']));

    output += "Digit-based (1) score: %.2f percent\n" % (stats['digit_1_score']*100);
    output += "Prediction size (1) histogram:   %s\n" % (str(stats['prediction_1_size_histogram']));
    output += "Digit (1) histogram:   %s\n" % (str(stats['prediction_1_histogram']));
    
    output += "Digit-based (2) score: %.2f percent\n" % (stats['digit_2_score']*100);
    output += "Prediction size (2) histogram:   %s\n" % (str(stats['prediction_2_size_histogram']));
    output += "Digit (2) histogram:   %s\n" % (str(stats['prediction_2_histogram']));
        
    output += "Digit-based score: %.2f percent\n" % (stats['digit_score']*100);
    output += "Prediction size histogram:   %s\n" % (str(stats['prediction_size_histogram']));
    output += "Digit histogram:   %s\n" % (str(stats['prediction_histogram']));
    
    output += "\n";
    print(output);
    
    return stats;

if __name__ == '__main__':
    theano.config.floatX = 'float32';
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
    if (not parameters['decoder']):
        print("WARNING! RUNNING WITHOUT DECODER!");
    
    # Construct models
    datasets, model = constructModels(parameters, 0, {});
    
    # Train on all datasets in succession
    # Print settings headers to raw results file
    print(str(parameters));
    
    dataset = datasets[0];
    reps = parameters['repetitions'];
    
    # Compute batching variables
    repetition_size = dataset.lengths[dataset.TRAIN];
    if (parameters['max_training_size'] is not False):
        repetition_size = min(parameters['max_training_size'],repetition_size);
    next_testing_threshold = parameters['test_interval'] * repetition_size;
    
    total_datapoints_processed = 0;
    b = 0;
    terminate = False;
    
    # Determine the minimum max_length needed to get batches quickly
    min_samples_required = dataset.lengths[dataset.TRAIN] * 0.10;
    max_length = model.n_max_digits;
    samples_available = dataset.expressionLengths[max_length];
    trimmed_from_max_length = 0;
    while (samples_available < min_samples_required):
        max_length -= 1;
        trimmed_from_max_length += 1;
        samples_available += dataset.expressionLengths[max_length];
    
    intervention_locations_train = {k: 0 for k in range(model.n_max_digits)};
    for r in range(reps):
        unused_in_rep = 0;
        total_error = 0.0;
        # Print progress and save to raw results file
        progress = "Batch %d (repetition %d of %d, dataset 1 of 1) (samples processed after batch: %d)" % \
            (r+1,r+1,reps,total_datapoints_processed+repetition_size);
        print(progress);
        
        # Train model per minibatch
        batch_range = range(0,repetition_size,model.minibatch_size);
        for k in batch_range:
            data, target, _, expressions, possibleInterventions, interventionLocation = \
                get_batch(True, dataset, model, 
                          parameters['intervention_range'], max_length, 
                          debug=parameters['debug'],
                          base_offset=parameters['intervention_base_offset'] - trimmed_from_max_length,
                          applyIntervention=parameters['train_interventions']);
            
            # Perform interventions
            if (parameters['train_interventions']):
                target, target_expressions, interventionLocation, emptySamples = \
                    dataset.insertInterventions(target, copy.deepcopy(expressions), 
                                                0,
                                                interventionLocation, 
                                                possibleInterventions);
                intervention_locations_train[interventionLocation] += 1;
                #differences = map(lambda (d,t): d == t, zip(np.argmax(data, axis=2), np.argmax(target, axis=2)));
            
            # Run training
            if (parameters['train_interventions']):
                outputs = \
                    model.sgd(dataset, data, target, parameters['learning_rate'],
                              emptySamples=emptySamples, expressions=expressions,
                              intervention_expressions=target_expressions, 
                              interventionLocation=interventionLocation,
                              intervention=parameters['train_interventions'],
                              fixedDecoderInputs=parameters['fixed_decoder_inputs']);
            else:
                outputs = \
                    model.sgd(dataset, data, target, parameters['learning_rate'],
                              emptySamples=emptySamples, expressions=expressions,
                              intervention=parameters['train_interventions'],
                              fixedDecoderInputs=parameters['fixed_decoder_inputs']);
            total_error += outputs[0];
            if (str(outputs[0]) == 'nan' and parameters['debug']):
                print("NaN at batch %d" % k);
                print("Cross entropy: " + str(outputs[1]));
                print("Label: " + str(np.sum(np.sum(outputs[3], axis=2), axis=1)));
                print("Smallest right hand value: " + np.array_str(np.min(outputs[2]), precision=16));                   
            
            if (str(outputs[0]) == 'nan'):
                # Terminate since we cannot work with NaN values
                print("ERROR! ENCOUNTERED NAN ERROR! TERMINATING RUN.")
                terminate = True;
                break;
            
            if ((k+model.minibatch_size) % 100 == 0):
                print("# %d / %d (error = %.2f)" % (k+model.minibatch_size, repetition_size, total_error));
#                 if (parameters['debug']):
#                     for p, l, e in zip(prediction_expressions, label_expressions, expressions):
#                         print("Input:\t%s\tPrediction:\t%s\tLabel used:\t%s" % (e, p, l));
            
        # Update stats
        total_datapoints_processed += repetition_size;
        
        # Report on error
        print("Total error: %.2f" % total_error);
        print("Intervention locations: %s" % (str(intervention_locations_train)));
        
        # Intermediate testing if this was not the last iteration of training
        # and we have passed the testing threshold
        #if (r != repetition_size-1):
        test(model, dataset, parameters, max_length, print_samples=parameters['debug'], 
             sample_size=parameters['sample_testing_size'], 
             trimmed_from_max_length=trimmed_from_max_length);
        # Save weights to pickles
        if (saveModels):
            saveVars = model.getVars();
            save_to_pickle('saved_models/%s_%d.model' % (name, r), saveVars, settings=parameters);
        
        if (terminate):
            break;
    
    if (terminate):
        print("Experiment terminated prematurely!");
    else:
        print("Training all datasets finished!");
    