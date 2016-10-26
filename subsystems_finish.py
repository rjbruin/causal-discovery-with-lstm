'''
Created on 9 sep. 2016

@author: Robert-Jan
'''

import time;
import sys;
from math import floor;

from tools.file import save_to_pickle, load_from_pickle_with_filename;
from tools.arguments import processCommandLineArguments;
from tools.model import constructModels, set_up_statistics;
from tools.gpu import using_gpu; # @UnresolvedImport

import numpy as np;
import theano;
import copy;
from profiler import profiler

def print_stats(stats, parameters, prefix=''):
    # Print statistics
    output = "\n";

    # Print statistics
    output += prefix + "Score: %.2f percent\n" % (stats['score']*100);
    
    if (not parameters['only_cause_expression']):
        output += prefix + "Structure score cause: %.2f percent\n" % (stats['structureScoreCause']*100);
        output += prefix + "Structure score effect: %.2f percent\n" % (stats['structureScoreEffect']*100);
        output += prefix + "Structure score top: %.2f percent\n" % (stats['structureScoreTop']*100);
        output += prefix + "Structure score bot: %.2f percent\n" % (stats['structureScoreBot']*100);
        output += prefix + "Structure score: %.2f percent\n" % (stats['structureScore']*100);
        output += prefix + "Effect score: %.2f percent\n" % (stats['effectScore']*100);
        output += prefix + "Effect score including no effect: %.2f percent\n" % (stats['allEffectScore']*100);
    
    output += prefix + "Valid: %.2f percent\n" % (stats['validScore']*100);
    if (not parameters['only_cause_expression']):
        output += prefix + "Structure valid cause: %.2f percent\n" % (stats['structureValidScoreCause']*100);
        output += prefix + "Structure valid effect: %.2f percent\n" % (stats['structureValidScoreEffect']*100);
        output += prefix + "Structure valid top: %.2f percent\n" % (stats['structureValidScoreTop']*100);
        output += prefix + "Structure valid bot: %.2f percent\n" % (stats['structureValidScoreBot']*100);
    
    output += prefix + "Intervention locations:   %s\n" % (str(stats['intervention_locations']));

    if (not parameters['only_cause_expression']):
        output += prefix + "Digit-based (1) score: %.2f percent\n" % (stats['digit_1_score']*100);
        output += prefix + "Prediction size (1) histogram:   %s\n" % (str(stats['prediction_1_size_histogram']));
        output += prefix + "Digit (1) histogram:   %s\n" % (str(stats['prediction_1_histogram']));
        
        output += prefix + "Digit-based (2) score: %.2f percent\n" % (stats['digit_2_score']*100);
        output += prefix + "Prediction size (2) histogram:   %s\n" % (str(stats['prediction_2_size_histogram']));
        output += prefix + "Digit (2) histogram:   %s\n" % (str(stats['prediction_2_histogram']));
        
    output += prefix + "Digit-based score: %.2f percent\n" % (stats['digit_score']*100);
    output += prefix + "Prediction size histogram:   %s\n" % (str(stats['prediction_size_histogram']));
    output += prefix + "Digit histogram:   %s\n" % (str(stats['prediction_histogram']));
    
    output += prefix + "Error margin 1 score: %.2f percent\n" % stats['error_1_score'];
    output += prefix + "Error margin 2 score: %.2f percent\n" % stats['error_2_score'];
    output += prefix + "Error margin 3 score: %.2f percent\n" % stats['error_3_score'];
    
    output += prefix + "All error margins: %s\n" % str(stats['error_histogram']);
    
    output += prefix + "Unique labels predicted: %d\n" % stats['unique_labels_predicted'];
    
    output += "\n";
    print(output);

def get_batch(isTrain, dataset, model, intervention_range, max_length, 
              debug=False, base_offset=12, applyIntervention=True, 
              seq2ndmarkov=False, bothcause=False):    
    # Reseed the random generator to prevent generating identical batches
    np.random.seed();
    
    if (isTrain):
        storage = dataset.expressionsByPrefix;
        if (seq2ndmarkov and not parameters['only_cause_expression']):
            storage_bot = dataset.expressionsByPrefixBot;
    else:
        storage = dataset.testExpressionsByPrefix;
        if (seq2ndmarkov and not parameters['only_cause_expression']):
            storage_bot = dataset.testExpressionsByPrefixBot;
    
    batch = [];
    interventionLocations = [];
    subbatch_size = parameters['subbatch_size'];
    while (len(batch) < model.minibatch_size):
        interventionLocation = np.random.randint(max_length-intervention_range-base_offset, 
                                                 max_length-base_offset);
        if (seq2ndmarkov):
            # Change interventionLocation to nearest valid location to the left 
            # of current location (which is the operator or the right argument)
            if (interventionLocation % 3 != 2 and interventionLocation % 3 != 1):
                # Offset is 2 for right argument and 1 for operator 
                offset = interventionLocation % 3;
                interventionLocation = int(floor((interventionLocation - offset) / 3) * 3) + offset;
        
        # Choose top or bottom cause
        if (not seq2ndmarkov or parameters['only_cause_expression'] is not False):
            topcause = True;
        else:
            topcause = np.random.randint(2) == 1;
        
        subbatch = [];
        fails = 0;
        limit = 1000;
        while (len(subbatch) < subbatch_size):
            if (seq2ndmarkov and not topcause):
                branch = storage_bot.get_random_by_length(interventionLocation, getStructure=True);
            else:
                branch = storage.get_random_by_length(interventionLocation, getStructure=True);
            validPrefixes = {};
            
            if (not seq2ndmarkov or not applyIntervention):
                if (len(branch.prefixedExpressions) >= 2):
                    for prefix in branch.prefixedExpressions:
                        symbolIndex = dataset.oneHot[prefix];
                        if (symbolIndex < dataset.EOS_symbol_index-4 and len(branch.prefixedExpressions[prefix].fullExpressions) >= 1):
                            # Check for valid intervention symbol: has to be the right
                            # symbol and has to have expressions
                            validPrefixes[prefix] = branch.prefixedExpressions[prefix];
            else:
                # We use all prefixes for seq2ndmarkov because we can only have valid intervention locations
                # We know this because the samples all have the same repeating symtax of 
                # <left><op><right><answer/left><op>...
                validPrefixes = {p: branch.prefixedExpressions[p] for p in branch.prefixedExpressions};
            
            if (applyIntervention):
                if (len(validPrefixes.keys()) >= 2):
                    # If there are at least two valid prefixes we will always be
                    # able to find an intervention sample for a random sample from
                    # this branch 
                    randomPrefix = validPrefixes.keys()[np.random.randint(0,len(validPrefixes.keys()))];
                    randomCandidate = np.random.randint(0,len(branch.prefixedExpressions[randomPrefix].fullExpressions));
                    # Keep the order of top and bottom
                    if (seq2ndmarkov and not topcause):
                        subbatch.append((branch.prefixedExpressions[randomPrefix].primedExpressions[randomCandidate],
                                         branch.prefixedExpressions[randomPrefix].fullExpressions[randomCandidate],
                                         validPrefixes.keys()));
                    else:
                        subbatch.append((branch.prefixedExpressions[randomPrefix].fullExpressions[randomCandidate],
                                         branch.prefixedExpressions[randomPrefix].primedExpressions[randomCandidate],
                                         validPrefixes.keys()));
                else:
                    fails += 1;
                    if (fails >= limit):
                        subbatch = [];
                        break;
            else:
                # No intervention, just the sample, but we still need to select a random sample
                if (len(validPrefixes.keys()) >= 1):
                    randomPrefix = validPrefixes.keys()[np.random.randint(0,len(validPrefixes.keys()))];
                    randomCandidate = np.random.randint(0,len(branch.prefixedExpressions[randomPrefix].fullExpressions));
                    subbatch.append((branch.prefixedExpressions[randomPrefix].fullExpressions[randomCandidate],
                                     branch.prefixedExpressions[randomPrefix].primedExpressions[randomCandidate],
                                     validPrefixes.keys()));
        
        # Add subbatch to batch
        batch.extend(subbatch);
        interventionLocations.extend(np.ones((len(subbatch)), dtype='int32') * interventionLocation);
    
    data = [];
    targets = [];
    labels = [];
    expressions = [];
    interventionSymbols = [];
    for (expression, expression_prime, possibleInterventions) in batch:
        if (seq2ndmarkov and not bothcause):
            if (parameters['only_cause_expression'] == 2):
                expression_prime = expression;
                expression = "";
            data, targets, labels, expressions, _ = dataset.processor(";".join([expression, expression_prime, str(int(topcause))]), 
                                                                      data,targets, labels, expressions);
        else:
            data, targets, labels, expressions, _ = dataset.processor(";".join([expression, expression_prime]), 
                                                                      data,targets, labels, expressions);
        # Convert symbols to indices
        if (applyIntervention):
            interventionSymbols.append(map(lambda s: dataset.oneHot[s], possibleInterventions));
    
    data = dataset.fill_ndarray(data, 1);
    targets = dataset.fill_ndarray(copy.deepcopy(targets), 1, fixed_length=model.n_max_digits);
    
    return data, targets, labels, expressions, interventionSymbols, interventionLocations, topcause;

def test(model, dataset, parameters, max_length, base_offset, intervention_range, print_samples=False, sample_size=False):
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
    total_labels_used = {k: 0 for k in range(30)};
    
    # Predict
    printed_samples = False;
    batch_range = range(0,total,model.minibatch_size);
    totalError = 0.0;
    for _ in batch_range:
        # Get data from batch
        test_data, test_targets, _, test_expressions, \
            possibleInterventions, interventionLocations, topcause = get_batch(False, dataset, model, 
                                                                              intervention_range, 
                                                                              max_length, debug=parameters['debug'],
                                                                              base_offset=base_offset,
                                                                              applyIntervention=parameters['test_interventions'],
                                                                              seq2ndmarkov=parameters['dataset_type'] == 1,
                                                                              bothcause=parameters['bothcause']);
        test_n = model.minibatch_size;
        for l in interventionLocations:
            stats['intervention_locations'][l] += 1;
        
        # Interventions are not optional in testing
        if (parameters['test_interventions']):
            test_targets, test_expressions, _ = \
                dataset.insertInterventions(test_targets, test_expressions, 
                                            topcause,
                                            interventionLocations, 
                                            possibleInterventions);
        
        predictions, other = model.predict(test_data, label=test_targets, 
                                           interventionLocations=interventionLocations,
                                           intervention=parameters['test_interventions'],
                                           fixedDecoderInputs=parameters['fixed_decoder_inputs']);
        totalError += other['error'];
        
        if (parameters['only_cause_expression']):
            prediction_1 = predictions;
            predictions = [predictions];
        else:
            prediction_1 = predictions[0];
            prediction_2 = predictions[1];
        
        profiler.start("test batch stats");
        labels_to_use = False;
        if (parameters['no_label_search']):
            # If we don't use label searching we need to provide labels_to_use
            labels_to_use = test_expressions;
        stats, labels_used = model.batch_statistics(stats, predictions, 
                                       test_expressions, interventionLocations, 
                                       other, test_n, dataset, 
                                       eos_symbol_index=dataset.EOS_symbol_index,
                                       topcause=topcause or parameters['bothcause'], # If bothcause then topcause = 1
                                       testExtraValidity=parameters['test_extra_validity'],
                                       bothcause=parameters['bothcause'],
                                       labels_to_use=labels_to_use);
        
        for j in range(model.minibatch_size):
            if (parameters['only_cause_expression'] is not False):
                total_labels_used[labels_used[j][0]] = True;
            else:
                total_labels_used[labels_used[j][0]+";"+labels_used[j][1]] = True;
        
        # Print samples
        if (print_samples and not printed_samples):
            for i in range(10):
                prefix = "# ";
#                 prefix = "";
                print(prefix + "Intervention location: %d" % interventionLocations[i]);
                print(prefix + "Original data 1: %s" % "".join((map(lambda x: dataset.findSymbol[x], 
                                                     np.argmax(test_data[i,:,:model.data_dim],len(test_data.shape)-2)))));
                print(prefix + "Interve. data 1: %s" % "".join((map(lambda x: dataset.findSymbol[x], 
                                                   np.argmax(test_targets[i,:,:model.data_dim],len(test_data.shape)-2)))));
                print(prefix + "Prediction    1: %s" % "".join(map(lambda x: dataset.findSymbol[x], prediction_1[i])));
                print(prefix + "Used label    1: %s" % labels_used[i][0]);
                
                if (not parameters['only_cause_expression']):
                    print(prefix + "Original data 2: %s" % "".join((map(lambda x: dataset.findSymbol[x], 
                                                         np.argmax(test_data[i,:,model.data_dim:],len(test_data.shape)-2)))));
                    print(prefix + "Interve. data 2: %s" % "".join((map(lambda x: dataset.findSymbol[x], 
                                                       np.argmax(test_targets[i,:,model.data_dim:],len(test_data.shape)-2)))));
                    print(prefix + "Prediction    2: %s" % "".join(map(lambda x: dataset.findSymbol[x], prediction_2[i])));
                    print(prefix + "Used label    2: %s" % labels_used[i][1]);
            printed_samples = True;

        if (stats['prediction_size'] % printing_interval == 0):
            print("# %d / %d" % (stats['prediction_size'], total));
        profiler.stop("test batch stats");
    
    profiler.profile();
    
    print("Total testing error: %.2f" % totalError);
    
    stats = model.total_statistics(stats, total_labels_used=total_labels_used);
    
    print_stats(stats, parameters);
    
    return stats;

if __name__ == '__main__':
    theano.config.floatX = 'float32';
    np.set_printoptions(precision=3, threshold=10000000);
    profiler.off();
    
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
    
    # Check for valid subbatch size
    if (parameters['minibatch_size'] % parameters['subbatch_size'] != 0):
        raise ValueError("Subbatch size is not compatible with minibatch size: m.size = %d, s.size = %d" % 
                            (parameters['minibatch_size'], parameters['subbatch_size']));
    
    # Construct models
    datasets, model = constructModels(parameters, 0, {});
    
    # Load pretrained only_cause_expression = 1 model
    if (parameters['load_cause_expression_1'] is not False):
        loadedVars, _ = load_from_pickle_with_filename("./saved_models/" + parameters['load_cause_expression_1']);
        if (model.loadPartialDataDimVars(dict(loadedVars), 0, model.data_dim)):
            print("Loaded pretrained model (expression 1) successfully!");
        else:
            raise ValueError("Loading pretrained model failed: wrong variables supplied!");
    
    # Load pretrained only_cause_expression = 2 model
    if (parameters['load_cause_expression_2'] is not False):
        loadedVars, _ = load_from_pickle_with_filename("./saved_models/" + parameters['load_cause_expression_2']);
        if (model.loadPartialDataDimVars(dict(loadedVars), model.data_dim, model.data_dim)):
            print("Loaded pretrained model (expression 2) successfully!");
        else:
            raise ValueError("Loading pretrained model failed: wrong variables supplied!");
    
    # Train on all datasets in succession
    # Print settings headers to raw results file
    print("# " + str(parameters));
    
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
    
    # Make the base_offset absorb the max length difference
    base_offset = parameters['intervention_base_offset'];
    intervention_range = parameters['intervention_range'];
    
    print("Adapted intervention range: %d" % intervention_range);
    print("Adapted base offset: %d" % base_offset);
    
    intervention_locations_train = {k: 0 for k in range(model.n_max_digits)};
    for r in range(reps):
        stats = set_up_statistics(dataset.output_dim, model.n_max_digits);
        unused_in_rep = 0;
        total_error = 0.0;
        # Print progress and save to raw results file
        progress = "Batch %d (repetition %d of %d, dataset 1 of 1) (samples processed after batch: %d)" % \
            (r+1,r+1,reps,total_datapoints_processed+repetition_size);
        print(progress);
        
        # Train model per minibatch
        batch_range = range(0,repetition_size,model.minibatch_size);
        for k in batch_range:
            profiler.start('train batch');
            profiler.start('get train batch');
            data, target, _, expressions, possibleInterventions, interventionLocations, topcause = \
                get_batch(True, dataset, model, 
                          intervention_range, max_length, 
                          debug=parameters['debug'],
                          base_offset=base_offset,
                          applyIntervention=parameters['train_interventions'],
                          seq2ndmarkov=parameters['dataset_type'] == 1,
                          bothcause=parameters['bothcause']);
            profiler.stop('get train batch');
            
            profiler.start('train interventions');
            # Perform interventions
            if (parameters['train_interventions']):
                target, target_expressions, interventionLocation = \
                    dataset.insertInterventions(target, copy.deepcopy(expressions), 
                                                topcause,
                                                interventionLocations, 
                                                possibleInterventions);
                for l in interventionLocations:
                    intervention_locations_train[l] += 1;
                #differences = map(lambda (d,t): d == t, zip(np.argmax(data, axis=2), np.argmax(target, axis=2)));
            profiler.stop('train interventions');
            
            # Run training
            profiler.start('train sgd');
            if (parameters['train_interventions']):
                outputs, predictions, new_targets, labels_to_use = \
                    model.sgd(dataset, data, target, parameters['learning_rate'],
                              emptySamples=[], expressions=expressions,
                              intervention_expressions=target_expressions, 
                              interventionLocations=interventionLocations,
                              intervention=parameters['train_interventions'],
                              fixedDecoderInputs=parameters['fixed_decoder_inputs'],
                              topcause=topcause or parameters['bothcause'], bothcause=parameters['bothcause']);
            else:
                outputs, predictions, new_targets, labels_to_use = \
                    model.sgd(dataset, data, target, parameters['learning_rate'],
                              emptySamples=[], expressions=expressions,
                              intervention_expressions=expressions, 
                              intervention=parameters['train_interventions'],
                              fixedDecoderInputs=parameters['fixed_decoder_inputs'],
                              topcause=topcause or parameters['bothcause'], bothcause=parameters['bothcause']);
            total_error += outputs[0];
            profiler.stop('train sgd');
            
            # Training prediction
            profiler.start('train stats');
            if (parameters['train_statistics'] and parameters['train_interventions']):
                stats, _ = model.batch_statistics(stats, predictions, 
                                               expressions, interventionLocations, 
                                               {}, len(expressions), dataset, 
                                               eos_symbol_index=dataset.EOS_symbol_index,
                                               labels_to_use=labels_to_use,
                                               training=True, topcause=topcause or parameters['bothcause'],
                                               testExtraValidity=parameters['test_extra_validity'],
                                               bothcause=parameters['bothcause']);
            
            if ((k+model.minibatch_size) % (model.minibatch_size*4) == 0):
                print("# %d / %d (error = %.2f)" % (k+model.minibatch_size, repetition_size, total_error));
            
            profiler.stop('train stats');
            profiler.stop('train batch');
        
        # Print sample of last training batch
#         if (parameters['debug'] and parameters['only_cause_expression'] == 1):
#             for i in range(10):
#                 prefix = "";
#                 print(prefix + "Intervention location: %d" % interventionLocations[i]);
#                 print(prefix + "Original data        : %s" % "".join((map(lambda x: dataset.findSymbol[x], 
#                                                      np.argmax(data[i,:,:model.data_dim],len(data.shape)-2)))));
#                 print(prefix + "Intervened data      : %s" % "".join((map(lambda x: dataset.findSymbol[x], 
#                                                    np.argmax(target[i,:,:model.data_dim],len(target.shape)-2)))));
#                 print(prefix + "Pre-int prediction   : %s" % "".join(map(lambda x: dataset.findSymbol[x], predictions[i])));
#                 print(prefix + "Used target          : %s" % "".join((map(lambda x: dataset.findSymbol[x], 
#                                                    np.argmax(new_targets[i],1)))));
#                 print(prefix + "Used label           : %s" % labels_to_use[i][0]);
            
        # Update stats
        total_datapoints_processed += repetition_size;
        
        # Report on error
        print("Total error: %.2f" % total_error);
        
        if (parameters['train_statistics']):
            stats = model.total_statistics(stats);
            print_stats(stats, parameters, prefix='TRAIN ');
        
        # Intermediate testing if this was not the last iteration of training
        # and we have passed the testing threshold
        #if (r != repetition_size-1):
        test(model, dataset, parameters, max_length, base_offset, intervention_range, print_samples=parameters['debug'], 
             sample_size=parameters['sample_testing_size']);
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
    
    