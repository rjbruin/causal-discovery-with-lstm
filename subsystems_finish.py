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

def addOtherInterventionLocations(intervention_locations, topcause):
    # Transform intervention locations to matrix where the 'other' locations
    # are location-1 because we don't want to use the label at the 
    # intervention location for the other expression
    matrix_intervention_locations = np.zeros((2, len(intervention_locations)), dtype='int32');
    matrix_intervention_locations[0 if topcause else 1,:] = np.array(intervention_locations, dtype='int32');
    # Negative intervention locations are allowed
    matrix_intervention_locations[1 if topcause else 0,:] = np.array(intervention_locations, dtype='int32') - 1;
    
    return matrix_intervention_locations;

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
    
    if (parameters['test_in_dataset']):
        output += prefix + "In dataset: %.2f percent\n" % (stats['inDatasetScore']*100);
    
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
    output += prefix + "Skipped because of zero prediction length: %d\n" % stats['skipped_because_intervention_location'];
    
    output += prefix + "! Samples correct: %s" % str(map(lambda (x,y): "%d,%d" % (int(x), int(y)),stats['samplesCorrect']));
    
    output += "\n";
    print(output);

def get_batch(isTrain, dataset, model, intervention_range, max_length, 
              debug=False, base_offset=12, 
              seq2ndmarkov=False, bothcause=False, homogeneous=False):    
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
    nrSamples = 0;
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
        while (len(subbatch) < subbatch_size):
            if (seq2ndmarkov and not topcause):
                branch = storage_bot.get_random_by_length(interventionLocation, getStructure=True);
            else:
                branch = storage.get_random_by_length(interventionLocation, getStructure=True);
            
            if (homogeneous):
                candidates = branch.fullExpressions;
                prime_candidates = branch.primedExpressions;
                if (len(candidates) <= subbatch_size):
                    # Use all available to fill subbatch
                    subbatch = zip(candidates,prime_candidates);
                    n_missing = subbatch_size - len(candidates);
                    nrSamples += len(candidates);
                    if (n_missing > 0):
                        subbatch.extend([('','') for _ in range(n_missing)]);
                else:
                    # Sample from available to fill subbatch
                    candidate_indices = range(len(candidates));
                    np.random.shuffle(candidate_indices);
                    nrSamples += subbatch_size;
                    for i in range(subbatch_size):
                        subbatch.append((candidates[candidate_indices[i]],prime_candidates[candidate_indices[i]]))
            else:
                nrSamples = model.minibatch_size;
                randomPrefix = np.random.randint(0,len(branch.fullExpressions));
                if (topcause):
                    subbatch.append((branch.fullExpressions[randomPrefix],
                                     branch.primedExpressions[randomPrefix]));
                else:
                    subbatch.append((branch.primedExpressions[randomPrefix],
                                     branch.fullExpressions[randomPrefix]));
        
        # Add subbatch to batch
        batch.extend(subbatch);
        interventionLocations.extend(np.ones((len(subbatch)), dtype='int32') * interventionLocation);
    
    data = [];
    targets = [];
    labels = [];
    expressions = [];
    for (expression, expression_prime) in batch:
        if (seq2ndmarkov and not bothcause):
            if (parameters['only_cause_expression'] == 2):
                expression_prime = expression;
                expression = "";
            data, targets, labels, expressions, _ = dataset.processor(";".join([expression, expression_prime, str(int(topcause))]), 
                                                                      data,targets, labels, expressions);
        else:
            data, targets, labels, expressions, _ = dataset.processor(";".join([expression, expression_prime]), 
                                                                      data,targets, labels, expressions);
    
    data = dataset.fill_ndarray(data, 1, fixed_length=model.n_max_digits);
    targets = dataset.fill_ndarray(copy.deepcopy(targets), 1, fixed_length=model.n_max_digits);
    
    return data, targets, labels, expressions, interventionLocations, topcause, nrSamples;

def test(model, dataset, parameters, max_length, base_offset, intervention_range, print_samples=False, 
         sample_size=False, homogeneous=False, returnTestSamples=False):
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
    totalError = 0.0;
    k = 0;
    testSamples = [];
    while k < total:
        # Get data from batch
        test_data, test_targets, _, test_expressions, \
            interventionLocations, topcause, nrSamples = get_batch(False, dataset, model, 
                                                                      intervention_range, 
                                                                      max_length, debug=parameters['debug'],
                                                                      base_offset=base_offset,
                                                                      seq2ndmarkov=parameters['dataset_type'] == 1,
                                                                      bothcause=parameters['bothcause'],
                                                                      homogeneous=parameters['homogeneous']);
        for l in interventionLocations:
            stats['intervention_locations'][l] += 1;
            
        # Make intervention locations into matrix
        interventionLocations = addOtherInterventionLocations(interventionLocations, topcause);
        
        # Create init hidden layers from abstract expressions
        abstractExpressions = np.zeros((model.minibatch_size, model.hidden_dim), dtype='float32');
        if (parameters['use_abstract']):
            for i in range(model.minibatch_size):
                # TODO: make this work with subsystems
                abstractExpr = dataset.abstractExpression(test_expressions[i][0]);
                abstractExpressions[i,:abstractExpr.shape[0]] = abstractExpr;
        
        # Append abstract to data dim
        if (parameters['append_abstract']):
            abstractExpressions = np.zeros((model.minibatch_size,13), dtype='float32');
            for i in range(model.minibatch_size):
                # TODO: make this work with subsystems
                abstractExpressions[i] = dataset.abstractExpression(target_expressions[i][0]);
        
        predictions, other = model.predict(test_data, test_targets, 
                                           interventionLocations=interventionLocations,
                                           nrSamples=nrSamples,
                                           abstractExpressions=abstractExpressions); 
        totalError += other['error'];
        
        if (parameters['only_cause_expression']):
            prediction_1 = predictions;
            predictions = [predictions];
        else:
            prediction_1 = predictions[0];
            prediction_2 = predictions[1];
        
        profiler.start("test batch stats");
        stats, labels_used = model.batch_statistics(stats, predictions, 
                                       test_expressions, interventionLocations, 
                                       other, nrSamples, dataset, test_expressions,
                                       topcause=topcause or parameters['bothcause'], # If bothcause then topcause = 1
                                       testInDataset=parameters['test_in_dataset'],
                                       bothcause=parameters['bothcause']);
        
        for j in range(nrSamples):
            if (parameters['only_cause_expression'] is not False):
                total_labels_used[labels_used[j][0]] = True;
            else:
                total_labels_used[labels_used[j][0]+";"+labels_used[j][1]] = True;
            
            # Save predictions to testSamples
            if (returnTestSamples):
                strData = map(lambda x: dataset.findSymbol[x], 
                              np.argmax(test_targets[j,:,:model.data_dim],len(test_targets.shape)-2));
                strPrediction = map(lambda x: dataset.findSymbol[x], prediction_1[j]);
                if (parameters['only_cause_expression'] is False):
                    strDataBot = map(lambda x: dataset.findSymbol[x], 
                                     np.argmax(test_targets[j,:,model.data_dim:],len(test_targets.shape)-2));
                    strPredictionBot = map(lambda x: dataset.findSymbol[x], prediction_2[j]);
                    testSamples.append((strData,strPrediction,strDataBot,strPredictionBot));
                else:
                    testSamples.append((strData,strPrediction));
        
        # Print samples
        if (print_samples and not printed_samples):
            for i in range(nrSamples):
                prefix = "# ";
                print(prefix + "Intervention location: %d" % interventionLocations[0,i]);
                print(prefix + "Data          1: %s" % "".join((map(lambda x: dataset.findSymbol[x], 
                                                   np.argmax(test_targets[i,:,:model.data_dim],len(test_data.shape)-2)))));
                print(prefix + "Prediction    1: %s" % "".join(map(lambda x: dataset.findSymbol[x], prediction_1[i])));
                print(prefix + "Used label    1: %s" % labels_used[i][0]);
                
                if (not parameters['only_cause_expression']):
                    print(prefix + "Data          2: %s" % "".join((map(lambda x: dataset.findSymbol[x], 
                                                       np.argmax(test_targets[i,:,model.data_dim:],len(test_data.shape)-2)))));
                    print(prefix + "Prediction    2: %s" % "".join(map(lambda x: dataset.findSymbol[x], prediction_2[i])));
                    print(prefix + "Used label    2: %s" % labels_used[i][1]);
            printed_samples = True;

        if (stats['prediction_size'] % printing_interval == 0):
            print("# %d / %d" % (stats['prediction_size'], total));
        profiler.stop("test batch stats");
        
        k += nrSamples;
    
    profiler.profile();
    
    print("Total testing error: %.2f" % totalError);
    
    stats = model.total_statistics(stats, total_labels_used=total_labels_used);
    print_stats(stats, parameters);
    
    if (returnTestSamples):
        return stats, testSamples;
    else:
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
    
    # Check for valid intervention ranges
    if (parameters['intervention_base_offset'] <= 0):
        raise ValueError("Invalid intervention base offset: is %d, must be at least 1." % parameters['intervention_base_offset']);
    
    
    
    # Construct models
    dataset, model = constructModels(parameters, 0, {});
    
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
    
    # Compute batching variables
    repetition_size = dataset.lengths[dataset.TRAIN];
    if (parameters['max_training_size'] is not False):
        repetition_size = min(parameters['max_training_size'],repetition_size);
    next_testing_threshold = parameters['test_interval'] * repetition_size;
    
    
    
    intervention_locations_train = {k: 0 for k in range(model.n_max_digits)};
    for r in range(parameters['repetitions']):
        stats = set_up_statistics(dataset.output_dim, model.n_max_digits);
        total_error = 0.0;
        # Print repetition progress and save to raw results file
        print("Batch %d (repetition %d of %d, dataset 1 of 1) (samples processed after batch: %d)" % \
                (r+1,r+1,parameters['repetitions'],(r+1)*repetition_size));
        
        # Train model per minibatch
        k = 0;
        printedProgress = -1;
        while k < repetition_size:
            profiler.start('train batch');
            profiler.start('get train batch');
            data, target, _, target_expressions, interventionLocations, topcause, nrSamples = \
                get_batch(True, dataset, model, 
                          parameters['intervention_range'], model.n_max_digits, 
                          debug=parameters['debug'],
                          base_offset=parameters['intervention_base_offset'],
                          seq2ndmarkov=parameters['dataset_type'] == 1,
                          bothcause=parameters['bothcause'],
                          homogeneous=parameters['homogeneous']);
            profiler.stop('get train batch');
            
            # Make intervention locations into matrix
            interventionLocations = addOtherInterventionLocations(interventionLocations, topcause);
            
            # Create init hidden layers from abstract expressions
            abstractExpressions = np.zeros((model.minibatch_size, model.hidden_dim), dtype='float32');
            if (parameters['use_abstract']):
                for i in range(model.minibatch_size):
                    # TODO: make this work with subsystems
                    abstractExpr = dataset.abstractExpression(target_expressions[i][0]);
                    abstractExpressions[i,:abstractExpr.shape[0]] = abstractExpr;
            
            # Append abstract to data dim
            if (parameters['append_abstract']):
                abstractExpressions = np.zeros((model.minibatch_size,13), dtype='float32');
                for i in range(model.minibatch_size):
                    # TODO: make this work with subsystems
                    abstractExpressions[i] = dataset.abstractExpression(target_expressions[i][0]);
            
            # Run training
            profiler.start('train sgd');
            outputs = model.sgd(dataset, data, target, parameters['learning_rate'],
                                  nrSamples=model.minibatch_size, expressions=target_expressions,
                                  interventionLocations=interventionLocations,
                                  topcause=topcause or parameters['bothcause'], bothcause=parameters['bothcause'],
                                  use_label_search=parameters['use_label_search'], abstractExpressions=abstractExpressions);
            total_error += outputs[0];
            profiler.stop('train sgd');
            
            # Print batch progress
            if ((k+model.minibatch_size) % (model.minibatch_size*4) < model.minibatch_size and \
                (k+model.minibatch_size) / (model.minibatch_size*4) > printedProgress):
                printedProgress = (k+model.minibatch_size) / (model.minibatch_size*4);
                print("# %d / %d (error = %.2f)" % (k+model.minibatch_size, repetition_size, total_error));
            
            profiler.stop('train batch');
            
            k += nrSamples;
        
        # Report on error
        print("Total error: %.2f" % total_error);
        
        # Intermediate testing if this was not the last iteration of training
        # and we have passed the testing threshold
        #if (r != repetition_size-1):
        test(model, dataset, parameters, model.n_max_digits, parameters['intervention_base_offset'], parameters['intervention_range'], print_samples=parameters['debug'], 
             sample_size=parameters['sample_testing_size'], homogeneous=parameters['homogeneous']);
        
        # Save weights to pickles
        if (saveModels):
            saveVars = model.getVars();
            save_to_pickle('saved_models/%s_%d.model' % (name, r), saveVars, settings=parameters);
    
    print("Training finished!");
    
    