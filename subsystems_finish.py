'''
Created on 9 sep. 2016

@author: Robert-Jan
'''

import time;
import sys, os;
from math import floor;
from collections import deque;
import subprocess;

from tools.file import save_to_pickle, load_from_pickle_with_filename;
from tools.arguments import processCommandLineArguments;
from tools.model import constructModels, set_up_statistics;
from tools.gpu import using_gpu; # @UnresolvedImport

import numpy as np;
import theano;
import copy;
from profiler import profiler

import trackerreporter;
from tools.arguments import processKeyValue

def addOtherInterventionLocations(intervention_locations, topcause):
    # Transform intervention locations to matrix where the 'other' locations
    # are location-1 because we don't want to use the label at the 
    # intervention location for the other expression
    matrix_intervention_locations = np.zeros((2, len(intervention_locations)), dtype='int32');
    matrix_intervention_locations[0 if topcause else 1,:] = np.array(intervention_locations, dtype='int32');
    # Negative intervention locations are allowed
    matrix_intervention_locations[1 if topcause else 0,:] = np.array(intervention_locations, dtype='int32') - 1;
    
    return matrix_intervention_locations;

def print_stats(stats, parameters, experimentId, currentIteration, prefix=''):
    # Print statistics
    printF("\n", experimentId, currentIteration);

    # Print statistics
    if (parameters['rnn_version'] != 1):
        printF(prefix + "Score: %.2f percent" % (stats['score']*100), experimentId, currentIteration);
    else:
        printF(prefix + "Score: %.2f percent" % (stats['digit_2_total_score']*100), experimentId, currentIteration);
    
    digit_score = (stats['digit_1_total_score']) * 100.;
    if (not parameters['only_cause_expression']):
        digit_score = (stats['digit_1_total_score'] + stats['digit_2_total_score']) * 50.;
    printF(prefix + "Digit-based score: %.2f percent" % (digit_score), experimentId, currentIteration);
    
    if (not parameters['only_cause_expression']):
        if (parameters['dataset_type'] != 3):
            printF(prefix + "Structure score cause: %.2f percent" % (stats['structureScoreCause']*100), experimentId, currentIteration);
            printF(prefix + "Structure score effect: %.2f percent" % (stats['structureScoreEffect']*100), experimentId, currentIteration);
        printF(prefix + "Structure score top: %.2f percent" % (stats['structureScoreTop']*100), experimentId, currentIteration);
        printF(prefix + "Structure score bot: %.2f percent" % (stats['structureScoreBot']*100), experimentId, currentIteration);
        if (parameters['dataset_type'] != 3):
            printF(prefix + "Structure score: %.2f percent" % (stats['structureScore']*100), experimentId, currentIteration);
            printF(prefix + "Effect score: %.2f percent" % (stats['effectScore']*100), experimentId, currentIteration);
            printF(prefix + "Effect score including no effect: %.2f percent" % (stats['allEffectScore']*100), experimentId, currentIteration);
    
    if (parameters['dataset_type'] != 3):
        if (not parameters['answering']):
            printF(prefix + "Valid: %.2f percent" % (stats['semantically_valid_score']*100), experimentId, currentIteration);
        printF(prefix + "Syntactically valid: %.2f percent" % (stats['syntactically_valid_score']*100), experimentId, currentIteration);
        if (not parameters['answering']):
            printF(prefix + "Valid left hand side: %.2f percent" % (stats['left_hand_valid_score']*100), experimentId, currentIteration);
            printF(prefix + "Valid right hand side: %.2f percent" % (stats['right_hand_valid_score']*100), experimentId, currentIteration);
            printF(prefix + "Score with valid left hand side: %.2f percent" % (stats['left_hand_valid_correct_score']*100), experimentId, currentIteration);
            printF(prefix + "Partially predicted left hand sides: %.2f percent" % ((stats['left_hand_valid_with_prediction_size']*100) / float(stats['prediction_size'])), experimentId, currentIteration);
            printF(prefix + "Valid left hand with partially predicted left hand side: %.2f percent" % (stats['valid_left_hand_with_prediction_score']*100.), experimentId, currentIteration);
            printF(prefix + "Score with partially predicted left hand side: %.2f percent" % (stats['left_hand_valid_with_prediction_score']*100), experimentId, currentIteration);
            printF(prefix + "Score with given left hand side: %.2f percent" % (stats['left_hand_given_score']*100), experimentId, currentIteration);
            printF(prefix + "Score with partially predicted valid left hand side: %.2f percent" % (stats['valid_left_hand_valid_with_prediction_score']*100), experimentId, currentIteration);
        
#         printF(prefix + "Local valid: %.2f percent" % (stats['localValidScore']*100), experimentId, currentIteration);
#         if (not parameters['only_cause_expression']):
#             printF(prefix + "Structure valid cause: %.2f percent" % (stats['structureValidScoreCause']*100), experimentId, currentIteration);
#             printF(prefix + "Structure valid effect: %.2f percent" % (stats['structureValidScoreEffect']*100), experimentId, currentIteration);
#             printF(prefix + "Structure valid top: %.2f percent" % (stats['structureValidScoreTop']*100), experimentId, currentIteration);
#             printF(prefix + "Structure valid bot: %.2f percent" % (stats['structureValidScoreBot']*100), experimentId, currentIteration);
#             printF(prefix + "Local valid cause: %.2f percent" % (stats['localValidScoreCause']*100), experimentId, currentIteration);
#             printF(prefix + "Local valid effect: %.2f percent" % (stats['localValidScoreEffect']*100), experimentId, currentIteration);
        printF(prefix + "Intervention locations:   %s" % (str(stats['intervention_locations'])), experimentId, currentIteration);
        if (parameters['test_in_dataset']):
            printF(prefix + "In dataset: %.2f percent" % (stats['inDatasetScore']*100), experimentId, currentIteration);

    if (not parameters['only_cause_expression']):
        printF(prefix + "Digit-based (1) score: %.2f percent" % (stats['digit_1_total_score']*100), experimentId, currentIteration);
        printF(prefix + "Digit-based (1) individual scores histogram: %s percent" % (str(stats['digit_1_score'])), experimentId, currentIteration);
        printF(prefix + "Digit prediction (1) histogram:   %s" % (str(stats['prediction_1_histogram'])), experimentId, currentIteration);
        
        printF(prefix + "Digit-based (2) score: %.2f percent" % (stats['digit_2_total_score']*100), experimentId, currentIteration);
        printF(prefix + "Digit-based (2) individual scores histogram: %s percent" % (str(stats['digit_2_score'])), experimentId, currentIteration);
        printF(prefix + "Digit prediction (2) histogram:   %s" % (str(stats['prediction_2_histogram'])), experimentId, currentIteration);
        
        if (parameters['dataset_type'] != 3):
            printF(prefix + "Prediction size (1) histogram:   %s" % (str(stats['prediction_1_size_histogram'])), experimentId, currentIteration);
            printF(prefix + "Prediction size (2) histogram:   %s" % (str(stats['prediction_2_size_histogram'])), experimentId, currentIteration);
        else:
            dp_length = 20 - 8;
            printF(prefix + "Digit-based score (1st quarter): %.2f percent" % (np.mean([stats['digit_1_score'][i]*50. + stats['digit_2_score'][i]*50. for i in range(int((0./4)*dp_length),int((1./4)*dp_length))])), experimentId, currentIteration);
            printF(prefix + "Digit-based score (2nd quarter): %.2f percent" % (np.mean([stats['digit_1_score'][i]*50. + stats['digit_2_score'][i]*50. for i in range(int((1./4)*dp_length),int((2./4)*dp_length))])), experimentId, currentIteration);
            printF(prefix + "Digit-based score (3rd quarter): %.2f percent" % (np.mean([stats['digit_1_score'][i]*50. + stats['digit_2_score'][i]*50. for i in range(int((2./4)*dp_length),int((3./4)*dp_length))])), experimentId, currentIteration);
            printF(prefix + "Digit-based score (4th quarter): %.2f percent" % (np.mean([stats['digit_1_score'][i]*50. + stats['digit_2_score'][i]*50. for i in range(int((3./4)*dp_length),int((4./4)*dp_length))])), experimentId, currentIteration);
        
#     printF(prefix + "Prediction size histogram:   %s\n" % (str(stats['prediction_size_histogram']));
    printF(prefix + "Digit histogram:   %s" % (str(stats['prediction_histogram'])), experimentId, currentIteration);
    
    if (parameters['dataset_type'] != 3 and 'label_size_score' in stats):
        printF(prefix + "Label sizes: %s" % (str(stats['label_sizes'])), experimentId, currentIteration);
        for size in stats['label_size_score'].keys():
            printF(prefix + "Score by label size = %d: %.2f percent" % (size, stats['label_size_score'][size]*100.), experimentId, currentIteration);
    if ('input_size_score' in stats):
        printF(prefix + "Input sizes: %s" % (str(stats['input_sizes'])), experimentId, currentIteration);
        for size in stats['input_size_score'].keys():
            printF(prefix + "Score by input size = %d: %.2f percent" % (size, stats['input_size_score'][size]*100.), experimentId, currentIteration);
    
    printF(prefix + "Error margin 1 score: %.2f percent" % (stats['error_1_score']*100.), experimentId, currentIteration);
    printF(prefix + "Error margin 2 score: %.2f percent" % (stats['error_2_score']*100.), experimentId, currentIteration);
    printF(prefix + "Error margin 3 score: %.2f percent" % (stats['error_3_score']*100.), experimentId, currentIteration);
    
    printF(prefix + "All error margins: %s" % str(stats['error_histogram']), experimentId, currentIteration);
    
    trueSizes = parameters['n_max_digits'];
    nrCorrects = parameters['n_max_digits'];
    if (parameters['answering']):
        trueSizes = 5;
        nrCorrects = 6;
    if (parameters['answering']):
        for trueSize in range(1,trueSizes+1):
            for nrCorrect in range(min(nrCorrects,trueSize)+1):
                printF(prefix + "Label size %d nr correct %d: %.2f (%d)" % (trueSize, nrCorrect, stats['correct_matrix_scores'][trueSize,nrCorrect] * 100., stats['correct_matrix'][trueSize,nrCorrect]), experimentId, currentIteration);
    
    if ('label_size_input_size_confusion_score' in stats):
        np.set_printoptions(precision=8);
        for i in range(stats['label_size_input_size_confusion_score'].shape[0]):
            printF(prefix + "Label / input size row %d: %s" % (i, np.array2string(stats['label_size_input_size_confusion_score'][i,:]).replace('\n', '')), experimentId, currentIteration);
        np.set_printoptions(precision=3);
    
    if (parameters['dataset_type'] != 3):
        printF(prefix + "Unique labels used: %d" % stats['unique_labels_predicted'], experimentId, currentIteration);
        printF(prefix + "Skipped because of zero prediction length: %d" % stats['skipped_because_intervention_location'], experimentId, currentIteration);
    
#     printF(prefix + "! Samples correct: %s" % str(map(lambda (x,y): "%d,%d" % (int(x), int(y)),stats['samplesCorrect']));
    
    printF("\n",experimentId, currentIteration);

def processSampleDiscreteProcess(line, data_dim, oneHot, EOS_symbol_index):
    """
    Data is ndarray of size (nr lines, sequence length, nr input vars).
    Targets is same as data.
    Labels is same as data.
    Expressions is string representation.
    """
    sample1, sample2 = line.split(";");
    encoding = np.zeros((len(sample1), data_dim*2), dtype='float32');
    
    for i in range(len(sample1)):
        encoding[i,oneHot[sample1[i]]] = 1.0;
    
    for i in range(len(sample2)):
        encoding[i,oneHot[sample2[i]]+data_dim] = 1.0;
    
    return encoding, encoding, (sample1, sample2);

def load_data(parameters, processor, dataset_model):
    f = open(os.path.join(parameters['dataset'],'all.txt'));
    
    dataset_data = {};
    label_index = {};
    i = 0;
    for line in f:
        packedData = processor(line.strip(), dataset_model.data_dim, dataset_model.oneHot, dataset_model.EOS_symbol_index);
#         dataset_data.append((data, labels));
        dataset_data[line.strip()] = packedData;
        label_index[i] = line.strip();
        i += 1;
    
    return dataset_data, label_index;

def get_sample_index(which_part, dataset_size, parameters):
    """
    Which_part: 0 = train, 1 = test, 2 = validation.
    Validation set offset is always right after test_offset.
    """
    val_offset = (parameters['test_offset'] + parameters['test_size']);
    
    test_sample_range = [parameters['test_offset']*dataset_size,parameters['test_offset']*dataset_size+parameters['test_size']*dataset_size];
    val_offset_range = [val_offset*dataset_size,val_offset*dataset_size+parameters['val_size']*dataset_size];
    
    sampleIndex = np.random.randint(0,dataset_size);
    while ((which_part == 0 and (sampleIndex >= test_sample_range[0] and sampleIndex < val_offset_range[1])) or
           (which_part == 1 and (sampleIndex < test_sample_range[0] or sampleIndex >= test_sample_range[1])) or
           (which_part == 2 and (sampleIndex < val_offset_range[0] or sampleIndex >= val_offset_range[1]))):
        sampleIndex = np.random.randint(0,dataset_size);
    
    return sampleIndex;

def dataset_health(dataset_set, label_index, n=100):
    # Draw n random samples to inspect
    dataset_size = len(label_index.keys());
    indices = 0.;
    for _ in range(n):
        sampleIndex = np.random.randint(0,dataset_size);
        encoded, _ = dataset_data[label_index[sampleIndex]];
        indices += np.sum(encoded);
    
    return indices;

def batch_health(data):
    indices = 0;
    for i in range(data.shape[0]):
        encoded = data[i];
        indices += np.sum(encoded);
    
    return indices;

def get_batch_unprefixed(which_part, dataset_model, dataset_data, label_index, parameters):    
    # Reseed the random generator to prevent generating identical batches
    np.random.seed();
    
    # Set range to sample from
    dataset_size = len(label_index.keys());
    
    data = [];
    targets = [];
    labels = [];
    expressions = [];
    while (len(data) < parameters['minibatch_size']):
        # Get random sample
        sampleIndex = get_sample_index(which_part, dataset_size, parameters);
        # Append to data
        encoded, encodedTargets, sampleLabels = dataset_data[label_index[sampleIndex]];
        data.append(encoded);
        targets.append(encodedTargets);
        labels.append(np.argmax(encodedTargets));
        expressions.append(sampleLabels);
    
    # Make data ndarray
#     data = np.array(data);
    data = dataset_model.fill_ndarray(data, 1, fixed_length=parameters['n_max_digits']);
    targets = np.array(targets, dtype='float32');
    
    return data, targets, labels, expressions, batch_health(data);

def get_batch_prefixed(isTrain, dataset, model, intervention_range, max_length, 
                       debug=False, base_offset=12, 
                       seq2ndmarkov=False, bothcause=False, homogeneous=False,
                       answering=False):    
    # Reseed the random generator to prevent generating identical batches
    np.random.seed();
    
    if (isTrain == 0):
        storage = dataset.expressionsByPrefix;
        if (seq2ndmarkov and not parameters['only_cause_expression']):
            storage_bot = dataset.expressionsByPrefixBot;
    elif (isTrain == 1):
        storage = dataset.testExpressionsByPrefix;
        if (seq2ndmarkov and not parameters['only_cause_expression']):
            storage_bot = dataset.testExpressionsByPrefixBot;
    else:
        storage = dataset.validateExpressionsByPrefix;
    
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
                    if (answering):
                        interventionLocation = branch.fullExpressions[randomPrefix].index("=");
                else:
                    subbatch.append((branch.primedExpressions[randomPrefix],
                                     branch.fullExpressions[randomPrefix]));
                    if (answering):
                        interventionLocation = branch.primedExpressions[randomPrefix].index("=");
        
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

def get_batch(isTrain, dataset, model, intervention_range, max_length, parameters, dataset_data, label_index,
              debug=False, base_offset=12, 
              seq2ndmarkov=False, bothcause=False, homogeneous=False,
              answering=False):    
    if (parameters['simple_data_loading']):
        data, targets, labels, expressions, health = get_batch_unprefixed(isTrain, dataset, dataset_data, label_index, parameters);
        return data, targets, labels, expressions, np.zeros((data.shape[0])), True, parameters['minibatch_size'], health;
    else:
        data, targets, labels, expressions, interventionLocations, topcause, nrSamples = \
            get_batch_prefixed(isTrain, dataset, model, intervention_range, max_length, debug, 
                               base_offset, seq2ndmarkov, bothcause, homogeneous, answering)
        return data, targets, labels, expressions, interventionLocations, topcause, nrSamples, 0;

def test(model, dataset, dataset_data, label_index, parameters, max_length, base_offset, intervention_range, print_samples=False, 
         sample_size=False, homogeneous=False, returnTestSamples=False):
    total = dataset.lengths[dataset.TEST];
    printing_interval = 1000;
    if (parameters['max_dataset_size'] is not False):
        printing_interval = 100;
    elif (sample_size != False):
        total = sample_size;
    
    # Test
    test_set_size = len(dataset.testExpressionsByPrefix.expressions);
    if (parameters['simple_data_loading']):
        test_set_size = parameters['test_size']*len(dataset_data);
    printF("Testing... %d from %d" % (total, test_set_size), experimentId, currentIteration);
    
    # Set up statistics
    stats = set_up_statistics(dataset.output_dim, model.n_max_digits, dataset.oneHot.keys());
    total_labels_used = {};
    
    # Predict
    printed_samples = False;
    totalError = 0.0;
    k = 0;
    testSamples = [];
    while k < total:
        # Get data from batch
        test_data, test_targets, _, test_expressions, \
            interventionLocations, topcause, nrSamples, _ = get_batch(1, dataset, model, 
                                                                      intervention_range, 
                                                                      max_length, 
                                                                      parameters, dataset_data, label_index, 
                                                                      debug=parameters['debug'],
                                                                      base_offset=base_offset,
                                                                      seq2ndmarkov=parameters['dataset_type'] == 1,
                                                                      bothcause=parameters['bothcause'],
                                                                      homogeneous=parameters['homogeneous'],
                                                                      answering=parameters['answering']);
        for l in interventionLocations:
            stats['intervention_locations'][l] += 1;
            
        # Make intervention locations into matrix
        interventionLocations = addOtherInterventionLocations(interventionLocations, topcause);
        
        predictions, other = model.predict(test_data, test_targets, 
                                           interventionLocations=interventionLocations,
                                           nrSamples=nrSamples); 
        totalError += other['summed_error'];
        
        if (parameters['only_cause_expression']):
            prediction_1 = predictions;
            predictions = [predictions];
        else:
            prediction_1 = predictions[0];
            prediction_2 = predictions[1];
        
        profiler.start("test batch stats");
        stats, labels_used, notInDataset = model.batch_statistics(stats, predictions, 
                                       test_expressions, interventionLocations, 
                                       other, nrSamples, dataset, test_expressions,
                                       dataset_data, parameters,
                                       topcause=topcause or parameters['bothcause'], # If bothcause then topcause = 1
                                       testInDataset=parameters['test_in_dataset'],
                                       bothcause=parameters['bothcause']);
        
        for j in range(nrSamples):
            if (parameters['only_cause_expression'] is not False):
                if (labels_used[j][0] not in total_labels_used):
                    total_labels_used[labels_used[j][0]] = True;
            else:
                if (labels_used[j][0]+";"+labels_used[j][1] not in total_labels_used):
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
                whitespaceprefix = "".join([" " for t in range(parameters['lag'])]);
                if (parameters['dataset_type'] != 3):
                    printF(prefix + "Intervention location: %d" % interventionLocations[0,i], experimentId, currentIteration);
                    whitespaceprefix = "";
                printF(prefix + "Data          1: %s" % "".join((map(lambda x: dataset.findSymbol[x], 
                                                   np.argmax(test_targets[i,:,:model.data_dim],len(test_data.shape)-2)))), experimentId, currentIteration);
                printF(prefix + "Prediction    1: %s" % (whitespaceprefix + "".join(map(lambda x: dataset.findSymbol[x], prediction_1[i]))), experimentId, currentIteration);
                printF(prefix + "Used label    1: %s" % labels_used[i][0], experimentId, currentIteration);
                
                if (not parameters['only_cause_expression']):
                    printF(prefix + "Data          2: %s" % "".join((map(lambda x: dataset.findSymbol[x], 
                                                       np.argmax(test_targets[i,:,model.data_dim:],len(test_data.shape)-2)))), experimentId, currentIteration);
                    printF(prefix + "Prediction    2: %s" % (whitespaceprefix + "".join(map(lambda x: dataset.findSymbol[x], prediction_2[i]))), experimentId, currentIteration);
                    printF(prefix + "Used label    2: %s" % labels_used[i][1], experimentId, currentIteration);
            printed_samples = True;

        if (stats['prediction_size'] % printing_interval == 0):
            printF("# %d / %d" % (stats['prediction_size'], total), experimentId, currentIteration);
        profiler.stop("test batch stats");
        
        k += nrSamples;
    
    profiler.profile();
    
    printF("Total testing error: %.2f" % totalError, experimentId, currentIteration);
    printF("Mean testing error: %.8f" % (totalError/float(k)), experimentId, currentIteration);
    
    stats = model.total_statistics(stats, dataset, total_labels_used=total_labels_used);
    print_stats(stats, parameters, experimentId, currentIteration);
    
    if (returnTestSamples):
        return stats, totalError, testSamples;
    else:
        return stats, totalError;

def validate(model, dataset, dataset_data, label_index, parameters, max_length, base_offset, intervention_range, print_samples=False, 
             sample_size=False, homogeneous=False):
    total = parameters['val_size']*np.sum(dataset.lengths);
    printing_interval = 1000;
    if (sample_size != False):
        total = sample_size;
        
    # Validate
    printF("Validating... %d from %d" % (total, parameters['val_size']*np.sum(dataset.lengths)), experimentId, currentIteration);
    
    # Predict
    totalError = 0.0;
    k = 0;
    while k < total:
        # Get data from batch
        val_data, val_targets, _, val_expressions, \
            interventionLocations, topcause, nrSamples, _ = get_batch(2, dataset, model, 
                                                                      intervention_range, 
                                                                      max_length, 
                                                                      parameters, dataset_data, label_index, 
                                                                      debug=parameters['debug'],
                                                                      base_offset=base_offset,
                                                                      seq2ndmarkov=parameters['dataset_type'] == 1,
                                                                      bothcause=parameters['bothcause'],
                                                                      homogeneous=parameters['homogeneous'],
                                                                      answering=parameters['answering']);
        for l in interventionLocations:
            stats['intervention_locations'][l] += 1;
            
        # Make intervention locations into matrix
        interventionLocations = addOtherInterventionLocations(interventionLocations, topcause);
        
        predictions, other = model.predict(val_data, val_targets, 
                                           interventionLocations=interventionLocations,
                                           nrSamples=nrSamples); 
        totalError += other['summed_error'];
        
        if (parameters['only_cause_expression']):
            prediction_1 = predictions;
            predictions = [predictions];
        else:
            prediction_1 = predictions[0];
            prediction_2 = predictions[1];

        if (stats['prediction_size'] % printing_interval == 0):
            printF("# %d / %d" % (stats['prediction_size'], total), experimentId, currentIteration);
        
        k += nrSamples;
    
    profiler.profile();
    
    printF("Total validation error: %.2f" % totalError, experimentId, currentIteration);
    printF("Mean validation error: %.8f" % (totalError/float(k)), experimentId, currentIteration);
    
    return stats, totalError, totalError/float(k);

if __name__ == '__main__':
    theano.config.floatX = 'float32';
    np.set_printoptions(precision=3, threshold=10000000);
    profiler.off();
    
    # Settings
    api_key = os.environ.get('TCDL_API_KEY');
    if (api_key is None):
        raise ValueError("No API key present for reporting to tracker!");
    score_types = {'Precision': 'Score',
                   'Training loss': 'Total error',
                   'Training loss (m)': 'Mean error',
                   'Testing loss': 'Total testing error',
                   'Testing loss (m)': 'Mean testing error',
                   'Validation loss': 'Total validation error',
                   'Validation loss (m)': 'Mean validation error',
                   'Digit precision': 'Digit-based score',
                   'Digit (1) precision': 'Digit-based (1) score',
                   'Digit (2) precision': 'Digit-based (2) score',
                   'Digit precision (1/4)': 'Digit-based score (1st quarter)',
                   'Digit precision (2/4)': 'Digit-based score (2nd quarter)',
                   'Digit precision (3/4)': 'Digit-based score (3rd quarter)',
                   'Digit precision (4/4)': 'Digit-based score (4th quarter)',
                   'Train Precision': 'TRAIN Score',
                   'Train Digit precision': 'TRAIN Digit-based score',
                   'Structure precision': 'Structure score',
                   'Structure pr. (t)': 'Structure score top',
                   'Structure pr. (b)': 'Structure score bot',
                   'Effect precision': 'Effect score',
                   'Mistake (1) precision': 'Error margin 1 score',
                   'Mistake (2) precision': 'Error margin 2 score',
                   'Mistake (3) precision': 'Error margin 3 score',
                   'Validity': 'Valid',
                   'Validity (c)': 'Structure valid cause',
                   'Validity (e)': 'Structure valid effect',
                   'Local validity': 'Local valid',
                   'Local validity (c)': 'Local valid cause',
                   'Local validity (e)': 'Local valid effect',
                   'In dataset': 'In dataset',
                   'Skipped': 'Skipped because of zero prediction length',
                   'Unique labels': 'Unique labels used',
                   'f-subs prediction score': 'f-subs prediction score',
                   'f-subs prediction cause score': 'f-subs prediction score (c)',
                   'f-subs prediction effect score': 'f-subs prediction score (e)',
                   'Mean data health': 'Average data health',
                   'Stddev data health': 'Stddev data health',
                   'Mean model health': 'Average model health',
                   'Stddev model health': 'Stddev model health',
                   'Syntax': 'Syntactically valid',
                   'Syntax (l)': 'Valid left hand side',
                   'Syntax (r)': 'Valid right hand side',
                   'Valid with valid left': 'Score with valid left hand side',
                   'Partial valid left hand': 'Valid left hand with partially predicted left hand side',
                   'Partial left hand score': 'Score with partially predicted left hand side',
                   'Full left hand score': 'Score with given left hand side',
                   'Partial valid left hand score': 'Score with partially predicted valid left hand side',
                   'Partial left hand': 'Partially predicted left hand sides'};
    for size in range(20):
        score_types['Label.size %d' % size] = 'Score by label size = %d' % size;
    for size in range(20):
        score_types['Inpt.size %d' % size] = 'Score by input size = %d' % size;
    for trueSize in range(20):
        for nrCorrect in range(20):
            score_types['T %d C %d' % (trueSize, nrCorrect)] = 'Label size %d nr correct %d' % (trueSize, nrCorrect);
    trackerreporter.init('http://rjbruin.nl/experimenttracker/api/',api_key);
    
    cmdargs = sys.argv[1:];
    # Check for experiment settings file argument and obtain new arguments
    allparameters = processCommandLineArguments(cmdargs);
    newparameters = [];
    if (allparameters[0]['debug']):
        newparameters = allparameters;
        for i in range(len(allparameters)):
            allparameters[i]['basename'] = allparameters[i]['name'];
    else:
        for i in range(len(allparameters)):
            iterative = False;
            # Ask for experiment base name
            basename = raw_input("Experiment %d name (%s): " % (i+1,allparameters[i]['name']));
            if (' ' in basename):
                raise ValueError("Experiment name cannot contain whitespace! Offending name: \"%s\"" % basename);
            allparameters[i]['basename'] = allparameters[i]['name'];
            if (basename != ''):
                allparameters[i]['basename'] = basename;
            allparameters[i]['name'] = allparameters[i]['basename'] + time.strftime("_%d-%m-%Y_%H-%M-%S");
            
            # Ask for iterative parameter
            iterativeArgs = raw_input("(optional) Add one iterative parameter where values are separated by commas (e.g. '--key value1,value2,value3'): ");
            if (iterativeArgs != ""):
                iterativeArgs = iterativeArgs.split(" ");
                extraArgs = [];
                key = iterativeArgs[0][2:];
                suffices = [];
                for k, val in enumerate(iterativeArgs[1].split(",")):
                    suffix = raw_input("Provide the suffix to the name for iteration %d: " % k);
                    newparams = copy.deepcopy(allparameters[i]);
                    newparams[key] = processKeyValue(key,val);
                    newparams['basename'] += suffix;
                    newparams['name'] += suffix;
                    newparameters.append(newparams);
            else:
                newparameters.append(allparameters[i]);
    
    allparameters = newparameters;
    for i in range(len(allparameters)):            
        # Construct output paths
        allparameters[i]['output_path'] = './raw_results/%s.txt' % (allparameters[i]['name']);
        while (os.path.exists(allparameters[i]['output_path'])):
            allparameters[i]['name'] += '-';
            allparameters[i]['output_path'] = './raw_results/%s.txt' % (allparameters[i]['name']);
    
    for parameters in allparameters:
        # Initiate experiment at tracker and obtain experiment ID
        if (parameters['report_to_tracker']):
            if ('multipart_dataset' in parameters):
                datasets = parameters['multipart_dataset'];
            else:
                datasets = 1;
            experimentId = trackerreporter.initExperiment(parameters['basename'], totalProgress=parameters['repetitions'], 
                                                totalDatasets=datasets, scoreTypes=score_types.keys(), 
                                                scoreIdentifiers=score_types);
            if (experimentId is False):
                print("WARNING! Experiment could not be posted to tracker!");
                experimentId = 0;
        else:
            experimentId = 0;
        currentIteration = 1;
        currentDataset = 1;        
        
        # Construct outputPath and new printing target
        name = parameters['name'];
        outputPath = parameters['output_path'];
        printf = open(outputPath, 'w');
        printf.close();
        saveModels = True;
        def printF(s, experimentId, currentIt):
            print(s);
            printf = open(outputPath, 'a');
            if (s != "" and s[0] != "#"):
                printf.write(s + "\n");
            printf.close();
            if (parameters['report_to_tracker']):
                trackerreporter.fromExperimentOutput(experimentId, s, atProgress=currentIt, atDataset=1);
        
        # Print parameters
        printF(str(parameters), experimentId, currentIteration);
        
        # Warn for unusual parameters
        if (parameters['max_dataset_size'] is not False):
            printF("WARNING! RUNNING WITH LIMIT ON DATASET SIZE!", experimentId, currentIteration);
        if (not using_gpu()):
            printF("WARNING! RUNNING WITHOUT GPU USAGE!", experimentId, currentIteration);
        
        # Check for valid subbatch size
        if (parameters['minibatch_size'] % parameters['subbatch_size'] != 0):
            raise ValueError("Subbatch size is not compatible with minibatch size: m.size = %d, s.size = %d" % 
                                (parameters['minibatch_size'], parameters['subbatch_size']));
        
        # Check for valid intervention ranges
        if (parameters['intervention_base_offset'] <= 0):
            raise ValueError("Invalid intervention base offset: is %d, must be at least 1." % parameters['intervention_base_offset']);
        
        # Set simple loading processor
        processor = None;
        if (parameters['dataset_type'] == 3):
            processor = processSampleDiscreteProcess;
        
        
        # Construct models
        dataset, model = constructModels(parameters, 0, {});
        
        # Load pretrained only_cause_expression = 1 model
        if (parameters['load_cause_expression_1'] is not False):
            loadedVars, _ = load_from_pickle_with_filename("./saved_models/" + parameters['load_cause_expression_1']);
            if (model.loadPartialDataDimVars(dict(loadedVars), 0, model.data_dim)):
                printF("Loaded pretrained model (expression 1) successfully!", experimentId, currentIteration);
            else:
                raise ValueError("Loading pretrained model failed: wrong variables supplied!");
        
        # Load pretrained only_cause_expression = 2 model
        if (parameters['load_cause_expression_2'] is not False):
            loadedVars, _ = load_from_pickle_with_filename("./saved_models/" + parameters['load_cause_expression_2']);
            if (model.loadPartialDataDimVars(dict(loadedVars), model.data_dim, model.data_dim)):
                printF("Loaded pretrained model (expression 2) successfully!", experimentId, currentIteration);
            else:
                raise ValueError("Loading pretrained model failed: wrong variables supplied!");
        
        # Train on all datasets in succession
        # Print settings headers to raw results file
        printF("# " + str(parameters), experimentId, currentIteration);
        
        # Compute batching variables
        repetition_size = dataset.lengths[dataset.TRAIN];
        next_testing_threshold = parameters['test_interval'] * repetition_size;
        
        dataset_data = None;
        label_index = None;
        if (parameters['simple_data_loading']):
            dataset_data, label_index = load_data(parameters, processor, dataset);
                
        if (not os.path.exists(os.path.join('.','figures'))):
            os.makedirs(os.path.join('.','figures'));
        model.plotWeights("%s_0" % (name));
        
        intervention_locations_train = {k: 0 for k in range(model.n_max_digits)};
        val_error_stack = deque();
        mean_error_stack = deque();
        last_val_error_avg = 0.0;
        for r in range(parameters['repetitions']):
            stats = set_up_statistics(dataset.output_dim, model.n_max_digits, dataset.oneHot.keys());
            total_error = 0.0;
            # Print repetition progress and save to raw results file
            train_set_size = len(dataset.expressionsByPrefix.expressions);
            if (parameters['simple_data_loading']):
                train_size = 1 - parameters['test_size'];
                if (parameters['early_stopping'] or parameters['force_validation']):
                    train_size -= parameters['val_size'];
                train_set_size = train_size*len(dataset_data);
            printF("Batch %d (repetition %d of %d, dataset 1 of 1) (samples processed after batch: %d from %d)" % \
                    (r+1,r+1,parameters['repetitions'],(r+1)*repetition_size, train_set_size), experimentId, currentIteration);
            currentIteration = r+1;
            currentDataset = 1;
            
            # Train model per minibatch
            k = 0;
            printedProgress = -1;
            data_healths = [];
            model_healths = [];
            while k < repetition_size:
                profiler.start('train batch');
                profiler.start('get train batch');
                data, target, _, target_expressions, interventionLocations, topcause, nrSamples, health = \
                    get_batch(0, dataset, model, 
                              parameters['intervention_range'], model.n_max_digits, 
                              parameters, dataset_data, label_index,
                              debug=parameters['debug'],
                              base_offset=parameters['intervention_base_offset'],
                              seq2ndmarkov=parameters['dataset_type'] == 1,
                              bothcause=parameters['bothcause'],
                              homogeneous=parameters['homogeneous'],
                              answering=parameters['answering']);
                data_healths.append(health);
                model_healths.append(model.modelHealth());
                profiler.stop('get train batch');
                
                # Make intervention locations into matrix
                interventionLocations = addOtherInterventionLocations(interventionLocations, topcause);
                
                # Run training
                profiler.start('train sgd');
                outputs = model.sgd(dataset, data, target, parameters['learning_rate'],
                                      nrSamples=model.minibatch_size, expressions=target_expressions,
                                      interventionLocations=interventionLocations,
                                      topcause=topcause or parameters['bothcause'], bothcause=parameters['bothcause']);
                total_error += outputs[1];
                profiler.stop('train sgd');
                
                # Print batch progress
                if ((k+model.minibatch_size) % (model.minibatch_size*4) < model.minibatch_size and \
                    (k+model.minibatch_size) / (model.minibatch_size*4) > printedProgress):
                    printedProgress = (k+model.minibatch_size) / (model.minibatch_size*4);
                    printF("# %d / %d (error = %.2f)" % (k+model.minibatch_size, repetition_size, total_error), experimentId, currentIteration);
                
                profiler.stop('train batch');
                
                k += nrSamples;
            
            # Report on error
            printF("Total error: %.2f" % total_error, experimentId, currentIteration);
            printF("Mean error: %.8f" % (total_error/float(k)), experimentId, currentIteration);
            if (parameters['simple_data_loading']):
                printF("Average data health: %.2f" % np.mean(data_healths), experimentId, currentIteration);
                printF("Stddev data health: %.2f" % np.std(data_healths), experimentId, currentIteration);
                printF("Average model health: %.2f" % np.mean(model_healths), experimentId, currentIteration);
                printF("Stddev model health: %.2f" % np.std(model_healths), experimentId, currentIteration);
            
            # Intermediate testing if this was not the last iteration of training
            # and we have passed the testing threshold
            sampleSize = parameters['sample_testing_size'];
            if (r == parameters['repetitions'] - 1):
                sampleSize = False;
            _, testError = test(model, dataset, dataset_data, label_index, parameters, model.n_max_digits, parameters['intervention_base_offset'], parameters['intervention_range'], print_samples=parameters['debug'], 
                                sample_size=sampleSize, homogeneous=parameters['homogeneous']);
            if (parameters['early_stopping'] or parameters['force_validation']):
                _, valError, meanValError = validate(model, dataset, dataset_data, label_index, parameters, model.n_max_digits, parameters['intervention_base_offset'], parameters['intervention_range'], print_samples=parameters['debug'], 
                                                     homogeneous=parameters['homogeneous']);
            
            # Save weights to pickles
            save_modulo = 50;
            if (saveModels and (r+1) % save_modulo == 0):
                saveVars = model.getVars();
                save_to_pickle('saved_models/%s_%d.model' % (name, r), saveVars, settings=parameters);
            
            model.plotWeights("%s_%d" % (name, r+1));
            
            # Check for early stopping
            if (parameters['early_stopping']):
                valErrorMovingAverageN = parameters['early_stopping_errors'];
                valErrorEpsilon = parameters['early_stopping_epsilon'];
                val_error_stack.append(meanValError);
                if (len(val_error_stack) >= valErrorMovingAverageN):
                    if (len(val_error_stack) > valErrorMovingAverageN):
                        val_error_stack.popleft();
                    # Only check for early stopping after queue is large enough
                    avg_error = np.mean(val_error_stack);
                    if (len(mean_error_stack) >= parameters['early_stopping_offset']):
                        error_diff = np.abs(avg_error - mean_error_stack.popleft());
                        if (error_diff < valErrorEpsilon):
                            printF("STOPPING EARLY at iteration %d with average error %.2f and difference %.2f!" % (r+1, avg_error, error_diff), experimentId, currentIteration);
                            # Perform final testing - will report stats for the same iteration so will overwrite in tracker
                            _, testError = test(model, dataset, dataset_data, label_index, parameters, model.n_max_digits, parameters['intervention_base_offset'], parameters['intervention_range'], print_samples=parameters['debug'], 
                                                sample_size=False, homogeneous=parameters['homogeneous']);
                            break;
                    mean_error_stack.append(avg_error);
        
        printF("Training finished!", experimentId, currentIteration);
        trackerreporter.experimentDone(experimentId);
    
    