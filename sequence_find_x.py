'''
Created on 9 sep. 2016

@author: Robert-Jan
'''

import time;
import sys, os;
import subprocess;

from tools.file import save_to_pickle, load_from_pickle_with_filename;
from tools.arguments import processCommandLineArguments;
from tools.model import constructModels, set_up_statistics;
from tools.gpu import using_gpu; # @UnresolvedImport

from subsystems_finish import get_batch_unprefixed, load_data;

import numpy as np;
import theano;\
import copy;
from profiler import profiler

import trackerreporter;
from tools.arguments import processKeyValue

def processSampleFindX(line, data_dim, oneHot, EOS_symbol_index):
    expression, _ = line.strip().split(";");
    
    x_position = np.random.randint(0,len(expression));
    x = expression[x_position];
    expression = expression[:x_position] + 'x' + expression[x_position+1:];
    
    expression_embeddings = np.zeros((len(expression)+1,data_dim));
    for i, literal in enumerate(expression):
        expression_embeddings[i,oneHot[literal]] = 1.0;
    expression_embeddings[i+1,EOS_symbol_index] = 1.0;
    
    target_embedding = np.zeros((data_dim));
    target_embedding[oneHot[x]] = 1.;
    
    return expression_embeddings, target_embedding, expression;

def print_stats(stats, parameters, prefix=''):
    # Print statistics
    printF("\n", experimentId, currentIteration);

    # Print statistics
    printF(prefix + "Score: %.2f percent" % (stats['score']*100), experimentId, currentIteration);
    printF(prefix + "Digit histogram:   %s" % (str(stats['prediction_histogram'])), experimentId, currentIteration);
    
    printF(prefix + "X side sizes: %s" % str(stats['x_hand_side_size']), experimentId, currentIteration);
    printF(prefix + "X on left hand side score: %.2f percent" % (stats['x_hand_side_score']['left']*100), experimentId, currentIteration);
    printF(prefix + "X on right hand side score: %.2f percent" % (stats['x_hand_side_score']['right']*100), experimentId, currentIteration);
    printF(prefix + "X on equals sign score: %.2f percent" % (stats['x_hand_side_score']['equals']*100), experimentId, currentIteration);
    
    printF(prefix + "Unique labels predicted: %d" % stats['unique_labels_predicted'], experimentId, currentIteration);
    
    if ('input_size_score' in stats):
        printF(prefix + "Input sizes: %s" % (str(stats['input_sizes'])), experimentId, currentIteration);
        for size in stats['input_size_score'].keys():
            printF(prefix + "Score by input size = %d: %.2f percent" % (size, stats['input_size_score'][size]*100.), experimentId, currentIteration);
    
    if ('x_offset_score' in stats):
        printF(prefix + "X offset sizes: %s" % (str(stats['x_offset_size'])), experimentId, currentIteration);
        for size in stats['x_offset_score'].keys():
            printF(prefix + "X offset = %d: %.2f percent" % (size, stats['x_offset_score'][size]*100.), experimentId, currentIteration);
    
    if ('symbol_score' in stats):
        printF(prefix + "Symbol sizes: %s" % (", ".join(["%s: %s" % (dataset.findSymbol[s], stats['symbol_size'][dataset.findSymbol[s]]) for s in sorted(dataset.findSymbol.keys())])), experimentId, currentIteration);
        for i in sorted(dataset.findSymbol.keys()):
            symbol = dataset.findSymbol[i];
            printF(prefix + "Symbol %d: %.2f percent" % (i, stats['symbol_score'][symbol]*100.), experimentId, currentIteration);
    
    for i in sorted(dataset.findSymbol.keys()):
        printF(prefix + "Symbol confusion row %d: %s" % (i, str(stats['symbol_confusion'][i,:])), experimentId, currentIteration);
    
#     printF(prefix + "! Samples correct: %s" % str(map(lambda (x,y): "%d,%d" % (int(x), int(y)),stats['samplesCorrect'])), experimentId, currentIteration);
    
    printF("\n", experimentId, currentIteration);

def get_batch_regular(isTrain, dataset, model, debug=False):    
    # Reseed the random generator to prevent generating identical batches
    np.random.seed();
    
    if (isTrain == 0):
        storage = dataset.expressionsByPrefix;
    elif (isTrain == 1):
        storage = dataset.testExpressionsByPrefix;
    else:
        storage = dataset.validationExpressionsByPrefix;
    
    batch = [];
    nrSamples = 0;
    while (len(batch) < model.minibatch_size):
        # Add subbatch to batch
        expression, _, _, _ = storage.get_random();
        batch.append(expression);
        nrSamples += 1;
    
    data = [];
    targets = [];
    labels = [];
    expressions = [];
    for expression in batch:
        data, targets, labels, expressions, _ = dataset.processor(expression + ";", data, targets, labels, expressions);
    
    data = dataset.fill_ndarray(data, 1, fixed_length=model.n_max_digits);
    targets = np.array(targets).astype('float32');
    
    return data, targets, labels, expressions, nrSamples;

def get_batch(isTrain, dataset, model, dataset_data, label_index, debug=False):    
    if (parameters['simple_data_loading']):
        data, targets, labels, expressions, health = get_batch_unprefixed(isTrain, dataset, dataset_data, label_index, parameters);
        return data, targets, labels, expressions, parameters['minibatch_size'], health;
    else:
        data, targets, labels, expressions, nrSamples = \
            get_batch_regular(isTrain, dataset, model, debug);
        return data, targets, labels, expressions, nrSamples, 0;

def test(model, dataset, parameters, max_length, print_samples=False, 
         sample_size=False, returnTestSamples=False):
    # Test
    printF("Testing...", experimentId, currentIteration);
        
    total = dataset.lengths[dataset.TEST];
    printing_interval = 1000;
    if (parameters['max_dataset_size'] is not False):
        printing_interval = 100;
    elif (sample_size != False):
        total = sample_size;
    
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
        test_data, test_targets, test_labels, test_expressions, \
            nrSamples, health = get_batch(1, dataset, model, dataset_data, label_index, debug=parameters['debug']);
        
        predictions, other = model.predict(test_data, test_targets, 
                                           nrSamples=nrSamples); 
        totalError += other['summed_error'];
        
        profiler.start("test batch stats");
        stats, _, _ = model.batch_statistics(stats, predictions, 
                                             test_labels, None,
                                             other, nrSamples, dataset, 
                                             None, None, parameters, data=test_data);
        
        for j in range(nrSamples):
            if (test_labels[j] not in total_labels_used):
                total_labels_used[test_labels[j]] = True;
            
            # Save predictions to testSamples
            if (returnTestSamples):
                strData = map(lambda x: dataset.findSymbol[x], 
                              np.argmax(test_data[j,:,:model.data_dim],len(test_data.shape)-2));
                strPrediction = dataset.findSymbol[predictions[j]];
                testSamples.append((strData,strPrediction));
        
        # Print samples
        if (print_samples and not printed_samples):
            for i in range(nrSamples):
                prefix = "# ";
                printF(prefix + "Data          1: %s" % "".join((map(lambda x: dataset.findSymbol[x], 
                                                   np.argmax(test_data[i],len(test_data.shape)-2)))), experimentId, currentIteration);
                printF(prefix + "Prediction    1: %s" % dataset.findSymbol[predictions[i]], experimentId, currentIteration);
                printF(prefix + "Used label    1: %s" % dataset.findSymbol[test_labels[i]], experimentId, currentIteration);
            printed_samples = True;

        if (stats['prediction_size'] % printing_interval == 0):
            printF("# %d / %d" % (stats['prediction_size'], total), experimentId, currentIteration);
        profiler.stop("test batch stats");
        
        k += nrSamples;
    
    profiler.profile();
    
    print(total_labels_used);
    print("Test: %d" % k);
    printF("Total testing error: %.2f" % totalError, experimentId, currentIteration);
    printF("Mean testing error: %.8f" % (totalError/float(k)), experimentId, currentIteration);
    
    stats = model.total_statistics(stats, dataset, total_labels_used=total_labels_used, digits=False);
    print_stats(stats, parameters);
    
    if (returnTestSamples):
        return stats, testSamples;
    else:
        return stats;

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
                   'Train Precision': 'TRAIN Score',
                   'Unique labels': 'Unique labels predicted',
                   'Mean data health': 'Average data health',
                   'Stddev data health': 'Stddev data health',
                   'Mean model health': 'Average model health',
                   'Stddev model health': 'Stddev model health',
                   'X left': 'X on left hand side score',
                   'X right': 'X on right hand side score',
                   'X equals': 'X on equals sign score'};
    for size in range(20):
        score_types['Size %d' % size] = 'Score by input size = %d:' % size;
    for symbolId in range(20):
        score_types['Symbol %d' % symbolId] = 'Symbol %d:' % symbolId;
    for trueSize in range(20):
        for nrCorrect in range(20):
            score_types['T %d C %d' % (trueSize, nrCorrect)] = 'Prediction size %d nr correct %d' % (trueSize, nrCorrect);
    trackerreporter.init('http://rjbruin.nl/experimenttracker/api/',api_key);
    
    cmdargs = sys.argv[1:];
    # Check for experiment settings file argument and obtain new arguments
    allparameters = processCommandLineArguments(cmdargs);
    newparameters = [];
    if (allparameters[0]['debug']):
        newparameters = allparameters;
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
        
        # Warn for unusual parameters
        if (parameters['max_dataset_size'] is not False):
            printF("WARNING! RUNNING WITH LIMIT ON DATASET SIZE!", experimentId, currentIteration);
        if (not using_gpu()):
            printF("WARNING! RUNNING WITHOUT GPU USAGE!", experimentId, currentIteration);
        
        # Set simple loading processor
        processor = processSampleFindX;
        
        # Construct models
        dataset, model = constructModels(parameters, 0, {});
        
        # Load pretrained only_cause_expression = 1 model
        if (parameters['load_cause_expression_1'] is not False):
            loadedVars, _ = load_from_pickle_with_filename("./saved_models/" + parameters['load_cause_expression_1']);
            if (model.loadPartialDataDimVars(dict(loadedVars), 0, model.data_dim)):
                printF("Loaded pretrained model (expression 1) successfully!", experimentId, currentIteration);
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
        
        
        for r in range(parameters['repetitions']):
            stats = set_up_statistics(dataset.output_dim, model.n_max_digits, dataset.oneHot.keys());
            total_error = 0.0;
            # Print repetition progress and save to raw results file
            printF("Batch %d (repetition %d of %d, dataset 1 of 1) (samples processed after batch: %d)" % \
                    (r+1,r+1,parameters['repetitions'],(r+1)*repetition_size), experimentId, currentIteration);
            currentIteration = r+1;
            currentDataset = 1;
            
            # Train model per minibatch
            k = 0;
            printedProgress = -1;
            while k < repetition_size:
                profiler.start('train batch');
                profiler.start('get train batch');
                data, target, test_labels, target_expressions, nrSamples, health = \
                    get_batch(0, dataset, model, dataset_data, label_index, 
                              debug=parameters['debug']);
                profiler.stop('get train batch');
                
                # Run training
                profiler.start('train sgd');
                outputs = model.sgd(dataset, data, target, parameters['learning_rate'],
                                      nrSamples=nrSamples);
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
            print("Train: %d" % k);
            printF("Total error: %.2f" % total_error, experimentId, currentIteration);
            printF("Mean error: %.8f" % (total_error/float(k)), experimentId, currentIteration);
            
            # Intermediate testing if this was not the last iteration of training
            # and we have passed the testing threshold
            sampleSize = parameters['sample_testing_size']
            if (r == parameters['repetitions'] - 1):
                sampleSize = False;
            test(model, dataset, parameters, model.n_max_digits, print_samples=parameters['debug'], 
                 sample_size=sampleSize);
            
            # Save weights to pickles
            save_modulo = 50;
            if (saveModels and (r+1) % save_modulo == 0):
                saveVars = model.getVars();
                save_to_pickle('saved_models/%s_%d.model' % (name, r), saveVars, settings=parameters);
        
        printF("Training finished!", experimentId, currentIteration);
    
    