'''
Created on 9 sep. 2016

@author: Robert-Jan
'''

import time;
import sys;

from tools.file import save_to_pickle, load_from_pickle_with_filename;
from tools.arguments import processCommandLineArguments;
from tools.model import constructModels, set_up_statistics;
from tools.gpu import using_gpu; # @UnresolvedImport

import numpy as np;
import theano;
from profiler import profiler

def print_stats(stats, parameters, prefix=''):
    # Print statistics
    output = "\n";

    # Print statistics
    output += prefix + "Score: %.2f percent\n" % (stats['score']*100);
    output += prefix + "Digit histogram:   %s\n" % (str(stats['prediction_histogram']));
    
    output += prefix + "Unique labels predicted: %d\n" % stats['unique_labels_predicted'];
    
    if ('prediction_size_score' in stats):
        output += prefix + "Prediction sizes: %s\n" % (str(stats['prediction_sizes']));
        for size in stats['prediction_size_score'].keys():
            output += prefix + "Score by prediction size = %d: %.2f percent\n" % (size, stats['prediction_size_score'][size]*100.);
    
#     output += prefix + "! Samples correct: %s" % str(map(lambda (x,y): "%d,%d" % (int(x), int(y)),stats['samplesCorrect']));
    
    output += "\n";
    print(output);

def get_batch(isTrain, dataset, model, debug=False):    
    # Reseed the random generator to prevent generating identical batches
    np.random.seed();
    
    if (isTrain):
        storage = dataset.expressionsByPrefix;
    else:
        storage = dataset.testExpressionsByPrefix;
    
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

def test(model, dataset, parameters, max_length, print_samples=False, 
         sample_size=False, returnTestSamples=False):
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
        test_data, test_targets, test_labels, test_expressions, \
            nrSamples = get_batch(False, dataset, model, debug=parameters['debug']);
        
        predictions, other = model.predict(test_data, test_targets, 
                                           nrSamples=nrSamples); 
        totalError += other['error'];
        
        profiler.start("test batch stats");
        stats, _ = model.batch_statistics(stats, predictions, 
                                       test_labels, None,
                                       other, nrSamples, dataset, 
                                       None, None, parameters, data=test_data);
        
        for j in range(nrSamples):
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
                print(prefix + "Data          1: %s" % "".join((map(lambda x: dataset.findSymbol[x], 
                                                   np.argmax(test_data[i],len(test_data.shape)-2)))));
                print(prefix + "Prediction    1: %s" % dataset.findSymbol[predictions[i]]);
                print(prefix + "Used label    1: %s" % dataset.findSymbol[test_labels[i]]);
            printed_samples = True;

        if (stats['prediction_size'] % printing_interval == 0):
            print("# %d / %d" % (stats['prediction_size'], total));
        profiler.stop("test batch stats");
        
        k += nrSamples;
    
    profiler.profile();
    
    print("Total testing error: %.2f" % totalError);
    
    stats = model.total_statistics(stats, total_labels_used=total_labels_used, digits=False);
    print_stats(stats, parameters);
    
    if (returnTestSamples):
        return stats, testSamples;
    else:
        return stats;

if __name__ == '__main__':
    theano.config.floatX = 'float32';
    np.set_printoptions(precision=3, threshold=10000000);
    profiler.off();
    
    # Process parameters
    parameters = processCommandLineArguments(sys.argv[1:]);
    
    # Specific settings - default name is time of experiment
    name = parameters['output_name'] + time.strftime("_%d-%m-%Y_%H-%M-%S");
    saveModels = True;
    
    # Warn for unusual parameters
    if (parameters['max_training_size'] is not False):
        print("WARNING! RUNNING WITH LIMIT ON TRAINING SIZE!");
    if (not using_gpu()):
        print("WARNING! RUNNING WITHOUT GPU USAGE!");
    
    # Construct models
    dataset, model = constructModels(parameters, 0, {});
    
    # Load pretrained only_cause_expression = 1 model
    if (parameters['load_cause_expression_1'] is not False):
        loadedVars, _ = load_from_pickle_with_filename("./saved_models/" + parameters['load_cause_expression_1']);
        if (model.loadPartialDataDimVars(dict(loadedVars), 0, model.data_dim)):
            print("Loaded pretrained model (expression 1) successfully!");
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
            data, target, test_labels, target_expressions, nrSamples = \
                get_batch(True, dataset, model, 
                          debug=parameters['debug']);
            profiler.stop('get train batch');
            
            # Run training
            profiler.start('train sgd');
            outputs = model.sgd(dataset, data, target, parameters['learning_rate'],
                                  nrSamples=nrSamples);
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
    
    print("Training finished!");
    
    