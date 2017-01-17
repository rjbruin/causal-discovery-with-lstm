'''
Created on 9 sep. 2016

@author: Robert-Jan
'''

import time;
import sys, os;
from math import floor;

from tools.file import save_to_pickle, load_from_pickle_with_filename;
from tools.arguments import processCommandLineArguments;
from tools.model import constructModels, set_up_statistics;
from tools.gpu import using_gpu; # @UnresolvedImport

import numpy as np;
import theano;
from profiler import profiler

def set_up_linear_stats(parameters):
    return {'meanDifferences': [], 'prediction_size': 0,
            #'meanDifferencesPerIteration': {k: [] for k in range(parameters['n_max_digits'])}
            'meanDifferencesFirstHalf': [], 'meanDifferencesSecondHalf': []};

def print_stats(stats, parameters, prefix=''):
    # Print statistics
    output = "\n";

    # Print statistics
    output += prefix + "Mean difference: %.4f\n" % (np.mean(stats['meanDifferences'])*100);
    output += prefix + "Mean difference first half: %.4f\n" % (np.mean(stats['meanDifferencesFirstHalf'])*100);
    output += prefix + "Mean difference second half: %.4f\n" % (np.mean(stats['meanDifferencesSecondHalf'])*100);
    
    output += "\n";
    print(output);

def processSampleLinearProcess(line, data):
        """
        Data is ndarray of size (nr lines, sequence length, nr input vars).
        Targets is same as data.
        Labels is same as data.
        Expressions is string representation.
        """
        _, samplesStr = line.split("|");
        samples = samplesStr.split(";");
        encoding = np.zeros((len(samples), len(samples[0].split(","))), dtype='float32');
        
        for i in range(len(samples)):
            vals = samples[i].split(",");
            for j in range(len(vals)):
                encoding[i,j] = float(vals[j]);
        data.append(encoding);
        
        return data;

def load_data(parameters):
    f = open(os.path.join(parameters['dataset'],'all.txt'));
    
    data = [];
    for line in f:
        linedata = processSampleLinearProcess(line, data);
        data.append(linedata);
    
    return data;

def get_batch(isTrain, data, parameters):    
    # Reseed the random generator to prevent generating identical batches
    np.random.seed();
    
    # Set range to sample from
    test_sample_range = [parameters['test_offset']*len(data),parameters['test_offset']*len(data)+parameters['test_size']*len(data)];
    
    batch = [];
    while (len(batch) < model.minibatch_size):
        # Get random sample
        sampleIndex = np.random.randint(0,len(data));
        while ((isTrain and sampleIndex >= test_sample_range[0] and sampleIndex < test_sample_range[1]) or
               (not isTrain and sampleIndex < test_sample_range[0] and sampleIndex >= test_sample_range[1])):
            sampleIndex = np.random.randint(0,len(data));
        # Append to data
        batch.append(data[sampleIndex]);
    
    # Make data ndarray
    encoded_batch = np.array(batch);
    
    return encoded_batch;

def test(model, dataset_data, parameters, print_samples=False, 
         sample_size=False):
    # Test
    print("Testing...");
        
    total = parameters['test_size']*len(dataset_data);
    printing_interval = 1000;
    if (parameters['max_testing_size'] is not False):
        total = parameters['max_testing_size'];
        printing_interval = 100;
    elif (sample_size != False):
        total = sample_size;
    
    # Set up statistics
    stats = set_up_linear_stats(parameters);
    
    # Predict
    printed_samples = False;
    totalError = 0.0;
    k = 0;
    while k < total:
        # Get data from batch
        test_data = get_batch(False, dataset_data, model);
        
        predictions, other = model.predict(test_data, test_data,
                                           nrSamples=parameters['minibatch_size']); 
        totalError += other['error'];
        
        if (parameters['only_cause_expression']):
            prediction_1 = predictions;
            predictions = [predictions];
        else:
            prediction_1 = predictions[0];
            prediction_2 = predictions[1];
        
        profiler.start("test batch stats");
        stats = model.batch_statistics(stats, predictions, 
                                       test_expressions, interventionLocations, 
                                       other, nrSamples, dataset, test_expressions,
                                       topcause=topcause or parameters['bothcause'], # If bothcause then topcause = 1
                                       testInDataset=parameters['test_in_dataset'],
                                       bothcause=parameters['bothcause']);
        
        # Print samples
        if (print_samples and not printed_samples):
            for i in range(nrSamples):
                prefix = "# ";
                print(prefix + "Data          1: %s" % str(test_data[:,:,0]));
                print(prefix + "Prediction    1: %s" % str(prediction_1[i]));
                print(prefix + "Data          2: %s" % str(test_data[:,:,1]));
                print(prefix + "Prediction    2: %s" % str(prediction_2[i]));
            printed_samples = True;

        if (stats['prediction_size'] % printing_interval == 0):
            print("# %d / %d" % (stats['prediction_size'], total));
        profiler.stop("test batch stats");
        
        k += nrSamples;
    
    profiler.profile();
    
    print("Total testing error: %.2f" % totalError);
    
    print_stats(stats, parameters);
    
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
    _, model = constructModels(parameters, 0, {}, noDataset=True);
    
    # Load data
    dataset_data = load_data(parameters);
    
    # Train on all datasets in succession
    # Print settings headers to raw results file
    print("# " + str(parameters));
    
    # Compute batching variables
    repetition_size = len(dataset_data);
    if (parameters['max_training_size'] is not False):
        repetition_size = min(parameters['max_training_size'],repetition_size);
    next_testing_threshold = parameters['test_interval'] * repetition_size;
    
    
    
    for r in range(parameters['repetitions']):
#         stats = set_up_statistics(dataset.output_dim, model.n_max_digits);
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
                get_batch(True, dataset_data, model);
            profiler.stop('get train batch');
            
            # Run training
            profiler.start('train sgd');
            outputs = model.sgd(dataset, data, target, parameters['learning_rate'],
                                  nrSamples=model.minibatch_size, expressions=target_expressions,
                                  interventionLocations=interventionLocations,
                                  topcause=topcause or parameters['bothcause'], bothcause=parameters['bothcause']);
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
        test(model, dataset_data, parameters, print_samples=parameters['debug'], sample_size=parameters['sample_testing_size']);
        
        # Save weights to pickles
        if (saveModels):
            saveVars = model.getVars();
            save_to_pickle('saved_models/%s_%d.model' % (name, r), saveVars, settings=parameters);
    
    print("Training finished!");
    
    