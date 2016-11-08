'''
Created on 4 nov. 2016

@author: Robert-Jan
'''

import time;
import sys;

from tools.file import save_to_pickle;
from tools.arguments import processCommandLineArguments;
from tools.model import constructModels, set_up_statistics;
from tools.gpu import using_gpu; # @UnresolvedImport

from models.Autoencoder import Autoencoder;

import numpy as np;
import theano;
from profiler import profiler

def print_stats(precision, digit_precision):
    # Print statistics
    output = "\n";

    # Print statistics
    output += "Score: %.2f percent\n" % (precision);
    output += "Digit-based score: %.2f percent\n" % (digit_precision);
    
    output += "\n";
    print(output);

def get_batch(isTrain, dataset, model, intervention_range, max_length, 
              debug=False, base_offset=12):    
    # Reseed the random generator to prevent generating identical batches
    np.random.seed();
    
    if (isTrain):
        storage = dataset.expressionsByPrefix;
    else:
        storage = dataset.testExpressionsByPrefix;
    
    batch = [];
    subbatch_size = parameters['subbatch_size'];
    while (len(batch) < model.minibatch_size):
        interventionLocation = np.random.randint(max_length-intervention_range-base_offset, 
                                                 max_length-base_offset);
        
        subbatch = [];
        while (len(subbatch) < subbatch_size):
            branch = storage.get_random_by_length(interventionLocation, getStructure=True);
            
            randomPrefix = np.random.randint(0,len(branch.fullExpressions));
            subbatch.append((branch.fullExpressions[randomPrefix],
                             branch.primedExpressions[randomPrefix]));
        
        # Add subbatch to batch
        batch.extend(subbatch);
    
    data = [];
    targets = [];
    labels = [];
    expressions = [];
    for (expression, expression_prime) in batch:
        data, _, _, expressions, _ = dataset.processor(";".join([expression, expression_prime]), 
                                                                  data,targets, labels, expressions);
    
    data = dataset.fill_ndarray(data, 1, fixed_length=model.n_max_digits);
    
    return data, expressions;

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
    stats = set_up_statistics(dataset.data_dim, model.n_max_digits);
    
    # Predict
    printed_samples = False;
    totalError = 0.0;
    k = 0;
    testSamples = [];
    precisions = [];
    digit_precisions = [];
    while k < total:
        # Get data from batch
        test_data, test_expressions = get_batch(False, dataset, model, 
                                                intervention_range, 
                                                max_length, debug=parameters['debug'],
                                                base_offset=base_offset);
        
        predictions, precision, digit_precision, error = model.predict(test_data); 
        precisions.append(precision);
        digit_precisions.append(digit_precision);
        totalError += error;
        
        if (parameters['only_cause_expression']):
            prediction_1 = predictions;
            predictions = [predictions];
        else:
            prediction_1 = predictions[0];
            prediction_2 = predictions[1];
        
        # Print samples
        if (print_samples and not printed_samples):
            for i in range(model.minibatch_size):
                prefix = "# ";
                if (parameters['only_cause_expression'] is not False):
                    print(prefix + "Data      : %s" % "".join((map(lambda x: dataset.findSymbol[x], 
                                                       np.argmax(test_data[i],len(test_data.shape)-2)))));
                    print(prefix + "Prediction: %s" % "".join(map(lambda x: dataset.findSymbol[x], prediction_1[i])));
                else:
                    print(prefix + "Data       1: %s" % "".join((map(lambda x: dataset.findSymbol[x], 
                                                       np.argmax(test_data[i,:,:model.data_dim/2],len(test_data.shape)-2)))));
                    print(prefix + "Prediction 1: %s" % "".join(map(lambda x: dataset.findSymbol[x], prediction_1[i])));
                    print(prefix + "Data       2: %s" % "".join((map(lambda x: dataset.findSymbol[x], 
                                                       np.argmax(test_data[i,:,model.data_dim/2:],len(test_data.shape)-2)))));
                    print(prefix + "Prediction 2: %s" % "".join(map(lambda x: dataset.findSymbol[x], prediction_2[i])));
            printed_samples = True;

        if (k % printing_interval == 0):
            print("# %d / %d" % (stats['prediction_size'], total));
        
        k += model.minibatch_size;
    
    profiler.profile();
    
    print("Total testing error: %.2f" % totalError);
    
    print_stats(np.mean(precisions), np.mean(digit_precisions));
    
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
    
    # Check for valid subbatch size
    if (parameters['minibatch_size'] % parameters['subbatch_size'] != 0):
        raise ValueError("Subbatch size is not compatible with minibatch size: m.size = %d, s.size = %d" % 
                            (parameters['minibatch_size'], parameters['subbatch_size']));
    
    # Check for valid intervention ranges
    if (parameters['intervention_base_offset'] <= 0):
        raise ValueError("Invalid intervention base offset: is %d, must be at least 1." % parameters['intervention_base_offset']);
    
    
    
    # Construct models
    dataset, _ = constructModels(parameters, 0, {}, noModel=True);
    actual_data_dim = dataset.data_dim;
    if (parameters['only_cause_expression'] is False):
        actual_data_dim *= 2;
    model = Autoencoder(actual_data_dim, parameters['hidden_dim'], parameters['minibatch_size'], parameters['n_max_digits'], 
                        parameters['learning_rate'], dataset.GO_symbol_index, dataset.EOS_symbol_index, parameters['only_cause_expression']);
    
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
        stats = set_up_statistics(dataset.data_dim, model.n_max_digits);
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
            data, target_expressions = \
                get_batch(True, dataset, model, 
                          parameters['intervention_range'], model.n_max_digits, 
                          debug=parameters['debug'],
                          base_offset=parameters['intervention_base_offset']);
            profiler.stop('get train batch');
            
            # Run training
            profiler.start('train sgd');
            error = model.sgd(data);
            total_error += error;
            profiler.stop('train sgd');
            
            # Print batch progress
            if ((k+parameters['minibatch_size']) % (parameters['minibatch_size']*4) < parameters['minibatch_size'] and \
                (k+parameters['minibatch_size']) / (parameters['minibatch_size']*4) > printedProgress):
                printedProgress = (k+parameters['minibatch_size']) / (parameters['minibatch_size']*4);
                print("# %d / %d (error = %.2f)" % (k+parameters['minibatch_size'], repetition_size, total_error));
            
            profiler.stop('train batch');
            
            k += parameters['minibatch_size'];
        
        # Report on error
        print("Total error: %.2f" % total_error);
        
        # Intermediate testing if this was not the last iteration of training
        # and we have passed the testing threshold
        #if (r != repetition_size-1):
        test(model, dataset, parameters, model.n_max_digits, parameters['intervention_base_offset'], parameters['intervention_range'], print_samples=parameters['debug'], 
             sample_size=parameters['sample_testing_size'], homogeneous=parameters['homogeneous']);
        
        # Do random walk
        print("Random walk: " + str(model.randomWalk(nrSamples=10)));
        
        # Save weights to pickles
        if (saveModels):
            saveVars = model.getVars();
            save_to_pickle('saved_models/%s_%d.model' % (name, r), saveVars, settings=parameters);
    
    print("Training finished!");
    
    