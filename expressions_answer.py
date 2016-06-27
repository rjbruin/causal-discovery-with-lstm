'''
Created on 16 feb. 2016

@author: Robert-Jan
'''

import time;
import sys;

import model.RecurrentNeuralNetwork as rnn;
import model.GeneratedExpressionDataset as ge_dataset;
import model.RandomBaseline as rb;

from tools.file import save_to_pickle;
from tools.arguments import processCommandLineArguments;
from tools.model import train, test_and_save;
from tools.gpu import using_gpu; # @UnresolvedImport

import theano;
#theano.config.mode = 'FAST_COMPILE'

if (__name__ == '__main__'):
    # Specific settings - default name is time of experiment
    raw_results_folder = './raw_results';
    name = time.strftime("%d-%m-%Y_%H-%M-%S");
    saveModels = True;
    
    # Process parameters
    parameters = processCommandLineArguments(sys.argv);
    
    # Ask for seed if running random baseline
    if (parameters['random_baseline']):
        seed = int(raw_input("Please provide an integer seed for the random number generation: ")); 
    
    # Warn for unusual parameters
    if (parameters['max_training_size'] is not None):
        print("WARNING! RUNNING WITH LIMIT ON TRAINING SIZE!");
    if (not using_gpu()):
        print("WARNING! RUNNING WITHOUT GPU USAGE!");
    
    # Construct models
    dataset = ge_dataset.GeneratedExpressionDataset(parameters['dataset'], 
                                                    add_x=parameters['find_x'],
                                                    single_digit=parameters['single_digit'], 
                                                    single_class=parameters['single_class'],
                                                    preload=parameters['preload'],
                                                    test_batch_size=parameters['test_batch_size'],
                                                    train_batch_size=parameters['train_batch_size'],
                                                    max_training_size=parameters['max_training_size'],
                                                    max_testing_size=parameters['max_testing_size'],
                                                    sample_testing_size=parameters['sample_testing_size']);
    if (parameters['random_baseline']):
        rnn = rb.RandomBaseline(parameters['single_digit'], seed);
    else:
        rnn = rnn.RecurrentNeuralNetwork(dataset.data_dim, parameters['hidden_dim'], dataset.output_dim, 
                                         lstm=parameters['lstm'], single_digit=parameters['single_digit'],
                                         minibatch_size=parameters['minibatch_size'],
                                         n_max_digits=5,
                                         time_training_batch=parameters['time_training_batch'],
                                         decoder=parameters['decoder']);
    
    ### From here the experiment should be the same every time
    
    # Start experiment clock
    start = time.clock();
    
    # Train
    train(rnn, dataset, parameters, name, start, saveModels=saveModels, targets=not parameters['single_digit']);
    
    # Final test
    test_and_save(rnn, dataset, parameters, start, show_prediction_conf_matrix=False);
    
    # Save weights to pickles
    if (saveModels):
        saveVars = rnn.vars.items();
        save_to_pickle('saved_models/%s.model' % name, saveVars);
    
