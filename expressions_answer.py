'''
Created on 16 feb. 2016

@author: Robert-Jan
'''

import time;
import sys, os;

import model.RecurrentNeuralNetwork as rnn;
import model.GeneratedExpressionDataset as ge_dataset;

from tools.file import save_to_pickle;
from tools.arguments import processCommandLineArguments;
from tools.data import create_batches;
from tools.model import train, test_and_save;

#theano.config.mode = 'FAST_COMPILE'

if (__name__ == '__main__'):
    # Specific settings - default name is time of experiment
    raw_results_folder = './raw_results';
    name = time.strftime("%d-%m-%Y_%H-%M-%S");
    saveModels = True;
    
    # Generated variables
    raw_results_filepath = os.path.join(raw_results_folder,name+'.txt');
    
    # Process parameters
    parameters = processCommandLineArguments(sys.argv);
    
    # Debug settings
    if (parameters['max_training_size'] is not None):
        print("WARNING! RUNNING WITH LIMIT ON TRAINING SIZE!");
    
    # Construct models
    dataset = ge_dataset.GeneratedExpressionDataset(parameters['dataset'], 
                                                    add_x=parameters['find_x'],
                                                    single_digit=parameters['single_digit'], 
                                                    single_class=parameters['single_class']);
    rnn = rnn.RecurrentNeuralNetwork(dataset.data_dim, parameters['hidden_dim'], dataset.output_dim, 
                                     lstm=parameters['lstm'], single_digit=parameters['single_digit']);
    
    # Prepare data
    if (parameters['single_digit']):
        targets = dataset.train_labels;
    else:
        targets = dataset.train_targets;
    
    ### From here the experiment should be the same every time
    
    # Create batches
    batches, repetition_size = create_batches(dataset.train, parameters);
    # Set up statistics
    start = time.clock();
    key_indices = {k: i for (i,k) in enumerate(dataset.operators)};
    # Train
    train(rnn, dataset, dataset.train, targets, dataset.test, dataset.test_targets, dataset.test_labels, batches, repetition_size, parameters, raw_results_filepath, key_indices, name, start, saveModels=saveModels);
    # Final test
    test_and_save(rnn, dataset, dataset.test, dataset.test_targets, dataset.test_labels, parameters, raw_results_filepath, key_indices, start, show_prediction_conf_matrix=False);
    # Save weights to pickles
    if (saveModels):
        saveVars = rnn.vars.items();
        save_to_pickle('saved_models/%s.model' % name, saveVars);
    
