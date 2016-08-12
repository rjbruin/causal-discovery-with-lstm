'''
Created on 16 feb. 2016

@author: Robert-Jan
'''

import time;
import sys;

import models.RecurrentNeuralNetwork as rnn;
import tools.model;

from tools.file import save_to_pickle;
from tools.arguments import processCommandLineArguments;
from tools.model import train, test;
from tools.gpu import using_gpu; # @UnresolvedImport

import numpy as np;
#import theano;
#theano.config.mode = 'FAST_COMPILE'

def writeToVerbose(verboseOutputter, s):
    f = verboseOutputter['f']();
    f.write(s + '\n');
    f.close();

if (__name__ == '__main__'):
    np.set_printoptions(precision=3, threshold=10000000);
    
    # Specific settings - default name is time of experiment
    name = time.strftime("%d-%m-%Y_%H-%M-%S");
    saveModels = True;
    
    # Process parameters
    parameters = processCommandLineArguments(sys.argv[1:]);
    
    # Set up extreme verbose output
    # Can always be called but will not do anything if not needed
    if (parameters['extreme_verbose']):
        verboseOutputter = {'name': './verbose_output/%s.debug' % name};
        verboseOutputter['f'] = lambda: open(verboseOutputter['name'],'a');
        verboseOutputter['write'] = lambda s: writeToVerbose(verboseOutputter, s);
    else:
        verboseOutputter = {'write': lambda s: False};
    
    # Ask for seed if running random baseline
    seed = 0;
    if (parameters['random_baseline']):
        seed = int(raw_input("Please provide an integer seed for the random number generation: ")); 
    
    # Warn for unusual parameters
    if (parameters['max_training_size'] is not False):
        print("WARNING! RUNNING WITH LIMIT ON TRAINING SIZE!");
    if (not using_gpu()):
        print("WARNING! RUNNING WITHOUT GPU USAGE!");
    
    # Construct models
    datasets, rnn = tools.model.constructModels(parameters, seed, verboseOutputter);
    
    ### From here the experiment should be the same every time
    
    # Start experiment clock
    start = time.clock();
    
    # Train on all datasets in succession
    train(rnn, datasets, parameters, name, start, saveModels=saveModels, targets=not parameters['single_digit'], verboseOutputter=verboseOutputter);
    
    print("Training all datasets finished!");
    
    # Final test on last dataset
    test(rnn, datasets[-1], parameters, start, show_prediction_conf_matrix=False);
    
    # Save weights to pickles
    if (saveModels):
        saveVars = rnn.vars.items();
        save_to_pickle('saved_models/%s.model' % name, saveVars);
