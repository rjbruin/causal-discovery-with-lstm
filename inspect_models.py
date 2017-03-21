'''
Created on 23 feb. 2016

Run this script in debug mode to inspect the variables loaded.

@author: Robert-Jan
'''

import sys;
import numpy as np;

from models.TheanoRecurrentNeuralNetwork import TheanoRecurrentNeuralNetwork;
from models.GeneratedExpressionDataset import GeneratedExpressionDataset;
from tools.file import load_from_pickle;
from tools.model import constructModels;

def load():
    modelName = raw_input("Please provide the name of the model you want to inspect:\n");
    return read_from_file(modelName);

def read_from_file(modelName):
    # Keep trying to get a right filename
    while (True):
        try:
            f = open('./saved_models/' + modelName,'r')
            break;
        except IOError:
            modelName = raw_input("This model does not exist! Please provide the name of the model you want to inspect:\n");
    
    savedVars, settings = load_from_pickle(f);
    
    print(settings);
    
    dataset, rnn = constructModels(settings, None, None);
    
    # Actually load variables
    rnn.loadVars(savedVars);
    
    return dataset, rnn, settings;

if __name__ == '__main__':
#     modelName = 'choose';
    modelName = 'f-seqs-s_05-03-2017_15-23-19-t4_149_from_floats.model';
    
    if (len(sys.argv) > 1):
        modelName = sys.argv[1];
    
    if (modelName == 'choose'):
        modelName = raw_input("Please provide the name of the model you want to inspect:\n");
    
    dataset, rnn, settings = read_from_file(modelName);
    
    print(rnn.only_cause_expression);
    
    # Ask for inputs
    inpt = raw_input("Give input sample: ");
    while (inpt.strip() != ''):
        # input to numpy object
        indices = dataset.strToIndices(inpt);
        inpt_np = np.zeros((rnn.minibatch_size,rnn.n_max_digits,dataset.data_dim), dtype='float32');
        for i, ind in enumerate(indices):
            inpt_np[0,i,ind] = 1.;
        # call predict
        outpt, _ = rnn.predict(inpt_np, interventionLocations=np.array([[len(inpt)-1 for _ in range(rnn.minibatch_size)],[len(inpt)-2 for _ in range(rnn.minibatch_size)]]));
        # output to string
        outpt = dataset.indicesToStr(outpt[0]);
        print(outpt);
        # Ask for inputs
        inpt = raw_input("Give input sample: ");
    
    