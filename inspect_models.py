'''
Created on 23 feb. 2016

Run this script in debug mode to inspect the variables loaded.

@author: Robert-Jan
'''

import sys;

from model.RecurrentNeuralNetwork import RecurrentNeuralNetwork;
from model.GeneratedExpressionDataset import GeneratedExpressionDataset;
from tools.file import load_from_pickle;

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
    
    # Settings
    if (settings['single_class'] != 'None'):
        settings['single_class'] = int(settings['single_class']);
    else:
        settings['single_class'] = None;
    
    dataset = GeneratedExpressionDataset(settings['dataset'], 
                                         single_digit=settings['single_digit'],
                                         single_class=settings['single_class']);
    rnn = RecurrentNeuralNetwork(dataset.data_dim, settings['hidden_dim'], dataset.output_dim, 
                                 lstm=settings['lstm'], weight_values=savedVars, 
                                 single_digit=settings['single_digit']);
    
    return dataset, rnn, f, settings;

if __name__ == '__main__':
    modelName = 'answer-lstm-multi_digit.model';
    
    if (len(sys.argv) > 1):
        modelName = sys.argv[1];
        if (modelName == 'choose'):
            modelName = raw_input("Please provide the name of the model you want to inspect:\n");
    
    dataset, rnn, f, settings = read_from_file(modelName);
    
    # Do stuff
    predictions = [];
    ground_truths = [];
    for i in range(0,100):
        prediction = rnn.predict(dataset.test[i], dataset.test_targets[i]);
        predictions.append(prediction[0]);
        ground_truths.append(dataset.test_labels[i]);
    
    for i,prediction in enumerate(predictions):
        print(", ".join(map(lambda x: dataset.findSymbol[int(x)],prediction)) + " should be " + ", ".join(map(str,ground_truths[i])));
