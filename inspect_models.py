'''
Created on 23 feb. 2016

Run this script in debug mode to inspect the variables loaded.

@author: Robert-Jan
'''

import pickle, sys;
from model.RecurrentNeuralNetwork import RecurrentNeuralNetwork;
from model.GeneratedExpressionDataset import GeneratedExpressionDataset;

if __name__ == '__main__':
    modelName = 'answer-lstm-multi_digit.model';
    dataset_path = './data/expressions_positive_integer_answer_shallow';
    hidden_dim = 128;
    single_digit = False;
    lstm = True;
    
    if (len(sys.argv) > 1):
        modelName = sys.argv[1];
        if (modelName == 'choose'):
            modelName = raw_input("Please provide the name of the model you want to inspect:\n");
    
    # Keep trying to get a right filename
    while (True):
        try:
            f = open('./saved_models/' + modelName,'r')
            break;
        except IOError:
            modelName = raw_input("This model does not exist! Please provide the name of the model you want to inspect:\n");
    
    savedVars = pickle.load(f);
    dataset = GeneratedExpressionDataset(dataset_path, single_digit=single_digit)
    rnn = RecurrentNeuralNetwork(dataset.data_dim, hidden_dim, dataset.output_dim, lstm=lstm, weight_values=savedVars, single_digit=single_digit);
    
    # Do stuff
    predictions = [];
    ground_truths = [];
    for i in range(0,100):
        prediction = rnn.predict(dataset.test[i], dataset.test_targets[i]);
        predictions.append(prediction);
        ground_truths.append(dataset.test_labels[i]);
    
    for i,(prediction,size) in enumerate(predictions):
        print(", ".join(map(lambda x: dataset.findSymbol[int(x)],prediction)) + " should be " + ", ".join(map(str,ground_truths[i])) + " (size = " + str(size) + ")");
    f.close();
