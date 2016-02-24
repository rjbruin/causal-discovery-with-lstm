'''
Created on 24 feb. 2016

@author: Robert-Jan
'''

import pickle;
import numpy as np;

import model.RecurrentNeuralNetwork as rnn;
import model.GeneratedExpressionDataset as ge_dataset;

if __name__ == '__main__':
    # Default settings
    dataset_path = './data/expressions_one_digit_answer_shallow';
    hidden_dim = 128;
    lstm = False;
    
    # Load models
    models_path = './saved_models/23-02-2016_14-21-19_29.model';
    f = open(models_path, 'r');
    models = pickle.load(f);
    f.close();
    models_dict = {};
    for (name,var) in models:
        models_dict[name] = var.get_value();
    
    # Construct models
    dataset = ge_dataset.GeneratedExpressionDataset(dataset_path);
    rnn = rnn.RecurrentNeuralNetwork(dataset.data_dim, hidden_dim, dataset.output_dim, 
                                     lstm=lstm, weight_values=models_dict);
    
    s = 1;
    for i in [1,3,7,9]:
        sentence_index = s; 
        missing_index = i;
        
        sentence = dataset.train[s];
        label = dataset.train_labels[s];
        actual_digit = np.argmax(sentence[missing_index,:]);
        sentence[missing_index,:] = np.ones(dataset.data_dim) * 0.1;
        
        print(rnn.find_x_gradient(sentence,np.array([label]),missing_index));
        print(str(actual_digit) + " vs. " + str(rnn.find_x(sentence,np.array([label]),missing_index)[0]));