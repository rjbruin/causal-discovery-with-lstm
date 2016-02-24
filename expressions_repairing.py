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
    
    s = range(1000);
    total_found = 0;
    correct = 0;
    for j in s:
        sentence = dataset.train[j];
        label = dataset.train_labels[j];
        
        predicted_digit = rnn.predict(sentence);
        #print("Prediction: " + str(predicted_digit) + " vs. " + str(label));
        
        for i in range(len(dataset.train_expressions[j])-1):
            # Reset sentence
            sentence = dataset.train[j];
            
            missing_index = i;
            actual_digit = np.argmax(sentence[missing_index,:]);
            #sentence[missing_index,:] = np.ones(dataset.data_dim) * 0.1;
            sentence[missing_index,:] = np.random.uniform(-1.0,1.0,dataset.data_dim) * 0.1;
            found_x = rnn.find_x(sentence,np.array([label]),missing_index)[0];
            score = actual_digit == found_x;
            print_score = "CORRECT" if score else "WRONG";
            
            # Stats
            total_found += 1;
            if (score):
                correct += 1;
            
            #print(rnn.find_x_gradient(sentence,np.array([label]),missing_index));
            #print("Find x - " + print_score + ": " + str(actual_digit) + " vs. " + str(found_x));
    
    print("%d of %d correct = %.2f percent" % (correct, total_found, (float(correct) / total_found)*100.0));