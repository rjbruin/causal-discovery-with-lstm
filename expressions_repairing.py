'''
Created on 24 feb. 2016

@author: Robert-Jan
'''

import pickle;
import numpy as np;

import models.TheanoRecurrentNeuralNetwork as rnn;
import models.GeneratedExpressionDataset as ge_dataset;

if __name__ == '__main__':
    # Default settings
    dataset_path = './data/expressions_one_digit_answer_shallow';
    hidden_dim = 128;
    lstm = False;
    analytical = True;
    
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
    rnn = rnn.TheanoRecurrentNeuralNetwork(dataset.data_dim, hidden_dim, dataset.output_dim, 
                                     lstm=lstm, weight_values=models_dict);
    
    s = range(len(dataset.test));
    total_found = 0;
    correct = 0;
    for j in s:
        if (j % 10 == 0):
            print("# %d / %d" % (j, len(s)));
            if (total_found > 0):
                print("# %d of %d correct = %.2f percent" % (correct, total_found, (float(correct) / total_found)*100.0));
        sentence = dataset.test[j];
        label = dataset.test_labels[j];
        
        predicted_digit = rnn.predict(sentence);
        #print("Prediction: " + str(predicted_digit) + " vs. " + str(label));
        
        for i in range(len(dataset.test_expressions[j])-1):
            # Reset sentence
            sentence = dataset.test[j];
            
            missing_index = i;
            actual_digit = np.argmax(sentence[missing_index,:]);
            
            if analytical:
                sentence[missing_index,:] = np.ones(dataset.data_dim) * 0.1;
                #sentence[missing_index,:] = np.random.uniform(-1.0,1.0,dataset.data_dim) * 0.1;
                found_x = rnn.find_x(sentence,np.array([label]),missing_index)[0];
                score = actual_digit == found_x;
                print_score = "CORRECT" if score else "WRONG";
                
                # Stats
                total_found += 1;
                if (score):
                    correct += 1;
            else:
                best_error = 1000000.0;
                best_symbol = -1;
                for symbol in range(17):
                    sentence[missing_index,:] = np.zeros(dataset.data_dim);
                    sentence[missing_index,symbol] = 1.0;
                    error = rnn.error(sentence,np.array([label]));
                    if (error < best_error):
                        best_error = error;
                        best_symbol = symbol;
                score = actual_digit == best_symbol;
                print_score = "CORRECT" if score else "WRONG";
                
                # Stats
                total_found += 1;
                if (score):
                    correct += 1;
    
    print("%d of %d correct = %.2f percent" % (correct, total_found, (float(correct) / total_found)*100.0));