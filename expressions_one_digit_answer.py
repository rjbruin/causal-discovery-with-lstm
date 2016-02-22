'''
Created on 16 feb. 2016

@author: Robert-Jan
'''

import time;
import sys;

import model.RecurrentNeuralNetwork as rnn;
import model.GeneratedExpressionDataset as ge_dataset;

from statistic_tools import confusion_matrix, accuracy_per_origin;

#theano.config.mode = 'FAST_COMPILE'
        
if (__name__ == '__main__'):
    
    # Default settings
    dataset_path = './data/expressions_one_digit_answer_shallow';
    repetitions = 3;
    hidden_dim = 128;
    learning_rate = 0.01;
    lstm = True;
    max_training_size = None;
    
    # Command-line settings
    if (len(sys.argv) > 1):
        dataset_path = sys.argv[1];
        if (len(sys.argv) > 2):
            repetitions = int(sys.argv[2]);
            if (len(sys.argv) > 3):
                hidden_dim = int(sys.argv[3]);
                if (len(sys.argv) > 4):
                    learning_rate = float(sys.argv[4]);
                    if (len(sys.argv) > 5):
                        lstm = sys.argv[5] == 'True';
                        if (len(sys.argv) > 6):
                            max_training_size = int(sys.argv[6]);
     
    # Debug settings
    if (max_training_size is not None):
        print("WARNING! RUNNING WITH LIMIT ON TRAINING SIZE!");
    
    # Construct models
    dataset = ge_dataset.GeneratedExpressionDataset(dataset_path);
    rnn = rnn.RecurrentNeuralNetwork(dataset.data_dim, hidden_dim, dataset.output_dim, lstm=lstm);
 
    # Start timing
    start = time.clock();
      
    # Train
    for r in range(repetitions):
        print("Repetition %d of %d" % (r+1,repetitions));
        rnn.train(dataset.train, dataset.train_labels, learning_rate, max_training_size);
     
    # Set up statistics
    key_indices = {k: i for (i,k) in enumerate(dataset.operators)};
      
    # Test
    score, prediction_histogram, groundtruth_histogram, prediction_confusion_matrix, op_scores = rnn.test(dataset.test, dataset.test_labels, dataset.test_expressions, dataset.operators, key_indices, dataset)

    print

    # Print statistics
    duration = time.clock() - start;
    print("Duration: %d seconds" % duration);
    print("Score: %.2f percent" % (score*100));
    print("Prediction histogram:   %s" % (str(prediction_histogram)));
    print("Ground truth histogram: %s" % (str(groundtruth_histogram)));
     
    print("Confusion matrix:");
    confusion_matrix(prediction_confusion_matrix);
     
    print("Accuracy per origin");
    accuracy_per_origin(op_scores, dataset.operators);
