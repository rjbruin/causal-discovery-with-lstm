'''
Created on 15 jun. 2016

@author: Robert-Jan
'''
import numpy as np;

import unittest
from model.RecurrentNeuralNetwork import RecurrentNeuralNetwork
from tools import model


class Test(unittest.TestCase):


    def testRecurrentNeuralNetwork(self):
        experiment_repetitions = 3;        
        experiments_to_run = [(False,False,2,3000),(False,True,2,2000)];
        #experiment_repetitions = 2;
        #experiments_to_run = [(False,True,2,2000)];
        
        for e, (single_digit, lstm, n_max_digits, repetitions) in enumerate(experiments_to_run):
            scores = [];
            for j in range(experiment_repetitions):
                stats = self.runTest(single_digit,lstm,n_max_digits,repetitions);
                scores.append(stats['score']);
                print("Iteration %d: %.2f percent" % (j+1, stats['score'] * 100));
            mean_score = np.mean(scores);
            print("Average score: %.2f" % (mean_score * 100));
            self.assertGreaterEqual(mean_score, 1.0, 
                                    "Experiment %d: mean score is not perfect: %.2f percent" % (e, mean_score * 100));

    def runTest(self, single_digit=False, lstm=True, n_max_digits=24, repetitions=3000):
        # Testcase settings
        learning_rate = 0.1;
        repetitions = repetitions;
        reporting_interval = repetitions / 5;
        
        # Model settings
        data_dim = 5;
        hidden_dim = 8;
        output_dim = 5;
        single_digit = single_digit;
        minibatch_size = 5;
        
        # Model initialization
        rnn = RecurrentNeuralNetwork(data_dim, hidden_dim, output_dim, 
                                     minibatch_size,
                                     single_digit=single_digit, lstm=lstm,
                                     n_max_digits=n_max_digits);
        
        # Data generation
        training_data = np.array([np.array([[1.0,0.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]]),
                                  np.array([[1.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]]),
                                  np.array([[0.0,1.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]]),
                                  np.array([[0.0,1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]]),
                                  np.array([[0.0,1.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0,0.0]])]);
        training_targets = np.array([np.array([1.0,0.0,0.0,0.0,0.0]),
                                         np.array([0.0,1.0,0.0,0.0,0.0]),
                                         np.array([0.0,0.0,1.0,0.0,0.0]),
                                         np.array([0.0,0.0,0.0,1.0,0.0]),
                                         np.array([0.0,0.0,0.0,1.0,0.0])]);
        training_targets_multi_digit = np.array([np.array([[1.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,1.0]]),
                                        np.array([[0.0,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,1.0]]),
                                        np.array([[0.0,0.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0,1.0]]),
                                        np.array([[0.0,0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0,1.0]]),
                                        np.array([[0.0,0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0,1.0]])]);
        training_labels = np.array([0,1,2,3,3]);
        training_labels_multi_digit = np.array([[0],[1],[2],[3],[3]]);
        
        # Train using n repetitions
        for k in range(repetitions):
            if (k % reporting_interval) == 0:
                print("# %d / %d" % (k, repetitions));
            
            # We pass the targets as labels if we are doing multi-digit 
            # prediction
            batch_indices = np.random.random_integers(0,len(training_data)-1,11);
            
            if (single_digit):
                training_labels = training_targets;
            else:
                training_labels = training_targets_multi_digit;
            rnn.train(training_data[batch_indices], training_labels[batch_indices], learning_rate, no_print=True);
        
        # Test
        stats = model.set_up_statistics(output_dim,[None]);
        test_data = training_data;
        batch_indices = np.random.random_integers(0,len(test_data)-1,100);
        if (single_digit):
            test_labels = training_labels;
            test_targets = training_targets;
        else:
            test_labels = training_labels_multi_digit;
            test_targets = training_targets_multi_digit;
        test_expressions = np.array(map(lambda e: "".join(map(lambda es: str(np.argmax(es)),e)), training_data));
        # We need to exclude statistic operator_scores to prevent usage of 
        # dataset
        stats = rnn.test(test_data[batch_indices], test_labels[batch_indices], 
                         test_targets[batch_indices], test_expressions[batch_indices], 
                         None, stats, ['operator_scores'], no_print_progress=True,
                         eos_symbol_index=4);
        
        return stats;

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testRecurrentNeuralNetwork']
    unittest.main()