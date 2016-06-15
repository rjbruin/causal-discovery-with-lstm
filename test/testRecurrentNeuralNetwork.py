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
        for _ in range(experiment_repetitions):
            self.runTest(True);
            self.runTest(False);

    def runTest(self, lstm=True):
        # Testcase settings
        learning_rate = 0.1;
        repetitions = 1000;
        
        # Model settings
        data_dim = 4;
        hidden_dim = 4;
        output_dim = 4;
        single_digit = False;
        
        # Model initialization
        rnn = RecurrentNeuralNetwork(data_dim, hidden_dim, output_dim, 
                                     single_digit=single_digit, lstm=lstm);
        
        # Data generation
        training_data = [np.array([[1.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0]]),
                         np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0]]),
                         np.array([[0.0,1.0,0.0,0.0],[1.0,0.0,0.0,0.0]]),
                         np.array([[0.0,1.0,0.0,0.0],[0.0,1.0,0.0,0.0]])];
        training_targets = [np.array([[1.0,0.0,0.0,0.0]]),
                            np.array([[0.0,1.0,0.0,0.0]]),
                            np.array([[0.0,0.0,1.0,0.0]]),
                            np.array([[0.0,0.0,0.0,1.0]])];
        training_labels = np.array([[0],[1],[2],[3]]);
        
        # Train using n repetitions
        for _ in range(repetitions):
            # We pass the targets as labels if we are doing multi-digit 
            # prediction
            if (not single_digit):
                training_labels = training_targets;
            rnn.train(training_data, training_labels, learning_rate);
        
        # Test
        stats = model.set_up_statistics(output_dim,[None]);
        test_data = training_data;
        test_labels = training_labels;
        test_targets = training_targets;
        test_expressions = map(lambda e: "".join(map(lambda es: str(np.argmax(es)),e)), training_data);
        # We need to exclude statistic operator_scores to prevent usage of 
        # dataset
        stats = rnn.test(test_data, test_labels, test_targets, test_expressions, None, stats, ['operator_scores']);
        
        print("Score: %.2f percent" % (stats['score'] * 100));
        
        self.assertEqual(stats['score'],1.0,"One of the {0} experiments did not get a 100% score:\n{1}".format('lstm' if lstm else 'rnn',str(stats)));

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testRecurrentNeuralNetwork']
    unittest.main()