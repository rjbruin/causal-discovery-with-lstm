'''
Created on 15 jun. 2016

@author: Robert-Jan
'''
import numpy as np;

import unittest
from models.RecurrentNeuralNetwork import RecurrentNeuralNetwork
from test.FakeDataset import FakeDataset;
from tools.model import train, test
import time
from tools import arguments


class Test(unittest.TestCase):


    def testRecurrentNeuralNetwork(self):
        experiment_repetitions = 1;        
        experiments_to_run = [(False,True,1,2,False,1000),(False,True,1,2,True,4000),(False,True,2,2,False,8000)];
        
        for e, (single_digit, lstm, layers, n_max_digits, decoder, repetitions) in enumerate(experiments_to_run):
            scores = [];
            for j in range(experiment_repetitions):
                stats = self.runTest(single_digit,lstm,layers,decoder,n_max_digits,repetitions);
                scores.append(stats['score']);
                print("Iteration %d: %.2f percent" % (j+1, stats['score'] * 100));
            mean_score = np.mean(scores);
            print("Average score: %.2f" % (mean_score * 100));
            self.assertGreaterEqual(mean_score, 1.0, 
                                    "Experiment %d: mean score is not perfect: %.2f percent" % (e, mean_score * 100));

    def runTest(self, single_digit=False, lstm=True, layers=1, decoder=False, n_max_digits=24, repetitions=3000):
        # Testcase settings
        parameters = arguments.defaults;
        parameters['train_batch_size'] = False;
        parameters['learning_rate'] = 0.1;
        parameters['repetitions'] = repetitions;
        
        # Model settings
        data_dim = 5;
        hidden_dim = 8;
        output_dim = 5;
        single_digit = single_digit;
        minibatch_size = 5;
        
        # Model initialization
        model = RecurrentNeuralNetwork(data_dim, hidden_dim, output_dim, 
                                       minibatch_size,
                                       single_digit=single_digit, lstm=lstm,
                                       decoder=decoder,
                                       n_max_digits=n_max_digits,
                                       layers=layers);
        
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
        training_expressions = np.array(map(lambda e: "".join(map(lambda es: str(np.argmax(es)),e)), training_data))
        
        if (single_digit):
            dataset = FakeDataset(training_data,training_targets,training_labels,training_expressions,
                                  training_data,training_targets,training_labels,training_expressions,
                                  EOS_symbol_index=training_targets.shape[-1]);
        else:
            dataset = FakeDataset(training_data,training_targets_multi_digit,training_labels_multi_digit,training_expressions,
                                  training_data,training_targets_multi_digit,training_labels_multi_digit,training_expressions, 
                                  EOS_symbol_index=training_targets_multi_digit.shape[-1]);
        
        start = time.clock();
        train(model, [dataset], parameters, 'testRNN', start, saveModels=False);
        stats = test(model, dataset, parameters, start);
        
        return stats;

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testRecurrentNeuralNetwork']
    unittest.main()