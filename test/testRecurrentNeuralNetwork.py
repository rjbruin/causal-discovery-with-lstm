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
        experiments_to_run = [(True,1,False,4,2000),(True,1,True,4,6000),(True,2,False,4,8000)];
        #experiments_to_run = [(True,1,True,4,6000),(True,2,False,4,8000)];
        
        for e, (lstm, layers, decoder, n_max_digits, repetitions) in enumerate(experiments_to_run):
            scores = [];
            for j in range(experiment_repetitions):
                stats = self.runTest(lstm,layers,decoder,n_max_digits,repetitions);
                scores.append(stats['score']);
                print("Iteration %d: %.2f percent" % (j+1, stats['score'] * 100));
            mean_score = np.mean(scores);
            print("Average score: %.2f" % (mean_score * 100));
            self.assertGreaterEqual(mean_score, 1.0, 
                                    "Experiment %d: mean score is not perfect: %.2f percent" % (e, mean_score * 100));

    def runTest(self, lstm=True, layers=1, decoder=False, n_max_digits=24, repetitions=3000):
        # Testcase settings
        parameters = arguments.defaults;
        parameters['train_batch_size'] = False;
        parameters['learning_rate'] = 0.1;
        parameters['repetitions'] = repetitions;
        
        # Model settings
        data_dim = 5;
        hidden_dim = 8;
        output_dim = 5;
        minibatch_size = 6;
        
        # Model initialization
        model = RecurrentNeuralNetwork(data_dim, hidden_dim, output_dim, 
                                       minibatch_size,
                                       single_digit=False, lstm=lstm,
                                       decoder=decoder,
                                       n_max_digits=n_max_digits,
                                       layers=layers);
        
        # Data generation
        training_data = np.array([np.array([[1.0,0.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,1.0]]),   # 004 => 0244
                                  np.array([[1.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,1.0]]),   # 014 => 1344
                                  np.array([[0.0,1.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,1.0]]),   # 104 => 2044
                                  np.array([[0.0,1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,1.0]]),   # 114 => 3144
                                  np.array([[1.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,1.0],[0.0,0.0,0.0,0.0,0.0]]),   # 04_ => 3014
                                  np.array([[0.0,0.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0,1.0],[0.0,0.0,0.0,0.0,0.0]])]); # 24_ => 0314
        training_targets_multi_digit = np.array([
                                        np.array([[1.0,0.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0,1.0],[0.0,0.0,0.0,0.0,1.0]]),
                                        np.array([[0.0,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0,1.0],[0.0,0.0,0.0,0.0,1.0]]),
                                        np.array([[0.0,0.0,1.0,0.0,0.0],[1.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,1.0],[0.0,0.0,0.0,0.0,1.0]]),
                                        np.array([[0.0,0.0,0.0,1.0,0.0],[0.0,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,1.0],[0.0,0.0,0.0,0.0,1.0]]),
                                        np.array([[0.0,0.0,0.0,1.0,0.0],[1.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,1.0]]),
                                        np.array([[1.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0,0.0],[0.0,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,1.0]])]);
        training_labels_multi_digit = np.array([[0,2],[1,3],[2,0],[3,1],[3,0,1],[0,3,1]]);
        training_expressions = np.array(map(lambda e: "".join(map(lambda es: str(np.argmax(es)),e)), training_data))
        
        dataset = FakeDataset(training_data,training_targets_multi_digit,training_labels_multi_digit,training_expressions,
                              training_data,training_targets_multi_digit,training_labels_multi_digit,training_expressions, 
                              EOS_symbol_index=training_targets_multi_digit.shape[-1]);
        
        start = time.clock();
        train(model, [dataset], parameters, 'testRNN', start, saveModels=False);
        stats = test(model, dataset, parameters, start);
        
        # Print sample predictions for every input sample
        prediction, _ = model.predict(training_data);
        for i, _ in enumerate(training_data):
            print("Sample %d: %s" % (i,str(prediction[i])));
        
        return stats;

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testRecurrentNeuralNetwork']
    unittest.main()