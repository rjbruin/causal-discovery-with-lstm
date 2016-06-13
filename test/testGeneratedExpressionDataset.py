'''
Created on 13 jun. 2016

@author: Robert-Jan
'''
import unittest
from model.GeneratedExpressionDataset import GeneratedExpressionDataset


class Test(unittest.TestCase):

    # Data for use case
    sourceFolder = '../data/expressions_positive_integer_answer_shallow';
    single_digit = False;
    oneHot = {str(i): i for i in range(10)};
    oneHot['+'] = 10;
    oneHot['-'] = 11;
    oneHot['*'] = 12;
    oneHot['/'] = 13;
    oneHot['('] = 14;
    oneHot[')'] = 15;
    oneHot['='] = 16;
    data_dim = 18;
    
    def testPreloadImporting(self):
        """
        Checks if all expressions get imported using preloading.
        """
        # Settings
        preload = True;
        
        # Construct
        dataset = GeneratedExpressionDataset(self.sourceFolder, single_digit=self.single_digit, preload=preload);
        
        # Basic variable checks
        self.assertEqual(self.oneHot, dataset.oneHot, "(preload)oneHot encoding mismatch!");
        self.assertEqual(self.data_dim, dataset.data_dim, "(preload)data_dim size mismatch!");
        
        # Checking size of importing datasets
        train_length = dataset.filelength(dataset.sources[dataset.TRAIN]);
        self.assertEqual(train_length, len(dataset.train), "(preload)dataset.train length mismatch!");
        test_length = dataset.filelength(dataset.sources[dataset.TEST]);
        self.assertEqual(test_length, len(dataset.test), "(preload)dataset.test length mismatch!");
        
        print("(preload) Imported train = %d, test = %d" % (train_length, test_length));
        
    def testBatchImporting(self):
        """
        Checks if all expressions get imported using batched importing.
        """
        # Settings
        preload = False;
        batch_size = 1000;
        
        # Construct
        dataset = GeneratedExpressionDataset(self.sourceFolder, single_digit=self.single_digit, preload=preload);
        
        # Basic variable checks
        self.assertEqual(self.oneHot, dataset.oneHot, "(batch) oneHot encoding mismatch!");
        self.assertEqual(self.data_dim, dataset.data_dim, "(batch) data_dim size mismatch!");
        
        # Checking batches
        train_length = dataset.filelength(dataset.sources[dataset.TRAIN]);
        print("(batch) Training size = %d" % (train_length));
        iterations = int(train_length/batch_size);
        # Running twice the amount of iterations that fit in the training 
        # dataset guerantees we can train the overflowing of the dataset
        for i in range(iterations*2): 
            train, t_targets, t_labels, t_expressions = dataset.batch(batch_size);
            self.assertEqual(batch_size,len(train),"(batch) Iteration %d: training batch returned is not of size batch_size!" % (i));
            self.assertEqual(True,all(map(lambda i: len(i) == len(train), [t_targets, t_labels, t_expressions])), "(batch) Iteration %d: not all variables by batching are the same size!" % (i));


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testProcessing']
    unittest.main()