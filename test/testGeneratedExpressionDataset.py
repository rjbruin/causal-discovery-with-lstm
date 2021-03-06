'''
Created on 13 jun. 2016

@author: Robert-Jan
'''
import unittest
from models.GeneratedExpressionDataset import GeneratedExpressionDataset

import numpy as np;

class Test(unittest.TestCase):

    # Data for use case
    sourceFolder = '../data/subsystems_shallow_simple_topcause';
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
        dataset = GeneratedExpressionDataset(self.sourceFolder + '/train.txt', self.sourceFolder + '/test.txt', 
                                             single_digit=self.single_digit, preload=preload);
        
        # Basic variable checks
        self.assertEqual(self.oneHot, dataset.oneHot, "(preload) oneHot encoding mismatch!");
        self.assertEqual(self.data_dim, dataset.data_dim, "(preload) data_dim size mismatch!");
        
        # Checking size of importing datasets
        train_length = dataset.filelength(dataset.sources[dataset.TRAIN]);
        self.assertEqual(train_length, len(dataset.train), "(preload) dataset.train length mismatch!");
        self.assertEqual(train_length, dataset.lengths[dataset.TRAIN], "(preload) Dataset train length and actual length mismatch!");
        test_length = dataset.filelength(dataset.sources[dataset.TEST]);
        self.assertEqual(test_length, len(dataset.test), "(preload) dataset.test length mismatch!");
        self.assertEqual(test_length, dataset.lengths[dataset.TEST], "(preload) Dataset test length and actual length mismatch!");
        
        print("(preload) Imported train = %d, test = %d" % (train_length, test_length));
        
    def testTrainBatchImporting(self):
        """
        Checks if all expressions get imported using batched importing.
        """
        # Settings
        preload = False;
        batch_size = 1000;
        
        # Construct
        dataset = GeneratedExpressionDataset(self.sourceFolder + '/train.txt', self.sourceFolder + '/test.txt', 
                                             single_digit=self.single_digit, preload=preload);
        
        # Basic variable checks
        self.assertEqual(self.oneHot, dataset.oneHot, "(batch) oneHot encoding mismatch!");
        self.assertEqual(self.data_dim, dataset.data_dim, "(batch) data_dim size mismatch!");
        
        # Checking batches
        train_length = dataset.filelength(dataset.sources[dataset.TRAIN]);
        self.assertEqual(train_length, dataset.lengths[dataset.TRAIN], "(batch) Dataset train length and actual length mismatch!");
        print("(batch) Training size = %d" % (train_length));
#         iterations = int(train_length/batch_size);
#         # Running twice the amount of iterations that fit in the training 
#         # dataset guerantees we can train the overflowing of the dataset
#         for i in range(iterations*2): 
#             train, t_targets, t_labels, t_expressions = dataset.get_train_batch(batch_size);
#             self.assertEqual(batch_size,len(train),"(batch) Iteration %d: training batch returned is not of size batch_size!" % (i));
#             self.assertEqual(True,all(map(lambda i: len(i) == len(train), [t_targets, t_labels, t_expressions])), "(batch) Iteration %d: not all variables by batching are the same size!" % (i));

    def testTestBatch(self):
        """
        Checks if test batching goes right.
        """
        # Settings
        preload = False;
        batch_size = 1000;
        
        # Construct
        dataset = GeneratedExpressionDataset(self.sourceFolder + '/train.txt', self.sourceFolder + '/test.txt', 
                                             single_digit=self.single_digit, preload=preload, test_batch_size=batch_size);
        
        # Basic variable checks
        self.assertEqual(self.oneHot, dataset.oneHot, "(test) oneHot encoding mismatch!");
        self.assertEqual(self.data_dim, dataset.data_dim, "(test) data_dim size mismatch!");
        self.assertEqual(dataset.preloaded, False, "(test) preloaded should be False!")
        
        # Check batches
        test_length = dataset.filelength(dataset.sources[dataset.TEST]);
        leftover = test_length % batch_size;
        self.assertEqual(test_length, dataset.lengths[dataset.TEST], "(test) Dataset test length and actual length mismatch!");
        print("(test) Test size = %d" % (test_length));
        iterations = int(test_length/batch_size);
        # Running twice the amount of iterations that fit in the test 
        # dataset guerantees we can train the overflowing of the dataset
        finished = False;
        i = 0;
        while not finished:
            results = dataset.get_test_batch();
            if (not results):
                finished = True;
                print("(test) Final batch = %d" % (i));
                # Assert that variables have been reset
                self.assertEqual(dataset.locations[dataset.TEST],0,"(test) Dataset location is not reset correctly after test is done!");
                self.assertEqual(dataset.test_done,False,"(test) Dataset test_done is not reset correctly after test is done!");
                continue;
            else:
                test, t_targets, t_labels, t_expressions = results;
            
            # Basic checks
            self.assertEqual(True,all(map(lambda i: len(i) == len(test), [t_targets, t_labels, t_expressions])), "(test) Iteration %d: not all variables by batching are the same size!" % (i));
            
            # Internal variable checks
            if (leftover > 0 and i == iterations) or (leftover == 0 and i == iterations-1):
                # This is the final iteration
                if (leftover > 0):
                    self.assertLessEqual(leftover, len(test), "(test) Final batch size is too large!");
                    self.assertGreaterEqual(leftover, len(test), "(test) Final batch size is too small!");
                self.assertEqual(dataset.test_done, True, "(test) Test is not marked as done after final batch!");
            else:
                # As this is not the final iteration, the size of the test 
                # batch should be equal to batch_size
                self.assertEqual(batch_size,len(test),"(test) Iteration %d: test batch returned is not of size batch_size!" % (i));
                self.assertEqual(dataset.locations[dataset.TEST] % batch_size, 0, "(test) Location is not updated correctly: leftover = %d" % (dataset.locations[dataset.TEST] % batch_size));
            
            i += 1;
    
    def testTestSampling(self):
        """
        Checks if test batching using sampling goes right.
        """
        # Settings
        preload = False;
        sample_batch_sizes = [1,99,1000,31233,100000];
        
        for sample_batch_size in sample_batch_sizes:
        
            # Construct
            dataset = GeneratedExpressionDataset(self.sourceFolder + '/train.txt', self.sourceFolder + '/test.txt',
                                                 single_digit=self.single_digit, preload=preload,
                                                 sample_testing_size=sample_batch_size);
            
            # Basic variable checks
            self.assertEqual(self.oneHot, dataset.oneHot, "(test sampling) oneHot encoding mismatch!");
            self.assertEqual(self.data_dim, dataset.data_dim, "(test sampling) data_dim size mismatch!");
            self.assertEqual(dataset.preloaded, False, "(test sampling) preloaded should be False!")
            
            # First request
            results = dataset.get_test_batch();
            self.assertNotEqual(results, False, "(test sampling) No batches generated!");
            test, t_targets, t_labels, t_expressions = results;
            
            # Basic checks
            self.assertEquals(test.shape[0], sample_batch_size, "(test sampling) Batch size does not match setting sample_batch_size: %d - %d" % (test.shape[0], sample_batch_size)); 
            self.assertEqual(True,all(map(lambda i: len(i) == len(test), [t_targets, t_labels, t_expressions])), "(test sampling) Not all variables by batching are the same size!");
            
            # Second request, should fail
            results = dataset.get_test_batch();
            self.assertEquals(results, False, "(test sampling) Second batch returned while using sampling!");
            
            # First request again
            results = dataset.get_test_batch();
            self.assertNotEqual(results, False, "(test sampling) Test batching does not reset!");
            results = dataset.get_test_batch();
            self.assertEquals(results, False, "(test sampling) Second batch returned while using sampling!");
            
            try:
                for i in range(100):
                    results = dataset.get_test_batch();
                    test, t_targets, t_labels, t_expressions = results;
                    self.assertGreater(len(test),0,"(test sampling) Batch batch %d of size %d was too small: %d" % (i, sample_batch_size, len(test)));
                    # Call again to reset
                    results = dataset.get_test_batch();
            except Exception as error:
                self.assertEqual(True,False,"(test sampling) Exception '%s' occurred in batch %d of size %d" % (error, i, sample_batch_size));
    
    def testAbstractExpression(self):
        # Construct
        dataset = GeneratedExpressionDataset(self.sourceFolder + '/train.txt', self.sourceFolder + '/test.txt', 'thisdoesnotexist',
                                             copyMultipleExpressions=True);
        
        samples = [('1=1',np.array([1,0,0,0,1,0,0,0,0,1,0,0,0])),
                   ('1+1=1',np.array([1,0,0,0,0,1,0,0,0,0,1,0,0])),
                   ('1+(2-3)=1',np.array([1,0,0,0,0,0,1,0,0,0,0,1,0])),
                   ('(3*3)+(2-3)=1',np.array([1,0,0,0,0,0,0,1,0,0,0,0,1])),
                   ('(3*3)+(2-3)=51',np.array([0,0,1,0,0,0,0,1,0,0,0,0,1]))];
        
        for i, (expression, abstractExpression) in enumerate(samples):
            encodedAbstract = dataset.abstractExpression(expression);
            self.assertEqual(np.array_equal(abstractExpression,encodedAbstract),True,"(%d) No match: %s vs. encoded %s" % (i, str(abstractExpression), str(encodedAbstract)));

    def testEffectMatcher(self):
        dataset = GeneratedExpressionDataset('../data/seq2ndmarkov_both/all.txt', '../data/seq2ndmarkov_both/test.txt', '../data/seq2ndmarkov_both/config.json',
                                             operators=2,
                                             digits=8,
                                             dataset_type=GeneratedExpressionDataset.DATASET_SEQ2NDMARKOV,
                                             bothcause=True,
                                             finishExpressions=True);
        
        for i in range(len(dataset.expressionsByPrefix.expressions)):
            top = dataset.encodeExpression(dataset.expressionsByPrefix.expressions[i]);
            bot = dataset.encodeExpression(dataset.expressionsByPrefix.primedExpressions[i]);
            
            test = dataset.effect_matcher_seq2ndmarkov_both(top, bot, 8, 2, True);
            self.assertEqual(True,test,"(effect) %d is wrong: top = %s, bot = %s" % (i, top, bot));
            if (not test):
                print("(effect) %d is wrong: top = %s, bot = %s" % (i, top, bot));

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testProcessing']
    unittest.main()