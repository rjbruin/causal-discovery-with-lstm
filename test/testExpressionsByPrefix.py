'''
Created on 8 sep. 2016

@author: Robert-Jan
'''
import unittest
from models import SequencesByPrefix.ExpressionsByPrefix
from models.GeneratedExpressionDataset import GeneratedExpressionDataset
from profiler import profiler


class Test(unittest.TestCase):


    def testSampleExpressions(self):
        expressions = ['1+2=3',
                       '1+(2-1)=2',
                       '1',
                       '2/2=1'
                       ];
        
        storage = SequencesByPrefix();
        for exp in expressions:
            storage.add(exp);
        
        queries = [('1+',2)];
        
        for i, (q, n) in enumerate(queries):
            result = storage.get(q, len(q));
            print(result);
            self.assertEqual(len(result),n,"Answer %d: wrong number of results: got %d, should be %d" % (i+1,len(result),n));
    
    def testOnDataset(self):
        dataset = GeneratedExpressionDataset('../data/expressions_positive_integer_answer_shallow/train.txt',
                                             '../data/expressions_positive_integer_answer_shallow/test.txt',
                                             preload=False, finishExpressions=True);
        
        _, _, _, train_expressions = \
            dataset.loadFile(dataset.sources[dataset.TRAIN], 
                          location_index=dataset.TRAIN, 
                          file_length=dataset.lengths[dataset.TRAIN]);
        
        print("Now querying %d expressions..." % (len(train_expressions)));
        
        profiler.start('querying');
        for exp in train_expressions:
            lookup = dataset.expressionsByPrefix.get(exp[:5], len(exp[:5]));
        profiler.stop('querying');
        
        profiler.profile();
        
    def testGetRandom(self):
        dataset = GeneratedExpressionDataset('../data/expressions_positive_integer_answer_shallow/train.txt',
                                             '../data/expressions_positive_integer_answer_shallow/test.txt',
                                             preload=False, finishExpressions=True);
        
        for _ in range(10):
            expressions = dataset.expressionsByPrefix.get_random(10);
            print(expressions);

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testExpressionsByPrefix']
    unittest.main()