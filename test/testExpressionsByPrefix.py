'''
Created on 8 sep. 2016

@author: Robert-Jan
'''
import unittest
from models.SequencesByPrefix import SequencesByPrefix
from models.GeneratedExpressionDataset import GeneratedExpressionDataset
from profiler import profiler
from tools.model import constructModels
from tools.arguments import processCommandLineArguments;

import numpy as np;

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
    
    def testExists(self):
        params = ['--finish_subsystems','True',
                  '--only_cause_expression','1',
                  '--dataset','../data/subsystems_shallow_simple_topcause',
                  "--sample_testing_size", "10000",
                  "--n_max_digits", "17",
                  "--intervention_base_offset", "0",
                  "--intervention_range", "17",
                  "--nesterov_optimizer", "True",
                  "--decoder", "True",
                  "--learning_rate", "0.005",
                  "--hidden_dim", "256"];
        params = processCommandLineArguments(params);
        datasets, _ = constructModels(params, 1234, {});
        dataset = datasets[0];
        
        storage = dataset.expressionsByPrefix;
        
        expressions = ["(3-9)*(0-3)=18","(4-7)+(6*5)=27","(0/6)+(2*8)=16","(1-4)+(3+6)=6","(6+0)+(0-1)=5"];
        for i, expr in enumerate(expressions):
            self.assertEqual(storage.exists(expr),True,"(exists) Failing exists lookup for sample %d" % i);
            _, _, _, _, branch = storage.get(expr[:4], alsoGetStructure=True);
            closest, _, _, _ = branch.get_closest(expr[4:]);
            self.assertNotEqual(closest,False,"(exists) Branch-based lookup failed with False for sample %d: %s" % (i, closest));
            self.assertEqual(closest,expr,"(exists) Failing branch-based lookup for sample %d: %s" % (i, closest));
            
            # Apply mutations and test if both methods get the same new label
            for n in range(20):
                intervention_location = np.random.randint(0,len(expr));
                new_symbol = np.random.randint(dataset.data_dim);
                new_expression = expr[:intervention_location] + dataset.findSymbol[new_symbol] + expr[intervention_location+1:];
                print("Old: %s\tNew: %s" % (expr, new_expression));
            
                _, _, valids, _, branch = storage.get(new_expression[:intervention_location+1], alsoGetStructure=True);
                if (new_expression not in valids and len(valids) > 0):
                    # Old method: compare all
                    profiler.start('old');
                    nearest = -1;
                    nearest_score = 100000;
                    for j, nexpr in enumerate(valids):
                        score = string_difference(new_expression, nexpr);
                        if (score < nearest_score):
                            nearest = j;
                            nearest_score = score;
                    closest_old = valids[nearest];
                    profiler.stop('old');
                    
                    profiler.start('new');
                    # New method: 
                    closest_new, _, _, _ = branch.get_closest(new_expression[intervention_location+1:]);
                    profiler.stop('new');
                    
                    if (closest_old != closest_new):
                        print("(exists) Intervened closest do not match for sample %d: loc %d / orig %s / int %s / old %s / new %s" % 
                                        (i, intervention_location, expr, new_expression, closest_old, closest_new));
#                     self.assertEqual(closest_old, closest_new, 
#                                      "(exists) Intervened closest do not match for sample %d: loc %d / orig %s / int %s / old %s / new %s" % 
#                                         (i, intervention_location, expr, new_expression, closest_old, closest_new));
        
        profiler.profile();

def string_difference(string1, string2):
    # Compute string difference
    score = 0;
    string1len = len(string1);
    k = 0;
    for k,s in enumerate(string2):
        if (string1len <= k):
            score += 1;
        elif (s != string1[k]):
            score += 1;
    score += max(0,len(string1) - (k+1));
    return score;

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testExpressionsByPrefix']
    unittest.main()