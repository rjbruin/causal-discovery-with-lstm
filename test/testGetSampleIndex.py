'''
Created on 7 mrt. 2017

@author: Robert-Jan
'''
import unittest
import numpy as np;

from subsystems_finish import get_sample_index;

class Test(unittest.TestCase):


    def testGetSampleIndex(self):
        np.random.seed();
        # which_part, dataset_size, test_size, test_offset, val_size, lower bound, upper bound
        cases = [(0,1000,0.1,0.0,0.1,200,1000),
                 (0,1000,0.2,0.0,0.1,300,1000),
                 (1,1000,0.1,0.0,0.1,0,100),
                 (2,1000,0.1,0.0,0.1,100,200)];
        
        for i, (which_part, dataset_size, test_size, test_offset, val_size, lower_bound, upper_bound) in enumerate(cases):
            parameters = {'test_size': test_size, 'test_offset': test_offset, 'val_size': val_size};
            for _ in range(2000):
                index = get_sample_index(which_part, dataset_size, parameters);
                self.assertGreaterEqual(index, lower_bound, "(%d) Lower bound violated! index = %d" % (i, index));
                self.assertLessEqual(index, upper_bound, "(%d) Upper bound violated! index = %d" % (i, index));


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testGetSampleIndex']
    unittest.main()