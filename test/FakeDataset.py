'''
Created on 25 jul. 2016

@author: Robert-Jan
'''

import numpy as np;

from model.Dataset import Dataset

class FakeDataset(Dataset):
    '''
    classdocs
    '''


    def __init__(self, data, targets, labels, expressions, test_data, test_targets, test_labels, test_expressions, EOS_symbol_index):
        '''
        Constructor
        '''
        self.data = data;
        self.targets = targets;
        self.labels = labels;
        self.expressions = expressions;
        self.test_data = test_data;
        self.test_targets = test_targets;
        self.test_labels = test_labels;
        self.test_expressions = test_expressions;
        
        self.lengths = [len(data), len(test_data)];
        self.data_dim = self.data.shape[-1];
        self.output_dim = self.targets.shape[-1];
        self.EOS_symbol_index = EOS_symbol_index;
        
        self.testStatus = False;
    
    def get_train_batch(self, size=None):
        if (size is None):
            size = self.lengths[0];
        batch_indices = np.random.random_integers(0,len(self.data)-1,size);
        return self.data[batch_indices], self.targets[batch_indices], self.labels[batch_indices], self.expressions[batch_indices];
    
    def get_test_batch(self):
        if (self.testStatus):
            self.testStatus = False;
            return False;
        self.testStatus = True;
        return self.test_data, self.test_targets, self.test_labels, self.test_expressions;