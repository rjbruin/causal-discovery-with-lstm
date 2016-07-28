'''
Created on 25 jul. 2016

@author: Robert-Jan
'''
from abc import ABCMeta, abstractmethod

class Dataset(object):
    '''
    classdocs
    '''
    __metaclass__ = ABCMeta
    
    TRAIN = 0;
    TEST = 1;

    def __init__(self):
        '''
        Constructor
        '''
    
    @abstractmethod
    def get_train_batch(self):
        pass
    
    @abstractmethod
    def get_test_batch(self):
        pass