'''
Created on 25 jul. 2016

@author: Robert-Jan
'''
from abc import ABCMeta, abstractmethod

class RecurrentModel(object):
    '''
    classdocs
    '''
    __metaclass__ = ABCMeta

    def __init__(self):
        '''
        Constructor
        '''
    
    @abstractmethod
    def sanityChecks(self, training_data, training_labels):
        pass
    
    @abstractmethod
    def sgd(self, data, labels, learning_rate):
        pass
    
    @abstractmethod
    def predict(self, data, labels, targets):
        pass