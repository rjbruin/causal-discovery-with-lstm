'''
Created on 25 jul. 2016

@author: Robert-Jan
'''
from abc import ABCMeta, abstractmethod

class RecurrentModel(object):
    '''
    Requires variables:
    - int minibatch_size
    - bool fake_minibatch
    - bool time_training_batch
    - {string,Theano.shared} vars
    - {string,verbose output functions} or None verboseOutput
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