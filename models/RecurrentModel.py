'''
Created on 25 jul. 2016

@author: Robert-Jan
'''
from abc import ABCMeta, abstractmethod
import numpy as np;

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
    
    def writeVerboseOutput(self, message):
        if (self.verboseOutputter is not None):
            self.verboseOutputter['write'](message);
    
    @abstractmethod
    def getVars(self):
        pass
    
    def batch_statistics(self, stats, prediction, labels, targets, expressions, 
                         other, test_n, dataset,
                         excludeStats=None, no_print_progress=False,
                         eos_symbol_index=None, print_sample=False):
        # Statistics
        for j in range(0,test_n):
            if (self.single_digit):
                if (prediction[j] == labels[j]):
                    stats['correct'] += 1;
                    stats['digit_correct'] += 1;
                stats['digit_prediction_size'] += 1;
                
                stats['prediction_histogram'][int(prediction[j])] += 1;
                stats['groundtruth_histogram'][int(labels[j])] += 1;
                stats['prediction_confusion_matrix']\
                    [int(labels[j]),int(prediction[j])] += 1;
            else:
                # Get the labels
                argmax_target = np.argmax(targets[j],axis=1);
                # Compute the length of the target answer
                target_length = np.argmax(argmax_target);
                if (target_length == 0):
                    # If no EOS is found, the target is the entire length
                    target_length = targets[j].shape[1];
                # Compute the length of the prediction answer
                prediction_length = np.argmax(prediction[j]);
                if (prediction_length == target_length and np.array_equal(prediction[j][:target_length],argmax_target[:target_length])):
                    # Correct if prediction and target length match and 
                    # prediction and target up to target length are the same
                    stats['correct'] += 1.0;
                for k,digit in enumerate(prediction[j][:len(argmax_target)]):
                    if (digit == np.argmax(targets[j][k])):
                        stats['digit_correct'] += 1.0;
                    stats['digit_prediction_size'] += 1;
                    
                # Taking argmax over symbols for each sentence returns 
                # the location of the highest index, which is the first 
                # EOS symbol
                eos_location = np.argmax(prediction[j]);
                # Check for edge case where no EOS was found and zero was returned
                if (eos_symbol_index is None):
                    eos_symbol_index = dataset.EOS_symbol_index;
                if (prediction[j,eos_location] != eos_symbol_index):
                    stats['prediction_size_histogram'][prediction[j].shape[0]] += 1;
                else:
                    stats['prediction_size_histogram'][int(eos_location)] += 1;
                for digit_prediction in prediction[j]:
                    stats['prediction_histogram'][int(digit_prediction)] += 1;
            stats['prediction_size'] += 1;
        
        return stats;
        
    def total_statistics(self, stats):
        """
        Adds general statistics to the statistics generated per batch.
        """
        stats['score'] = stats['correct'] / float(stats['prediction_size']);
        stats['digit_score'] = stats['digit_correct'] / float(stats['digit_prediction_size']);
        
        return stats;