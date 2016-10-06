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
                         eos_symbol_index=None, print_sample=False,
                         emptySamples=None):
        # Statistics
        for j in range(0,test_n):
            if (emptySamples is not None and j in emptySamples):
                continue;
            
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
            for k,digit in enumerate(prediction[j][:target_length]):
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
        stats['prediction_histogram'] = dict(stats['prediction_1_histogram'], **stats['prediction_2_histogram']);
        stats['prediction_size_histogram'] = dict(stats['prediction_1_size_histogram'], **stats['prediction_2_size_histogram']);
        
        stats['digit_correct'] = stats['digit_1_correct'] + stats['digit_2_correct'];
        stats['digit_prediction_size'] = stats['digit_1_prediction_size'] + stats['digit_2_prediction_size'];
        
        if (stats['digit_1_prediction_size'] > 0):
            stats['digit_1_score'] = stats['digit_1_correct'] / float(stats['digit_1_prediction_size']);
        else:
            stats['digit_1_score'] = 0.0;
        if (stats['digit_2_prediction_size'] > 0):
            stats['digit_2_score'] = stats['digit_2_correct'] / float(stats['digit_2_prediction_size']);
        else:
            stats['digit_2_score'] = 0.0;
        
        if (stats['prediction_size'] > 0):
            stats['score'] = stats['correct'] / float(stats['prediction_size']);
            stats['effectScore'] = stats['effectCorrect'] / float(stats['prediction_size']);
            stats['causeScore'] = stats['causeCorrect'] / float(stats['prediction_size']);
            stats['validScore'] = stats['valid'] / float(stats['prediction_size']);
            stats['causeValidScore'] = stats['causeValid'] / float(stats['prediction_size']);
            stats['effectValidScore'] = stats['effectValid'] / float(stats['prediction_size']);
        else:
            stats['score'] = 0.0;
            stats['effectScore'] = 0.0;
            stats['causeScore'] = 0.0;
            stats['validScore'] = 0.0;
            stats['causeValidScore'] = 0.0;
            stats['effectValidScore'] = 0.0;
        if (stats['digit_prediction_size'] > 0):
            stats['digit_score'] = stats['digit_correct'] / float(stats['digit_prediction_size']);
        else:
            stats['digit_score'] = 0.0;
        
        stats['error_1_score'] = 0.0;
        stats['error_2_score'] = 0.0;
        stats['error_3_score'] = 0.0;
        if (stats['prediction_size'] > 0):
            stats['error_1_score'] = (stats['correct'] + stats['error_histogram'][1]) / float(stats['prediction_size']);
            stats['error_2_score'] = (stats['correct'] + stats['error_histogram'][1] + \
                                      stats['error_histogram'][2]) / float(stats['prediction_size']);
            stats['error_3_score'] = (stats['correct'] + stats['error_histogram'][1] + \
                                      stats['error_histogram'][2] + stats['error_histogram'][3]) / float(stats['prediction_size']); 
        
        return stats;