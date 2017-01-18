'''
Created on 25 jul. 2016

@author: Robert-Jan
'''
from abc import ABCMeta, abstractmethod
import numpy as np;
import copy;

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
        
    def total_statistics(self, stats, total_labels_used={}):
        """
        Adds general statistics to the statistics generated per batch.
        """
        stats['prediction_histogram'] = RecurrentModel.addDicts(stats['prediction_1_histogram'], stats['prediction_2_histogram']);
#         stats['prediction_size_histogram'] = RecurrentModel.addDicts(stats['prediction_1_size_histogram'], stats['prediction_2_size_histogram']);
        
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
            stats['structureScoreCause'] = stats['structureCorrectCause'] / float(stats['prediction_size']);
            stats['structureScoreEffect'] = stats['structureCorrectEffect'] / float(stats['prediction_size']);
            stats['structureScoreTop'] = stats['structureCorrectTop'] / float(stats['prediction_size']);
            stats['structureScoreBot'] = stats['structureCorrectBot'] / float(stats['prediction_size']);
            stats['structureScore'] = stats['structureCorrect'] / float(stats['prediction_size']);
            stats['allEffectScore'] = stats['effectCorrect'] / float(stats['prediction_size']);
            stats['noEffectScore'] = stats['noEffect'] / float(stats['prediction_size']);
            stats['validScore'] = stats['valid'] / float(stats['prediction_size']);
            stats['structureValidScoreCause'] = stats['structureValidCause'] / float(stats['prediction_size']);
            stats['structureValidScoreEffect'] = stats['structureValidEffect'] / float(stats['prediction_size']);
            stats['structureValidScoreTop'] = stats['structureValidTop'] / float(stats['prediction_size']);
            stats['structureValidScoreBot'] = stats['structureValidBot'] / float(stats['prediction_size']);
            stats['inDatasetScore'] = stats['inDataset'] / float(stats['prediction_size']);
            stats['prediction_size_score'] = {};
            for size, val in stats['prediction_size_correct'].items():
                if (stats['prediction_sizes'][size] > 0):
                    stats['prediction_size_score'][size] = val / float(stats['prediction_sizes'][size]);
                else:
                    stats['prediction_size_score'][size] = 0.
        else:
            stats['score'] = 0.0;
            stats['structureScoreCause'] = 0.0;
            stats['structureScoreEffect'] = 0.0;
            stats['structureScoreTop'] = 0.0;
            stats['structureScoreBot'] = 0.0;
            stats['structureScore'] = 0.0;
            stats['allEffectScore'] = 0.0;
            stats['noEffectScore'] = 0.0;
            stats['validScore'] = 0.0;
            stats['structureValidScoreCause'] = 0.0;
            stats['structureValidScoreEffect'] = 0.0;
            stats['structureValidScoreTop'] = 0.0;
            stats['structureValidScoreBot'] = 0.0;
            stats['inDatasetScore'] = 0.0;
        
        if (stats['localSize'] > 0):
            stats['localValidScore'] = stats['localValid'] / float(stats['localSize']);
            stats['localValidScoreCause'] = stats['localValidCause'] / float(stats['localSize'])
            stats['localValidScoreEffect'] = stats['localValidEffect'] / float(stats['localSize']);
        else:
            stats['localValidScore'] = 0.0;
            stats['localValidScoreCause'] = 0.0;
            stats['localValidScoreEffect'] = 0.0;
        
        if (stats['digit_prediction_size'] > 0):
            stats['digit_score'] = stats['digit_correct'] / float(stats['digit_prediction_size']);
        else:
            stats['digit_score'] = 0.0;
        stats['effectScore'] = stats['allEffectScore'] - stats['noEffectScore'];
        
        stats['error_1_score'] = 0.0;
        stats['error_2_score'] = 0.0;
        stats['error_3_score'] = 0.0;
        if (stats['prediction_size'] > 0):
            stats['error_1_score'] = (stats['correct'] + stats['error_histogram'][1]) / float(stats['prediction_size']);
            stats['error_2_score'] = (stats['correct'] + stats['error_histogram'][1] + \
                                      stats['error_histogram'][2]) / float(stats['prediction_size']);
            stats['error_3_score'] = (stats['correct'] + stats['error_histogram'][1] + \
                                      stats['error_histogram'][2] + stats['error_histogram'][3]) / float(stats['prediction_size']); 
        
        stats['unique_labels_predicted'] = len(total_labels_used.keys());
        
        return stats;
    
    @staticmethod
    def addDicts(dict1, dict2):
        newDict = copy.deepcopy(dict1);
        for key in dict2:
            if (key not in newDict):
                newDict[key] = 0;
            newDict[key] += dict2[key];
        return newDict;
        