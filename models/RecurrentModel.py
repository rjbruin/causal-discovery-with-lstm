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
        
    def total_statistics(self, stats, dataset, parameters, total_labels_used={}, digits=True):
        """
        Adds general statistics to the statistics generated per batch.
        """
        stats['prediction_histogram'] = RecurrentModel.addDicts(stats['prediction_1_histogram'], stats['prediction_2_histogram']);
#         stats['prediction_sizes'] = RecurrentModel.addDicts(stats['prediction_1_size_histogram'], stats['prediction_2_size_histogram']);
        
        for i in range(parameters['n_max_digits']+1):
            if (stats['digit_1_prediction_size'][i] > 0):
                stats['digit_1_score'][i] = stats['digit_1_correct'][i] / float(stats['digit_1_prediction_size'][i]);
            else:
                stats['digit_1_score'][i] = 0.0;
            if (stats['digit_2_prediction_size'][i] > 0):
                stats['digit_2_score'][i] = stats['digit_2_correct'][i] / float(stats['digit_2_prediction_size'][i]);
            else:
                stats['digit_2_score'][i] = 0.0;
        
        if (digits):
            # Using fixed values!
            dp_length = self.n_max_digits;
            
            d1_size = np.sum([stats['digit_1_prediction_size'][i] for i in range(dp_length)]);
            d2_size = np.sum([stats['digit_2_prediction_size'][i] for i in range(dp_length)]);
            stats['digit_prediction_size'] = d1_size + d2_size;
            if (d1_size > 0):
                stats['digit_1_total_score'] = np.sum([stats['digit_1_score'][i] * (stats['digit_1_prediction_size'][i] / float(d1_size)) for i in range(dp_length)]);
            else:
                stats['digit_1_total_score'] = 0.;
            if (d2_size > 0):
                stats['digit_2_total_score'] = np.sum([stats['digit_2_score'][i] * (stats['digit_2_prediction_size'][i] / float(d2_size)) for i in range(dp_length)]);
            else:
                stats['digit_2_total_score'] = 0.;
        
            # PRINT Digit precision per index = digit_2_score
            # PRINT First wrong prediction = first_error_score
            # Recovery percentage per errors = recovery_score_by_error
            if (d2_size > 0):
                stats['first_error_score'] = {k: 0. for k in range(-1,9)};
                stats['recovery_score'] = {k: 0. for k in range(8)};
                stats['error_size'] = {k: 0. for k in range(8)};
                stats['first_error_score'][-1] = stats['first_error'][-1] / d2_size;
                for i in range(8):
                    stats['first_error_score'][i] = stats['first_error'][i] / d2_size;
                    stats['error_size'][i] = float(stats['recovery'][i] + stats['no_recovery'][i]) / d2_size;
                    if (stats['recovery'][i] + stats['no_recovery'][i] > 0):
                        stats['recovery_score'][i] = stats['recovery'][i] / float(stats['recovery'][i] + stats['no_recovery'][i]);
                    else:
                        stats['recovery_score'][i] = 0.;
                stats['first_error_score'][8] = stats['first_error'][8] / d2_size;
        
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
            stats['label_size_score'] = {};
            for size, val in stats['label_size_correct'].items():
                if (stats['label_sizes'][size] > 0):
                    stats['label_size_score'][size] = val / float(stats['label_sizes'][size]);
                else:
                    stats['label_size_score'][size] = 0.
            stats['input_size_score'] = {};
            for size, val in stats['input_size_correct'].items():
                if (stats['input_sizes'][size] > 0):
                    stats['input_size_score'][size] = val / float(stats['input_sizes'][size]);
                else:
                    stats['input_size_score'][size] = 0.
            stats['label_size_input_size_confusion_score'] = np.zeros((parameters['n_max_digits']+1,parameters['n_max_digits']+1), dtype='float32');
            stats['left_missing_vs_left_size_score'] = np.zeros((parameters['n_max_digits']+1,parameters['n_max_digits']+1), dtype='float32');
            for l in range(parameters['n_max_digits']+1):
                for i in range(parameters['n_max_digits']+1):
                    if (stats['label_size_input_size_confusion_size'][l,i] > 0):
                        stats['label_size_input_size_confusion_score'][l,i] = stats['label_size_input_size_confusion_correct'][l,i] / float(stats['label_size_input_size_confusion_size'][l,i]);
                    else:
                        stats['label_size_input_size_confusion_score'][l,i] = 0.;
                    if (stats['left_missing_vs_left_size_size'][l,i] > 0):
                        stats['left_missing_vs_left_size_score'][l,i] = stats['left_missing_vs_left_size_correct'][l,i] / float(stats['left_missing_vs_left_size_size'][l,i]);
                    else:
                        stats['left_missing_vs_left_size_score'][l,i] = 0.;
            
            stats['x_hand_side_score'] = {};
            if (stats['x_hand_side_size']['left'] > 0):
                stats['x_hand_side_score']['left'] = stats['x_hand_side_correct']['left'] / float(stats['x_hand_side_size']['left']);
            else:
                stats['x_hand_side_score']['left'] = 0.;
            if (stats['x_hand_side_size']['right'] > 0):
                stats['x_hand_side_score']['right'] = stats['x_hand_side_correct']['right'] / float(stats['x_hand_side_size']['right']);
            else:
                stats['x_hand_side_score']['right'] = 0.;
            if (stats['x_hand_side_size']['equals'] > 0):
                stats['x_hand_side_score']['equals'] = stats['x_hand_side_correct']['equals'] / float(stats['x_hand_side_size']['equals']);
            else:
                stats['x_hand_side_score']['equals'] = 0.;
            
            stats['x_offset_score'] = {};
            for size, val in stats['x_offset_correct'].items():
                if (stats['x_offset_size'][size] > 0):
                    stats['x_offset_score'][size] = val / float(stats['x_offset_size'][size]);
                else:
                    stats['x_offset_score'][size] = 0.
            
            stats['syntactically_valid_score'] = stats['syntactically_valid'] / float(stats['prediction_size']);
            stats['semantically_valid_score'] = stats['semantically_valid'] / float(stats['prediction_size']);
            stats['left_hand_valid_score'] = stats['left_hand_valid'] / float(stats['prediction_size']);
            if (stats['left_hand_valid'] > 0):
                stats['left_hand_valid_correct_score'] = stats['left_hand_valid_correct'] / float(stats['left_hand_valid']);
            else:
                stats['left_hand_valid_correct_score'] = 0.;
            stats['right_hand_valid_score'] = stats['right_hand_valid'] / float(stats['prediction_size']);
            
            if (stats['left_hand_valid_with_prediction_size'] > 0):
                stats['left_hand_valid_with_prediction_score'] = stats['left_hand_valid_with_prediction_correct'] / float(stats['left_hand_valid_with_prediction_size']);
                stats['valid_left_hand_with_prediction_score'] = stats['valid_left_hand_valid_with_prediction_size'] / float(stats['left_hand_valid_with_prediction_size']);
            else:
                stats['left_hand_valid_with_prediction_score'] = 0.;
                stats['valid_left_hand_with_prediction_score'] = 0.;
            if (stats['valid_left_hand_valid_with_prediction_size'] > 0):
                stats['valid_left_hand_valid_with_prediction_score'] = stats['valid_left_hand_valid_with_prediction_correct'] / float(stats['valid_left_hand_valid_with_prediction_size']);
            else:
                stats['valid_left_hand_valid_with_prediction_score'] = 0.;
            if (stats['left_hand_given_size'] > 0):
                stats['left_hand_given_score'] = stats['left_hand_given_correct'] / float(stats['left_hand_given_size']);
            else:
                stats['left_hand_given_score'] = 0.;
            
            
            stats['symbol_score'] = {};
            for symbol in dataset.oneHot.keys():
                if (stats['symbol_size'][symbol] > 0):
                    stats['symbol_score'][symbol] = stats['symbol_correct'][symbol] / float(stats['symbol_size'][symbol]);
                else:
                    stats['symbol_score'][symbol] = 0.;
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
            stats['syntactically_valid_score'] = 0.0;
            stats['semantically_valid_score'] = 0.0;
            stats['left_hand_valid_score'] = 0.0;
            stats['left_hand_valid_correct_score'] = 0.0;
            stats['right_hand_valid_score'] = 0.0;
        
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
        
        if ('subsPredictionSize' in stats):
            if (stats['subsPredictionSize'] > 0):
                stats['subsPredictionScore'] = stats['subsPredictionCorrect'] / float(stats['subsPredictionSize']);
                stats['subsPredictionCauseScore'] = stats['subsPredictionCauseCorrect'] / float(stats['subsPredictionSize']/2.);
                stats['subsPredictionEffectScore'] = stats['subsPredictionEffectCorrect'] / float(stats['subsPredictionSize']/2.);
            else:
                stats['subsPredictionScore'] = 0.;
                stats['subsPredictionCauseScore'] = 0.;
                stats['subsPredictionEffectScore'] = 0.;
        
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
        
        stats['correct_matrix_scores'] = np.zeros_like(stats['correct_matrix'], dtype='float32');
        # Changing division size for empty sizes to avoid division by zero error
        for i in range(stats['correct_matrix_sizes'].shape[0]):
            if (stats['correct_matrix_sizes'][i] == 0):
                stats['correct_matrix_scores'][i,:] = np.zeros(stats['correct_matrix_scores'].shape[1]);
            else:
#                 stats['correct_matrix_scores'][i,:] = stats['correct_matrix'][i,:] / float(stats['correct_matrix_sizes'][i]);
                stats['correct_matrix_scores'][i,:] = stats['correct_matrix'][i,:] / np.sum(stats['correct_matrix'][i,:]);
#         stats['correct_matrix_scores'] = (stats['correct_matrix'] / stats['correct_matrix_sizes'].reshape(1,stats['correct_matrix_sizes'].shape[0]).astype('float32')) * 100.;
        
        return stats;
    
    @staticmethod
    def addDicts(dict1, dict2):
        newDict = copy.deepcopy(dict1);
        for key in dict2:
            if (key not in newDict):
                newDict[key] = 0;
            newDict[key] += dict2[key];
        return newDict;
        