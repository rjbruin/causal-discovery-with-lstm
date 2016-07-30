'''
Created on 21 jun. 2016

@author: Robert-Jan
'''

import numpy as np;

from model.RecurrentModel import RecurrentModel

class RandomBaseline(RecurrentModel):
    '''
    classdocs
    '''


    def __init__(self, single_digit, seed, dataset, n_max_digits=24, minibatch_size=10):
        '''
        Nothing much to do here.
        '''
        self.single_digit = single_digit;
        np.random.seed(seed);
        self.minibatch_size = minibatch_size;
        self.n_max_digits = n_max_digits;
        self.dataset = dataset;
        
        # Variables required by interface
        self.fake_minibatch = False;
        self.time_training_batch = False;
        self.vars = {};
        self.verboseOutputter = None;
    
    def sanityChecks(self, training_data, training_labels):
        pass
    
    def sgd(self, data, labels, learning_rate):
        pass
    
    def predict(self, data, labels, targets):
        if (self.single_digit):
            prediction = self.single_digit_predict();
        else:
            prediction = self.multi_digit_predict();
        
        return prediction, {};
    
    def single_digit_predict(self):
        """
        Return one random integer.
        """
        return np.random.randint(0,10,(self.minibatch_size));
    
    def multi_digit_predict(self):
        """
        Return a one to four random integers.
        """
        prediction = np.zeros((self.minibatch_size,self.n_max_digits));
        lengths = np.random.randint(0,self.n_max_digits,(self.minibatch_size));
        for i in range(prediction.shape[0]):
            prediction[i,:lengths[i]] = np.random.randint(0,self.dataset.EOS_symbol_index+1,(lengths[i]));
            prediction[i,lengths[i]] = self.dataset.EOS_symbol_index;
        return prediction;
    
    def batch_statistics(self, stats, prediction, labels, targets, expressions, 
                         other, test_n, dataset,
                         excludeStats=None, no_print_progress=False,
                         eos_symbol_index=None, print_sample=False):
        # Statistics
        for j in range(0,test_n):
            if (self.single_digit):
                if (prediction[j] == np.argmax(labels[j])):
                    stats['correct'] += 1;
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
                    
            if (self.single_digit):
                stats['prediction_histogram'][int(prediction[j])] += 1;
                stats['groundtruth_histogram'][np.argmax(labels[j])] += 1;
                stats['prediction_confusion_matrix']\
                    [np.argmax(labels[j]),int(prediction[j])] += 1;
                if ('operator_scores' not in excludeStats):
                    stats['operator_scores'] = \
                        self.operator_scores(expressions[j], 
                                             int(prediction[j])==np.argmax(labels[j]),
                                             dataset.operators,
                                             dataset.key_indices,
                                             stats['operator_scores']);
            else:
                # Taking argmax over symbols for each sentence returns 
                # the location of the highest index, which is the first 
                # EOS symbol
                eos_location = np.argmax(prediction[j]);
                # Check for edge case where no EOS was found and zero was returned
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
        if (not self.single_digit):
            stats['digit_score'] = stats['digit_correct'] / float(stats['digit_prediction_size']);
        
        return stats;
    