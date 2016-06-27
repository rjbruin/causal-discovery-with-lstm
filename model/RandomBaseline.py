'''
Created on 21 jun. 2016

@author: Robert-Jan
'''

import numpy as np;

class RandomBaseline(object):
    '''
    classdocs
    '''


    def __init__(self, single_digit, seed):
        '''
        Nothing much to do here.
        '''
        self.single_digit = single_digit;
        np.random.seed(seed);
        # Create dummy vars for compatibility with saving models
        self.vars = {};
    
    def train(self, data, labels, learning_rate):
        """
        Joke's on you! I'm not doing anything!
        """
        pass
    
    def test(self, test_data, test_labels, test_targets, test_expressions, dataset, stats):
        # Set printing interval
        total = len(test_data);
        printing_interval = 1000;
        if (total <= printing_interval * 10):
            # Make printing interval always at least one
            printing_interval = max(total / 10,1);
        
        for j in range(len(test_data)):
            data = test_data[j];
            
            if (self.single_digit):
                prediction = self.single_digit_predict();
            else:
                prediction = self.multi_digit_predict();
                right_hand_size = len(prediction);
            
            # Statistics
            if (self.single_digit):
                if (prediction == np.argmax(test_labels[j])):
                    stats['correct'] += 1;
            else:
                # Get the labels
                argmax_target = np.argmax(test_targets[j],axis=1);
                # Compute the length of the target answer
                target_length = np.argmax(argmax_target);
                if (target_length == 0):
                    # If no EOS is found, the target is the entire length
                    target_length = test_targets[j].shape[1];
                # Compute the length of the prediction answer
                prediction_length = len(prediction);
                if (prediction_length == target_length and np.array_equal(prediction[:target_length],argmax_target[:target_length])):
                    # Correct if prediction and target length match and 
                    # prediction and target up to target length are the same
                    stats['correct'] += 1.0;
                for k,digit in enumerate(prediction[:len(argmax_target)]):
                    if (digit == np.argmax(test_targets[j][k])):
                        stats['digit_correct'] += 1.0;
                    stats['digit_prediction_size'] += 1;
                    
            if (self.single_digit):
                stats['prediction_histogram'][int(prediction)] += 1;
                stats['groundtruth_histogram'][np.argmax(test_labels[j])] += 1;
                stats['prediction_confusion_matrix']\
                    [np.argmax(test_labels[j]),int(prediction)] += 1;
            else:
                # Taking argmax over symbols for each sentence returns 
                # the location of the highest index, which is the first 
                # EOS symbol
                eos_location = len(prediction);
                stats['prediction_size_histogram'][int(eos_location)] += 1;
                for digit_prediction in prediction:
                    stats['prediction_histogram'][int(digit_prediction)] += 1;
            stats['prediction_size'] += 1;
            
            if (stats['prediction_size'] % printing_interval == 0):
                print("# %d / %d" % (stats['prediction_size'], total));
        
        stats['score'] = stats['correct'] / float(stats['prediction_size']);
        if (not self.single_digit):
            stats['digit_score'] = stats['digit_correct'] / float(stats['digit_prediction_size']);
        
        return stats;
    
    def single_digit_predict(self):
        """
        Return one random integer.
        """
        return np.random.randint(0,10);
    
    def multi_digit_predict(self):
        """
        Return a one to four random integers.
        """
        return np.random.randint(0,10,(np.random.randint(0,5)))
    