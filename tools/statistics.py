'''
Created on 17 feb. 2016

@author: Robert-Jan
'''

import numpy as np;
import time;

def confusion_matrix(data):
    output = "";
    
    # Data must be square
    if (data.shape[0] != data.shape[1]):
        raise ValueError("Data must be square!");
    
    dim = data.shape[0];
    
    # Print headers
    output += "\t";
    for i in range(dim):
        output += "%d\t" % (i+1);
    output += "\n";
    # Print data
    for i in range(dim):
        digit_i = i+1;
        output += "%d\t" % digit_i;
        for j in range(dim):
            output += "%d\t" % int(data[i,j]);
        output += "\n";
    
    return output;

def accuracy_per_origin(data, keys):
    # Print headers
    output = "\t";
    for index in range(data.shape[0]):
        output += "%s\t" % keys[index];
    output += "\n";
    
    # Print corrects
    output += "CORRECT\t";
    for index in range(data.shape[0]):
        output += "%d\t" % data[index,0];
    output += "\n";
    
    # Print totals
    output += "TOTAL\t";
    for index in range(data.shape[0]):
        output += "%d\t" % data[index,1];
    output += "\n";
    
    return output;

def str_statistics(start, score, prediction_histogram=None, groundtruth_histogram=None, prediction_confusion_matrix=None, digit_score=None, prediction_size_histogram=None):
    output = "\n";

    # Print statistics
    duration = time.clock() - start;
    output += "Duration: %d seconds\n" % duration;
    output += "Score: %.2f percent\n" % (score*100);

    if (digit_score is not None):
        output += "Digit-based score: %.2f percent\n" % (digit_score*100);
    
    if (prediction_size_histogram is not None):
        output += "Prediction size histogram:   %s\n" % (str(prediction_size_histogram));
    
    if (prediction_histogram is not None):
        output += "Prediction histogram:   %s\n" % (str(prediction_histogram));
        
    if (groundtruth_histogram is not None):
        output += "Ground truth histogram: %s\n" % (str(groundtruth_histogram));
    
    if (prediction_confusion_matrix is not None):
        output += "Confusion matrix:\n";
        output += confusion_matrix(prediction_confusion_matrix);
    
    output += "\n";
    
    return output;

if __name__ == '__main__':
    dim = 10;
    scores = np.random.uniform(0,10,(dim,2));
    keys = range(10);
    accuracy_per_origin(scores, keys)