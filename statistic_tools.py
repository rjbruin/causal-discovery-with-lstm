'''
Created on 17 feb. 2016

@author: Robert-Jan
'''

import numpy as np;

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

if __name__ == '__main__':
    dim = 10;
    scores = np.random.uniform(0,10,(dim,2));
    keys = range(10);
    accuracy_per_origin(scores, keys)