'''
Created on 17 feb. 2016

@author: Robert-Jan
'''

import numpy as np;

def confusion_matrix(data):
    # Data must be square
    if (data.shape[0] != data.shape[1]):
        raise ValueError("Data must be square!");
    
    dim = data.shape[0];
    
    # Print headers
    output = "\t";
    for i in range(dim):
        output += "%d\t" % (i+1);
    print(output);
    # Print data
    for i in range(dim):
        digit_i = i+1;
        output = "%d\t" % digit_i;
        for j in range(dim):
            output += "%d\t" % int(data[i,j]);
        print(output);

if __name__ == '__main__':
    dim = 10;
    prediction_confusion_matrix = np.random.uniform(0,10,(dim,dim));
    confusion_matrix(prediction_confusion_matrix)