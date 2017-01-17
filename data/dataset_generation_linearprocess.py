'''
Created on 17 jan. 2017

@author: Robert-Jan
'''

import sys, os;
import numpy as np;

def generateSample(length, max_lag, inputrange, weightsrange, crosslinks=True):
    var1 = [];
    var2 = [];
    weights1 = [];
    weights2 = [];
    weights1to2 = [];
    weights2to1 = [];
    
    # Fill given values at beginning of sequence and determine stationary process
    for j in range(max_lag):
        var1.append(np.random.uniform(inputrange[0], inputrange[1]));
        var2.append(np.random.uniform(inputrange[0], inputrange[1]));
        weights1.append(np.random.uniform(weightsrange[0], weightsrange[1]));
        weights2.append(np.random.uniform(weightsrange[0], weightsrange[1]));
        if (crosslinks):
            weights1to2.append(np.random.uniform(weightsrange[0], weightsrange[1]));
            weights2to1.append(np.random.uniform(weightsrange[0], weightsrange[1]));
        else:
            weights1to2.append(0.);
            weights2to1.append(0.);
    
    for j in range(length - max_lag):
        var1.append(np.sum(map(lambda (v,w,v2,w2): v*w + v2*w2, zip(var1[-max_lag:],weights1,var2[-max_lag:],weights2to1))));
        var2.append(np.sum(map(lambda (v,w,v1,w1): v*w + v1*w1, zip(var2[-max_lag:],weights2,var1[-max_lag-1:-1],weights1to2))));
    
    # Convert to string
    weights = [];
    for j in range(len(weights1)):
        weights.append("%.4f,%.4f,%.4f,%.4f" % (weights1[j], weights2[j], weights1to2[j], weights2to1[j]));
    
    weightsStr = "";
    if (len(weights) > 0):
        weightsStr = ";".join(weights) + "|";
    
    sample = [];
    for j in range(len(var1)):
        sample.append("%.4f,%.4f" % (var1[j], var2[j]));
        
    return weightsStr + ";".join(sample);

if __name__ == '__main__':
    # Define dataset settings
    n = 1000000;
    length = 20;
    max_lag = 8;
    inputrange = [-1.,1.];
    weightsrange = [-1.,1.];
    progressPrintInterval = 10000;
    
    # DEBUG
    n = 1000;
    progressPrintInterval = 10;
    
    # Get folder name
    foldername = sys.argv[1];
    # Create folder if doesn't exist
    if (not os.path.exists(foldername)):
        os.makedirs(foldername);
    
    # Check for existing all.txt file - ask for overwrite
    stop = False;
    if (os.path.exists(os.path.join(foldername, 'all.txt'))):
        inp = raw_input("Dataset already present! Continue? ([y]/n) ");
        if inp != 'y':
            print("Terminated.");
            stop = True;
    
    if (not stop):
        f = open(os.path.join(foldername, 'all.txt'), 'w');
        
        # Generate linear processes and write to file
        for i in range(n):
            sample = generateSample(length, max_lag, inputrange, weightsrange);
            f.write(sample + "\n");
            if (i % progressPrintInterval == 0):
                print("%.2f%%..." % ((i/float(n))*100.))
        
        # Close file
        f.close();