'''
Created on 17 jan. 2017

@author: Robert-Jan
'''

import sys, os;
import numpy as np;

def generateWeights(length, max_lag, weightsrange, crosslinks=True, homogeneous_weights=False):
    weights1 = [];
    weights2 = [];
    weights1to2 = [];
    weights2to1 = [];
    
    # Determine stationary process
    for i in range(max_lag):
        if (homogeneous_weights and i != 0):
            weights1.append(weights1[-1]);
            weights2.append(weights2[-1]);
            if (crosslinks):
                weights1to2.append(weights1to2[-1]);
                weights2to1.append(weights2to1[-1]);
            else:
                weights1to2.append(0);
                weights2to1.append(0);
        else:
            weights1.append(np.random.randint(weightsrange[0], weightsrange[1]));
            weights2.append(np.random.randint(weightsrange[0], weightsrange[1]));
            if (crosslinks):
                weights1to2.append(np.random.randint(weightsrange[0], weightsrange[1]));
                weights2to1.append(np.random.randint(weightsrange[0], weightsrange[1]));
            else:
                weights1to2.append(0);
                weights2to1.append(0);
    
    return weights1, weights2, weights1to2, weights2to1;

def generateSample(length, max_lag, inputrange, weights1, weights2, weights1to2, weights2to1, sampleStorage, noise_prob):
    var1 = [];
    var2 = [];
    
    # Fill given values at beginning of sequence and determine stationary process
    for j in range(max_lag):
        var1.append(np.random.randint(inputrange[0], inputrange[1]));
        var2.append(np.random.randint(inputrange[0], inputrange[1]));
    
    for j in range(length - max_lag):
        var1.append(np.sum(map(lambda (v,w,v2,w2): (v*w + v2*w2), zip(var1[-max_lag:],weights1,var2[-max_lag:],weights2to1))) % (inputrange[1]+1));
        var2.append(np.sum(map(lambda (v,w,v1,w1): (v*w + v1*w1), zip(var2[-max_lag:],weights2,var1[-max_lag-1:-1],weights1to2))) % (inputrange[1]+1));
    
    sample1 = [];
    sample2 = [];
    for j in range(len(var1)):
        var1val = var1[j];
        var2val = var2[j];
        if (np.random.random() < noise_prob):
            var1val = np.random.randint(inputrange[0], inputrange[1]);
            var2val = np.random.randint(inputrange[0], inputrange[1]);
        
        sample1.append("%d" % (var1val));
        sample2.append("%d" % (var2val));
        
    # Check sampleStorage, recurse into new sample generation if sample exists already
    seedString = "".join(sample1[:max_lag]) + "".join(sample2[:max_lag]);
    if (seedString not in sampleStorage):
        sampleStorage[seedString] = True;
        sample = "".join(sample1) + ";" + "".join(sample2)
    else:
        sample, sampleStorage = generateSample(length, max_lag, inputrange, weights1, weights2, weights1to2, weights2to1, sampleStorage, noise_prob);
        
    return sample, sampleStorage;

if __name__ == '__main__':
    # Define dataset settings
    n = 1000000;
    length = 20;
    max_lag = 5;
    inputrange = [0,9];
    weightsrange = [-1,2];
    progressPrintInterval = 10000;
    crosslinks = True;
    homogeneous_weights = False;
    
    # Noise settings
    noise_prob = 0;
    
    # DEBUG
#     n = 1000;
#     progressPrintInterval = 10;
    
    if (weightsrange[1] - weightsrange[0] == 1 and np.power(len(range(inputrange[0], inputrange[1])),max_lag) < n):
        # Weights are fixed so unique samples amount is limited
        raise ValueError("ERROR! Input range and max lag define number of unique samples smaller than n: %d" % np.power(len(range(inputrange[0], inputrange[1]))));
    
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
        
        weights1, weights2, weights1to2, weights2to1 = generateWeights(length, max_lag, weightsrange, crosslinks=crosslinks, homogeneous_weights=homogeneous_weights);
        
        # Convert weights to string
        weights = [];
        for j in range(len(weights1)):
            weights.append("%d,%d,%d,%d" % (weights1[j], weights2[j], weights1to2[j], weights2to1[j]));
        
#         weightsStr = "";
#         if (len(weights) > 0):
#             weightsStr = ";".join(weights) + "|";
        
        print(weights);
        
        f_weights = open(os.path.join(foldername, 'weights.txt'), 'w');
        f_weights.write(str(weights));
        f_weights.close();
        
        sampleStorage = {};
        # Generate linear processes and write to file
        for i in range(n):
            sample, sampleStorage = generateSample(length, max_lag, inputrange, weights1, weights2, weights1to2, weights2to1, sampleStorage, noise_prob);
            f.write(sample + "\n");
            if (i % progressPrintInterval == 0):
                print("%.2f%%..." % ((i/float(n))*100.))
        
        # Close file
        f.close();
        
        print(weights);