'''
Created on 23 feb. 2017

@author: Robert-Jan
'''

import numpy as np;

def combine_stats():
#     delimiter = raw_input("Give delimiter without brackets [,]: ");
#     if (delimiter.strip() == ''):
#         delimiter = "),(";
#     else:
#         delimiter = ")" + delimiter + "(";
    delimiter = "),(";
    
#     divide = raw_input("Divide? [1.]: ");
#     if (divide.strip() == ''):
#         divide = 1.;
#     else:
#         divide = float(divide);
    divide = 1.;
    
#     finaldivide = raw_input("Divide final iteration? [divide]: ");
#     if (finaldivide.strip() == ''):
#         finaldivide = divide;
#     else:
#         finaldivide = float(finaldivide);
    finaldivide = divide;
    
#     maxit = raw_input("Truncate to iteration: ");
#     if (maxit.strip() == ''):
#         maxit = 1000000;
#     else:
#         maxit = int(maxit);
    maxit = 1000000;
    
    line = raw_input("Give stats separated by delimiter, one set per line (ENTER for stop): ");
    scores = {};
    while (line.strip() != ""):
        lines = line.split("\n");
        for line in lines:
            vals = line.strip().split(delimiter);
            for part in vals:
                key, val = part.split(",");
                it = int(key);
                if (it <= maxit):
                    if (it not in scores):
                        scores[it] = [];
                    scores[it].append(float(val));
        line = raw_input("Give stats separated by delimiter (ENTER for stop): ");
    
    output = [];
    its = sorted(scores.keys());
    for i, it in enumerate(its):
        vals = np.array(scores[it]);
        if (i == len(its)-1):
            vals /= finaldivide;
        else:
            vals /= divide;
        mean = np.mean(vals);
        std = np.std(vals);
        output.append("%d,%.8f) +- (%.8f,%.8f" % (it, mean, std, std));
    
    return ") (".join(output);

if __name__ == '__main__':
    print(combine_stats());