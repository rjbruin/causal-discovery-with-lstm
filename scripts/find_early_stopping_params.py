'''
Created on 1 mrt. 2017

@author: Robert-Jan
'''

import numpy as np;

def get_early_stopping(errors, epsilon, offset, values):
    avgs = [];
    for i in range(errors,len(values)):
        avgRange = values[i-errors:i];
        avg = np.mean(avgRange);
        
#         if (len(avgs) > 0):
#             print("avgs: %s\tavg: %.4f\tdiff: %.4f" % (str(avgs), avg, abs(avgs[0]-avg)));
#         else:
#             print("avgs: %s\tavg: %.4f" % (str(avgs), avg));
        
        if (len(avgs) >= offset and abs(avgs[0]-avg) <= epsilon):
            return i;
        avgs.append(avg);
        if (len(avgs) > offset):
            avgs = avgs[1:];
    return i;

if __name__ == '__main__':
    delimiter = "),(";
    values1 = map(lambda s: float(s.split(",")[1]),raw_input("Values for sequence 1 (separated by \"%s\"): " % delimiter).split(delimiter));
    values2 = map(lambda s: float(s.split(",")[1]),raw_input("Values for sequence 2 (separated by \"%s\"): " % delimiter).split(delimiter));
    
    inpt = raw_input("Errors & epsilon & offset (separated by ,): ");
    while (inpt.strip() != ""):
        errors, epsilon, offset = inpt.strip().split(",");
        errors = int(errors);
        epsilon = float(epsilon);
        offset = int(offset);
        
        iteration1 = get_early_stopping(errors, epsilon, offset, values1);
        iteration2 = get_early_stopping(errors, epsilon, offset, values2);
        
        print("Iterations %d and %d" % (iteration1, iteration2));
        
        inpt = raw_input("Errors & epsilon (separated by ,): ");
