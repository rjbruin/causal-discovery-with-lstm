'''
Created on 23 feb. 2016

Run this script in debug mode to inspect the variables loaded.

@author: Robert-Jan
'''

import pickle, sys;

if __name__ == '__main__':
    modelName = '23-02-2016_11-27-42.model';
    if (len(sys.argv) > 1):
        modelName = sys.argv[1];
        if (modelName == 'choose'):
            modelName = raw_input("Please provide the name of the model you want to inspect:\n");
    
    # Keep trying to get a right filename
    while (True):
        try:
            f = open('./saved_models/' + modelName,'r')
            break;
        except IOError:
            modelName = raw_input("This model does not exist! Please provide the name of the model you want to inspect:\n");
    
    savedVars = pickle.load(f);
    f.close();