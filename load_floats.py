'''
Created on 27 sep. 2016

@author: Robert-Jan
'''

import os, sys;
from tools.file import load_from_pickle_with_filename, save_to_pickle
from tools.model import constructModels;
import theano;

if __name__ == '__main__':
    theano.config.floatX = 'float32';
    
    name = sys.argv[1];
    
    filepath = "./saved_models/%s.floats" % name;
    if (os.path.isfile(filepath)):
        modelName = name;
        result = load_from_pickle_with_filename(filepath);
        if (result is not False):
            savedFloats, settings = result;
            dataset, rnn = constructModels(settings, 0, None);
            savedVars = zip(sorted(rnn.vars.keys()),savedFloats);
            modelSet = rnn.loadVars(dict(savedVars), floats=True); 
            if (modelSet):
                saveVars = rnn.getVars();
                save_to_pickle('saved_models/%s_from_floats.model' % (name), saveVars, settings=settings);
                print("Success!");
            else:
                print("Error (3)");
        else:
            print("Error (2)");
    else:
        print("Error (1)");