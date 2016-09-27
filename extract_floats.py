'''
Created on 27 sep. 2016

@author: Robert-Jan
'''

import os, sys;
import pickle;
from tools.file import load_from_pickle_with_filename
from tools.model import constructModels;
import theano;

if __name__ == '__main__':
    theano.config.floatX = 'float32';
    
    name = sys.argv[1];
    
    filepath = "./saved_models/%s.model" % name;
    if (os.path.isfile(filepath)):
        modelName = name;
        result = load_from_pickle_with_filename(filepath);
        if (result is not False):
            savedVars, settings = result;
            datasets, rnn = constructModels(settings, 0, None);
            dataset = datasets[-1];
            modelSet = rnn.loadVars(dict(savedVars)); 
            if (modelSet):
                modelInfo = settings;
                floats = []
                for key in sorted(rnn.vars.keys()):
                    floats.append(rnn.vars[key].get_value().astype('float32'));
            
                f_model = open(filepath);
                _ = f_model.readline();
                settingsLine = f_model.readline();
                f_model.close();
                
                f = open('./saved_models/%s.floats' % name, 'w');
                f.writelines(['###\n',settingsLine]);
                pickle.dump(floats,f);
                f.close();
                print("Success!");
            else:
                print("Error (3)");
        else:
            print("Error (2)");
    else:
        print("Error (1)");