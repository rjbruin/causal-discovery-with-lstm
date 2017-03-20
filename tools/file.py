'''
Created on 4 mrt. 2016

@author: Robert-Jan
'''

import pickle;
import tools.arguments as arg;

def append_to_file(filepath, string):
    f = open(filepath, 'a');
    f.write(string);
    f.close();

def save_to_pickle(filepath, variables, settings={}):
    """
    Saves models to pickle with settings as header.
    Provide all settings as strings.
    """
    f = open(filepath, 'wb');
    
    f.write("### %d\n" % len(settings));
    f.write(" ".join(arg.parametersArguments(settings)) + "\n");
#     for key,value in settings.items():
#         f.write("# %s:%s\n" % (key,value));
    
    pickle.dump(variables,f);
    f.close();

def load_from_pickle(f):
    """
    Reads models from pickle with settings as header.
    Settings are pre-populated with default values from tools.arguments.
    """
    # Skip deprecated first line
    _ = f.readline();
    settings = arg.processCommandLineArguments(f.readline().strip().split(" "), None);
    
    try:
        savedVars = pickle.load(f);
    except IndexError:
        return False;
    
    f.close();
    
    return savedVars, settings[0];

def load_from_pickle_with_filename(filepath):
    f = open(filepath, 'rb');
    return load_from_pickle(f);
