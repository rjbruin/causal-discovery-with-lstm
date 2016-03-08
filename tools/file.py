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

def save_to_pickle(filepath, vars, settings={}):
    """
    Saves models to pickle with settings as header.
    Provide all settings as strings.
    """
    f = open(filepath, 'w');
    
    f.write("### %d" % len(settings));
    for key,value in settings.items():
        f.write("# %s:%s\n" % (key,value));
    
    pickle.dump(vars,f);
    f.close();

def load_from_pickle(f):
    """
    Reads models from pickle with settings as header.
    Settings are pre-populated with default values from tools.arguments.
    """
    firstLine = f.readline();
    if (firstLine[:3] != "###"):
        raise ValueError("Model file header is missing!");
    nrSettings = int(firstLine.split(" ")[1]);
    
    settings = arg.defaults;
    for _ in range(nrSettings):
        line = f.readline()[1:].strip();
        key,value = map(lambda x: x.strip(), line.split(":"));
        settings[key] = arg.processKeyValue(key, value);
    
    savedVars = pickle.load(f);
    
    f.close();
    
    return savedVars, settings;

def load_from_pickle_with_filename(filepath):
    f = open(filepath, 'r');
    return load_from_pickle(f);
