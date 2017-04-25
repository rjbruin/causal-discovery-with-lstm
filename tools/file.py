'''
Created on 4 mrt. 2016

@author: Robert-Jan
'''

import pickle;
import tools.arguments as arg;
import os;

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
    
    return dict(savedVars), settings[0];

def load_from_pickle_with_filename(filepath):
    f = open(filepath, 'rb');
    return load_from_pickle(f);

def load_parameters_with_filename(filepath):
    f = open(filepath, 'r');
    _ = f.readline();
    settings = arg.processCommandLineArguments(f.readline().strip().split(" "), None);
    f.close();
    return settings[0];

def save_for_continuing(name, repetition, save_modulo, modelVariables, otherVariables, settings={},
                        remove=True, saveModel=True):
    filename = 'saved_models/%s_%d' % (name, repetition);
    previous_filename = 'saved_models/%s_%d' % (name, repetition-1);
    
    # Save new model vars
    if (saveModel):
        save_to_pickle(filename + '.model', modelVariables, settings=settings);
        # Find earlier saved model vars and delete
        if (remove and repetition % save_modulo != 0 and os.path.isfile(previous_filename + '.model')):
            os.remove(previous_filename + '.model');
    
    # Save new other vars
    save_to_pickle(filename + '.other', otherVariables, settings=settings);
    # Find earlier saved other vars and delete
    if (remove and os.path.isfile(previous_filename + '.other')):
        os.remove(previous_filename + '.other');

def remove_for_continuing(name, repetition):
    previous_filename = 'saved_models/%s_%d' % (name, repetition-1);
    # Find earlier saved other vars and delete
    if (os.path.isfile(previous_filename + '.other')):
        os.remove(previous_filename + '.other');
