'''
Created on 4 mrt. 2016

@author: Robert-Jan
'''

import pickle;

def append_to_file(filepath, string):
    f = open(filepath, 'a');
    f.write(string);
    f.close();

def save_to_pickle(filepath, vars):
    f = open(filepath, 'w');
    pickle.dump(vars,f);
    f.close();