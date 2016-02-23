'''
Created on 16 feb. 2016

@author: Robert-Jan
'''

import subprocess;
from subprocess import PIPE, STDOUT;
import os, sys;
import json;

if __name__ == '__main__':
    experiments_file = 'experiments.json';
    if (len(sys.argv) > 1):
        if (sys.argv[1] == 'choose'):
            experiments_file = raw_input("Please provide the name of the experiment settings JSON file:");
        else:
            experiments_file = sys.argv[1];
    
    f = open("./experiment_settings/" + experiments_file,'r');
    experiments = json.load(f);
    f.close();
    
    # Check if values can be stored
    for exp in experiments:
        name = exp['name'];
        outputPath = 'raw_results/' + name + '.txt';
        # http://stackoverflow.com/questions/273192/in-python-check-if-a-directory-exists-and-create-it-if-necessary
        if (os.path.exists(outputPath)):
            raise ValueError("Experiment already exists!");
    
    # Run experiments    
    for exp in experiments:
        name = exp['name'];
        args = ['python',exp['script']];
        for key,value in exp.items():
            if (key not in ['script','name']):
                args.append("--" + key);
                args.append(value);
        p = subprocess.Popen(args,stdout=PIPE,stderr=STDOUT,shell=True);
        outputs = [];
        while (p.poll() == None):
            out = p.stdout.readline();
            print(out.strip());
            if (out != '' and out[0] != '#'):
                outputs.append(out.strip());
        
        # Write to file
        outputPath = 'raw_results/' + name + '.txt';
        f = open(outputPath,'w');
        f.write("\n".join(outputs));
        f.close();