'''
Created on 16 feb. 2016

@author: Robert-Jan
'''

import subprocess;
from subprocess import PIPE, STDOUT;
import os;
import json;

if __name__ == '__main__':
    f = open('experiments.json','r');
    experiments = json.load(f);
    f.close();
    
    # Check if values can be stored
    for name, exp in experiments.items():
        outputPath = 'raw_results/' + name + '.txt';
        # http://stackoverflow.com/questions/273192/in-python-check-if-a-directory-exists-and-create-it-if-necessary
        if (os.path.exists(outputPath)):
            raise ValueError("Experiment already exists!");
    
    # Run experiments    
    for name, exp in experiments.items():
        p = subprocess.Popen(['python',exp['script'],exp['dataset'],exp['repetitions'],exp['hidden_dim'],exp['learning_rate'],exp['lstm']],stdout=PIPE,stderr=STDOUT,shell=True);
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