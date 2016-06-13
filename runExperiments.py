'''
Created on 16 feb. 2016

@author: Robert-Jan
'''

import subprocess;
from subprocess import PIPE, STDOUT;
import os, sys;
import json, time;

if __name__ == '__main__':
    experiments_file = 'choose';
    if (len(sys.argv) > 1):
        experiments_file = sys.argv[1];
    if (experiments_file  == 'choose'):
        experiments_file = raw_input("Please provide the name of the experiment settings JSON file:\n");

    # Append json extension if not present
    if (experiments_file[-5:] != '.json'):
        experiments_file += '.json';

    f = open("./experiment_settings/" + experiments_file,'r');
    experiments = json.load(f);
    f.close();
    
    # Check if values can be stored
    experiment_outputPaths = [];
    for exp in experiments:
        name = exp['name'];
        outputPath = './raw_results/%s-%s.txt' % (name, time.strftime("%d-%m-%Y_%H-%M-%S"));
        # http://stackoverflow.com/questions/273192/in-python-check-if-a-directory-exists-and-create-it-if-necessary
        if (os.path.exists(outputPath)):
            name = name + '-';
        experiment_outputPaths.append(outputPath);
    
    # Run experiments    
    for i,exp in enumerate(experiments):
        outputPath = experiment_outputPaths[i];
        args = ['python',exp['script']];
        for key,value in exp.items():
            if (key not in ['script','name']):
                args.append("--" + key);
                args.append(value);
        print("Command string: %s" % (" ".join(args)));
        p = subprocess.Popen(" ".join(args),stdout=PIPE,stderr=STDOUT,shell=True);
        while (p.poll() == None):
            out = p.stdout.readline().strip();
            if (len(out) > 0):
                print(out);
                if (out != '' and out[0] != '#'):
                    # Write to file
                    f = open(outputPath,'a');
                    f.write(out.strip() + "\n");
                    f.close();
