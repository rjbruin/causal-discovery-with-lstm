'''
Created on 16 feb. 2016

@author: Robert-Jan
'''

import subprocess;
from subprocess import PIPE, STDOUT;
import os, sys;
import json, time;
import requests;

if __name__ == '__main__':
    # Settings
    report_to_tracker_criteria = [lambda s: s[0] != '#'];
    api_key = os.environ.get('TCDL_API_KEY');
    if (api_key is None):
        raise ValueError("No API key present for reporting to tracker!");
    
    experiments_file = 'choose';
    if (len(sys.argv) > 1):
        experiments_file = sys.argv[1];
    if (experiments_file  == 'choose'):
        experiments_file = raw_input("Please provide the name of the experiment settings JSON file:\n");

    if (len(sys.argv) > 2):
        gpu = sys.argv[2] == 'gpu';
    else:
        gpu = raw_input("Use GPU? (y/n)") == 'y';

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
        report = True;
        if ('report_to_tracker' in exp):
            report = exp['report_to_tracker'] == 'True';
        if (report):
            requests.post("http://rjbruin.nl/experimenttracker/api/postExperiment.php", {'exp': exp['name'], 'key': api_key});
            
        outputPath = experiment_outputPaths[i];
        args = ['python',exp['script']];
        for key,value in exp.items():
            if (key not in ['script','name']):
                args.append("--" + key);
                args.append(str(value));
        joined_args = " ".join(args);
        if (gpu):
            joined_args = "THEANO_FLAGS=device=gpu " + joined_args;
        print("Command string: %s" % (joined_args));
        p = subprocess.Popen(joined_args,stdout=PIPE,stderr=STDOUT,shell=True);
        
        currentBatch = 1;
        while (p.poll() == None):
            out = p.stdout.readline().strip();
            if (len(out) > 0):
                print(out);
                if (out[:5] == 'Batch'):
                    currentBatch = int(out.split(" ")[1]);
                if (report and all(map(lambda f: f(out), report_to_tracker_criteria))):
                    requests.post("http://rjbruin.nl/experimenttracker/api/post.php", {'exp': exp['name'], 'msg': out, 'atProgress': currentBatch, 'key': api_key});
                if (out != '' and out[0] != '#'):
                    # Write to file
                    f = open(outputPath,'a');
                    f.write(out.strip() + "\n");
                    f.close();
