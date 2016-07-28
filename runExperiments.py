'''
Created on 16 feb. 2016

@author: Robert-Jan
'''

import subprocess;
from subprocess import PIPE, STDOUT;
import os, sys;
import json, time;
import requests;

request_headers = {'user-agent': 'Chrome/51.0.2704.103'};

def progressStackToTracker(stack):
    newStack = [];
    for data in stack:
        if ('url' in data):
            try:
                r = requests.post(data['url'], data, headers=request_headers);
                if (r.status_code != 200):
                    print("Posting to tracker failed with status code! Added data to stack.");
                    newStack.append(data);
            except Exception:
                print("Posting to tracker failed! Added data to stack.");
                newStack.append(data);
    return newStack;

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
        # Ask for name
        newName = raw_input("Experiment 1 name (%s): " % exp['name']);
        if (newName != ''):
            exp['name'] = newName;
        outputPath = './raw_results/%s-%s.txt' % (exp['name'], time.strftime("%d-%m-%Y_%H-%M-%S"));
        # http://stackoverflow.com/questions/273192/in-python-check-if-a-directory-exists-and-create-it-if-necessary
        if (os.path.exists(outputPath)):
            exp['name'] = exp['name'] + '-';
        experiment_outputPaths.append(outputPath);
    
    # Run experiments
    trackerStack = [];
    for i,exp in enumerate(experiments):
        report = True;
        if ('report_to_tracker' in exp):
            report = exp['report_to_tracker'] == 'True';
        if (report):
            data = {'url': "http://rjbruin.nl/experimenttracker/api/postExperiment.php", 
                    'exp': exp['name'], 'key': api_key, 'totalProgress': exp['repetitions']}
            try:
                r = requests.post(data['url'], data, headers=request_headers);
                if (r.json() != "false"):
                    experimentId = r.json()['id'];
                else:
                    print("WARNING! Experiment could not be posted to tracker!");
                    experimentId = -1;
            except Exception as e:
                print(e);
                raise ValueError("Posting experiment to tracker failed!");
            
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
        currentIteration = 1;
        while (p.poll() == None):
            out = p.stdout.readline().strip();
            if (len(out) > 0):
                print(out);
                if (out[:5] == 'Batch'):
                    currentBatch = int(out.split(" ")[1]);
                    currentIteration = int(out.split(" ")[5][:-1]);
                if (report and all(map(lambda f: f(out), report_to_tracker_criteria))):
                    # Compose data object
                    data = {'url': "http://rjbruin.nl/experimenttracker/api/post.php", 
                            'exp': experimentId, 'msg': out, 'atProgress': currentIteration, 'key': api_key};
                    # Add data to stack of data to send
                    trackerStack.append(data);
                    # Retrieve new stack containing all failed requests
                    trackerStack = progressStackToTracker(trackerStack);
                        
                if (out != '' and out[0] != '#'):
                    # Write to file
                    f = open(outputPath,'a');
                    f.write(out.strip() + "\n");
                    f.close();