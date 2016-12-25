'''
Created on 16 feb. 2016

@author: Robert-Jan
'''

import subprocess;
from subprocess import PIPE, STDOUT;
import os, sys;
import json, time;
import trackerreporter;

request_headers = {'user-agent': 'Chrome/51.0.2704.103'};

if __name__ == '__main__':
    # Settings
    api_key = os.environ.get('TCDL_API_KEY');
    if (api_key is None):
        raise ValueError("No API key present for reporting to tracker!");
    score_types = {'Precision': 'Score',
                   'Training loss': 'Total error',
                   'Testing loss': 'Total testing error',
                   'Digit precision': 'Digit-based score',
                   'Train Precision': 'TRAIN Score',
                   'Train Digit precision': 'TRAIN Digit-based score',
                   'Structure precision': 'Structure score',
                   'Structure pr. (c)': 'Structure score cause',
                   'Structure pr. (e)': 'Structure score effect',
                   'Effect precision': 'Effect score',
                   'Mistake (1) precision': 'Error margin 1 score',
                   'Mistake (2) precision': 'Error margin 2 score',
                   'Mistake (3) precision': 'Error margin 3 score',
                   'Validity': 'Valid',
                   'Validity (c)': 'Structure valid cause',
                   'Validity (e)': 'Structure valid effect',
                   'Local validity': 'Local valid',
                   'Local validity (c)': 'Local valid cause',
                   'Local validity (e)': 'Local valid effect',
                   'In dataset': 'In dataset',
                   'Skipped': 'Skipped because of zero prediction length',
                   'Unique predictions': 'Unique labels predicted'};
    trackerreporter.init('http://rjbruin.nl/experimenttracker/api/',api_key);
    
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
    experiment_args = [];
    iterative_args = {};
    for i, exp in enumerate(experiments):
        # Ask for name
        newName = raw_input("Experiment %d name (%s): " % (i+1,exp['name']));
        if (newName != ''):
            exp['name'] = newName;
        output_name = exp['name'];
        if (' ' in output_name):
            raise ValueError("Experiment name cannot contain whitespace! Offending name: \"%s\"" % output_name);
        outputPath = './raw_results/%s_%s.txt' % (exp['name'], time.strftime("%d-%m-%Y_%H-%M-%S"));
        # http://stackoverflow.com/questions/273192/in-python-check-if-a-directory-exists-and-create-it-if-necessary
        if (os.path.exists(outputPath)):
            exp['name'] = exp['name'] + '-';
        experiment_outputPaths.append(outputPath);
        
        # Ask for extra args
        extraArgs = raw_input("(optional) Add extra/overwriting parameters (e.g. '--key value'): ").split(" ");
        experiment_args.append(extraArgs);
        if ('--repetitions' in extraArgs):
            index = extraArgs.index('--repetitions');
            experiments[i]['repetitions'] = int(extraArgs[index+1]);
        
        iterativeArgs = raw_input("(optional) Add one iterative parameter where values are separated by commas (e.g. '--key value1,value2,value3'): ").split(" ");
        explodedArgs = [];
        key = iterativeArgs[0];
        for val in iterativeArgs[1].split(","):
            explodedArgs.append("%s %s" % (key, val));
        iterative_args[i] = explodedArgs;
    
    # Run experiments
    trackerStack = [];
    for i,exp in enumerate(experiments):
        # Check for iterative args
        if (i not in iterative_args):
            iterative_args[i] = [''];
        
        for j, it_args in iterative_args[i]:
            print("Beginning experiment %s\n" % exp['name']);
            
            extraArgs = experiment_args[i];
            report = True;
            if ('report_to_tracker' in exp):
                report = exp['report_to_tracker'] == 'True';
            if ('--report_to_tracker' in extraArgs):
                index = extraArgs.index('--report_to_tracker');
                report = extraArgs[index+1] == 'True';
            if (report):
                if ('multipart_dataset' in exp):
                    datasets = exp['multipart_dataset'];
                else:
                    datasets = 1;
                experimentId = trackerreporter.initExperiment(exp['name'], totalProgress=exp['repetitions'], 
                                                    totalDatasets=datasets, scoreTypes=score_types.keys(), 
                                                    scoreIdentifiers=score_types);
                if (experimentId is False):
                    print("WARNING! Experiment could not be posted to tracker!");
                
            outputPath = experiment_outputPaths[i];
            args = ['python',exp['script'],'--output_name',output_name];
            for key,value in exp.items():
                if (key not in ['script','name']):
                    args.append("--" + key);
                    args.append(str(value));
            joined_args = " ".join(args + extraArgs + it_args);
            if (gpu):
                joined_args = "THEANO_FLAGS='device=gpu,floatX=float32' " + joined_args;
            print("Command string: %s" % (joined_args));
            p = subprocess.Popen(joined_args,stdout=PIPE,stderr=STDOUT,shell=True);
            
            currentBatch = 1;
            currentIteration = 1;
            currentDataset = 1;
            while (p.poll() == None):
                out = p.stdout.readline().strip();
                if (len(out) > 0):
                    print(out);
                    if (out[:5] == 'Batch'):
                        currentBatch = int(out.split(" ")[1]);
                        currentIteration = int(out.split(" ")[3]);
                        currentDataset = int(out.split(" ")[7]);
                    # Compose data object
                    if (report):
                        trackerreporter.fromExperimentOutput(experimentId, out, 
                            atProgress=currentIteration, atDataset=currentDataset);
                            
                    if (out != '' and out[0] != '#'):
                        # Write to file
                        f = open(outputPath,'a');
                        f.write(out.strip() + "\n");
                        f.close();
                    