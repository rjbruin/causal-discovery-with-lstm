'''
Created on 16 feb. 2016

@author: Robert-Jan
'''

import subprocess;
from subprocess import PIPE, STDOUT;
import os;

if __name__ == '__main__':
    experiments = {'one_digit_answer_2_256_001_1': {
                        'script': 'expressions_one_digit_answer.py',
                        'dataset': './data/expressions_one_digit_answer_large',
                        'repetitions': '2',
                        'hidden_dim': '256',
                        'learning_rate': '0.01' },
                   'one_digit_answer_2_256_001_2': {
                        'script': 'expressions_one_digit_answer.py',
                        'dataset': './data/expressions_one_digit_answer_large',
                        'repetitions': '2',
                        'hidden_dim': '256',
                        'learning_rate': '0.01' },
                   'one_digit_answer_2_256_001_3': {
                        'script': 'expressions_one_digit_answer.py',
                        'dataset': './data/expressions_one_digit_answer_large',
                        'repetitions': '2',
                        'hidden_dim': '256',
                        'learning_rate': '0.01' },
                   'one_digit_answer_2_256_001_4': {
                        'script': 'expressions_one_digit_answer.py',
                        'dataset': './data/expressions_one_digit_answer_large',
                        'repetitions': '2',
                        'hidden_dim': '256',
                        'learning_rate': '0.01' },
                   'one_digit_answer_2_256_001_5': {
                        'script': 'expressions_one_digit_answer.py',
                        'dataset': './data/expressions_one_digit_answer_large',
                        'repetitions': '2',
                        'hidden_dim': '256',
                        'learning_rate': '0.01' },
                   'one_digit_answer_2_256_001_6': {
                        'script': 'expressions_one_digit_answer.py',
                        'dataset': './data/expressions_one_digit_answer_large',
                        'repetitions': '2',
                        'hidden_dim': '256',
                        'learning_rate': '0.01' },
                   'one_digit_answer_2_256_001_7': {
                        'script': 'expressions_one_digit_answer.py',
                        'dataset': './data/expressions_one_digit_answer_large',
                        'repetitions': '2',
                        'hidden_dim': '256',
                        'learning_rate': '0.01' },
                   'one_digit_answer_2_256_001_8': {
                        'script': 'expressions_one_digit_answer.py',
                        'dataset': './data/expressions_one_digit_answer_large',
                        'repetitions': '2',
                        'hidden_dim': '256',
                        'learning_rate': '0.01' },
                   'one_digit_answer_2_256_001_9': {
                        'script': 'expressions_one_digit_answer.py',
                        'dataset': './data/expressions_one_digit_answer_large',
                        'repetitions': '2',
                        'hidden_dim': '256',
                        'learning_rate': '0.01' },
                   'one_digit_answer_2_256_001_10': {
                        'script': 'expressions_one_digit_answer.py',
                        'dataset': './data/expressions_one_digit_answer_large',
                        'repetitions': '2',
                        'hidden_dim': '256',
                        'learning_rate': '0.01' },
                   };
    
    # Check if values can be stored
    for name, exp in experiments.items():
        outputPath = 'raw_results/' + name + '.txt';
        # http://stackoverflow.com/questions/273192/in-python-check-if-a-directory-exists-and-create-it-if-necessary
        if (os.path.exists(outputPath)):
            raise ValueError("Experiment already exists!");
    
    # Run experiments    
    for name, exp in experiments.items():
        p = subprocess.Popen(['python',exp['script'],exp['dataset'],exp['repetitions'],exp['hidden_dim'],exp['learning_rate']],stdout=PIPE,stderr=STDOUT,shell=True);
        outputs = [];
        while (p.poll() == None):
            out = p.stdout.readline();
            print(out.strip());
            if (out != ''):
                outputs.append(out.strip());
        
        # Write to file
        outputPath = 'raw_results/' + name + '.txt';
        f = open(outputPath,'w');
        f.write("\n".join(outputs));
        f.close();