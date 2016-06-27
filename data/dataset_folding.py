'''
Created on 27 jun. 2016

@author: Robert-Jan
'''

import sys, os.path;

if __name__ == '__main__':
    sourceFolder = sys.argv[1];
    
    # Check if dataset is present
    mandatoryFiles = ['train.txt','test.txt'];
    if (not all(map(lambda f: os.path.isfile(os.path.join(sourceFolder,f)), mandatoryFiles))):
        raise ValueError("Not all mandatory files are present in this dataset!");
    
    f = open(os.path.join(sourceFolder,'train.txt'));
    train_expressions = f.readlines();
    f.close();
    
    f = open(os.path.join(sourceFolder,'test.txt'));
    test_expressions = f.readlines();
    test_size = len(test_expressions);
    f.close();
    
    all_expressions = train_expressions + test_expressions;
    total_size = len(all_expressions);
    test_fraction = total_size / float(test_size);
    
    fold_test_starting_indices = range(0,total_size,test_size);
    
    for i, n in enumerate(fold_test_starting_indices):
        fold_test_expressions = all_expressions[n:n+test_size];
        fold_train_expressions = all_expressions[:n] + all_expressions[n+test_size:];
        
        f_train = open(os.path.join(sourceFolder,'train_%d.txt' % i), 'w');
        f_train.writelines(fold_train_expressions);
        f_train.close();
         
        f_test = open(os.path.join(sourceFolder,'test_%d.txt' % i), 'w');
        f_test.writelines(fold_test_expressions);
        f_test.close();