'''
Created on 7 mrt. 2016

@author: Robert-Jan
'''

import os;

if __name__ == '__main__':
    dataset = './expressions_positive_integer_answer_shallow';
    new_dataset = './expressions_positive_integer_answer_shallow_max100';
    
    filters = [lambda expression: int(expression.split("=")[1]) < 100]; 
    
    # http://stackoverflow.com/questions/273192/in-python-check-if-a-directory-exists-and-create-it-if-necessary
    if (not os.path.exists(new_dataset)):
        os.makedirs(new_dataset);
    
    for file in ['train.txt', 'test.txt']:
        f_old = open(os.path.join(dataset, file), 'r');
        f_new = open(os.path.join(new_dataset, file), 'w');
        
        line = f_old.readline();
        while (line != ""):
            if (all(map(lambda fx: fx(line), filters))):
                f_new.write(line);
            line = f_old.readline();
    
    f_old.close();
    f_new.close();