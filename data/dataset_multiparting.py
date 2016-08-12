'''
Created on 7 mrt. 2016

@author: Robert-Jan
'''

import os;

if __name__ == '__main__':
    dataset = './expressions_positive_integer_answer_deep';
    
    # [(filter function, postfix)]
    filters = [(lambda expression: expression.count('(') == 0,'_depth_0'),
               (lambda expression: expression.count('(') <= 1,'_depth_1'),
               (lambda expression: expression.count('(') <= 2,'_depth_2'),
               (lambda expression: expression.count('(') <= 3,'_depth_3'),
               (lambda expression: expression.count('(') <= 4,'_depth_4'),
               (lambda expression: expression.count('(') <= 5,'_depth_5')]; 
    
    for file in ['train', 'test']:
        for func, postfix in filters:
            f_old = open(os.path.join(dataset, file + '.txt'), 'r');
            f_new = open(os.path.join(dataset, file + postfix + '.txt'), 'w');
        
            line = f_old.readline();
            while (line != ""):
                if (func(line)):
                    f_new.write(line);
                line = f_old.readline();
    
    f_old.close();
    f_new.close();