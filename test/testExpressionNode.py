'''
Created on 15 aug. 2016

@author: Robert-Jan
'''
import unittest

from models.GeneratedExpressionDataset import ExpressionNode;


class Test(unittest.TestCase):


    def testStringToValue(self):
        expressions = [('1+1',2),
                       ('(1+3)/2',2),
                       ('(8/(2+(3-1)))*4',8),
                       ('((7/7)*3)/1',3)];
        
        for i, (expression,answer) in enumerate(expressions):
            self.assertEqual(ExpressionNode.fromStr(expression).getValue(),answer,"(strToVal) Answer #%d not correct!" % (i+1));
    
    def testSolve(self):
        expressions = [
                        ('1+1',3,4),
                        ('1+(1+1)',4,15),
                        ('1',11,0),
                        ('0-0',1,9),
                        ('1+(1-1)',2,3),
                       ('1*1',2,2),
                       ('1*1',0,19),
                       ('1/1',0,9),
                       ('1/1',3,3)];
        
        for i, (expression,answer,nrAnswers) in enumerate(expressions):
            answers = ExpressionNode.fromStr(expression).solveAll(answer);
            print(map(str, answers));
            self.assertEqual(len(answers),nrAnswers,"(solve) Answer #%d not correct: length %d should be %d!" % (i+1,len(answers),nrAnswers));
            if (nrAnswers > 0):
                self.assertEqual(all(map(lambda n: n.getValue() == answer, answers)),True,"(solve) Answer #%d not correct!" % (i+1));


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testExpressionNode']
    unittest.main()