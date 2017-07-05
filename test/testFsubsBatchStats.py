'''
Created on 5 apr. 2017

@author: Robert-Jan
'''
import unittest

from tools.model import set_up_statistics;

class Test(unittest.TestCase):


    def testFsubsBatchStats(self):
        # Label, prediction, first_error, recovery_index, no_recovery_index, errors
        samples = [('131','121',1,1,None,1),
                   ('131','131',-1,None,None,0),
                   ('131','221',0,0,None,2),
                   ('131','223',0,None,0,3),
                   ('131','134',2,None,2,1)];
        
        for i, (label, pred, tFirstError, tRecovIndex, tNoRecovIndex, tErrors) in enumerate(samples):
            stats = set_up_statistics(10, 20, []);
            comparison = map(lambda (p,l): p == l, zip(pred, label));
            recovery_index = None;
            no_recovery_index = None;
                    
            i = 0;
            errors = 0;
            first_error = -1;
            correct_after_first_error = False;
            for i, v in enumerate(comparison):
                if (v):
                    stats['digit_2_correct'][i] += 1.0;
                    if (first_error != -1):
                        correct_after_first_error = True;
                else:
                    errors += 1;
                    if (first_error == -1):
                        first_error = i;
                    elif (correct_after_first_error):
                        correct_after_first_error = False;
                stats['digit_2_prediction_size'][i] += 1;
            
            if (first_error < 8):
                stats['first_error'][first_error] += 1.0;
            else:
                stats['first_error'][8] += 1.0;
            
            if (first_error != -1):
                if (correct_after_first_error and first_error < len(comparison)-1):
                    stats['recovery'][first_error] += 1.0;
                    recovery_index = first_error;
                else:
                    stats['no_recovery'][first_error] += 1.0;
                    no_recovery_index = first_error;
            
            if (errors > 8):
                errors = 8;
            stats['error_size'][errors] += 1.0;
            stats['prediction_size'] += 1.0;
            
            self.assertEqual(tFirstError, first_error, "(%d) first_error: is %d, should be %d" % (i, first_error, tFirstError));
            self.assertEqual(tRecovIndex, recovery_index, "(%d) recovery_index: is %s, should be %s" % (i, str(recovery_index), str(tRecovIndex)));
            self.assertEqual(tNoRecovIndex, no_recovery_index, "(%d) no_recovery_index: is %s, should be %s" % (i, str(no_recovery_index), str(tNoRecovIndex)));
            self.assertEqual(tErrors, errors, "(%d) errors: is %d, should be %d" % (i, errors, tErrors));


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testFsubsBatchStats']
    unittest.main()