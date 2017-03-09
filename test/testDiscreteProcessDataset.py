'''
Created on 9 mrt. 2017

@author: Robert-Jan
'''
import unittest


class Test(unittest.TestCase):


    def testDiscreteProcessDataset(self):
        # Weights ([1 to 1, 2 to 2, 1 to 2, 2 to 1])
        weights = [[0,-2,-1,0], [1,2,-1,2], [2,-1,0,0], [0,0,1,-2], [-1,-1,-2,-2], [-2,-2,1,-2], [2,-1,0,-2], [2,1,1,-1]];
        dataset = '../data/discreteprocess_complex_nonoise/all.txt';
        sequence_length = 20;
        
        lag = len(weights);
        
        f = open(dataset);
        total = 0;
        right = 0;
        for xi, line in enumerate(f):
            seq1, seq2 = line.strip().split(";");
            seq1 = map(int,seq1);
            seq2 = map(int,seq2);
            
            for i in range(lag,sequence_length):
                total += 2;
                sym1sums = map(lambda (j,v1,v2): weights[j][0]*v1 + weights[j][3]*v2, zip(range(lag),seq1[i-lag:i],seq2[i-lag:i]));
                sym1target = sum(sym1sums) % 10;
                sym2sums = map(lambda (j,v1,v2): weights[j][2]*v1 + weights[j][1]*v2, zip(range(lag),seq1[i-lag:i],seq2[i-lag:i]));
                sym2target = sum(sym2sums) % 10;
                
                if (sym1target == seq1[i]):
                    right += 1;
                if (sym2target == seq2[i]):
                    right += 1;
                
#                 self.assertEqual(sym1target,seq1[i],"(%d) Top sequence symbol %d invalid: \ntarget = %d, pred = %d, \nseq1 = %s, \nseq1sums = %s,\nseq2 = %s,\nseq2sums = %s" % 
#                                  (xi, i-lag+1, sym1target, seq1[i], str(seq1), str(sym1sums), str(seq2), str(sym2sums)));
#                 self.assertEqual(sym2target,seq2[i],"(%d) Bot sequence symbol %d invalid: \ntarget = %d, pred = %d, \nseq1 = %s, \nseq1sums = %s,\nseq2 = %s,\nseq2sums = %s" % 
#                                  (xi, i-lag+1, sym2target, seq2[i], str(seq1), str(sym1sums), str(seq2), str(sym2sums)));

        print("Total: %d, right: %d" % (total, right));

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testDiscreteProcessDataset']
    unittest.main()