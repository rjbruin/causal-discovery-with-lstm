# Results for expression answer prediction
Script name: `expression_one_digit_answer`
Prediction of the final symbol in an expression of mathematical symbols using one-hot vector encoding. 
Model: Recurrent Neural Network.

## Common settings
Dataset: `expression_one_digit_answer_large`
* Train size: 1,000,000
* Test size: 100,000
Data units: 16
Output units: 16

## 16-02-2016 - 1
### Settings
Hidden units: 32
Repetitions: 1
Learning rate: 0.1

### Results
Duration: 4117 seconds
Score: **33.39 percent**
Prediction histogram:   {0: 87664, 1: 790, 2: 0, 3: 1139, 4: 184, 5: 4184, 6: 3966, 7: 723, 8: 192, 9: 1158, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0}
Ground thuth histogram: {0: 27122, 1: 10085, 2: 8834, 3: 8361, 4: 8117, 5: 7192, 6: 8154, 7: 6972, 8: 7868, 9: 7295, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0}

## 16-02-2016 - 2
### Settings
Hidden units: 32
Repetitions: 1
Learning rate: _0.05_

### Results
Duration: 1422 seconds
Score: **66.11 percent**
Prediction histogram:   {0: 25074, 1: 6886, 2: 15151, 3: 4877, 4: 8581, 5: 3689, 6: 23041, 7: 2866, 8: 5498, 9: 4337, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0}
Ground thuth histogram: {0: 27122, 1: 10085, 2: 8834, 3: 8361, 4: 8117, 5: 7192, 6: 8154, 7: 6972, 8: 7868, 9: 7295, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0}

## 16-02-2016 - 3
### Settings
Hidden units: _64_
Repetitions: 1
Learning rate: 0.05

### Results
Duration: 2074 seconds
Score: **77.34 percent**
Prediction histogram:   {0: 25454, 1: 9731, 2: 11386, 3: 7688, 4: 8562, 5: 6868, 6: 10385, 7: 3911, 8: 8593, 9: 7422, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0}
Ground thuth histogram: {0: 27122, 1: 10085, 2: 8834, 3: 8361, 4: 8117, 5: 7192, 6: 8154, 7: 6972, 8: 7868, 9: 7295, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0}

## 16-02-2016 - 4
### Settings
Hidden units: 32
Repetitions: 1
Learning rate: _0.01_

### Results
Duration: 1404 seconds
Score: **79.30 percent**
Prediction histogram:   {0: 26572, 1: 9783, 2: 9651, 3: 8867, 4: 7588, 5: 7711, 6: 10642, 7: 4796, 8: 6282, 9: 8108, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0}
Ground thuth histogram: {0: 27122, 1: 10085, 2: 8834, 3: 8361, 4: 8117, 5: 7192, 6: 8154, 7: 6972, 8: 7868, 9: 7295, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0}