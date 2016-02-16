# Results for expression answer prediction
Script name: `expression_one_digit_answer`
Prediction of the final symbol in an expression of mathematical symbols using one-hot vector encoding. 

## 16-02-2016
### Settings
Dataset: `expression_one_digit_answer_large`
* Train size: 1,000,000
* Test size: 100,000
Data units: 16
Hidden units: 32
Output units: 16

### Results
Duration: 4117 seconds
Score: 33.39 percent
Prediction histogram:   {0: 87664, 1: 790, 2: 0, 3: 1139, 4: 184, 5: 4184, 6: 3966, 7: 723, 8: 192, 9: 1158, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0}
Ground thuth histogram: {0: 27122, 1: 10085, 2: 8834, 3: 8361, 4: 8117, 5: 7192, 6: 8154, 7: 6972, 8: 7868, 9: 7295, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0}