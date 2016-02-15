# Results of neural network on MINST 60,000
MINIBATCH SIZE 	- LEARNING RATE 	- INIT 			- ERROR					- HIDDEN DIM	- REPETITIONS

5  				- 0.1 				- SQUARE ROOT 	- LOG LIKELIHOOD		- 10			- 1
21.27 percent

10 				- 0.1 				- SQUARE ROOT 	- LOG LIKELIHOOD		- 10			- 1
37.10 percent / 1015 seconds / W:-2728.76125742 / V:1.17961196366e-13 
37.49 percent / 1080 seconds / W:-2083.40081331 / V:2.3941959526e-13
41.17 percent / 47   seconds / W:74.0418733563  / V:3.01980662698e-13

10 				- 0.1 				- SQUARE ROOT 	- CROSS ENTROPY SUMMED	- 10			- 1
14.30 percent / 44 seconds   / W:-43400.9325321 / V:-1.76346715008e-11

10 				- 0.1 				- SQUARE ROOT 	- CROSS ENTROPY MEAN	- 10			- 1
65.04 percent / 30 seconds   / W:-1600.45990321 / V:2.67896815842e-13
48.17 percent / 60 seconds   / W:-1758.18560455 / V:3.02813329967e-13

10 				- 0.01 				- SQUARE ROOT 	- CROSS ENTOPY MEAN		- 10			- 1
92.54 percent / 34 seconds   / W:-73.3439852134 / V:1.61010094146e-13

20 				- 0.01 				- SQUARE ROOT 	- CROSS ENTOPY MEAN		- 10			- 1
93.18 percent / 29 seconds   / W:-21.8747913119 / V:4.42701431069e-15

20 				- 0.01 				- SQUARE ROOT 	- CROSS ENTOPY MEAN		- 50			- 1
94.07 percent / 97 seconds   / W:-290.766139246 / V:-5.41199030035e-14

20 				- 0.01 				- SQUARE ROOT 	- CROSS ENTOPY MEAN		- 100			- 1
94.09 percent / 247 seconds  / W:-924.584483495 / V:6.03336824945e-14

20 				- 0.01 				- SQUARE ROOT 	- CROSS ENTOPY MEAN		- 10			- 2
94.51 percent / 39 seconds   / W:-65.7327088224 / V:6.04793992665e-14
