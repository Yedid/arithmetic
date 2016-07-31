This repo contains code for our paper 'Visual Learning of Arithmetic Operations', Y. Hoshen and S. Peleg, AAAI'16, Phoenix, Feb 2016.

The paper can be found at: http://www.cs.huji.ac.il/~peleg/papers/AAAI16-Arithmetic.pdf

The code depends on keras, numpy and PIL. The visualization in test.py requires matplotlib.

The code can be run without a GPU (although it helps). The network takes about 30 minutes to train on my laptop's CPU. 

Thr current configuration is for visual addition of 7 digit numbers. But all the other decimal operations presented in the paper can be tested with minor modifcations.

Running the code  

Prepare the dataset  
$ python get_data.py  
Train the network  
$ python train.py  
Show the network in action  
$ python test.py  

Citation

If this code was helpful to you please consider citing our paper "Visual Learning of Arithmetic Operations, Y. Hoshen and S. Peleg, AAAI'16, Phoenix, Feb 2016"

Errata

Please report all bugs to ydidh@cs.huji.ac.il
