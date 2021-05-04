from __future__ import print_function
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import fileinput
from itertools import islice

def main():
    N = 740684
    batch = 196
    a = -1
    b = 1
    iterations = N / batch # 3779
    with open('rand_normalized.vectors', 'w') as newf:
        with open('rand.vectors', 'r') as f:  
            i = 0
            while True:
                lines_gen = islice(f, batch)
                if not lines_gen:
                    print("nope")
                    break
                for _,line in enumerate(lines_gen):
                    l = line.strip().split()
                    word = l[0]
                    vect = [float(x) for x in l[1:]]
                    vect = np.array(vect)
                    xmin = np.amin(vect)
                    xmax = np.amax(vect)
                    denom = (xmax - xmin) + 1e-5
                    vect = (b-a)*(vect - xmin) /denom  + a
                    vect_str = ' '.join([str(num) for num in vect])
                    newline = word + ' ' + vect_str
                    newf.write(newline+'\n') 
                print("#batch: ",i, "/",iterations)
                i += 1

if __name__ == '__main__':
    main()
