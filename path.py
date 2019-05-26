#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 20:23:57 2019

@author: pushap
"""
import numpy as np
from processing import preprocessing
path="/home/pushap/newdata/capturedImages/"
x,y=preprocessing(path)
np.save('x.npy',x)
np.save('y.npy',y)