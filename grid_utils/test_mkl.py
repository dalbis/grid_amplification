#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 11:36:05 2019

@author: dalbis
"""

import numpy as np
import ctypes
from multiprocessing import Pool
import os
import psutil
import socket

def dummy_job(sim_idx,iterations=1000):
  print 'Running simulation %d'%sim_idx
  
  for i in range(iterations):
    a=np.random.randn(400,400)
    a*=a
  
  
def get_optimal_thread_num(num_sims,num_cores):
  """
  Returns the total number of thread per process to maximize load.
  Parameters:
    - num_sim : number of simulations to be run
    - num_cores: number of cores available on the machine
  """
  num_threads=1
  while num_threads<=num_cores:
    if num_sims*num_threads>num_cores:
      return num_threads-1
    num_threads+=1
  return num_threads-1  



# compute number of processes and threads for the current job
num_sims=10
nice_level=19
host=socket.gethostname()  
num_cores=psutil.cpu_count()
num_threads= get_optimal_thread_num(num_sims,num_cores)
machine_load=num_threads*num_sims*100./num_cores

print 
print '======================================================'
print 'Host machine: %s '%host
print 'Multiprocess pool size: %d '%num_cores
print 'Nice level: %d '%nice_level
print 'MKL threads per process: %d '%num_threads  
print 'Machine load: %.1f %%'%machine_load
print '======================================================'
print

# set nice level and number of MKL threads
os.nice(nice_level)
mkl_rt = ctypes.CDLL('libmkl_rt.so')
mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(num_threads)))

# create pool of processes and run
pool=Pool(processes=num_cores)
for sim_idx in range(num_sims):  
  pool.apply_async(dummy_job, args=(sim_idx,))
pool.close()
pool.join()    
    