# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:57:51 2014

@author: dalbis
"""
import sys
import os
from time import time
from numpy import floor,remainder,float64
from collections import namedtuple
import numpy as np
import json


def get_project_root():
  """
  Returns the root of the current project (parent directory of this Python file)
  """
  return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def get_config(param):
  """
  Reads a parameter from the configuration file.
  The configuration file is a JSON file named config.json and saved in the project's root directory
  """
  config_path= os.path.join(get_project_root(),'config.json')
  if config_path:
    with open(config_path, 'r') as f:
        config = json.load(f)
  else:
   raise Exception('Configuration file not found: %s'%config_path)      
   
  if param in config.keys():
    return config[param]
  else:
    raise Exception("%s is not a configuration parameter"%param )

def get_results_path():
  """
  Returns the path to the results folder (read from the configuration file)
  """
  return get_config('RESULTS_PATH')

def get_figures_path():
  """
  Returns the path to the figures folder (read from the configuration file)
  """
  return get_config('FIGURES_PATH')


def get_params(paramMap):
  """
  Returns a namedtuple from a parameters' dictionary
  """
  Params = namedtuple('Params', paramMap.keys())
  return Params(**paramMap)  


def load_data(data_path):
  """
  Load data to a target dictionary
  """
  
  # load data
  print 'Loading: %s'%data_path
  assert(os.path.exists(data_path))

  data=np.load(data_path,mmap_mode='r',allow_pickle=True)
  
  # load parameters into locals
  paramMap=data['paramMap'][()]
  p=get_params(paramMap)

  Results = namedtuple('Params', data.keys())
  r=Results(**data)
  
    #print '==========    DATA    =========='
  #print '\n'.join(data.keys())
  #print
  
  return p,r

def map_merge(*args):
  """
  Merges two or more dictionaries
  """
  tot_list=[]
  for arg in args:
    tot_list=tot_list+arg.items()    
  return dict(tot_list)

  
def run_from_ipython():
  """
  Checks whether the current program is running within ipython
  """
  try:
      __IPYTHON__
      return True
  except NameError:
      return False
        
  
       
def format_elapsed_time(delta_clock):
  """
  Formats elapsed time in a human readable format
  """
  hours = floor(delta_clock/3600)
  minutes = floor(delta_clock/60-hours*60)
  seconds = floor(delta_clock-hours*3600-minutes*60)
  
  string=''
  if hours>0:
     string+=' %dh'%hours
  if minutes>0:
     string+=' %dm'%minutes
     
  string+=' %ds'%seconds
  return string
  
def print_progress(snap_idx,num_snaps,start_clock=None,step=None):
  """
  Prints a progress bar to the console
  """
    
  if step is None:
    step=num_snaps/100.0
    
  if snap_idx>0:
    snap_idx+=1
    if remainder(snap_idx,float(step))==0:
      progress = int(snap_idx/float(num_snaps)*100)
      string = '\r[{:20s}] {:3d}% complete'.format('#'*(progress/5), progress)
    
      if progress>0 and start_clock is not None:
        cur_clock = time()
        elapsed_time = cur_clock-start_clock
        remaining_time = (elapsed_time/progress)*(100-progress)
        string+=',%s elapsed, about%s remaining' %(format_elapsed_time(elapsed_time),format_elapsed_time(remaining_time))
  
      if progress == 100:
        print string
      else:
        print string,
        
      if hasattr(sys.stdout,'flush'):
        sys.stdout.flush()
          


def ensureParentDir(path):
  """
  Ensures that the parent directory to 'path' is present.
  If not the directory is created.
  """
  parentDir = os.path.realpath(path+'/..')
  if not os.path.exists(parentDir):
    os.makedirs(parentDir)

def ensureDir(path):
  """
  Ensures that the directory 'path' is present.
  If not the directory is created.
  """
  if not os.path.exists(path):
    os.makedirs(path)
    

def gen_hash_id(string):
  """
  Returns and MD5 hash representation of a string.
  This function is used to generate unique file names.
  The input string is built from the parameters that generated the data to be saved.
  """
  import hashlib  
  hash_object = hashlib.md5(string.encode())
  return str(hash_object.hexdigest())



def gen_string_id(paramMap,key_params=None,sep='_'):
  """
  Returns a string representation of a dictionary of parameters.
  This is used to then generate a unique MD5 hash to use as filename for data storage.
  """
  str_id=''
  if key_params is None:
    keys=paramMap.keys()
  else:
    keys=key_params
  for key in sorted(keys): #keys
    str_id+='%s=%s%s'%(key,format_val(paramMap[key]),sep) 
  str_id=str_id[:-1]
  return str_id


def format_val(val):
  """
  Formats in a string formats numerical values of different typse
  """
  if (type(val)==float64 or type(val)==float) and (abs(val)<1e-3 or abs(val)>1e3):
    val_str='%.3e'%val 
  else:
    val_str=str(val)
     
  return val_str
    
def params_to_str(paramMap,keyParams=None,compact=False,to_exclude=[]):
  """
  Human-readable string representation of a dictionary of parameters
  """
  
  if keyParams is None:
    keys=paramMap.keys()
  else:
    keys=keyParams
    
  if compact is False:
    logStr='\n'
    logStr+='========== PARAMETERS ==========\n'
    delimiter='\n'
    equal=' = '
  else:
    logStr=''
    delimiter=', '
    equal='='
  for key in sorted(keys):
    if key not in to_exclude:
     val=paramMap[key]
     val_str=format_val(val)
     logStr+=key+equal+val_str+delimiter
     
  if compact is False:    
    logStr+='\n'
  else:
    logStr=logStr[0:-len(delimiter)]
  return logStr    
    
def logSim(simId,simTitle,tsStr,teStr,elapsedTime,paramMap,paramsPath,doPrint=True):
  """
  Logs simulation parameters and simulation time to a string.
  The string is also saved to a text file stored in paramsPath
  """
  import socket

  logStr=''
  logStr+='Simulation Title: %s \n'%simTitle
  logStr+='Simulation Id: %s \n' %simId
  logStr+='Running on: %s\n'% socket.gethostname()
  logStr+= 'Simulation started: %s\n'%tsStr
  logStr+= 'Simulation ended: %s\n'%teStr
  logStr+='Elapsed time: %s \n' %format_elapsed_time(elapsedTime)
  logStr+=params_to_str(paramMap)
  if doPrint is True:
    print logStr
  ensureParentDir(paramsPath)
  f=open(paramsPath,'w')
  f.write(logStr)
  f.close()
  return logStr
  
class Tee(object):
  """
  Class for duplicating stdout to log file
  """  
  def __init__(self, name):
      self.file = open(name, 'w')
      self.stdout = sys.stdout
      sys.stdout = self
  def __del__(self):
      sys.stdout = self.stdout
      self.file.close()
  def write(self, data):
      self.file.write(data)
      self.file.flush()
      self.stdout.write(data)
      self.stdout.flush()

def get_unique_path(baseName):
  """
  Returns a unique path name by appending an integer at the end of the file name.
  """  

  suffix=''
  idx=1
  while True:
    if not os.path.exists(baseName+suffix):
      return baseName+suffix
    else:
      suffix='_%d'%idx
      idx+=1


