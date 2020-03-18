# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:04:57 2016

@author: dalbis
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:30:16 2015

@author: dalbis
"""


from multiprocessing import Pool

import itertools
import socket
import traceback
import os
import numpy as np
import datetime,time
import simlib as sl
import psutil
import ctypes
import pandas as pd
import sys


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
      return max(1,num_threads-1)
    num_threads+=1
  return max(1,num_threads-1)
    

def run_simulation(sim,methods_to_run):
  """
  sim: simulation object
  methods_to_run: list of tuples, the first entry of each tuple id the method name (string), the second is a map of arguments (None if no argument is accepted)
  e.g. 
    
  methods_to_run= (('run',None),('post_run',post_run_arg_map))
  
  will result in:
    
  sim.run()
  sim.post_run(**post_run_arg_map)
  
  """

  try:
    
    for method,args in methods_to_run:
      assert(hasattr(sim,method))
      
      if args is not None:
          getattr(sim, method)(**args)
      else:
          getattr(sim, method)()

  
  except Exception:
    print
    print 'Exception'
    traceback.print_exc()
  


class BatchSimulator(object): 
  
  
  def __init__(self,sim_class,
                    batch_default_map,
                    batch_override_map,
                    batch_data_folder,
                    methods_to_run,
                    all_combinations=True,
                    suffix='data'):
    
    self.sim_class=sim_class
    self.batch_data_folder=batch_data_folder
    self.batch_default_map=batch_default_map
    self.batch_override_map=batch_override_map
    
    self.startTimeStr=''
    self.endTimeStr=''
    self.elapsedTime=0
    self.methods_to_run=methods_to_run
    self.all_combinations=all_combinations
    self.suffix=suffix
    

  
  def get_path_by_hash(self,chash,suffix='data'):
    return os.path.join(self.sim_class.results_path,'%s_%s_%s.npz'%(self.sim_class.__name__,chash,suffix))

  def get_path_by_pars(self,pars):
    #if type(pars) is not tuple:
    #  pars=(pars,)
    chash=self.pars_to_hash_map[str(pars)]
    return self.get_path_by_hash(chash)    
    
  def get_data_paths(self):
    return [ self.get_path_by_hash(chash) for chash in self.hashes]


  def post_init(self,force=False,do_print=False):

    
    # create simulation objects for each simulation to be run
    
    self.sims=[]
    self.hashes=[]
    self.pars_to_hash_map={}
  
    # in this case we simulate all parameter combinations, i.e. (a,b); (1,2)  -> (a,1); (a,2); (b,1); (b,2)
    if self.all_combinations is True:
      
      self.all_par_values=sorted(itertools.product(*self.batch_override_map.values()))  
      
    # in this case we run one simulation per parameter set, i.e., (a,b); (1,2)  -> (a,1); (b,2)
    else:
      
      # check that all parameters have the same number of values
      all_val_nums=np.array([ len(vals) for vals in self.batch_override_map.values()])
      num_vals=all_val_nums[0]
      assert(np.all(all_val_nums==num_vals))
      
      
      self.all_par_values=[]
      for i in xrange(num_vals):
        par_comb=[]
        
        for key in self.batch_override_map.keys():
          par_comb.append(self.batch_override_map[key][i])
        
        self.all_par_values.append(par_comb)   
    
    self.batch_override_str=' '.join([ '%s (%s-%s)'%(key,sl.format_val(min(values)),                                            
sl.format_val(max(values))) for key,values in self.batch_override_map.items()])
    
    # loop over all combinations of paramater values
    for par_values in self.all_par_values:
  
      override_param_map={k:v for (k,v) in zip(self.batch_override_map.keys(),par_values)} 
      
      parMap=sl.map_merge(self.batch_default_map,override_param_map)
           
      sim=self.sim_class(parMap)    
      
      # always run if do_run attribute is not present
      if not hasattr(sim,'do_run'):
        sim.do_run=True
        
      #print sim.hash_id+' Run: %s'%sim.do_run
              
      if sim.do_run is True or force is True:
        self.sims.append(sim)
        
      self.hashes.append(sim.hash_id)
      self.pars_to_hash_map[str(par_values)]=sim.hash_id
     
    # generate batch hash
    self.batch_hash=sl.gen_hash_id('_'.join(self.hashes))
    self.batch_data_path=os.path.join(self.batch_data_folder,'%s_%s_%s.hd5'%(self.sim_class.__name__,self.batch_hash,self.suffix))
    self.batch_params_path=os.path.join(self.batch_data_folder,'%s_%s_%s.txt'%(self.sim_class.__name__,self.batch_hash,self.suffix))
    
       
    self.batch_summary_str=\
    """
BATCH SIMULATION CLASS: %s
    
BATCH HASH: %s
    
BATCH PARAMS = %s"""%\
    (
        str(self.sim_class),
        self.batch_hash,
        self.batch_override_str
     )
    
    if do_print:    
      print self.batch_summary_str
    
    self.toSaveMap={'hashes':self.hashes,
                    'batch_override_map':self.batch_override_map,
                    'batch_default_map':self.batch_default_map
                    }
    
    if os.path.exists(self.batch_data_path) and not force:
      if do_print:    
        print """
  *** BATCH DATA PRESENT ***
        
        """ 
        print self.batch_data_path
        print
      return False
    else:
      if do_print:    
        print """
  *** BATCH DATA NOT PRESENT ***
        
        """ 
        print self.batch_data_path
        print
        print '%d/%d simulations to be run'%(len(self.sims),len(self.all_par_values))
  
        print      
        print 'Parameters'
        print self.batch_override_map.keys()
        print 
        
        for sim,par_values in zip(self.sims,self.all_par_values):
          print str(par_values)+'  :   '+sim.hash_id
        
      return True
     
     

    
     
  def run(self,nice_level=9):
    
    
    ##############################################################################
    ###### CREATE AND RUN POOL OF PROCESSES (ONE PER SIMULATION)
    ##############################################################################
    
    # compute number of processes and threads for the current job
    num_sims=len(self.sims)
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

    # create pool of processes
    self.pool=Pool(processes=num_cores)
      
    
    startClock=time.time()
    startTime=datetime.datetime.fromtimestamp(time.time())
    self.startTimeStr=startTime.strftime('%Y-%m-%d %H:%M:%S')
    
        
    for sim in self.sims:  
      self.pool.apply_async(run_simulation, args=(sim,self.methods_to_run))
  
    self.pool.close()
    self.pool.join()      
  
  
    # logging simulation end
    endTime=datetime.datetime.fromtimestamp(time.time())
    self.endTimeStr=endTime.strftime('%Y-%m-%d %H:%M:%S')
    self.elapsedTime =time.time()-startClock
  
    print 'Batch simulation ends: %s'%self.endTimeStr
    print 'Elapsed time: %s\n' %sl.format_elapsed_time(self.elapsedTime)
      
      
      
  def load_dataframe(self):
    self.df=pd.read_hdf(self.batch_data_path)    
      
  def save_dataframe(self):
    print
    print 'BATCH HASH: %s'%self.batch_hash
    sl.ensureParentDir(self.batch_data_path)
    
    import warnings
    warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)
    self.df.to_hdf(self.batch_data_path, key='df', mode='w')
    
    print
    print 'Batch data saved in: %s\n'%self.batch_data_path
    print  
    
  def __merge_and_save(self,outputs_to_merge,merge_functions={},suffix='data'):
    """
    outputs_to_merge is a list of attributes to be merged in a single output map
    Each output must be saved in the simulation results or a specific function for its computation 
    must be provided in the merge_functions map (key is the output to compute, value is the function).
    The merge function takes as input the results data dictionary and outputs the value to be saved
    in the merged batch file.
    """
    
      
    #############################################################################
    ##### MERGE DATA
    #############################################################################
  

    print 'Merging data...\n'
    sys.stdout.flush()
     
    data_list=[]
    start_clock=time.time()
    

    for idx,(chash,par_values) in enumerate(zip(self.hashes,self.all_par_values)):

      #print '%d/%d '%(idx,len(self.all_par_values))
      sl.print_progress(idx,len(self.all_par_values),start_clock=start_clock,step=len(self.all_par_values)/20.)
      sys.stdout.flush()
  
      try:
        dataPath=os.path.join(self.sim_class.results_path,
                              '%s_%s_%s.npz'%(self.sim_class.__name__,chash,suffix))       
        data=np.load(dataPath,mmap_mode='r',allow_pickle=True)
        print 'Loading %s',dataPath
        
      except Exception:
        print 'error in loading: %s %s'%(str(par_values),chash)
        import traceback
        traceback.print_exc()
        return
        
      # construct a data record for the current combination of parameters        
      record={}
      
      ## add paramter values 
      for idx,param_name in enumerate(self.batch_override_map.keys()):
        record[param_name]=par_values[idx]
        
      ## add output data to merge  
      for output_name in outputs_to_merge:
        if output_name in data.keys():
          record[output_name]=data[output_name]
        elif output_name in merge_functions.keys():
          record[output_name]=merge_functions[output_name](data)
        else:
          raise Exception('Output to merge non present in archive and merge function not found.')
        
      data_list.append(record)         

    # create data frame
    self.df = pd.DataFrame(data_list)
            
    # save      
    sl.logSim(self.batch_hash,self.batch_override_str,self.startTimeStr,self.endTimeStr,self.elapsedTime,self.batch_default_map,self.batch_params_path,doPrint=False)
    
    self.save_dataframe()

  def merge(self,outputs_to_merge,merge_functions={},suffix='data',force=False):
    
    if force or not os.path.exists(self.batch_data_path):
      self.__merge_and_save(outputs_to_merge,merge_functions,suffix)
    else:
      self.load_dataframe()    

