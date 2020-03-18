# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:11:39 2016

@author: dalbis
"""

import time
import numpy as np
from numpy.linalg import norm
import os
from simlib import print_progress,gen_hash_id,gen_string_id
import simlib as sl

def get_pos_idx(p,pos):
  
  p_dist=((np.array(p)-pos)**2).sum(axis=1)  
  p_idx = np.argmin(p_dist)
  return p_idx
  
def inside_arena(p,arena_shape,L):
  """
  Check if the rat is in the arena
  """
  if arena_shape == 'square':
    if p[0]>=-L/2. and p[0]<L/2. and p[1]>=-L/2. and p[1]<L/2.:
      return True
    else:
      return False
  elif arena_shape == 'circle':
    if norm(p)<L/2.:
      return True
    else:
      return False
      
def bounce_fun(p,v,arena_shape,L,dt,theta_sigma=0):
  """
  Bounce against walls.  Usage:
  v_new=bounce_fun(p,v,arena_shape,L)
  theta=arctan2(v_new[1],v_new[0])
  """        
  if arena_shape == 'square':
    if p[0]<-L/2. or p[0]>=L/2.:
      v_new= np.array([-v[0],v[1]])
    elif p[1]<-L/2. or p[1]>=L/2.:
      v_new= np.array([v[0],-v[1]]) 
    else:
      v_new= v
  elif arena_shape == 'circle':
    n = p/norm(p)
    v_new=v-2*n*np.dot(n,v)
  else:
    v_new = v
  
  theta=np.arctan2(v_new[1],v_new[0])
  if theta_sigma>0:
    theta=theta_sigma*np.randn()+theta

  p=p+v_new*dt
  return p,theta

def periodic_pos(p,L) :
  for idx in 0,1:
    if p[idx]<-L/2.:
      p[idx]+=L
    elif p[idx]>=L/2.:
      p[idx]-=L
  return p
    
    
# key parameters
key_params_const_speed=['arena_shape','L','speed','theta_sigma','position_dt','nx',
                        'walk_seed','walk_time','periodic_walk','bounce','bounce_theta_sigma',
                        'virtual_bound_ratio','init_p','init_theta']

var_speed_params=['variable_speed','speed_theta','speed_sigma']


    
class RandomWalk(object):

  results_path=os.path.join(sl.get_results_path(),'random_walk')

  
  @staticmethod
  def get_id(paramMap):  
    string_id=gen_string_id(paramMap,key_params=RandomWalk.get_key_params(paramMap))
    #print string_id
    return gen_hash_id(string_id)

    
  @staticmethod  
  def get_data_path(paramMap):
    return os.path.join(RandomWalk.results_path,RandomWalk.get_id(paramMap)+'_data.npz')
    
  @staticmethod
  def get_key_params(paramMap):
    key_params=key_params_const_speed
    
    if 'variable_speed' in paramMap.keys() and paramMap['variable_speed'] is True:
      key_params+=var_speed_params

    if 'run_in_circle' in paramMap.keys() and paramMap['run_in_circle'] is True:
      key_params+=['run_in_circle',]
      
    if 'sweep' in paramMap.keys() and paramMap['sweep'] is True:
      key_params+=['sweep',]
      
      
    return key_params
    
    
  def __init__(self,keyParamMap,do_print=True,force=False):
               #init_p=np.array([0.,0.]),init_theta=0.0):
            
    # import parameters
    for param in RandomWalk.get_key_params(keyParamMap):
      setattr(self,param,keyParamMap[param])
      #print 'setting %s %s'%(param,keyParamMap[param])
          

    #self.init_theta=init_theta
    #self.init_p=init_p
    
    self.circle_radius=0.397
    self.circle_omega=self.speed/self.circle_radius
      
    if not hasattr(self,'variable_speed'):
      self.variable_speed=False
      
    if not hasattr(self,'run_in_circle'):
      self.run_in_circle=False       

    if not hasattr(self,'sweep'):
      self.sweep=False     
      
    self.id=RandomWalk.get_id(keyParamMap)
    self.paramsPath=os.path.join(self.results_path,self.id+'_log.txt')
    self.dataPath=os.path.join(self.results_path,self.id+'_data.npz')   

    if force or not os.path.exists(self.dataPath):
      
      # generate and save data
      self.gen_data(do_print)

    # load data 
    self.load_data(do_print)
        

        
  def load_data(self,do_print):
    """
    Loads data from disk
    """
    
    if do_print:
      print
      print 'Loading walk data, Id = %s'%self.id
    
    data=np.load(self.dataPath,allow_pickle=True)
    
    for k,v in data.items():
      setattr(self,k,v)

    if do_print:     
      print 'Loaded variables: '+' '.join(data.keys())


  def gen_data(self,do_print):
    """
    Generates walk data and saves it to disk
    """

    if do_print:
      print
      print 'Generating walk data, Id = %s'%self.id
    
    self.post_init()
    self.run()

    toSaveMap={'pos':self.pos,'pidx_vect':self.pidx_vect,'walk_steps':self.walk_steps,'speed_vect':self.speed_vect}               
    np.savez(self.dataPath,**toSaveMap)

    if do_print:    
      print 'Result saved in: %s\n'%self.dataPath

      
  def post_init(self):

    #self.dx=self.position_dt
    #self.nx=int(self.L/self.position_dt)
    
    self.dx=float(self.L)/self.nx
    
    
    self.walk_steps = int(self.walk_time/self.position_dt)
    
    self.current_speed=self.speed
    
    np.random.seed(self.walk_seed)

    X,Y=np.mgrid[-self.L/2:self.L/2:self.dx,-self.L/2:self.L/2:self.dx]
    iX,iY=np.mgrid[-self.nx/2:self.nx/2:1,-self.nx/2:self.nx/2:1]    
    
    self.pos=np.array([np.ravel(X), np.ravel(Y)]).T
    self.ipos=np.array([np.ravel(iX), np.ravel(iY)]).T
        
    self.startClock=time.time()

  def update_speed(self):

    self.current_speed = self.current_speed+self.speed_theta*(self.speed-self.current_speed)*self.position_dt+self.speed_sigma*np.sqrt(self.position_dt)*np.random.randn()
    self.current_speed=self.current_speed.clip(min=0)
    
  def update_position(self):
    
    # choose next running direction and position
    p0=self.p  
    while True:
      
      # update theta    
      self.theta = self.theta+self.theta_sigma*np.sqrt(self.position_dt)*np.random.randn()
        
      self.v = np.array([np.cos(self.theta),np.sin(self.theta)])*self.current_speed
      self.p=p0+self.v*self.position_dt
  
      # boundary conditions
      if inside_arena(self.p,self.arena_shape,self.L*self.virtual_bound_ratio):
        break
      else:
        
        if self.periodic_walk is True:
          self.p
          self.p=periodic_pos(self.p,self.L)
          break
        
        if self.bounce is True:
          self.p,self.theta=bounce_fun(self.p,self.v,self.arena_shape,self.L*self.virtual_bound_ratio,self.position_dt,theta_sigma=self.bounce_theta_sigma)
          break
            


  def update_position_circle(self):
      
      self.circle_angle+=self.circle_omega*self.position_dt
      self.p=self.circle_radius*np.array([np.cos(self.circle_angle),np.sin(self.circle_angle)])
      
      
            
  def update_position_sweep(self):

      self.v = np.array([np.cos(self.theta),np.sin(self.theta)])*self.current_speed
      self.p_next=self.p+self.v*self.position_dt
    
      # got to a boundary
      if not inside_arena(self.p_next,self.arena_shape,self.L*self.virtual_bound_ratio):
        down_theta=-np.pi/2.
        
        self.v = np.array([np.cos(down_theta),np.sin(down_theta)])*self.current_speed
        self.p_next=self.p+self.v*self.position_dt

        # changing direction        
        if self.theta==0:
          self.theta=-np.pi
        elif self.theta==-np.pi:
          self.theta=0.
          
                      
        if not inside_arena(self.p_next,self.arena_shape,self.L*self.virtual_bound_ratio):
          print 'sweep end'
          self.sweep_end=True
          return
                  
      self.p=self.p_next
      
      
  def run(self):
        
    self.p = self.init_p
    #print self.p
    
    if self.sweep:
      self.p=np.array([-self.L/2.,self.L/2.])
      self.sweep_end=False
    
    elif self.run_in_circle:
      self.circle_angle=0.
      self.p=np.array([self.circle_radius,0.])
      
    self.theta=self.init_theta

    self.pidx_vect=np.zeros((self.walk_steps),dtype=np.int32)    

    progress_clock=time.time()

    snap_idx=0
    num_snaps=100
    delta_snap=self.walk_steps/num_snaps
    self.speed_vect=np.zeros(num_snaps)
    
    for step_idx in xrange(self.walk_steps):

      
      
      if self.variable_speed is True:
        self.update_speed()
      
      if self.sweep is True:
        if self.sweep_end: 
          break
        else:
          self.update_position_sweep()
          
      elif self.run_in_circle is True:
        self.update_position_circle()
      else:
        self.update_position()
      
      if np.remainder(step_idx,delta_snap)==0:
        if snap_idx<num_snaps:
          self.speed_vect[snap_idx]=self.current_speed
        print_progress(snap_idx,num_snaps,progress_clock)
        snap_idx+=1
        
      
      pidx=get_pos_idx(self.p,self.pos)
      self.pidx_vect[step_idx]=pidx
            
  def plot_occupancy(self):
    import pylab as pl
    from plotlib import custom_axes

    p_hist,x,y = np.histogram2d(self.pos[self.pidx_vect,0],self.pos[self.pidx_vect,1],
                                range=[[-self.L/2, self.L/2], [-self.L/2, self.L/2]],bins=50)
    
    pl.figure()
    # plot the results
    pl.subplot(111,aspect='equal')
    custom_axes()
    pl.xlim(-self.L/2,self.L/2)
    pl.ylim(-self.L/2,self.L/2)
    
    pl.pcolormesh(x,y,p_hist)
    pl.colorbar()
    pl.xlabel('X bin')
    pl.ylabel('Y bin')
    pl.title('Visits')

    pl.figure()
    pl.hist(self.pidx_vect,color='k',bins=100)

    pl.figure()
    pl.plot(self.pidx_vect,'-k')


  def plot(self,num_steps=1000):
    import pylab as pl
    from plotlib import custom_axes
    
    pl.figure()
    pl.subplot(111,aspect='equal')
    custom_axes()
    pl.xlim(-self.L/2,self.L/2)
    pl.ylim(-self.L/2,self.L/2)
    
    if num_steps is not None:
      pl.plot(self.pos[self.pidx_vect[0:num_steps],0],self.pos[self.pidx_vect[0:num_steps],1],'.k',ms=2)  
    else:
      pl.plot(self.pos[self.pidx_vect,0],self.pos[self.pidx_vect,1],'.k',ms=2)  
        
    pl.figure(figsize=(10,4))
    pl.subplot(121)
    time=np.linspace(0,self.walk_time,len(self.speed_vect))
    pl.plot(time,self.speed_vect,'-k')
    pl.xlabel('Time [s]')
    pl.ylabel('Speed [m/s]')
    custom_axes()
    pl.xlim(0,5)

    pl.subplot(122)
    pl.hist(self.speed_vect,color='k',bins=100)
    pl.xlabel('Speed [m/s]')
    pl.ylabel('Count')
    custom_axes()
    pl.title('Mean = %.2f  Var = %.3e'%(self.speed_vect.mean(),self.speed_vect.var()))

  def remove_data(self):
    os.remove(self.dataPath)
    
if __name__ == '__main__':
  
  ## TESTING
  
  L=1.5
  params_map = { 'arena_shape':'square',
                'L':L,
                'speed':0.25,
                'theta_sigma':0.7,
                'position_dt':L/50.,
                'walk_seed':0,
                'walk_time':300,
                'periodic_walk':False,
                'bounce':True,
                'bounce_theta_sigma':0.,
                'virtual_bound_ratio':1.0,
                'variable_speed':False,
                'run_in_circle':False
              }        

  tw=RandomWalk(params_map,force=True)
  tw.plot_occupancy()
  