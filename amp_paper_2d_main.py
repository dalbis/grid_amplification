#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 09:38:37 2018

@author: dalbis
"""


import warnings
warnings.simplefilter('ignore',FutureWarning)

import os
import numpy as np

import grid_utils.simlib as sl
from grid_utils.spatial_inputs import InputType
from grid_utils.batch_simulator import BatchSimulator
from recamp_2pop import RecAmp2PopSteady,RecAmp2PopLearn


# folder to save collected batch data
recamp_batch_data_folder=os.path.join(sl.get_results_path(),'recamp2pop_batch')

# ------------------------- DEFAULT PARAMETERS -------------------------------       

def_recamp_params={ 

                                                                                                
   # general paramters           
   'n_e':30, 
   'n_i':15,    
   'L':2., 
   'seed':1,
   'dt':0.002,
   'tau':0.01,
   'r0':0.,

   # connectivity
   'frac_conns_ee':0.1,
   'frac_conns_ii':0.4,
   'frac_conns_ei':0.4,
   'frac_conns_ie':0.4,
   'W_tot_ee': 2.,
   'W_tot_ei': .4,                     
   'W_tot_ie': 10.,                      
   'W_tot_ii': 1.,                     
                            
   
   # walk parameters
   'arena_shape':'square',
   'nx':100,
   'speed':0.25,
   'theta_sigma': 0.7,
   'walk_seed':0,
   'periodic_bounds':False,
   'bounce':True,
   'bounce_theta_sigma':0.,
   'variable_speed':False,
   'periodic_walk':False,
   'virtual_bound_ratio':1.,

    # inputs parameters
   'inputs_type':InputType.INPUT_NOISY_GRID,
   'input_mean':5.,        
   'inputs_seed':1,
   'grid_T':0.5, 
   'grid_angle':0.,
   'signal_weight': 0.35,
   
   'grid_T_sigma':0.03,
   'grid_angle_sigma':0.03,
   
   'noise_sigma_x': 0.3, 
   'noise_sigma_phi':0.1, # note that dphi ~0.2 therefore this is uncorrelated!
   
   'jitter_variance': 0.,
   'jitter_sigma_phi':0.,    
   
   'same_fixed_norm':False,  
   'fixed_norm':6.,
   
   'zero_phase':False,
   'scale_field_size_ratio':0.3  
             
}    


def_recamp_learn_params=sl.map_merge(def_recamp_params,{
    
    # learning recurrent connectivity
   'start_with_zero_connectivity':False,   
   'learn_with_recurrent_input':False,    # If set to True the system may be unstable if W_tot_ee is to large
   'learn_rate':2e-5,    
   'learn_num_snaps':200,   
   'learn_walk_time':1000,    
   'use_speed_input':False,
    
    })



def_recamp_steady_params=sl.map_merge(def_recamp_params,{
   
    # running recurrent dynamics (without walk)
   'recdyn_time':0.2,   
   'recdyn_num_snaps':20,
   'use_learned_recurrent_weights':True,
   'recurrent_weights_path': RecAmp2PopLearn(def_recamp_learn_params).data_path, 
})

if def_recamp_steady_params['use_learned_recurrent_weights'] is False:
  def_recamp_steady_params=sl.map_merge(def_recamp_steady_params,{'fixed_connectivity_tuning':1.0,})
  
  
  
# ---------------- PARAMETER RANGES -------------------------------------------
  
jitter_variance_ran=np.linspace(0.,0.005,30,endpoint=True)
signal_weight_ran=np.linspace(0.,1.,30,endpoint=True)
min_sigma_x=def_recamp_params['L']/def_recamp_params['nx']
noise_sigma_x_ran=np.logspace(np.log10(min_sigma_x),np.log10(30),30,endpoint=True)
noise_sigma_phi_ran=np.logspace(np.log10(0.1),np.log10(30),30,endpoint=True)
jitter_sigma_phi_ran=np.linspace(0.,.6,10,endpoint=True)
W_tot_ee_ran=np.linspace(0.,1.,10,endpoint=True)

inputs_seed_ran=np.arange(10)

grid_T_sigma_ran=[ 0.03, 0.05, 0.1, 0.5]
grid_angle_sigma_ran=[ 0.03, 0.05, 0.1, 0.5]
                                


# ------------------- BATCH SIMULATORS ----------------------------------------


### LEARNING RECURRENT WEIGHTS ############################################################


# different input seeds
batch_learn_input_seed=BatchSimulator(RecAmp2PopLearn,
                       def_recamp_learn_params,
                       { 'inputs_seed':inputs_seed_ran},
                       recamp_batch_data_folder,
                       (
                           ('learn_recurrent_weights',None),
                       ))
batch_learn_input_seed.post_init()    

# signal weight --------------------------------------------------------------
  
# signal_weight learn weights
batch_signal_weight_learn=BatchSimulator(RecAmp2PopLearn,
                       def_recamp_learn_params,
                       {'signal_weight':signal_weight_ran},
                       recamp_batch_data_folder,
                       (
                           ('learn_recurrent_weights',None),
                       ))
batch_signal_weight_learn.post_init()    

# signal_weight learn weights: multiple input seeds
batch_signal_weight_learn_input_seed=BatchSimulator(RecAmp2PopLearn,
                       def_recamp_learn_params,
                       {'signal_weight':signal_weight_ran, 'inputs_seed':inputs_seed_ran},
                       recamp_batch_data_folder,
                       (
                           ('learn_recurrent_weights',None),
                       ),all_combinations=True)
batch_signal_weight_learn_input_seed.post_init()    

# sigma_x --------------------------------------------------------------

# sigma_x learn weights
batch_sigma_x_learn=BatchSimulator(RecAmp2PopLearn,
                       def_recamp_learn_params,
                       {'noise_sigma_x':noise_sigma_x_ran},
                       recamp_batch_data_folder,
                       (   
                           ('learn_recurrent_weights',None),
                       ))
batch_sigma_x_learn.post_init()

# sigma_phi --------------------------------------------------------------


# sigma_phi learn weights
batch_sigma_phi_learn=BatchSimulator(RecAmp2PopLearn,
                       def_recamp_learn_params,
                       {'noise_sigma_phi':noise_sigma_phi_ran},
                       recamp_batch_data_folder,
                       (   
                           ('learn_recurrent_weights',None),
                       ))
batch_sigma_phi_learn.post_init()
                      

# sigma_phi learn weights: multiple input seeds
batch_sigma_phi_learn_input_seed=BatchSimulator(RecAmp2PopLearn,
                       def_recamp_learn_params,
                       {'noise_sigma_phi':noise_sigma_phi_ran, 'inputs_seed':inputs_seed_ran},
                       recamp_batch_data_folder,
                       (   
                           ('learn_recurrent_weights',None),
                       ),all_combinations=True)
batch_sigma_phi_learn_input_seed.post_init()



# uniform angles --------------------------------------------------------------
batch_uniform_angles_learn=BatchSimulator(RecAmp2PopLearn,
                       sl.map_merge(def_recamp_learn_params,{'inputs_type':InputType.INPUT_NOISY_GRID_UNIFORM_ANGLES,
                                                             'signal_weight':1.                                                             
                                                             }),
                       {
                           'inputs_seed':inputs_seed_ran
                       },
                       recamp_batch_data_folder,
                       (   
                           ('learn_recurrent_weights',None),
                       ),all_combinations=True)
batch_uniform_angles_learn.post_init()


# jitter variance --------------------------------------------------------------

batch_jitter_variance_learn=BatchSimulator(RecAmp2PopLearn,
                       def_recamp_learn_params,
                      {'jitter_variance':jitter_variance_ran},
                      recamp_batch_data_folder,
                      (
                          ('learn_recurrent_weights',None),
                      ))
batch_jitter_variance_learn.post_init()



batch_sigma_phi_learn_no_signal=BatchSimulator(RecAmp2PopLearn,
                       sl.map_merge(def_recamp_learn_params,{'signal_weight':0.}),
                       {'noise_sigma_phi':noise_sigma_phi_ran},
                       recamp_batch_data_folder,
                       (
                           ('learn_recurrent_weights',None),
                       ))


batch_W_tot_ee=BatchSimulator(RecAmp2PopLearn,
                       def_recamp_learn_params,
                       {'W_tot_ee':W_tot_ee_ran},
                       recamp_batch_data_folder,
                       (
                           ('learn_recurrent_weights',None),
                       ))

 


### AMPLIFICATION ################################################################

# using the connectivity learned with same parameters                      
batch_signal_weight=BatchSimulator(RecAmp2PopSteady,
                       def_recamp_steady_params,                      
                       {
                        'signal_weight':signal_weight_ran,
                       },
                       recamp_batch_data_folder,
                       (
                           ('compute_and_save_steady_output',None), 
                       ), all_combinations=False )



# using all learned connectivities 
batch_signal_weight_all_learned_weights=BatchSimulator(RecAmp2PopSteady,
                       def_recamp_steady_params,                      
                       {
                        'signal_weight':signal_weight_ran,
                        'recurrent_weights_path': batch_signal_weight_learn.get_data_paths()
                       },
                       recamp_batch_data_folder,
                       (
                           ('compute_and_save_steady_output',None), 
                       ), all_combinations=True )


# sigma_x --------------------------------------------------------------


# using the connectivity learned with default parameters
batch_noise_x=BatchSimulator(RecAmp2PopSteady,
                       def_recamp_steady_params,
                       {'noise_sigma_x':noise_sigma_x_ran},
                       recamp_batch_data_folder,
                       (
                           ('compute_and_save_steady_output',None),
                       ))

# sigma_phi --------------------------------------------------------------

# using the connectivity learned with default parameters
batch_noise_phi=BatchSimulator(RecAmp2PopSteady,
                       def_recamp_steady_params,
                       {'noise_sigma_phi':noise_sigma_phi_ran},
                       recamp_batch_data_folder,
                       (
                           ('compute_and_save_steady_output',None), 
                       ))

# using the connectivity learned with default parameters, multiple inputs seed
batch_noise_phi_input_seed=BatchSimulator(RecAmp2PopSteady,
                       def_recamp_steady_params,
                       {'noise_sigma_phi':noise_sigma_phi_ran,'inputs_seed':inputs_seed_ran},
                       recamp_batch_data_folder,
                       (
                           ('compute_and_save_steady_output',None), 
                       ),all_combinations=True)


# using all learned connectivities 
def get_batch(inputs_seed):
  batch=BatchSimulator(RecAmp2PopSteady,
           sl.map_merge(def_recamp_steady_params,{'inputs_seed':inputs_seed}),        
           {
            'noise_sigma_phi':noise_sigma_phi_ran,
            'recurrent_weights_path': batch_sigma_phi_learn.get_data_paths()
           },
           recamp_batch_data_folder,
           (
                 ('compute_and_save_steady_output',None),
           ), all_combinations=True )
  return batch
batches_noise_phi_all_learned_weights=[get_batch(inputs_seed) for inputs_seed in np.arange(6)]





#------------------------------- MAIN PROGRAM --------------------------------

if __name__ == '__main__':
 
    
  # simulations in which we learn the synaptic weights
  #learn_batches=[batch_signal_weight_learn,batch_sigma_phi_learn,batch_sigma_phi_learn_no_signal]         
  
  # simulations in which we estimate the amplification for the default connectitivty 
  amp_batches=[batch_signal_weight,batch_noise_phi_input_seed]

  # simulation in which we estimate the amplification for all connectivities
  amp_all_weights_batches=[batch_signal_weight_all_learned_weights]+batches_noise_phi_all_learned_weights

  #batches=learn_batches+amp_batches+amp_all_weights_batches  

  batches=[batch_learn_input_seed]
  #batches=amp_batches+amp_all_weights_batches
  
  ### RUN ALL BATCHES IN SEQUENCE
  
  for batch in batches:
    batch.post_init(do_print=True)
    batch.run()



