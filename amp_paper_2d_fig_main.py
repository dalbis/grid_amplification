#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:36:15 2019

@author: dalbis
"""


import pylab as pl
import numpy as np
import grid_utils.plotlib as pp
import grid_utils.gridlib as gl
import grid_utils.simlib as sl

from recamp_2pop import RecAmp2PopLearn,RecAmp2PopSteady
import amp_paper_2d_main as apm


input_color='m'


#%%
### LOAD DEFAULT SIMULATION DATA

extra_params={}
sim_conn=RecAmp2PopLearn(sl.map_merge(apm.def_recamp_learn_params,extra_params))

sim=RecAmp2PopSteady(sl.map_merge(apm.def_recamp_steady_params,
extra_params,{'recurrent_weights_path':sim_conn.data_path}))

sim.post_init()
sim.load_weights_from_data_path(sim.recurrent_weights_path)
sim.load_steady_outputs()
sim.load_steady_scores()



#%% ========================================================================================

### EXAMPLE INPUTS

input_idxs=[1,2,3]  
dx=sim.L/sim.nx
xran=np.arange(sim.nx)*sim.L/sim.nx-sim.L/2-dx/2.

pl.figure(figsize=(5.5,3))
pl.subplots_adjust(wspace=0.2)

for idx,cell_idx in enumerate(input_idxs):
  pl.subplot(2,5,idx+1,aspect='equal')
  r_map=sim.inputs.inputs_flat[:,cell_idx].reshape(sim.nx,sim.nx).T
  pl.pcolormesh(xran,xran,r_map,vmin=0,rasterized=True)
  
  pl.title('%-18.1f '%(r_map.max()),fontsize=8)
  pp.noframe()  


fname = 'fig2d_model_example_inputs' 
pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])

#%%

### PLOT EXAMPLE EXCITATORY/INHIBITORY OUTPUTS ===========================================

output_scores_to_show=[0.35,0.4,0.45,0.5,0.6]
exc_cell_idxs=[np.argmin(np.abs(sim.grid_tuning_out-out_score)) for out_score in output_scores_to_show]
inhib_cell_idxs=np.array([0,1,2,3,4])+sim.N_e
  
dx=sim.L/sim.nx
xran=np.arange(sim.nx)*sim.L/sim.nx-sim.L/2-dx/2.

plot_scores=True
vmax=None

for inhibitory in False,True:
  
  cell_idxs = inhib_cell_idxs if inhibitory else exc_cell_idxs


  pl.figure(figsize=(5.5,3))
  pl.subplots_adjust(wspace=0.2)
  for idx,cell_idx in enumerate(cell_idxs):
    pl.subplot(2,5,idx+1,aspect='equal')
    r_map=sim.r[cell_idx,:].reshape(sim.nx,sim.nx).T

      
    pl.pcolormesh(xran,xran,r_map,rasterized=True,vmin=0,vmax=vmax)
    
    if plot_scores is True:      
        if inhibitory is True:          
            pl.title('%2.1f    $\\bf{%2.2f}$'%(r_map.max(),sim.grid_tuning_in[cell_idx-sim.N_e]),fontsize=8)            
        else:
            pl.title('%2.1f    $\\bf{%2.2f}$'%(r_map.max(),sim.grid_tuning_out[cell_idx]),fontsize=8)
    
    pp.noframe()
        
  fname = 'fig2d_model_inhib_outputs' if inhibitory else 'fig2d_model_exc_outputs'
  pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])
  

  
#%% =================================================================================

### PLOT RECURRENT CONNECTIVITY 
  

for learned in True,False:  

  W=sim.W_ee if learned else sim.W_ee0
  pp.plot_recurrent_weights(W,sim.gp,vmax=sim.W_max_ee,ms=5,figsize=(3.2,3.5))
  tuning_index= gl.get_recurrent_matrix_tuning_index(W,sim.gp)

  fname = 'fig2d_model_rec_weights_learned' if learned else 'fig2d_model_rec_weights_init'
  pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])
  
  print 'Connnectivity tuning index: %.3f'%tuning_index

#%%

  
### PLOT TUNING INDEX HISTOGRAMS 
  
  
pl.rc('font',size=10)

bins = pl.histogram_bin_edges(sim.grid_tuning_in,bins=100,range=[0,1])
#logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

pl.figure(figsize=(2.7,2.3))
pl.subplots_adjust(left=0.3,bottom=0.26,right=0.95)
pl.hist(sim.grid_tuning_out_inhib,bins=bins,color='dodgerblue',histtype='stepfilled',weights=np.ones_like(sim.grid_tuning_out_inhib)/float(len(sim.grid_tuning_out_inhib)),alpha=1)
pl.hist(sim.grid_tuning_in,bins=bins,color=input_color,histtype='stepfilled',weights=np.ones_like(sim.grid_tuning_in)/float(len(sim.grid_tuning_in)),alpha=1)
pl.hist(sim.grid_tuning_out,bins=bins,color='black',histtype='stepfilled',weights=np.ones_like(sim.grid_tuning_out)/float(len(sim.grid_tuning_out)),alpha=1)
pl.hist(sim.grid_tuning_out_inhib,bins=bins,color='dodgerblue',histtype='step',weights=np.ones_like(sim.grid_tuning_out_inhib)/float(len(sim.grid_tuning_out_inhib)),alpha=1)

pp.custom_axes()
pl.xlim(0,0.7)
pl.xlabel('Grid tuning index')
pl.ylabel('Fraction of cells')

print 'Mean input grid tuning index: %.2f'%np.mean(sim.grid_tuning_in)
print 'Mean output grid tuning index: %.2f'%np.mean(sim.grid_tuning_out)


fname = 'fig2d_model_grid_tuning_hists'

pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])





