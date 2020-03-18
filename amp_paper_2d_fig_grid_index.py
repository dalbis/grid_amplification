#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:29:54 2020

@author: dalbis
"""

import pylab as pl
import numpy as np
import grid_utils.plotlib as pp
import grid_utils.gridlib as gl
import grid_utils.simlib as sl
import amp_paper_2d_main as apm

from recamp_2pop import RecAmp2PopSteady,RecAmp2PopLearn
from grid_utils.spatial_inputs import SpatialInputs,InputType,gen_corr_noise_2d



def plot_score_test(inputs,test_parameter,test_scores,test_tuning_indexes,test_parameter_label,same_vmax=True,tuning_index_lims=(-0.03,0.6)):
  """
  Plot gridness scores and grid-tuning indexes agains a test parameter (e.g., modulation depth, tuning strengths)
  """
  ms=5

  fig=pl.figure(figsize=(4,2.5))
  pl.subplots_adjust(left=0.2,bottom=0.2,right=0.8)
  
  num_cells=6
  gs = pl.GridSpec(3,num_cells,hspace=0.35,wspace=0.1)
  
  
  cell_idxs=np.linspace(0,n**2-1,num_cells).astype(int)
  vmax=inputs.max()
  for idx,cell_idx in enumerate(cell_idxs):
    ax=fig.add_subplot(gs[0,idx],aspect='equal')
    r_map=inputs[:,cell_idx].reshape(nx,nx).T
    if same_vmax is True:
      ax.pcolormesh(r_map,rasterized=True,vmin=0,vmax=vmax)
    else:
      ax.pcolormesh(r_map,rasterized=True,vmin=0)
      
    pp.noframe(ax)  
    
  ax1 = fig.add_subplot(gs[1:3,0:num_cells])
  
  ax1=pl.gca()
  
  color = 'tab:red'
  ax1.set_xlabel(test_parameter_label)
  ax1.set_ylabel('Grid tuning index', color=color)
  ax1.plot(test_parameter, test_tuning_indexes, color=color,marker='o',lw=0,alpha=.5,ms=ms,mec='none')
  ax1.tick_params(axis='y', labelcolor=color)
  ax1.set_ylim(tuning_index_lims)
  ax1.spines['top'].set_color('none')
  
  pl.plot(test_parameter[cell_idxs],np.ones_like(test_parameter[cell_idxs])*0,'vk')
    
    
  ax2 = ax1.twinx()  
  
  color = 'tab:orange'
  ax2.set_ylabel('Gridness score', color=color)  
  ax2.plot(test_parameter, test_scores, color=color,marker='o',lw=0,alpha=.5,ms=ms,mec='none')
  ax2.tick_params(axis='y', labelcolor=color)
  ax2.set_ylim(-1,2)
  ax2.spines['top'].set_color('none')


#  ax_joint=fig.add_subplot(gs[4:7,0:num_cells])
#  pl.sca(ax_joint)
#  pl.scatter(test_tuning_indexes,test_scores)
#  pp.custom_axes()
#  pl.xlabel('Grid tuning index')
#  pl.ylabel('Gridness score')

#%%
### LOAD DEFAULT SIMULATION DATA

sim_conn=RecAmp2PopLearn(sl.map_merge(apm.def_recamp_learn_params,{}))
sim=RecAmp2PopSteady(sl.map_merge(apm.def_recamp_steady_params,{},{'recurrent_weights_path':sim_conn.data_path}))

sim.post_init()
sim.load_weights_from_data_path(sim.recurrent_weights_path)
sim.load_steady_outputs()

# recompute amplification index
#sim.recompute_and_save_amplification_index()

sim.load_steady_scores()

# recompute gridness scores
#sim.compute_steady_scores(force_input_scores=True)
#sim.save_steady_scores()


#%%

### COMPARE GRIDNESS SCORE AND GRID-TUNING INDEX

fig=pl.figure(figsize=(8,3.5))
pl.subplots_adjust(top=0.95,right=0.95,bottom=0.15,left=0.15)
gs = pl.GridSpec(3,3,hspace=0.55,wspace=0.35)

ax_joint = fig.add_subplot(gs[1:3,0:2])
ax_marg_x = fig.add_subplot(gs[0,0:2],sharex=ax_joint)
ax_marg_y = fig.add_subplot(gs[1:3,2],sharey=ax_joint)

xlims=0,0.6
ylims=-0.3,1.7
ms=5

ax_joint.plot(sim.grid_tuning_out_inhib,sim.ri_scores,color='dodgerblue',marker='o',lw=0,alpha=0.5,ms=ms,mec='none')
ax_joint.plot(sim.grid_tuning_in,sim.he_scores,color='m',marker='o',lw=0,alpha=0.5,ms=ms,mec='none')
ax_joint.plot(sim.grid_tuning_out,sim.re_scores,color='black',marker='o',lw=0,alpha=0.5,ms=ms,mec='none')
ax_joint.set_xlabel('Grid tuning index')
ax_joint.set_ylabel('Gridness score')

pp.custom_axes(ax_joint)
pp.custom_axes(ax_marg_x)
pp.custom_axes(ax_marg_y)

ax_joint.set_xlim(xlims)
ax_joint.set_ylim(ylims)

pl.sca(ax_marg_x)
bins = pl.histogram_bin_edges(sim.grid_tuning_in,bins=100,range=[0,1])
pl.hist(sim.grid_tuning_out_inhib,bins=bins,color='dodgerblue',histtype='stepfilled',weights=np.ones_like(sim.grid_tuning_out_inhib)/float(len(sim.grid_tuning_out_inhib)),alpha=1,lw=1)
pl.hist(sim.grid_tuning_in,bins=bins,color='m',histtype='stepfilled',weights=np.ones_like(sim.grid_tuning_in)/float(len(sim.grid_tuning_in)),alpha=1,lw=1)
pl.hist(sim.grid_tuning_out,bins=bins,color='black',histtype='stepfilled',weights=np.ones_like(sim.grid_tuning_out)/float(len(sim.grid_tuning_out)),alpha=1,lw=1)
pl.hist(sim.grid_tuning_out_inhib,bins=bins,color='dodgerblue',histtype='step',weights=np.ones_like(sim.grid_tuning_out_inhib)/float(len(sim.grid_tuning_out_inhib)),alpha=1,lw=1)
pl.ylabel('Fraction\nof cells')
pl.plot(np.median(sim.grid_tuning_out_inhib),0.45,color='dodgerblue',marker='v')
pl.plot(np.median(sim.grid_tuning_in),0.45,'vm')
pl.plot(np.median(sim.grid_tuning_out),0.45,'vk')

pl.sca(ax_marg_y)
bins = pl.histogram_bin_edges(sim.he_scores,bins=30,range=[-0.2,1.7])
pl.hist(sim.ri_scores,orientation='horizontal',bins=bins,color='dodgerblue',histtype='stepfilled',weights=np.ones_like(sim.ri_scores)/float(len(sim.ri_scores)),alpha=1)
pl.hist(sim.he_scores,orientation='horizontal',bins=bins,color='m',histtype='stepfilled',weights=np.ones_like(sim.he_scores)/float(len(sim.he_scores)),alpha=1)
pl.hist(sim.re_scores,orientation='horizontal',bins=bins,color='black',histtype='stepfilled',weights=np.ones_like(sim.re_scores)/float(len(sim.re_scores)),alpha=1)
pl.hist(sim.he_scores,orientation='horizontal',bins=bins,color='m',histtype='step',weights=np.ones_like(sim.he_scores)/float(len(sim.he_scores)),alpha=1)
pl.hist(sim.ri_scores,orientation='horizontal',bins=bins,color='dodgerblue',histtype='step',weights=np.ones_like(sim.ri_scores)/float(len(sim.ri_scores)),alpha=1)
pl.xlabel('Fraction of cells')

pl.plot(0.52,np.median(sim.ri_scores),color='dodgerblue',marker=(3, 0, 90))
pl.plot(0.52,np.median(sim.he_scores),color='m',marker=(3, 0, 90))
pl.plot(0.52,np.median(sim.re_scores),'vk',marker=(3, 0, 90))

fname = 'fig2d_compare_grid_scores'
pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])

#%%
from scipy.stats import spearmanr
r_in,p_in=spearmanr(sim.grid_tuning_in,sim.he_scores)
r_out,p_out=spearmanr(sim.grid_tuning_out,sim.re_scores)
r_out_inhib,p_out_inhib=spearmanr(sim.grid_tuning_out_inhib,sim.ri_scores)

print 'correlation feed-forward inputs: r=%.2f p=%e'%(r_in,p_in)
print 'correlation excitatory outputs: r=%.2f p=%e'%(r_out,p_out)
print 'inhibitory outputs: r=%.2f p=%e'%(r_out_inhib,p_out_inhib)

print 'mean tuning feed %.2f'%sim.grid_tuning_in.mean()


#%%
#
#mod_depth=sim.r.max(axis=1)-sim.r.min(axis=1)
#
#pl.figure()
#pl.hist(mod_depth[:sim.n_e**2])
#pl.hist(mod_depth[sim.n_e**2:])
#
##%%
#
#pl.figure()
#pl.subplot(121)
#pl.plot(mod_depth[:sim.n_e**2],sim.re_scores,color='black',marker='o',lw=0,alpha=0.5,ms=ms,mec='none')
#pl.plot(mod_depth[sim.n_e**2:],sim.ri_scores,color='dodgerblue',marker='o',lw=0,alpha=0.5,ms=ms,mec='none')
#pl.xlabel('Modulation depth [spike/s]')
#pl.ylabel('Gridness score')
#pl.subplot(122)
#pl.plot(mod_depth[:sim.n_e**2],sim.grid_tuning_out,color='black',marker='o',lw=0,alpha=0.5,ms=ms,mec='none')
#pl.plot(mod_depth[sim.n_e**2:],sim.grid_tuning_out_inhib,color='dodgerblue',marker='o',lw=0,alpha=0.5,ms=ms,mec='none')
#pl.xlabel('Modulation depth [spike/s]')
#pl.ylabel('Grid tuning index')
##pp.custom_axes()

    
#%%
### GENERATE GRIDS FOR GRIDNESS SCORE TESTING


n=20
nx=100
L=2.
input_mean=5.
noise_sigma_x=0.3
noise_sigma_phi=0.1

mod_depth_params={
                                                                                                
   # general paramters           
   'n':n, 
   'nx':nx,
   'L':L,
   
   'jitter_variance':0.,
   'jitter_sigma_phi':0.,
   
    # inputs parameters
   'inputs_type':InputType.INPUT_NOISY_GRID,
   'input_mean':5.,        
   'inputs_seed':1,
   'grid_T':0.5, 
   'grid_angle':0.,
   'signal_weight':1.,
   
   'grid_T_sigma':0.,
   'grid_angle_sigma':0.,
     
   'noise_sigma_x': noise_sigma_x, 
   'noise_sigma_phi':noise_sigma_phi,
   
   'same_fixed_norm':False,  
   'fixed_norm':6.,
   
   'zero_phase':True,
   'scale_field_size_ratio':0.3  
   }

inputs=SpatialInputs(mod_depth_params)

#%%

### GENERATE GRIDS WITH DIFFERENT MODULATION DEPTHS
mod_depth_scaled_inputs=inputs.inputs_flat*np.linspace(0.1,1,n**2)
mod_depth_scaled_inputs+=input_mean-mod_depth_scaled_inputs.mean(axis=0)
mod_depth_scaled_inputs=mod_depth_scaled_inputs.clip(min=0)

### COMPUTE SCORES WITH DIFFERENT MODULATION DEPTHS
mod_depth_grid_tuning=gl.comp_grid_tuning_index(L,nx,mod_depth_scaled_inputs)
mod_depth_scores,spacings=gl.gridness_evo(np.reshape(mod_depth_scaled_inputs,(nx,nx,n**2)),L/nx,num_steps=10)
mod_depth=mod_depth_scaled_inputs.max(axis=0)-mod_depth_scaled_inputs.min(axis=0)


#%%

### PLOT SCORES WITH DIFFERENT MODULATION DEPTHS  
plot_score_test(mod_depth_scaled_inputs,mod_depth,mod_depth_scores,mod_depth_grid_tuning,'Modulation depth [spike/s]',tuning_index_lims=[-0.05,1.05])
fname = 'fig2d_grid_score_mod_depth'
pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])

    
#%%

### GENERATE GRIDS WITH DIFFERENT TUNING STRENGTHS

xi=gen_corr_noise_2d(L,nx,n,inputs.inputs_flat.var(),0.1,0.3)
signal_weights=np.linspace(0.1,1,n**2)
noisy_grids=np.zeros_like(inputs.inputs_flat)

for idx,signal_weight in enumerate(signal_weights):
  grid=inputs.inputs_flat[:,idx]
  c=0
  while (True):
    noisy_grid=(grid*signal_weight+xi[:,idx]*(1-signal_weight)+c).clip(min=0)
    if noisy_grid.mean()>=input_mean:
      break
    c+=0.1
  noisy_grids[:,idx]=noisy_grid
  
  
### COMPUTE SCORES WITH DIFFERENT TUNING STRENGTHS
signal_weight_grid_tuning=gl.comp_grid_tuning_index(L,nx,noisy_grids)
signal_weight_scores,spacings=gl.gridness_evo(np.reshape(noisy_grids,(nx,nx,n**2)),L/nx,num_steps=10)

#%%

### PLOT SCORES WITH DIFFERENT TUNING STRENGTHS
plot_score_test(noisy_grids,signal_weights,signal_weight_scores,signal_weight_grid_tuning,'Tuning strength',same_vmax=False,tuning_index_lims=[-0.05,1.05])
fname = 'fig2d_grid_score_tuning_strength'
pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])



#%%

### GENERATE GRIDS WITH DIFFERENT SCALES 

n=20
nx=100
L=2.
input_mean=5.
noise_sigma_x=0.3
noise_sigma_phi=0.1

grid_T_min=0.5
grid_T_max=2.

scale_gradient_params={
                                                                                                
   # general paramters           
   'n':n, 
   'nx':nx,
   'L':L,
   
   'jitter_variance':0.,
   'jitter_sigma_phi':0.,
   
    # inputs parameters
   'inputs_type':InputType.INPUT_NOISY_GRID_SCALEGRADIENT,
   'input_mean':5.,        
   'inputs_seed':1,
   'grid_T':0.5, 
   'grid_angle':0.,
   'signal_weight':1.,
     
   'noise_sigma_x': noise_sigma_x, 
   'noise_sigma_phi':noise_sigma_phi,
   
   'same_fixed_norm':False,  
   'fixed_norm':6.,
   
   'zero_phase':True,
   'scale_field_size_ratio':0.3,
   
   'grid_T_min':0.5,
   'grid_T_max':2.
   }

inputs_scales=SpatialInputs(scale_gradient_params)

### COMPUTE SCORES WITH DIFFERENT SCALES
grid_tuning_scales=gl.comp_grid_tuning_index(L,nx,inputs_scales.inputs_flat.reshape(nx**2,n**2),verbose=False,do_plot=False,warnings=True)
grid_scores_scales,est_spacings=gl.gridness_evo(np.reshape(inputs_scales.inputs_flat,(nx,nx,n**2)),L/nx,num_steps=10)


#%%
### PLOT SCORES WITH DIFFERENT SCALES
plot_score_test(inputs_scales.inputs_flat,inputs_scales.grid_T_vect/L,grid_scores_scales,grid_tuning_scales,'Grid spacing relative to arena size',tuning_index_lims=[-0.05,1.05])
fname = 'fig2d_grid_score_scales'
pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])

#%%

### GENERATE GRIDS WITH DIFFERENT FIELD-SIZE RATIOS 

n=20
nx=100
L=2.
input_mean=5.
noise_sigma_x=0.3
noise_sigma_phi=0.1

field_size_params={
                                                                                                
   # general paramters           
   'n':n, 
   'nx':nx,
   'L':L,
   
   'jitter_variance':0.,
   'jitter_sigma_phi':0.,
   
    # inputs parameters
   'inputs_type':InputType.INPUT_NOISY_GRID_FIELD_SIZE_GRADIENT,
   'input_mean':5.,        
   'inputs_seed':1,
   'grid_T':0.5, 
   'grid_angle':0.,
   'signal_weight':1.,
     
   'noise_sigma_x': noise_sigma_x, 
   'noise_sigma_phi':noise_sigma_phi,
   
   'same_fixed_norm':False,  
   'fixed_norm':6.,
   
   'zero_phase':True,
   
   'field_size_ratio_min':0.1,
   'field_size_ratio_max':0.35,
   
   'grid_T_sigma':0.,
   'grid_angle_sigma':0.,
   }

inputs_fild_sizes=SpatialInputs(field_size_params)


#%%
### COMPUTE SCORES WITH DIFFERENT FIELD SIZES
grid_tuning_field_sizes=gl.comp_grid_tuning_index(L,nx,inputs_fild_sizes.inputs_flat.reshape(nx**2,n**2),verbose=False,do_plot=False,warnings=True)
grid_scores_field_sizes,est_spacings=gl.gridness_evo(np.reshape(inputs_fild_sizes.inputs_flat,(nx,nx,n**2)),L/nx,num_steps=10)

#%%

### PLOT SCORES WITH DIFFERENT FIELD SIZES
plot_score_test(inputs_fild_sizes.inputs_flat,inputs_fild_sizes.scale_field_size_ratio_vect,grid_scores_field_sizes,grid_tuning_field_sizes,'Field size relative to grid spacing',same_vmax=False,tuning_index_lims=[-0.05,1.05])
fname = 'fig2d_grid_score_field_sizes'
pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])

#%%

dataPath=apm.batch_uniform_angles_learn.get_data_paths()[9]
data=np.load(dataPath,allow_pickle=True)
paramMap=data['paramMap'].flat[0]



#sim_conn=RecAmp2PopLearn(apm.def_recamp_learn_params)
sim_conn=RecAmp2PopLearn(paramMap)

sim_conn.post_init()
sim_conn.load_learned_recurrent_weights()
sim_conn.plot_recurrent_connectivity()


angle_diff=np.remainder(sim_conn.inputs.grid_angle_vect-sim_conn.inputs.grid_angle_vect.T,np.pi/3)
angle_diff[angle_diff>np.pi/6]-=np.pi/3

phase_dist=np.zeros((sim_conn.n_e**2,sim_conn.n_e**2))
for idx,phase in enumerate(sim_conn.gp.phases):
  phase_dist[idx,:]=gl.get_periodic_dist_on_rhombus(sim_conn.n_e,phase,sim_conn.gp.phases,sim_conn.gp.u1,sim_conn.gp.u2)






#%%
 
phase_idx=gl.get_pos_idx([0,0],sim_conn.gp.phases)
pl.figure(figsize=(8,4))
pl.subplots_adjust(wspace=0.4,bottom=0.2)
pl.subplot(121,aspect='equal')
pp.plot_on_rhombus(sim_conn.gp.R_T,1,0,sim_conn.N_e,sim_conn.gp.phases,
                                phase_dist[phase_idx,:],plot_axes=False,plot_rhombus=False,
                                plot_cbar=True)

pl.subplot(122)
pl.hist(phase_dist[phase_idx,:])
pl.xlabel('Phase distance')



 #%%

weights=sim_conn.W_ee0
weights=sim_conn.W_ee


fig=pl.figure(figsize=(12,7))


#pl.subplots_adjust(wspace=0.5,bottom=0.25)

gs = pl.GridSpec(2,12,hspace=1.,wspace=0.4,left=0.1,right=1)
  
fig.add_subplot(gs[0,0:3])
pl.hist(sim_conn.inputs.grid_angle_vect*180/np.pi,weights=sim_conn.N_e*[1./sim_conn.N_e,],bins=30)
pl.xlabel('Grid angle [deg]')
pl.ylabel('Fraction of inputs')
pl.xlim(-30,30)
pl.xticks([-30,-15,0,15,30])
pp.custom_axes()

fig.add_subplot(gs[0,4:7])
nbins=100
n,bins_angle=pl.histogram(angle_diff.ravel()*180/np.pi,bins=nbins)
nw,bins_angle=pl.histogram(angle_diff.ravel()*180/np.pi,weights=weights.ravel(),bins=nbins)
pl.bar(bins_angle[0:-1],nw/n)
pl.xlabel('Angle difference [deg]')
pl.ylabel('Mean synaptic weight')
pl.xticks([-30,-15,0,15,30])
pp.custom_axes()

fig.add_subplot(gs[0,8:11])
n,bins_phase=pl.histogram(phase_dist.ravel(),bins=30)
nw,bins_phase=pl.histogram(phase_dist.ravel(),weights=weights.ravel(),bins=30)
pl.bar(bins_phase[:-1],nw/n,width=0.01)
pl.xlabel('Phase distance')
pl.ylabel('Mean synaptic weight')
pp.custom_axes()

idxs_angle=np.argsort(sim_conn.inputs.grid_angle_vect[:,0])
sorted_W_angle=weights.copy()
sorted_W_angle=sorted_W_angle[idxs_angle,:]
sorted_W_angle=sorted_W_angle[:,idxs_angle]

fig.add_subplot(gs[1,0:3],aspect='equal')
pl.pcolormesh(sorted_W_angle,cmap='binary')
pp.custom_axes()
pl.title('Synaptic weights\ninputs sorted by orientation\n')
pl.xlabel('Input neuron index')
pl.ylabel('Input neuron index')
pp.colorbar()

fig.add_subplot(gs[1,3:9],aspect='equal')
pl.pcolormesh(weights,cmap='binary')
pp.custom_axes()
pl.title('Synaptic weights\ninputs sorted by phase\n')
pl.xlabel('Input neuron index')
pl.ylabel('Input neuron index')
pp.colorbar()

#%%

## COMPUTE OUTPUT
sim=RecAmp2PopSteady(sl.map_merge(apm.def_recamp_steady_params,{   'inputs_type':InputType.INPUT_NOISY_GRID_UNIFORM_ANGLES,'signal_weight':1.,
},{'recurrent_weights_path':sim_conn.data_path}))

sim.post_init()
sim.load_weights_from_data_path(sim.recurrent_weights_path)
sim.compute_and_save_steady_output()
sim.load_steady_outputs()

#sim.plot_example_outputs()
#sim.comp_amplification_index()

#%%
pl.figure()
pl.hist(sim.grid_tuning_in)
pl.hist(sim.grid_tuning_out)

#%%

out_angles=np.zeros(sim.N_e)
for out_idx in xrange(sim.N_e):
  grid=sim.r[out_idx,:].reshape(sim.nx,sim.nx)
  
  cx=gl.norm_autocorr(grid)
  est_angle,est_spacing=gl.get_grid_spacing_and_orientation(cx,float(sim.L)/sim.nx)
  out_angles[out_idx]=est_angle

##%%
#in_angles=np.zeros(sim.N_e)
#for in_idx in xrange(sim.N_e):
#  grid=sim.inputs.inputs_flat[:,in_idx].reshape(sim.nx,sim.nx)
#  
#  cx=gl.norm_autocorr(grid)
#  est_angle,est_spacing=gl.get_grid_spacing_and_orientation(cx,float(sim.L)/sim.nx)
#  in_angles[in_idx]=est_angle

#%%

out_angles=out_angles[~np.isnan(out_angles)]
out_angles[out_angles>np.pi/3]-=np.pi/3
out_angles[out_angles>np.pi/6]-=np.pi/3


in_angles=sim.inputs.grid_angle_vect
in_angles=in_angles[~np.isnan(in_angles)]
in_angles[in_angles>np.pi/3]-=np.pi/3
in_angles[in_angles>np.pi/6]-=np.pi/3

  
pl.figure(figsize=(10,5))
pl.subplot(121)
pl.hist(in_angles*180/np.pi,bins=20)
pl.title('Input')
pl.xlabel('Grid angle [deg]')
pl.subplot(122)
pl.title('Output')
pl.hist(out_angles*180/np.pi,bins=20)
pl.xlabel('Grid angle [deg]')