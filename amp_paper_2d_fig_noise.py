#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 10:32:19 2019

@author: dalbis
"""

import pylab as pl
import numpy as np

import grid_utils.plotlib as pp
import grid_utils.gridlib as gl
import grid_utils.simlib as sl
import amp_paper_2d_main as apm

from grid_utils.spatial_inputs import SpatialInputs
from recamp_2pop import RecAmp2PopLearn,RecAmp2PopSteady
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from collections import namedtuple

# color of the feed-forward input
input_color='m'

# extracts an array from a dataframe grouping by one column
get_array_from_df= lambda  df,group_col,sel_col:  np.array(df.groupby([group_col])[sel_col].apply(lambda x: x.values.tolist()).tolist())


def get_percs(data,axis,interpolation='linear'):
  """
  Computes distribution median and percentiles
  """

  med=np.median(data,axis=axis)
  p25=np.percentile(data,25,axis=axis,interpolation=interpolation)
  p75=np.percentile(data,75,axis=axis,interpolation=interpolation)
  
  Dist=namedtuple('Dist',['med','p25','p75'])
  return Dist(med,p25,p75)

def plot_sample_maps(batch,param_name,plot_values,map_type,plot_type,inputs_seed=1):
  """
  Plots example input maps (in space or phase) for different values of 'param_name'.
  The paramater values for the plots are given in 'plot_values'
  """
  
  assert map_type in ('noise', 'input' ,'output')
  assert plot_type in ('cell', 'pop')
    
  if plot_type == 'pop':  
    pl.figure(figsize=(6,2))
    pl.subplots_adjust(wspace=0.25)
  else:
    pl.figure(figsize=(5,2))
  
  x_position=[0.0,0.0]
  phase=[0.,0.]

  plot_idx=1    
  for plot_value in plot_values:
    
    pl.subplot(1,len(plot_values),plot_idx,aspect='equal')
    sim=RecAmp2PopSteady(sl.map_merge(batch.batch_default_map,{param_name:plot_value,'inputs_seed':inputs_seed}))
    sim.post_init()  
    sim.load_steady_outputs()
      
    if map_type == 'noise':
      inputs=sim.inputs.noise_flat
    elif map_type == 'input':
      inputs=sim.inputs_flat
    elif map_type == 'output':
      inputs=(sim.r[0:sim.n_e**2,:]).T
      
    vmin=0
    if plot_type == 'pop':
      x,img=pp.plot_population_activity(inputs.T,sim.L,x_position,vmin=vmin)
      #pl.title('%.1f'%plot_value,fontsize=11)
   
    else:
      phase_idx=gl.get_pos_idx(phase,sim.gp.phases)
      pl.pcolormesh(inputs[:,phase_idx].reshape(sim.nx,sim.nx),vmin=vmin,rasterized=True)
      pp.noframe()    
      if map_type=='input':
        pl.title('%.2f'%sim.grid_tuning_in[phase_idx],fontsize=9)
      else:
        pl.title('%.2f'%sim.grid_tuning_out[phase_idx],fontsize=9)
            
    plot_idx+=1



#%%
      
### **** AMPLIFICATION ACROSS SIGNAL STRENGTHS WITH FIXED WEIGHTS
      
batch_sig=apm.batch_signal_weight
batch_sig.post_init()

batch_sig.merge(['grid_tuning_in','grid_tuning_out','grid_amp_index',
                 ],force=True)
  
grid_tuning_in=np.stack(np.array(batch_sig.df['grid_tuning_in']))
grid_tuning_out=np.stack(np.array(batch_sig.df['grid_tuning_out']))
grid_amp_index=np.stack(np.array(batch_sig.df['grid_amp_index']))

dist_tuning_in=get_percs(grid_tuning_in,1)
dist_tuning_out=get_percs(grid_tuning_out,1)
dist_ampindex=get_percs(grid_amp_index,1)
plot_values_beta=apm.signal_weight_ran[0:-1:6]

pl.figure(figsize=(8,2.7))
pl.subplots_adjust(left=0.1,bottom=0.3,right=0.95,wspace=0.6,hspace=0.9,top=0.95)


#------ PLOT INPUT/OUTPUT INDEXES ---------------------------------------------

pl.subplot(1,2,1)
pl.fill_between(apm.signal_weight_ran,dist_tuning_in.p25,dist_tuning_in.p75,facecolor=input_color,alpha=0.2, edgecolor=None,linewidth=0)
pl.plot(apm.signal_weight_ran,dist_tuning_in.med,color=input_color,lw=2,label='Input')

pl.fill_between(apm.signal_weight_ran,dist_tuning_out.p25,dist_tuning_out.p75,color='k',alpha=0.2,edgecolor=None,linewidth=0)
pl.plot(apm.signal_weight_ran,dist_tuning_out.med,color='k',lw=2,label='Output')

pp.custom_axes()
pl.xlabel('Input-tuning strength $\\beta $')
pl.ylabel('Grid tuning index')

pl.axvline(apm.def_recamp_params['signal_weight'],color='k',ls=pp.linestyles['dashed'])

# mark plot values
ax=pl.gca()
ymin,ymax=ax.get_ylim()
pl.plot(plot_values_beta,np.ones_like(plot_values_beta)*ymin,'vk')

#------ PLOT INDEX RATIO   ----------------------------------------------------

pl.subplot(1,2,2)
pl.fill_between(apm.signal_weight_ran,dist_ampindex.p25,dist_ampindex.p75,facecolor='k',alpha=0.2, edgecolor=None,linewidth=0)
pl.plot(apm.signal_weight_ran,dist_ampindex.med,color='k',lw=2,label='Fixed weights')

pp.custom_axes()
pl.yticks(np.arange(1,9))
pl.xlabel('Input-tuning strength  $\\beta$ ')
pl.ylabel('Grid amplification index')
pl.axvline(apm.def_recamp_params['signal_weight'],color='k',ls=pp.linestyles['dashed'])

fname = 'fig2d_input_tuning_signal_strength_fixed_raw' 
pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])

#%%
#------ FIRING RATE MAPS EXAMPLES ---------------------------------------------
plot_sample_maps(batch_sig,'signal_weight',plot_values_beta,'input','cell')

fname = 'fig2d_input_tuning_noise_fixed_weights_signal_weight_input_examples' 
pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])


plot_sample_maps(batch_sig,'signal_weight',plot_values_beta,'output','cell')
fname = 'fig2d_input_tuning_noise_fixed_weights_signal_weight_output_examples' 
pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])


#%%
#------ PLOT EXAMPLE CONNECTIVITIES LEARNED WITH DIFFERENT INPUT TUNING STRENGTHS  ------------------------------------------

for plot_idx,plot_value in enumerate(plot_values_beta):
  conn=RecAmp2PopLearn(sl.map_merge(apm.def_recamp_learn_params,{'inputs_seed':1,'signal_weight':plot_value}))
  conn.post_init()  
  conn.load_learned_recurrent_weights()
  pp.plot_recurrent_weights(conn.W_ee,conn.gp,vmax=conn.W_max_ee,ms=0,figsize=(2,2)) 
  pp.save_fig(sl.get_figures_path(),'fig2d_tuning_learning_conn_example_%d'%plot_idx,exts=['png','svg'])
  

#%%

###  **** LEARNING WEIGHTS WITH DIFFERENT SIGNAL STRENGTH AND PERFORMANCES WITH ALL CONNECTIVITIES

plot_values_beta=apm.signal_weight_ran[0:-1:6]


conn_batch_sig=apm.batch_signal_weight_learn
conn_batch_sig.post_init()
conn_batch_sig.merge(['W_ee','conn_tuning_index','conn_trans_index'],force=True)
conn_index_sig=get_array_from_df(conn_batch_sig.df,'signal_weight','conn_tuning_index')


# indexes to sort the learned connectivity by signal weight (otherwise sorted by hash string)
sort_idxs=np.argsort(conn_batch_sig.hashes)
inv_sort_idxs=np.argsort(sort_idxs)

# batch in which we test performances for all connectivites and all signal weights
batch_sig_all_weights=apm.batch_signal_weight_all_learned_weights
batch_sig_all_weights.post_init()
batch_sig_all_weights.merge(['grid_tuning_in','grid_tuning_out','grid_amp_index'],force=True)

# collect amp indexes grouped by connectivity and sort them accorting to the signal weight during learning
grid_amp_index=get_array_from_df(batch_sig_all_weights.df,'recurrent_weights_path','grid_amp_index')
grid_amp_index=np.mean(grid_amp_index,axis=2)
grid_amp_index=grid_amp_index[inv_sort_idxs,:]


# collect grid-tuning out grouped by connectivity and sort them accorting to the signal weight during learning
grid_out_index=get_array_from_df(batch_sig_all_weights.df,'recurrent_weights_path','grid_tuning_out')
grid_out_index=np.mean(grid_out_index,axis=2)
grid_out_index=grid_out_index[inv_sort_idxs,:]



# collect grid-tuning out grouped by connectivity and sort them accorting to the signal weight during learning
grid_in_index=get_array_from_df(batch_sig_all_weights.df,'recurrent_weights_path','grid_tuning_in')
grid_in_index=np.mean(grid_in_index,axis=2)
grid_in_index=grid_in_index[inv_sort_idxs,:]


# find the index of the default signal weight and of the resulting connectivity tuning index
def_sig_weight_index=np.argmin(np.abs(apm.signal_weight_ran-apm.def_recamp_params['signal_weight']))
def_conn_index=conn_index_sig[def_sig_weight_index]

fig=pl.figure(figsize=(8,3.5))
pl.subplots_adjust(left=0.1,bottom=0.3,right=0.98,wspace=0.5,hspace=0.9,top=0.9)


#------ PLOT CONNECTIVITY TUNING INDEX ---------------------------------------

pl.subplot(1,2,1)
pl.plot(apm.signal_weight_ran,conn_index_sig,color='k',lw=2,label='Output')
pl.ylim(0,1)

# mark plot values
ax=pl.gca()
ymin,ymax=ax.get_ylim()
pl.plot(plot_values_beta,np.ones_like(plot_values_beta)*ymin+0.025,'vk')

pl.axvline(apm.def_recamp_params['signal_weight'],color='k',ls=pp.linestyles['dashed'])
pl.axhline(def_conn_index,color='k',ls=pp.linestyles['dashed'])

pl.xlabel('Input-tuning strength $\\beta$\n(during learning)')
pl.ylabel('Connectivity tuning index')
pp.custom_axes()


#------ PLOT GRID AMP INDEX COLORMAP ------------------------------------------

ax=pl.subplot(1,2,2)
#levels=np.linspace(0,0.7,6)
CS=pl.contourf(apm.signal_weight_ran,apm.signal_weight_ran,grid_amp_index.T,
            cmap='viridis',#pp.get_cmap_from_color('k'),
            )
            #levels=levels)

pl.plot(apm.signal_weight_ran,apm.signal_weight_ran,'-k')
pl.axvline(apm.def_recamp_params['signal_weight'],color='k',ls=pp.linestyles['dashed'])
pl.axhline(apm.def_recamp_params['signal_weight'],color='k',ls=pp.linestyles['dashed'])

pp.custom_axes()

pl.xlim(0,1)
pl.ylim(0,1)

pl.xticks(np.arange(0,1.2,0.25))
pl.yticks(np.arange(0,1.2,0.25))
pl.xlabel('Input-tuning strength $\\beta$\n(during learning)')
pl.ylabel('Input-tuning strength $\\beta$\n(after learning)')

pl.colorbar()



fname = 'fig2d_tuning_learning_signal_strength_conn_raw'   
pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])



#%%
#### ALTERNATIVE PLOT: OUTPUT GRID-TUNING AND AMPLIFICATION INDEXES FOR DIFFERENT CONNECTIVITIES 1D PLOTS

grid_out_index=get_array_from_df(batch_sig_all_weights.df,'recurrent_weights_path','grid_tuning_out')
grid_out_index=np.median(grid_out_index,axis=2)
grid_out_index=grid_out_index[inv_sort_idxs,:]


sel_betas_idxs=[0,6,8,9,10,11,29]
sel_betas=apm.signal_weight_ran[sel_betas_idxs]


x_idx=np.argmin(np.abs(apm.signal_weight_ran-0.4))

fig=pl.figure(figsize=(8,3.5))
pl.subplots_adjust(left=0.1,bottom=0.3,right=0.98,wspace=0.5,hspace=0.9,top=0.9)

pl.subplot(121)
pl.plot(apm.signal_weight_ran,dist_tuning_in.med,color=input_color,lw=2,label='Input')
pl.plot(apm.signal_weight_ran,grid_out_index[sel_betas_idxs,:].T,'-k',lw=1)


for sel_beta,sel_beta_idx in zip(sel_betas,sel_betas_idxs):
  pl.annotate('$%.2f$'%sel_beta,(0.4,grid_out_index[sel_beta_idx,x_idx]),fontsize=9)
  
pp.custom_axes()
pl.xlabel('Input-tuning strength $\\beta$  \n(after learning)')  
pl.ylabel('Grid tuning index')  


x_idx=np.argmin(np.abs(apm.signal_weight_ran-0.25))

pl.subplot(122)
pl.plot(apm.signal_weight_ran,grid_amp_index[sel_betas_idxs,:].T,'-k',lw=1)


for sel_beta,sel_beta_idx in zip(sel_betas,sel_betas_idxs):
  pl.annotate('$%.2f$'%sel_beta,(0.25,grid_amp_index[sel_beta_idx,x_idx]),fontsize=9)
  
pp.custom_axes()
pl.xlabel('Input-tuning strength $\\beta$  \n(after learning)')  
pl.ylabel('Grid amplification index')  



#%%

### **** AMPLIFICATION ACROSS SIGMA PHI WITH FIXED WEIGHTS
  
# batch for different levels of noise_sigma_phi and for 10 different realizations of the inputs
batch_phi=apm.batch_noise_phi_input_seed
batch_phi.post_init()

batch_phi.merge(['grid_tuning_in','grid_tuning_out','grid_amp_index' ],force=True)
  
# get basic parameter values  
n=apm.def_recamp_params['n_e']
num_sigmas=len(apm.noise_sigma_phi_ran)
num_seeds=len(batch_phi.batch_override_map['inputs_seed'])
nx=batch_phi.batch_default_map['nx']
L=batch_phi.batch_default_map['L']

# get tuning indexes 
grid_tuning_in=get_array_from_df(batch_phi.df,'noise_sigma_phi','grid_tuning_in').reshape(num_sigmas,n**2*num_seeds)
grid_tuning_out=get_array_from_df(batch_phi.df,'noise_sigma_phi','grid_tuning_out').reshape(num_sigmas,n**2*num_seeds)
grid_amp_index=get_array_from_df(batch_phi.df,'noise_sigma_phi','grid_amp_index').reshape(num_sigmas,n**2*num_seeds)

# compute distributions
dist_tuning_in=get_percs(grid_tuning_in,1)
dist_tuning_out=get_percs(grid_tuning_out,1)
dist_ampindex=get_percs(grid_amp_index,1)

# values to which we show example maps
plot_values_phi=batch_phi.batch_override_map['noise_sigma_phi'][0:-1:6]

# compute input power in phase (for the inset)  
H1_phi=[]
for sigma_phi in batch_phi.batch_override_map['noise_sigma_phi']:
  parMap=sl.map_merge(batch_phi.batch_default_map,{'noise_sigma_phi':sigma_phi,
                                                   'n':apm.def_recamp_params['n_e']})
  spat_inputs = SpatialInputs(parMap)
  hran,mean_pw=gl.get_mean_pw_on_phase_rhombus(n,nx,L,spat_inputs.noise_flat.T,max_harmonic=5)
  H1_pw=(mean_pw[hran==0,hran==1]+mean_pw[hran==1,hran==0]+mean_pw[hran==1,hran==1])/3.
  H1_phi.append(H1_pw)
  
# sigma_phi at which the power at the first harmonic is maximal  
sigma_max=apm.noise_sigma_phi_ran[np.argmax(H1_phi)]

#%%
pl.figure(figsize=(8.,3))
pl.subplots_adjust(left=0.1,bottom=0.3,right=0.9,wspace=0.4,hspace=0.9,top=0.95)


#------ PLOT INPUT/OUTPUT INDEXES ---------------------------------------------

pl.subplot(1,2,1)
pl.fill_between(apm.noise_sigma_phi_ran,dist_tuning_in.p25,dist_tuning_in.p75,facecolor=input_color,alpha=0.2, edgecolor=None,linewidth=0)
pl.fill_between(apm.noise_sigma_phi_ran,dist_tuning_out.p25,dist_tuning_out.p75,color='k',alpha=0.2,edgecolor=None,linewidth=0)

pl.axvline(sigma_max,color='k',lw=1.5,ls=pp.linestyles['densely dotted'])
pl.plot(apm.noise_sigma_phi_ran,dist_tuning_in.med,color=input_color,lw=2,label='Input')
pl.plot(apm.noise_sigma_phi_ran,dist_tuning_out.med,color='k',lw=2,label='Output')

pp.custom_axes()
pl.xlabel('Noise correlation length $\sigma_\\vec{\\varphi}$\n(after learning)')
pl.ylabel('Grid tuning index')

pl.axvline(apm.def_recamp_params['noise_sigma_phi'],color='k',ls=pp.linestyles['dashed'])
ax=pl.gca()
ax.set_xscale('log')

# mark plot values
ax=pl.gca()
ymin,ymax=ax.get_ylim()
pl.plot(plot_values_phi,np.ones_like(plot_values_phi)*ymin,'vk')


#------ PLOT INDEX RATIO   ----------------------------------------------------

pl.subplot(1,2,2)
pl.fill_between(apm.noise_sigma_phi_ran,dist_ampindex.p25,dist_ampindex.p75,facecolor='k',alpha=0.2, edgecolor=None,linewidth=0)

pl.axvline(sigma_max,color='k',lw=1.5,ls=pp.linestyles['densely dotted'])
pl.plot(apm.noise_sigma_phi_ran,dist_ampindex.med,color='k',lw=2)

pp.custom_axes()
pl.xlabel('Noise correlation length $\sigma_\\vec{\\varphi}$\n(after learning)')
pl.ylabel('Grid amplification index')
pl.axvline(apm.def_recamp_params['noise_sigma_phi'],color='k',ls=pp.linestyles['dashed'])

ax=pl.gca()
ax.set_xscale('log')

#------ INSET ----------------------------------------------------------------

axins = inset_axes(ax, width=1.15, height=0.7)
pl.axvline(sigma_max,color='k',lw=1.5,ls=pp.linestyles['densely dotted'])
pl.plot(apm.noise_sigma_phi_ran,H1_phi,color='k',lw=2)
axins.set_xscale('log')
axins.xaxis.set_tick_params(labelsize=9)
axins.yaxis.set_tick_params(labelsize=9)
pl.ylabel('Noise power\n at $\mathcal{H}_1$ [1/s$^2$]',fontsize=9)
pl.xlabel('$\sigma_\\vec{\\varphi}$',fontsize=9)
pp.custom_axes()

fname = 'fig2d_correlations_noise_phi_fixed_raw' 
pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])

#%%

#------ FIRING RATE MAPS EXAMPLES ---------------------------------------------
    

plot_sample_maps(batch_phi,'noise_sigma_phi',plot_values_phi,'input','pop',inputs_seed=0)
fname = 'fig2d_correlations_noise_sigma_phi_input_examples' 
pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])

#%%
plot_sample_maps(batch_phi,'noise_sigma_phi',plot_values_phi,'output','pop',inputs_seed=0)
fname = 'fig2d_correlations_noise_sigma_phi_output_examples' 
pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])

#%%

###  **** LEARNING WEIGHTS WITH DIFFERENT SIGMA PHI AND PERFORMANCES WITH ALL CONNECTIVITIES

# learning connectivity weights batch (with signal, one input seed)
conn_batch_phi=apm.batch_sigma_phi_learn
conn_batch_phi.post_init()
conn_batch_phi.merge(['conn_tuning_index'],force=True)
conn_index_phi=get_array_from_df(conn_batch_phi.df,'noise_sigma_phi','conn_tuning_index')

# learning connectivity weights batch (no signal, one input seed)
conn_batch_phi_no_signal=apm.batch_sigma_phi_learn_no_signal
conn_batch_phi_no_signal.post_init()
conn_batch_phi_no_signal.merge(['conn_tuning_index'],force=True)
conn_index_phi_no_signal=get_array_from_df(conn_batch_phi_no_signal.df,'noise_sigma_phi','conn_tuning_index')


# indexes to sort the learned connectivity by sigma phi (otherwise sorted by hash string)
sort_idxs=np.argsort(apm.batch_sigma_phi_learn.hashes)
inv_sort_idxs=np.argsort(sort_idxs)


# batches with all connectivities and all values of noise_sigma_phi (one batch per input realizations, total=10)
amp_index_seed_list=[]
grid_out_index_seed=[]

for batch in apm.batches_noise_phi_all_learned_weights: 
  batch.post_init()
  batch.merge(['grid_amp_index','grid_tuning_out'],force=True)
  
  grid_amp_index=get_array_from_df(batch.df,'recurrent_weights_path','grid_amp_index')
  grid_amp_index=np.median(grid_amp_index,axis=2)
  grid_amp_index=grid_amp_index[inv_sort_idxs,:]
  amp_index_seed_list.append(grid_amp_index)   

  grid_out_index=get_array_from_df(batch.df,'recurrent_weights_path','grid_tuning_out')
  grid_out_index=np.median(grid_out_index,axis=2)
  grid_out_index=grid_out_index[inv_sort_idxs,:]
  grid_out_index_seed.append(grid_out_index)   


# compute the grand median of the data  
grid_amp_index_seed_mean=np.mean(np.array(amp_index_seed_list),axis=0)
grid_out_index_seed_mean=np.mean(np.array(grid_out_index_seed),axis=0)

# values to which we show example maps
plot_values_phi=apm.noise_sigma_phi_ran[0:-1:6]


# default values
def_sigma_phi=apm.noise_sigma_phi_ran[0]
def_conn_index=conn_index_phi[0]

fig=pl.figure(figsize=(8,3.5))
pl.subplots_adjust(left=0.08,bottom=0.3,right=0.98,wspace=0.5,hspace=0.9,top=0.9)


#------ PLOT CONNECTIVITY TUNING INDEX ---------------------------------------

pl.subplot(1,2,1)

pl.axvline(sigma_max,color='k',lw=1.5,ls=pp.linestyles['densely dotted'])
pl.plot(apm.noise_sigma_phi_ran,conn_index_phi_no_signal,color='gray',lw=2)
pl.plot(apm.noise_sigma_phi_ran,conn_index_phi,color='k',lw=2)
pl.ylim(0,1.)


pl.axvline(apm.def_recamp_params['noise_sigma_phi'],color='k',ls=pp.linestyles['dashed'])
pl.axhline(def_conn_index,color='k',ls=pp.linestyles['dashed'])



pl.xlabel('Noise correlation length $\sigma_\\vec{\\varphi}$\n(during learning)')
pl.ylabel('Connectivity tuning index')

pp.custom_axes()

ax=pl.gca()
ax.set_xscale('log')

# mark plot values
ax=pl.gca()
ymin,ymax=ax.get_ylim()
pl.plot(plot_values_phi,np.ones_like(plot_values_phi)*ymin+0.025,'vk')


#------ PLOT GRID AMP INDEX COLORMAP ------------------------------------------
 
pl.subplot(1,2,2)

y_conn =False


CS=pl.contourf(apm.noise_sigma_phi_ran,apm.noise_sigma_phi_ran,grid_amp_index_seed_mean.T,
             cmap='viridis')

pl.plot(apm.noise_sigma_phi_ran,apm.noise_sigma_phi_ran,'-k')

ax=pl.gca()
ax.set_xscale('log')
ax.set_yscale('log')

pl.colorbar()


pl.axvline(sigma_max,color='k',lw=1.5,ls=pp.linestyles['densely dotted'])
pl.axhline(sigma_max,color='k',lw=1.5,ls=pp.linestyles['densely dotted'])

pl.xlabel('Noise correlation length $\sigma_\\vec{\\varphi}$\n(during learning)')
pl.ylabel('Noise correlation length $\sigma_\\vec{\\varphi}$\n(after learning)')



fname = 'fig2d_corr_learning_sigma_phi_conn_raw'   
pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])

#%%

#------ PLOT EXAMPLE CONNECTIVITIES  ------------------------------------------


for plot_idx,plot_value in enumerate(plot_values_phi):
  conn=RecAmp2PopLearn(sl.map_merge(apm.def_recamp_learn_params,{'inputs_seed':1,
                                                                 'noise_sigma_phi':plot_value,
                                                                 }))
  conn.post_init()  
  conn.load_learned_recurrent_weights()
  pp.plot_recurrent_weights(conn.W_ee,conn.gp,vmax=conn.W_max_ee,ms=0,figsize=(2,2))
 
  pp.save_fig(sl.get_figures_path(),'fig2d_corr_learning_conn_example_%d'%plot_idx,exts=['png','svg'])





