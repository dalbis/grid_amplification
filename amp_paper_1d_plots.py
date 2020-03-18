#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:25:37 2019

@author: dalbis
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 08:59:17 2018

@author: dalbis
"""

import numpy as np
import pylab as pl
import amp_paper_1d_plot_functions as pf
import grid_utils.plotlib as pp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from recamp_1d import RunMode,RecAmp1D
import simlib as sl

ms=4  # marker size
fs=11 # font size
pl.rc('font',size=fs)

save_figs=False


#%%-----------------------------------------------------------------------------
### PLOT INPUT EXAMPLES
#-------------------------------------------------------------------------------

ri=RecAmp1D(RunMode.RUN_MODE_INPUT_SAMPLES)

pl.figure(figsize=(3.5,3.5),facecolor='w')
pl.subplots_adjust(left=0.1,right=0.9,wspace=0.1,hspace=0.1,bottom=0.1,top=0.9)  
pl.subplot(111)
pl.xlim([-ri.L/2,ri.L/2])
pp.noframe()   

offset=4.
plot_B_idx=0
plot_sigma_x_idx=0
plot_sigma_phi_idx=0
plot_mc_sample_idx=0
plot_mc_sample_idx=0


phi_idxs=[0,50,100]

h = ri.h_mat[plot_B_idx,plot_sigma_phi_idx,plot_sigma_x_idx,plot_mc_sample_idx,:,:]
v = ri.v_mat[plot_B_idx,plot_sigma_phi_idx,plot_sigma_x_idx,plot_mc_sample_idx,:,:]


for idx,phi_idx in enumerate(phi_idxs):
  pl.plot(ri.x_ran-ri.L/2,ri.tuningx_fun(ri.x_ran-ri.L/2,ri.phi_ran[phi_idx],ri.B_ran[plot_B_idx])+offset*(10-idx),linewidth=1.5,color=pp.red,linestyle='-')
  pl.plot(ri.x_ran-ri.L/2,np.roll(h[phi_idx,:],ri.n_x/2)+offset*(10-idx),'-k',linewidth=1.5)

fname = 'fig1d_inputs'
if save_figs:
  pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])
  

#%%-----------------------------------------------------------------------------
### CONNECTIVITY FUNCTION
#-------------------------------------------------------------------------------

pl.figure(figsize=(3.,2.5))
pl.subplot(111)
pl.subplots_adjust(left=.3, bottom=0.2)
pp.custom_axes()
ax=pl.gca()
pp.adjust_spines(offset=10)
pp.set_axes_width(.7)
pp.set_tick_size(3)
pl.plot(ri.phi_ran-np.pi,np.roll(np.cos(ri.phi_ran),ri.n_phi/2),linewidth=2,color='k' )
pl.xticks(np.arange(-2*np.pi/2,2*np.pi*3/4.,2*np.pi/4),['-$\pi$','-$\pi/2$','0','$\pi/2$','$\pi$'])
pl.xlim([-np.pi,np.pi])

ax=pl.gca()
pl.xticks([-np.pi,0,np.pi],['-$\pi$','0','$\pi$'])
pl.yticks(np.arange(-1,2,1),['$\\frac{A^{-1}-1}{\\pi}$','0','$\\frac{1-A^{-1}}{\\pi}$'])

fname = 'fig1d_connectivity'
if save_figs:
  pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])

        

#%%-----------------------------------------------------------------------------
### EXPLAIN INPUT NOISE
#-------------------------------------------------------------------------------

rn=RecAmp1D(RunMode.RUN_MODE_INPUT_NOISE)

plot_mc_sample_idx=0
phi_idxs,step = np.linspace(0,len(rn.phi_ran)/2,6,retstep=True)
phi_idxs=phi_idxs.astype(int)

offset=5.5
ylims=np.array([-5,41])

pl.figure(figsize=(6.5,5.8),facecolor='w')
pl.subplots_adjust(left=0.1,right=0.9,wspace=0.15,hspace=0.1,bottom=0.1,top=0.9)  

center=offset*10/2

sigma_idx=0
for plot_sigma_phi_idx in np.flipud(xrange(len(rn.sigma_phi_ran))):
  for plot_sigma_x_idx in xrange(len(rn.sigma_x_ran)):
    pl.subplot(len(rn.sigma_phi_ran),len(rn.sigma_x_ran),sigma_idx+1)

    
    pl.xlim([-rn.L/2,rn.L/2])
    pl.ylim(ylims)
    
    span_phi=rn.sigma_phi_ran[plot_sigma_phi_idx]*3*(offset*10)/(np.pi/2.)
    
        
    pp.noframe()   
    xi = rn.xi_mat[plot_B_idx,plot_sigma_phi_idx,plot_sigma_x_idx,plot_mc_sample_idx,:,:]
    sigma_idx+=1  
      
    for idx,phi_idx in enumerate(phi_idxs):
      pl.plot(rn.x_ran-rn.L/2,offset*idx+np.roll(xi[phi_idx,:],rn.n_x/2),color=[0,0,0],lw=1) 

fname = 'fig1d_noise_raw'
if save_figs:
  pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])

#%%

## PLOT VON MISES CORRELATION FUNCTIONS

# correlation in phase
pl.figure(figsize=(4.8,.9),facecolor='w')
pl.subplots_adjust(left=0.1,right=0.9,wspace=0.8,hspace=0.1,bottom=0.1,top=0.9) 

plot_idx=1
for sigma_phi in rn.sigma_phi_ran:
  pl.subplot(1,3,plot_idx)
  
  pl.plot(rn.phi_ran-np.pi,rn.teo_xi_corr_phi(rn.phi_ran-np.pi,sigma_phi),'-k',lw=1.5)
  pl.xlim(-np.pi,np.pi)
  plot_idx+=1
  pp.noframe()  
  
fname = 'fig1d_noise_corr_phi_raw'
#pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])  


# correlation in space
pl.figure(figsize=(6.3,0.9),facecolor='w')
pl.subplots_adjust(left=0.1,right=0.9,wspace=0.4,hspace=0.1,bottom=0.1,top=0.9) 

plot_idx=1
for sigma_x in rn.sigma_x_ran:
  pl.subplot(1,3,plot_idx)
  
  pl.plot(rn.x_ran-rn.L/2.,rn.teo_xi_corr_x(rn.x_ran-rn.L/2.,sigma_x),'-k',lw=1.5)
  pl.xlim(-rn.L/2,rn.L/2)
  plot_idx+=1
  pp.noframe()  
  
fname = 'fig1d_noise_corr_x_raw'
if save_figs:
  pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])


#%%============================================================================
#### POPULATION AMPLIFICATION EXAMPLE 
#==============================================================================

rb=RecAmp1D(RunMode.RUN_MODE_BASE_CASE)


fig=pl.figure(figsize=(8,4.5),facecolor='w')
pl.subplots_adjust(left=0.08,right=0.99,wspace=0.5,hspace=0.9,bottom=0.1,top=0.92)  


pf.plot_amp_res(rb.phi_ran,rb.w_ran_phi,rb.n_phi,rb.dphi,
                rb.h_phi,rb.k_phi_est,rb.k_phi,rb.v_phi,
                rb.h_phi_pw,rb.h_phi_mean_pw,
                rb.k_phi_pw,rb.k_phi_mean_pw,
                rb.v_phi_pw,rb.v_phi_mean_pw,
                rb.teo_h_phi_pw,rb.teo_k_phi_pw,rb.teo_v_phi_pw,
                ran_ticks=[-np.pi,0,np.pi],ran_ticks_labels=['','0',''],delta_label='$1/\Delta\phi$',
                split_v=[0.1,0.2],
                ytick_bottom_h=0.1,
                ytick_top_k=6,ytick_bottom_k=1,yticks_k=[1,5,9],
                ytick_top_v=1,ytick_bottom_v=0.1,
                fig=fig,lw=1.2,
                ymax_v=3,
                ymax_h=0.3,
                tuning_curve=rb.tuning_fun(rb.phi_ran,rb.B),ms=ms
)

fname = 'fig1d_pop_raw'
if save_figs:
  pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])

#%%=============================================================================
#### SINGLE-NEURON AMPLIFICATION EXAMPLE  
#===============================================================================


fig=pl.figure(figsize=(8,4.5),facecolor='w')
pl.subplots_adjust(left=0.08,right=0.99,wspace=0.5,hspace=0.9,bottom=0.1,top=0.92)  


pf.plot_amp_res(rb.x_ran,rb.w_ran_x,rb.n_x,rb.dx,
                rb.h_x,rb.k_x_est,rb.k_x,rb.v_x,
                rb.h_x_pw,rb.h_x_mean_pw,
                rb.k_x_pw,rb.k_x_mean_pw,
                rb.v_x_pw,rb.v_x_mean_pw,
                rb.teo_h_x_pw,rb.teo_k_x_pw,rb.teo_v_x_pw,
                ran_ticks=np.arange(-2,3),ran_ticks_labels=np.arange(-2,3),delta_label='$1/\Delta{x}$',
                split_k=[2,4],
                split_v=[0.1,1],
                ymax_h=0.25,ymax_k=16,ymax_v=2,
                ytick_top_h=0.3,ytick_bottom_h=0.05,
                ytick_top_k=6,ytick_bottom_k=1,
                ytick_top_v=1,ytick_bottom_v=0.05,
                fig=fig,lw=1.5,tuning_curve=rb.tuningx_fun(rb.x_ran,0,rb.B),ms=ms)
                


fname = 'fig1d_cell_raw'
if save_figs:
  pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])

#%%===========================================================================
### GRID TUNING INDEX AS A FUNCTION OF INPUT-TUNING STRENGTH
#=============================================================================

for clip_rates in False,True:
  rt=RecAmp1D(RunMode.RUN_MODE_TUNING_STRENGTH,clip_rates)
  
  ms=3.5
  
  if clip_rates:
    ran=np.arange(0,len(rt.B_ran))
  else:
    ran=np.arange(0,len(rt.B_ran)-1)
  
  B_ran_cont = np.arange(0,.98,0.001)
    
  pl.figure(figsize=(8,3.2))
  pl.subplots_adjust(left=0.1,bottom=0.3,right=0.95,wspace=0.6,hspace=0.9,top=0.95)
  
  pl.rcParams['ytick.minor.size'] = 3
  pl.rcParams['ytick.major.size'] = 6
  
  pl.subplot(121)
  
  if clip_rates is False:
    pl.plot(B_ran_cont,rt.teo_1d_grid_tuning_in(B_ran_cont,rt.def_sigma_x,rt.variance),color='m')
    pl.plot(rt.B_ran[ran],rt.grid_index_in_mat[ran,0,0],marker='s',mfc='m',mec='m',lw=0,ms=ms,label='feed-forward input')
  else:
    pl.plot(rt.B_ran[ran],rt.grid_index_in_mat[ran,0,0],marker='s',mfc='m',mec='m',lw=0,ms=ms,label='feed-forward input')
  
  
  if clip_rates is False:
    pl.plot(B_ran_cont,rt.teo_1d_grid_tuning_out(B_ran_cont,rt.def_sigma_phi,rt.def_sigma_x,rt.variance),color='k')
    pl.plot(rt.B_ran[ran],rt.grid_index_out_mat[ran,0,0],marker='s',mfc='k',mec='k',lw=0,ms=ms, label='steady-state output')
  else:
    pl.plot(rt.B_ran[ran],rt.grid_index_out_mat[ran,0,0],marker='s',mfc='k',mec='k',lw=0,ms=ms, label='steady-state output')
    pl.yticks([0,0.2,0.4,0.6])
  pl.xlim(-0.05,1.05)
  
  pl.xlabel('Input tuning strength B')
  pl.ylabel('1D grid tuning index')
  
  if clip_rates is False:
    ax=pl.gca()
    ax.set_yscale('log')
  
  pp.custom_axes()
  pl.axvline(rt.def_B,color='k',ls=pp.linestyles['dashed'])
  
  
  pl.subplot(122)
  
  if clip_rates is False:
    pl.plot(B_ran_cont,rt.teo_1d_grid_tuning_out(B_ran_cont,rt.def_sigma_phi,rt.def_sigma_x,rt.variance)/rt.teo_1d_grid_tuning_in(B_ran_cont,rt.def_sigma_x,rt.variance),color='k')
    pl.plot(rt.B_ran[ran],rt.grid_index_out_mat[ran,0,0]/rt.grid_index_in_mat[ran,0,0],marker='s',mfc='k',mec='k',lw=0,ms=ms)
    pl.axhline(rt.A_pop/rt.A_noise_fun(rt.def_sigma_phi),color='k',ls=pp.linestyles['densely dotted'])
    pl.yticks(np.arange(1,9))
  else:
    pl.plot(rt.B_ran[ran],rt.grid_index_out_mat[ran,0,0]/rt.grid_index_in_mat[ran,0,0],marker='s',mfc='k',mec='k',lw=0,ms=ms)
    
  pl.xlim(-0.05,1.05)
  pl.xlabel('Input tuning strength B')
  pl.ylabel('1D grid amplification index')
  pp.custom_axes()
  pl.axvline(rt.def_B,color='k',ls=pp.linestyles['dashed'])
  
  if clip_rates is False:
    fname = 'fig1d_amp_input_tuning_linear'
  else:
    fname = 'fig1d_amp_input_tuning_clipped'
    
    if save_figs:
      pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg']) 




#%%===========================================================================
### GRID TUNING INDEX AS A FUNCTION NOISE CORRELATION LENGTH ACROSS NEURONS
#=============================================================================
for clip_rates in False,True:
  
  rp=RecAmp1D(RunMode.RUN_MODE_NOISE_NEURONS,clip_rates)

  ms=3.5
  
  sigma_phi_ran_cont=np.logspace(np.log10(0.02),np.log10(100),100)
  
  
  pl.figure(figsize=(8,3.2))
  pl.subplots_adjust(left=0.1,bottom=0.3,right=0.98,wspace=0.45,hspace=0.9,top=0.95)
  
  pl.rcParams['ytick.minor.size'] = 3
  pl.rcParams['ytick.major.size'] = 6
  pl.rcParams['xtick.minor.size'] = 3
  pl.rcParams['xtick.major.size'] = 6
  
  
  pl.subplot(121)
  
  ax=pl.gca()
  ax.set_xscale('log')
  
  # input
  if clip_rates is False:
    pl.plot(sigma_phi_ran_cont,rp.teo_1d_grid_tuning_in(rp.def_B,rp.def_sigma_x,rp.variance)*np.ones_like(sigma_phi_ran_cont),color='m')
    pl.plot(rp.sigma_phi_ran,rp.grid_index_in_mat[0,:,0],marker='s',mfc='m',mec='m',lw=0,ms=ms,label='feed-forward input')
    pl.ylim(0,55)
  
  else:
    pl.plot(rp.sigma_phi_ran,rp.grid_index_in_mat[0,:,0],marker='s',mfc='m',mec='m',lw=0,ms=ms,label='feed-forward input')
  
  # output
  if clip_rates is False:
    pl.plot(sigma_phi_ran_cont,rp.teo_1d_grid_tuning_out(rp.def_B,sigma_phi_ran_cont,rp.def_sigma_x,rp.variance),color='k')
    pl.plot(rp.sigma_phi_ran,rp.grid_index_out_mat[0,:,0],marker='s',mfc='k',mec='k',lw=0,ms=ms, label='steady-state output')
  else:
    pl.plot(rp.sigma_phi_ran,rp.grid_index_out_mat[0,:,0],marker='s',mfc='k',mec='k',lw=0,ms=ms, label='steady-state output')
    
  # default value
  pl.axvline(rp.def_sigma_phi,color='k',ls=pp.linestyles['dashed'])
  
  # theoretical value of the minimum
  pl.axvline(rp.sigma_phi_peak,color='k',lw=1.5,ls=pp.linestyles['densely dotted'])
  
  pl.xlabel('Noise correlation length $\\sigma_\\varphi$')
  pl.ylabel('1D grid tuning index')
  pp.custom_axes()
  pl.xlim(0.02,100)
  

  pl.subplot(122)
  
  ax=pl.gca()
  ax.set_xscale('log')
  
  if clip_rates is False:
    pl.plot(sigma_phi_ran_cont,rp.teo_1d_grid_tuning_out(rp.def_B,sigma_phi_ran_cont,rp.def_sigma_x,rp.variance)/rp.teo_1d_grid_tuning_in(rp.def_B,rp.def_sigma_x,rp.variance),color='k')
    pl.plot(rp.sigma_phi_ran,rp.grid_index_out_mat[0,:,0]/rp.grid_index_in_mat[0,:,0],marker='s',mfc='k',mec='k',lw=0,ms=ms)
    pl.ylim(0,8)
  else:
    pl.plot(rp.sigma_phi_ran,rp.grid_index_out_mat[0,:,0]/rp.grid_index_in_mat[0,:,0],marker='s',mfc='k',mec='k',lw=0,ms=ms)
  
  # default value
  pl.axvline(rp.def_sigma_phi,color='k',ls=pp.linestyles['dashed'])
  
  # theoretical value of the minimum
  pl.axvline(rp.sigma_phi_peak,color='k',lw=1.5,ls=pp.linestyles['densely dotted'])
  
  pl.xlabel('Noise correlation length $\\sigma_\\varphi$')
  pl.ylabel('1D grid amplification index')
  pp.custom_axes()
  pl.xlim(0.02,100)
  
  
  pl.rcParams['ytick.major.size'] = 3
  pl.rcParams['xtick.major.size'] = 3
  axins = inset_axes(ax, width=0.9, height=0.6)
  pl.xlim(0.01,100)
  pl.xticks([0.01,1,10,100])
  pl.plot(sigma_phi_ran_cont,rp.variance*rp.teo_xi_pw_phi(1,sigma_phi_ran_cont),'-k')
  pl.axvline(rp.sigma_phi_peak,color='k',lw=1.5,ls=pp.linestyles['densely dotted'])
  
  axins.set_xscale('log')
  axins.xaxis.set_tick_params(labelsize=9)
  axins.yaxis.set_tick_params(labelsize=9)
  pl.ylabel('$S_{pop}^\\xi(1)$',fontsize=9)
  pl.xlabel('$\sigma_{\\varphi}$',fontsize=9)
  pp.custom_axes()
  
  if clip_rates is False:
    fname = 'fig1d_amp_sigma_phi_linear'
  else:
    fname = 'fig1d_amp_sigma_phi_clipped'
    
  if save_figs:
    pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])


#%%

#%%==========================================================================================
### GRID TUNING INDEX AS A FUNCTION OF NOISE CORRELATION LENGTH IN SPACE AND ACROSS NEURONS
#============================================================================================

# use base case here, because the plot is fully analytical
rb=RecAmp1D(RunMode.RUN_MODE_BASE_CASE)


B_small=0.2
B_large=0.8
sigma_phi_ran=np.logspace(np.log10(0.01),np.log10(100),100)[:,np.newaxis]
sigma_x_ran=np.logspace(np.log10(0.001),np.log10(10),100)[np.newaxis,:]
A_mat_B_small=rb.teo_amp_index(B_small,sigma_phi_ran,sigma_x_ran,0.5)
A_mat_B_def=rb.teo_amp_index(rb.def_B,sigma_phi_ran,sigma_x_ran,0.5)
A_mat_B_large=rb.teo_amp_index(B_large,sigma_phi_ran,sigma_x_ran,0.5)



pl.rcParams['ytick.minor.size'] = 3
pl.rcParams['ytick.major.size'] = 6
pl.rcParams['xtick.minor.size'] = 3
pl.rcParams['xtick.major.size'] = 6


pl.figure(figsize=(8,2.8))
pl.subplots_adjust(left=0.1,bottom=0.3,right=0.9,wspace=0.45,hspace=0.9,top=0.9)

pl.subplot(131)
pl.contourf(sigma_x_ran[0,:],sigma_phi_ran[:,0],A_mat_B_small,rasterized=True)


ax=pl.gca()
ax.set_xscale('log')
ax.set_yscale('log')
pp.custom_axes()
pl.xlabel('Noise correlation length $\\sigma_x$')
pl.ylabel('Noise correlation length $\\sigma_\\varphi$')

pl.title('B=%.1f'%B_small)
pl.axvline(rb.sigma_x_peak,color='w',lw=1.5,ls=pp.linestyles['densely dotted'])
pl.axhline(rb.sigma_phi_peak,color='w',lw=1.5,ls=pp.linestyles['densely dotted'])

pl.axvline(rb.def_sigma_x,color='w',lw=1.5,ls=pp.linestyles['dashed'])
pl.axhline(rb.def_sigma_phi,color='w',lw=1.5,ls=pp.linestyles['dashed'])

import matplotlib 
locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(10)*0.1,numticks=12)
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
pl.xticks([0.001,0.01,0.1,1,10])
ax.tick_params(axis='x', rotation=45)

pl.subplot(132)
pl.contourf(sigma_x_ran[0,:],sigma_phi_ran[:,0],A_mat_B_def,rasterized=True)

ax=pl.gca()
ax.set_xscale('log')
ax.set_yscale('log')
pp.custom_axes()

pl.xlabel('Noise correlation length $\\sigma_x$')


pl.title('B=%.1f'%rb.def_B)

pl.axvline(rb.sigma_x_peak,color='w',lw=1.5,ls=pp.linestyles['densely dotted'])
pl.axhline(rb.sigma_phi_peak,color='w',lw=1.5,ls=pp.linestyles['densely dotted'])

pl.axvline(rb.def_sigma_x,color='w',lw=1.5,ls=pp.linestyles['dashed'])
pl.axhline(rb.def_sigma_phi,color='w',lw=1.5,ls=pp.linestyles['dashed'])

locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(10)*0.1,numticks=12)
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
pl.xticks([0.001,0.01,0.1,1,10])
ax.tick_params(axis='x', rotation=45)

pl.subplot(133)
pl.contourf(sigma_x_ran[0,:],sigma_phi_ran[:,0],A_mat_B_large)


ax=pl.gca()
ax.set_xscale('log')
ax.set_yscale('log')
pp.custom_axes()

pl.title('B=%.1f'%B_large)

pl.axvline(rb.sigma_x_peak,color='w',lw=1.5,ls=pp.linestyles['densely dotted'])
pl.axhline(rb.sigma_phi_peak,color='w',lw=1.5,ls=pp.linestyles['densely dotted'])

pl.axvline(rb.def_sigma_x,color='w',lw=1.5,ls=pp.linestyles['dashed'])
pl.axhline(rb.def_sigma_phi,color='w',lw=1.5,ls=pp.linestyles['dashed'])

pl.xlabel('Noise correlation length $\\sigma_x$')

locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(10)*0.1,numticks=12)
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
pl.xticks([0.001,0.01,0.1,1,10])
ax.tick_params(axis='x', rotation=45)

pl.rcParams['ytick.major.size'] = 3
pl.rcParams['xtick.major.size'] = 3

cbar_ax = pl.gcf().add_axes([0.92, 0.3, 0.01, 0.3 ]) 
pp.colorbar(cax=cbar_ax)

fname = 'fig1d_amp_sigma_x_sigma_phi'
if save_figs:
  pp.save_fig(sl.get_figures_path(),fname,exts=['png','svg'])



