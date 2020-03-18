# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:24:41 2018

@author: dalbis
"""


import pylab as pl
import numpy as np
import grid_utils.plotlib as pp

#==============================================================================
### PLOT POPULATION-LEVEL AMPLIFICATION
#==============================================================================

def plot_spectrum(w_ran,pw,mean_pw,teo_pw,ymax_top=None,
                  ytick_top=None,ytick_bottom=None,yticks=None,
                  split_lims=None,bar_color=[0.7,0.7,0.7],
                  line_color='black',lw=1,ms=5):
                    
  if split_lims is not None:
    y_max_bottom,y_min_top = split_lims[0],split_lims[1]
  
    xlim=[-.5,9.5]
    if ymax_top is None:
      ymax_top=teo_pw.max()
      
    if ytick_bottom is None:      
      ytick_bottom=split_lims[0]
      
    if ytick_top is None:      
      ytick_top=ymax_top/2
      
    ylim_top = [y_min_top,ymax_top]
    ylim_bottom  = [0, y_max_bottom]
        
    ax_top,ax_bottom=pp.broken_axis(pl.gca(),xlim,ylim_top,ylim_bottom,'',ratio=0.5)
    
    pp.bar(pw,N=10,ax=ax_top,color=bar_color,edgecolor=bar_color)
    pp.bar(pw,N=10,ax=ax_bottom,color=bar_color,edgecolor=bar_color)
    ax_top.set_xticks([])
    ax_top.yaxis.set_major_locator( pl.MaxNLocator(nbins=3) )
  
    ax_bottom.plot(mean_pw,marker='s',markersize=ms,linewidth=0,
            color=line_color,markeredgecolor=line_color)
    ax_top.plot(mean_pw,marker='s',markersize=ms,linewidth=0,
           color=line_color,markeredgecolor=line_color)
    ax_bottom.yaxis.set_major_locator( pl.MaxNLocator(nbins=3) )

    ax_bottom2=ax_bottom.twiny()
    ax_bottom2.set_ylim(ylim_bottom)
    ax_bottom2.plot(w_ran,teo_pw,color=line_color,linestyle='-',linewidth=lw)
    ax_bottom2.set_xlim(-0.5,9.5)
    ax_bottom2.set_xticks([])
    ax_bottom2.set_yticks([0,ytick_bottom])
    pp.custom_axes(ax_bottom2)       
    
    ax_top2=ax_top.twiny()
    ax_top2.set_ylim(ylim_top)
    ax_top2.plot(w_ran,teo_pw,color=line_color,linestyle='-',linewidth=lw)
    ax_top2.set_xticks([])
    ax_top2.set_yticks([ytick_top,ymax_top])
    ax_top2.spines['bottom'].set_color('none')
    ax_top2.set_xlim(-0.5,9.5)
    pp.custom_axes(ax_top2)
  

  else:
    
    pp.custom_axes()
    pp.bar(pw,N=10,color=bar_color,edgecolor=bar_color)
    pl.plot(mean_pw,marker='s',markersize=ms,linewidth=0,color=line_color,markeredgecolor=line_color)
    pl.plot(w_ran,teo_pw,color=line_color,linestyle='-',linewidth=lw)
    if ymax_top is not None:
      pl.ylim(0,ymax_top)
      
    if yticks is not None:
      pl.yticks(yticks)
      
      
def plot_amp_res(ran,w_ran,n,ds,
                 h,k_est,k,v,
                 h_pw,h_mean_pw,
                 k_pw,k_mean_pw,
                 v_pw,v_mean_pw,
                 teo_h_pw,teo_k_pw,teo_v_pw,
                 ran_ticks=[],ran_ticks_labels=[],delta_label='',
                 split_h=None,split_k=None,split_v=None,
                 ymax_h=None,ymax_k=None,ymax_v=None,
                 ytick_top_h=None,ytick_bottom_h=None,
                 ytick_top_k=None,ytick_bottom_k=None,yticks_k=None,
                 ytick_top_v=None,ytick_bottom_v=None,
                 lw=2,fig=None,tuning_curve=None,ms=5):
  
 
  if fig is None:
    pl.figure(figsize=(10,6))
    pl.subplots_adjust(left=0.1,right=0.99,wspace=0.4,hspace=.8,bottom=0.1,top=0.95)  
  
  # --------------------------------- Input ------------------------------------

  half_ran=(ran[-1]+ds)/2
  
  pl.subplot(231)
  pp.custom_axes()
  if tuning_curve is not None:
    pl.plot(ran-half_ran,np.roll(tuning_curve,n/2),'-k',color=[.7,.7,.7],lw=lw)  
  pl.plot(ran-half_ran,np.roll(h,n/2),'-k',lw=lw)  
  pl.xlim([-half_ran,half_ran])
  pl.xticks(ran_ticks,ran_ticks_labels)
  pl.ylim([-2,2])
  pl.gca().yaxis.set_major_locator(pl.MaxNLocator(3))


  # --------------------------------- Filter ------------------------------------

  ax=pl.subplot(232,frameon=False)
  xlim=[-half_ran,half_ran]
  ylim_top = [20, np.ceil(1/ds)+10]
  ylim_bottom  = [-1.5, 1.5]
  xlabel=''
  ax_top,ax_bottom=pp.broken_axis(ax,xlim,ylim_top,ylim_bottom,xlabel,ratio=0.5)
  ax_top.set_yticks([1/ds,])
  ax_top.set_yticklabels([delta_label],fontsize=12)

  
  ax_bottom.set_yticks([-1,1])
  pl.sca(ax_bottom)
  pl.xticks(ran_ticks,ran_ticks_labels)
  
  #ax_top.plot(ran-half_ran,np.roll(k_est,n/2),color=[0.6,0.6,0.6],lw=lw,ls='-')
  #ax_bottom.plot(ran-half_ran,np.roll(k_est,n/2),color=[0.6,0.6,0.6],lw=lw,ls='-')

  ax_top.plot(ran-half_ran,np.roll(k,n/2),'-k',lw=lw)
  ax_bottom.plot(ran-half_ran,np.roll(k,n/2),'-k',lw=lw)

  
  # --------------------------- Output -----------------------------------------
  
  pl.subplot(233)
  pp.custom_axes()
  #if tuning_curve is not None:
  #  pl.plot(ran-half_ran,np.roll(tuning_curve,n/2),'-k',color=[0.7,0.7,0.7],lw=lw*3)  
  pl.plot(ran-half_ran,np.roll(v,n/2),'-k',lw=lw)
  pl.xlim([-half_ran,half_ran])
  pl.ylim([-2,2])
  pl.xticks(ran_ticks,ran_ticks_labels)
  pl.gca().yaxis.set_major_locator(pl.MaxNLocator(3))

 
  #----------------------------------------------------------------------------
  # =========== FREQUENCY DOMAIN ==============================================
  #----------------------------------------------------------------------------
    
  #--------------------------- Input -------------------------------------------
  
  ax=pl.subplot(234,frameon=split_h is None)
  
  plot_spectrum(w_ran,h_pw,h_mean_pw,teo_h_pw,split_lims=split_h,ymax_top=ymax_h,
                ytick_top=ytick_top_h,ytick_bottom=ytick_bottom_h,ms=ms)
  
  
  #---------------- Equivalent Feed-farward filter -----------------------------
  
  ax=pl.subplot(235,frameon=split_k is None)
  
  plot_spectrum(w_ran,k_pw,k_mean_pw,teo_k_pw,split_lims=split_k,ymax_top=ymax_k,
                ytick_top=ytick_top_k,ytick_bottom=ytick_bottom_k,yticks=yticks_k,ms=ms)
  
  
  #--------------------------- Output-------------------------------------------

  ax=pl.subplot(236,frame_on=split_v is None)
      
  plot_spectrum(w_ran,v_pw,v_mean_pw,teo_v_pw,split_lims=split_v,ymax_top=ymax_v,
                ytick_top=ytick_top_v,ytick_bottom=ytick_bottom_v,ms=ms)


