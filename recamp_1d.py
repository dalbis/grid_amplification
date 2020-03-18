#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:04:23 2019

@author: dalbis

"""


import numpy as np
from numpy.random import randn
from numpy.fft import fft,ifft
from scipy.special import ive
from scipy.optimize import minimize

class RunMode:
  RUN_MODE_INPUT_SAMPLES='RunModeInputSamples'
  RUN_MODE_INPUT_NOISE='RunModeInputNoise'
  RUN_MODE_BASE_CASE='RunModeBaseCase'
  RUN_MODE_TUNING_STRENGTH='RunModeTuninglStength'
  RUN_MODE_NOISE_NEURONS='RunModeNoiseNeurons'
  RUN_MODE_NOISE_SPACE='RunModeNoiseSpace'


class RecAmp1D():
  
  
  
  def __init__(self, run_mode, clip_rates=False):
    
    # check that run_mode is valid
    valid_run_modes=dict(filter(lambda item: item[0].startswith('RUN'),RunMode.__dict__.items())).values()
    assert(run_mode in valid_run_modes)
    
    np.random.seed(102)

    self.run_mode=run_mode
    self.clip_rates=clip_rates
    
    # ---- PARAMETERS ---------------------------------------------------------

    self.L=5.                               # length of the track [m]
    self.n_x=1000                           # number of space samples
    self.n_phi=200                          # number of neurons
    self.tau=0.01                           # time constant
    self.dt=0.002                           # time step
    
    self.dx = self.L/self.n_x                              # space sampling interval
    self.dphi=2*np.pi/self.n_phi                           # phase sampling interval
    
    self.f=1                                               # frequency of the oscillatory signal [1/m]
    self.variance=.5                                       # variance of the noise 
    
    self.M_max=2./3 if self.clip_rates is False else 6.    # peak of the connectivity function
    self.monte_carlo_samples=80                            # number of stochastic realizations of the inputs
    
    self.def_sigma_x=0.1                                   # default noise correlation length in space
    self.def_sigma_phi=self.dphi                           # default noise correlation length in phase
    self.def_B=0.4                                         # default strength of the signal
    
    self.tuning_harmonic_phi=1                             # tuning harmonic in phase
    self.tuning_harmonic_x=int(self.L*self.f)              # tuning harmonic in space
    self.max_harmonic=30                                   # maximal harmonic to consider numerically stable


    # grid signal tuning function
    self.tuning_fun = lambda phi,B: B*np.cos(phi)
    self.tuningx_fun = lambda x,phi,B: B*(np.cos(2*np.pi*self.f*x+phi))

    # connectivity functions as a function of phi
    self.m_fun = lambda phi: 2*self.M_max*np.cos(phi)/self.n_phi
    
    # connectivity functions as a function of x
    self.mx_fun= lambda x: 2*self.M_max*np.cos(2*np.pi*self.f*x)/self.n_x
 
    # Stength of population level amplification
    self.A_pop=1./(1.-self.M_max)**2
    
    # samples in space and phase
    self.x_ran=np.arange(0,self.L,self.dx)
    self.phi_ran=np.arange(0,2*np.pi,self.dphi)
    
    # connectivity matrix
    phi_diff=self.phi_ran[:,np.newaxis]-self.phi_ran[np.newaxis,:]
    self.M=self.m_fun(phi_diff)
    
    # equivalent filter in space
    self.F_x = 2*(np.sqrt(self.A_pop)-1)/self.L*np.cos(2*np.pi*self.f*self.x_ran)*self.dx
        
    # equivalent feed-forward filter in phase
    I=np.diag(np.ones(self.n_phi))
    self.K = np.linalg.inv(I-self.M)
    
    # define theory functions
    self.define_theory_funtions()

    # set parameter ranges
    self.set_parameter_ranges()
    
    # run the main body of the simulation    
    self.run_main()

    if run_mode==RunMode.RUN_MODE_BASE_CASE:
        # post-processing, e.g. computation of power spectra, analytical solutions, etc.
        self.post_run_for_plotting()


  def define_theory_funtions(self):
    """
    Define functions for theoretical curves
    """
  
    # theoretical noise correlation function in phase/space (von Mises) 
    self.teo_xi_corr_phi = lambda phi,sigma: np.exp((np.cos(phi)-1)/sigma**2)
    self.teo_xi_corr_x   = lambda x,  sigma: np.exp((np.cos(2*np.pi*x/self.L)-1)/sigma**2)
    
    # theoretical correlation power in phase/space (von Mises)
    self.teo_xi_pw_phi = lambda k_phi, sigma: 2*np.pi*ive(k_phi,1/sigma**2) 
    self.teo_xi_pw_x   = lambda k_x, sigma: self.L*ive(k_x,1/sigma**2) 
    
    # sigma at which the noise power of the given harmonic ub phase/space is maximal
    def get_sigma_peak(harmonic):
      fun= lambda sigma: -self.teo_xi_pw_phi(harmonic,sigma)        # invert sign because we want to maximize the function
      return minimize(fun,0.01).x[0]                                # initial guess
    
    self.sigma_phi_peak=get_sigma_peak(1)
    self.sigma_x_peak=get_sigma_peak(self.L*self.f)
      
    # theoretical value of the input grid tuning index  
    self.teo_1d_grid_tuning_in = lambda B,sigma_x,variance: (B**2*self.L/4+(1-B)**2*variance*self.teo_xi_pw_x(self.L*self.f,sigma_x))/((1-B)**2*variance*self.teo_xi_pw_x(0,sigma_x))
  
    # theoretical value of A_noise (note that it does not depend on the noise variance)
    self.A_noise_fun = lambda sigma_phi: 1+(self.A_pop-1)/np.pi*self.teo_xi_pw_phi(1,sigma_phi)
    
    # theoretical value of A_cell
    self.A_cell_fun = lambda B,sigma_phi,sigma_x,variance: (B**2*self.A_pop*self.L/4+(1-B)**2*self.A_noise_fun(sigma_phi)*variance*self.teo_xi_pw_x(self.L*self.f,sigma_x))/(B**2*self.L/4+(1-B)**2*variance*self.teo_xi_pw_x(self.L*self.f,sigma_x))
    
    # theoretical value of the output grid tuning index
    self.teo_1d_grid_tuning_out = lambda B,sigma_phi,sigma_x,variance: (B**2*self.A_pop*self.L/4+(1-B)**2*self.A_noise_fun(sigma_phi)*variance*self.teo_xi_pw_x(self.L*self.f,sigma_x))/((1-B)**2*variance*self.teo_xi_pw_x(0,sigma_x)*self.A_noise_fun(sigma_phi))
    
    # theoretical grid amplification index
    self.teo_amp_index = lambda B,sigma_phi,sigma_x,variance: self.A_cell_fun(B,sigma_phi,sigma_x,variance)/self.A_noise_fun(sigma_phi)
    

  def set_parameter_ranges(self):
    """
    Sets the parameter ranges depending if the run mode
    """
     
    ## BASE CASE
    self.B_ran=[self.def_B]
    self.sigma_phi_ran=[self.def_sigma_phi]
    self.sigma_x_ran=[self.def_sigma_x]
        
    # INPUT EXAMPLES
    if self.run_mode==RunMode.RUN_MODE_INPUT_SAMPLES:
      self.monte_carlo_samples=1
      self.variance=0.1

    # INPUT NOISE
    elif self.run_mode==RunMode.RUN_MODE_INPUT_NOISE:
      self.sigma_x_ran = [self.dx,0.05,0.1]
      self.sigma_phi_ran = [self.dphi,1,2]
      self.monte_carlo_samples=1

    # TUNING STRENGTH
    elif self.run_mode==RunMode.RUN_MODE_TUNING_STRENGTH:
      self.B_ran=np.linspace(0,1,30)

    # NOISE CORRELATIONS NEURONS
    elif self.run_mode==RunMode.RUN_MODE_NOISE_NEURONS:
      self.sigma_phi_ran=np.logspace(np.log10(self.dphi),np.log10(50),30)
          
    # NOISE CORRELATIONS SPACE
    elif self.run_mode==RunMode.RUN_MODE_NOISE_SPACE:
      self.sigma_x_ran=np.logspace(np.log10(self.dx),np.log10(2),30)



  def get_steady_output(self,h):
    """
    Simulates the steady-state output of the network activity in the non-linear scenaio
    """
    v=np.zeros_like(h)
    
    for time_idx in xrange(int(self.tau*5./self.dt)):
      v+=(self.dt/self.tau)*(-v+(h+np.dot(self.M,h)).clip(min=0))
    return v


  def gen_corr_noise(self,variance,sigma_phi,sigma_x,n_phi,n_x,L,use_teo_pw=True):
    """
    Generates correlated noise
    
    Note that using the theoretical power gives better matching with the analytics but gives numerical problems
    in the generation of the noise for large sigma_x for example. Need to understand what we can do about it
    
    """
  
    # we only consider strictly positive autocorrelation lengths, for sigma_phi,sigma_x = 0 the autocorrelation is not defined
    assert(sigma_phi>0 and sigma_x>0)
  
    # sampling intervals in phase/space  
    dphi=2*np.pi/n_phi
    dx=L/n_x
      
    # generate white noise with the prescribed variance (note the normalization by dx and dphi, see Dynan Abbot book page 22.)
    xi=np.sqrt(variance/(dx*dphi))*randn(n_phi,n_x)        
    
    # The filter in frequency domain is the square root of the theoretical noise power spectrum normalized to variance 1
    if use_teo_pw is True:
      
      # harmonics for power spectra
      k_phi=np.fft.fftshift(np.arange(n_phi)-n_phi/2.)                 
      k_x=np.fft.fftshift(np.arange(n_x)-n_x/2.)                 
    
      # theoretical noise power in phase/space (spectrum is even, and numerically is more stable for positive harmonics)
      pw_phi=self.teo_xi_pw_phi(np.abs(k_phi),sigma_phi)
      pw_x=self.teo_xi_pw_x(np.abs(k_x),sigma_x)
      
      
    # Here we compute the power spectrum numerically from the autocorrelation
    else:
      
      phi_ran=np.linspace(-np.pi,np.pi,n_phi)
      x_ran=np.linspace(-L/2.,L/2.,n_x)
  
      # numerical power in phase/space
      pw_phi=np.fft.fft(np.fft.fftshift(self.teo_xi_corr_phi(phi_ran,sigma_phi))).real*dphi
      pw_x=np.fft.fft(np.fft.fftshift(self.teo_xi_corr_x(x_ran,sigma_x))).real*dx
  
      
    # cut out negative values (shall be small and due to sampling)  
    pw_phi[pw_phi<0]=0
    pw_x[pw_x<0]=0
  
    # filter spectrum in phase/space
    filt_phi_dft=np.sqrt(pw_phi)       
    filt_x_dft=np.sqrt(pw_x)
  
  
    ### ================ INTRODUCE NOISE CORRELATIONS IN SPACE ==================================================
    
    # filter the noise in frequency domain
    xi_x_dft=fft(xi,axis=1)*dx
    xi_x_filt_dft=np.multiply(xi_x_dft,filt_x_dft[np.newaxis,:])
  
    # transform back to time domain  
    xi=np.real(ifft(xi_x_filt_dft,axis=1)/dx)
  
  
    ### ================ INTRODUCE NOISE CORRELATIONS ACROSS NEURONS ==============================================
    
    # filter the noise in frequency domain
    xi_phi_dft=fft(xi,axis=0)*dphi
    xi_phi_filt_dft=np.multiply(xi_phi_dft,filt_phi_dft[:,np.newaxis])
    
    # transform back to time domain
    xi=np.real(ifft(xi_phi_filt_dft,axis=0)/dphi)
    
    # enforce fixed variance 
    xi_norm=xi/xi.std()*np.sqrt(variance)
  
  
    return xi_norm
  
  
  def run_main(self):
    """
    Rund the main body of the simulation
    
    """
    
    self.xi_mat = np.zeros((len(self.B_ran),len(self.sigma_phi_ran),len(self.sigma_x_ran),self.monte_carlo_samples,self.n_phi,self.n_x))
    self.h_mat = np.zeros((len(self.B_ran),len(self.sigma_phi_ran),len(self.sigma_x_ran),self.monte_carlo_samples,self.n_phi,self.n_x))
    self.v_mat = np.zeros((len(self.B_ran),len(self.sigma_phi_ran),len(self.sigma_x_ran),self.monte_carlo_samples,self.n_phi,self.n_x))
    self.v_t_mat = np.zeros((len(self.B_ran),len(self.sigma_phi_ran),len(self.sigma_x_ran),self.monte_carlo_samples,self.n_phi,self.n_x))
                     
    for B_idx,B in enumerate(self.B_ran):
      
      print 'B_idx = %d/%d, value=%.3f'%(B_idx,len(self.B_ran),B)
      
      # grid signal
      g=np.cos(2*np.pi*self.f*self.x_ran[np.newaxis,:]+self.phi_ran[:,np.newaxis])  
    
    
      for sigma_phi_idx,sigma_phi in enumerate(self.sigma_phi_ran):
        
        print 'sigma_phi_idx = %d/%d, value=%.3f'%(sigma_phi_idx,len(self.sigma_phi_ran),sigma_phi)
      
        for sigma_x_idx,sigma_x in enumerate(self.sigma_x_ran):
          
          print 'sigma_x_idx = %d/%d, value=%.3f'%(sigma_x_idx,len(self.sigma_x_ran),sigma_x)
              
          # loop across different np.realization of the noise      
          for ms_idx in xrange(self.monte_carlo_samples):
            
            #print 'ms_idx = %d/%d'%(ms_idx,monte_carlo_samples)          
                
            # generate noise
            xi = self.gen_corr_noise(self.variance,sigma_phi,sigma_x,self.n_phi,self.n_x,self.L) 
  
            # total feed-forward input (signal+noise)
            h = B*g+(1-B)*xi 
            
            if self.clip_rates:
              
              v=self.get_steady_output(h)  
              h=h.clip(min=0)
              # output
            else:
              v=np.dot(self.K,h)           
              
            self.xi_out=np.dot(self.K,xi)
            self.g_out=np.dot(self.K,g)
  
            # save input and outputs
            self.xi_mat[B_idx,sigma_phi_idx,sigma_x_idx,ms_idx,:,:]=xi
            self.h_mat[B_idx,sigma_phi_idx,sigma_x_idx,ms_idx,:,:]=h
            self.v_mat[B_idx,sigma_phi_idx,sigma_x_idx,ms_idx,:,:]=v
  
            
    
    #% estimate A_noise and A_cell from the numerical simulations
    self.A_noise_est_mat = np.zeros((len(self.B_ran),len(self.sigma_phi_ran),len(self.sigma_x_ran)))
    self.A_noise_est_pw_mat =  np.zeros((len(self.B_ran),len(self.sigma_phi_ran),len(self.sigma_x_ran))) # this is a semi-analytical estimation (numerially more stable)
    self.A_cell_est_mat=np.zeros((len(self.B_ran),len(self.sigma_phi_ran),len(self.sigma_x_ran)))
  
    # estimate input and output grid-tuning indexes
    self.grid_index_in_mat = np.zeros((len(self.B_ran),len(self.sigma_phi_ran),len(self.sigma_x_ran)))
    self.grid_index_out_mat = np.zeros((len(self.B_ran),len(self.sigma_phi_ran),len(self.sigma_x_ran)))
      
    for B_idx,B in enumerate(self.B_ran):
      for sigma_phi_idx,sigma_phi in enumerate(self.sigma_phi_ran):
        for sigma_x_idx,sigma_x in enumerate(self.sigma_x_ran):
          
          # mean power of the noise in phase
          xi_phi_pool=np.swapaxes(self.xi_mat,4,5)[B_idx,sigma_phi_idx,sigma_x_idx,:,:,:].reshape(self.monte_carlo_samples*self.n_x,self.n_phi) 
          xi_phi_pool_dft=fft(xi_phi_pool,axis=1)*self.dphi
          self.xi_phi_mean_pw=np.mean(np.real(xi_phi_pool_dft*np.conjugate(xi_phi_pool_dft))/(2*np.pi),axis=0)
  
          # mean power of the noise in space        
          xi_x_pool=self.xi_mat[B_idx,sigma_phi_idx,sigma_x_idx,:,:,:].reshape(self.monte_carlo_samples*self.n_phi,self.n_x)
          xi_x_pool_dft=fft(xi_x_pool,axis=1)*self.dx
          self.xi_x_mean_pw=np.mean(np.real(xi_x_pool_dft*np.conjugate(xi_x_pool_dft))/self.L,axis=0)
  
          # mean power of the input in space        
          h_x_pool=self.h_mat[B_idx,sigma_phi_idx,sigma_x_idx,:,:,:].reshape(self.monte_carlo_samples*self.n_phi,self.n_x)
          h_x_pool_dft=fft(h_x_pool,axis=1)*self.dx
          self.h_x_mean_pw=np.mean(np.real(h_x_pool_dft*np.conjugate(h_x_pool_dft))/self.L,axis=0)
    
          # mean power of the output in space        
          v_x_pool=self.v_mat[B_idx,sigma_phi_idx,sigma_x_idx,:,:,:].reshape(self.monte_carlo_samples*self.n_phi,self.n_x)
          v_x_pool_dft=fft(v_x_pool,axis=1)*self.dx
          self.v_x_mean_pw=np.mean(np.real(v_x_pool_dft*np.conjugate(v_x_pool_dft))/self.L,axis=0)
      
  
          # power of the equivalent feed-forward filter in space
          self.k_x_mean_est_pw = self.v_x_mean_pw/self.h_x_mean_pw
          
          # set to 1 high harmonics to avoid numerical instability
          self.k_x_mean_est_pw[self.max_harmonic:-self.max_harmonic+1]=1
  
          # numerical estimate of A_cell
          self.A_cell_est_mat[B_idx,sigma_phi_idx,sigma_x_idx]=self.k_x_mean_est_pw[self.tuning_harmonic_x]        
          
          # harmonics over which to average the filter spectrum to estimate A_noise
          harmonics_for_A_noise=np.setdiff1d(np.arange(2*self.max_harmonic)-self.max_harmonic,[self.tuning_harmonic_x,-self.tuning_harmonic_x])        
  
          # estimate A_noise from the numerical equivalent filter (this is still numerically unstable)
          self.A_noise_est_mat[B_idx,sigma_phi_idx,sigma_x_idx]=np.mean(self.k_x_mean_est_pw[harmonics_for_A_noise])
          
          # estimate A_noise from noise power (this is numerically stable but is semi analytical)
          self.A_noise_est_pw_mat[B_idx,sigma_phi_idx,sigma_x_idx]=1+(self.A_pop-1)/np.pi*self.xi_phi_mean_pw[1]/self.variance
                  
          # 1d grid tuning indexes
          self.grid_index_in_mat[B_idx,sigma_phi_idx,sigma_x_idx]=self.h_x_mean_pw[self.tuning_harmonic_x]/self.h_x_mean_pw[0]
          self.grid_index_out_mat[B_idx,sigma_phi_idx,sigma_x_idx]=self.v_x_mean_pw[self.tuning_harmonic_x]/self.v_x_mean_pw[0]
    

  def post_run_for_plotting(self):
    
    # select one single example
    plot_B_idx=0
    plot_sigma_phi_idx=0
    plot_sigma_x_idx=0
    plot_mc_sample_idx=0
            
          
    self.B=self.B_ran[plot_B_idx]
    self.sigma_x=self.sigma_x_ran[plot_sigma_x_idx]
    self.sigma_phi=self.sigma_phi_ran[plot_sigma_phi_idx]
    
    self.h = self.h_mat[plot_B_idx,plot_sigma_phi_idx,plot_sigma_x_idx,plot_mc_sample_idx,:,:]
    self.v = self.v_mat[plot_B_idx,plot_sigma_phi_idx,plot_sigma_x_idx,plot_mc_sample_idx,:,:]
    
    self.m=self.M[0,:]
    
    self.h_phi = self.h[:,0]
    self.h_x =self.h[0,:]
    self.v_phi = self.v[:,0]
    self.v_x = self.v[0,:]
    
    # power of single input and output examples
    
    h_phi_dft=fft(self.h_phi)*self.dphi
    self.h_phi_pw=np.real(h_phi_dft*np.conjugate(h_phi_dft))/(2*np.pi)
    
    h_x_dft=fft(self.h_x)*self.dx
    self.h_x_pw = np.real(h_x_dft*np.conjugate(h_x_dft))/self.L
    
    v_phi_dft=fft(self.v_phi)*self.dphi
    self.v_phi_pw=np.real(v_phi_dft*np.conjugate(v_phi_dft))/(2*np.pi)
    
    v_x_dft=fft(self.v_x)*self.dx
    self.v_x_pw=np.real(v_x_dft*np.conjugate(v_x_dft))/self.L
      
    self.m_dft=fft(self.m)

    
    # average input power in space
    h_x_pool=self.h_mat[plot_B_idx,plot_sigma_phi_idx,plot_sigma_x_idx,:,:,:].reshape(self.monte_carlo_samples*self.n_phi,self.n_x) 
    h_x_pool_dft=fft(h_x_pool,axis=1)*self.dx
    self.h_x_mean_pw=np.mean(np.real(h_x_pool_dft*np.conjugate(h_x_pool_dft))/self.L,axis=0)
    
    # average output power in space
    v_x_pool=self.v_mat[plot_B_idx,plot_sigma_phi_idx,plot_sigma_x_idx,:,:,:].reshape(self.monte_carlo_samples*self.n_phi,self.n_x)
    v_x_pool_dft=fft(v_x_pool,axis=1)*self.dx
    self.v_x_mean_pw=np.mean(np.real(v_x_pool_dft*np.conjugate(v_x_pool_dft))/self.L,axis=0)
    
    # average input power across neurons (phase)
    h_phi_pool=np.swapaxes(self.h_mat,4,5)[plot_B_idx,plot_sigma_phi_idx,plot_sigma_x_idx,:,:,:].reshape(self.monte_carlo_samples*self.n_x,self.n_phi) 
    h_phi_pool_dft=fft(h_phi_pool,axis=1)*self.dphi
    self.h_phi_mean_pw=np.mean(np.real(h_phi_pool_dft*np.conjugate(h_phi_pool_dft))/(2*np.pi),axis=0)
    
    # average output power across neurons (phase)
    v_phi_pool=np.swapaxes(self.v_mat,4,5)[plot_B_idx,plot_sigma_phi_idx,plot_sigma_x_idx,:,:,:].reshape(self.monte_carlo_samples*self.n_x,self.n_phi)
    v_phi_pool_dft=fft(v_phi_pool,axis=1)*self.dphi
    self.v_phi_mean_pw=np.mean(np.real(v_phi_pool_dft*np.conjugate(v_phi_pool_dft))/(2*np.pi),axis=0)
    
    
    #-------------------------------------------------
    # Equivalent filters
    #-------------------------------------------------
    
    # here we estimate input/output filters by looking at the average frequency responses
    # in the first max_harmonic frequency components. For the other components it is assumed no 
    # filtering (i.e. gain=1, phase=0). This restriction is to avoid numerical
    # instability when the amplitude of the input is close to zero (division by zero problem)
    
    # equivalent filter across neurons (phase)
    self.k_phi_pw=self.v_phi_pw/self.h_phi_pw
    self.k_phi_mean_pw = self.v_phi_mean_pw/self.h_phi_mean_pw
    self.k_phi_mean_pw[self.max_harmonic:-self.max_harmonic+1]=1
    self.k_phi_est = np.real(ifft(np.sqrt(self.k_phi_mean_pw)))/self.dphi
    
    
    # equivalent filter in space 
    self.k_x_pw=self.v_x_pw/self.h_x_pw
    self.k_x_mean_pw = self.v_x_mean_pw/self.h_x_mean_pw
    self.k_x_mean_pw[self.max_harmonic:-self.max_harmonic+1]=1
    self.k_x_est = np.real(ifft(np.sqrt(self.k_x_mean_pw)))/self.dx
    
    
    #-----------------------------------------------
    # Theoretical curves
    #-----------------------------------------------
    
    self.A_noise = self.A_noise_fun(self.sigma_phi)  
    self.A_cell = self.A_cell_fun(self.B,self.sigma_phi,self.sigma_x,self.variance)
      
    
    # Phase domain
    #-----------------------------------------------
    
    self.w_ran_phi=np.arange(0,9,0.01)
    peak_phi_idx=np.argmin(abs(self.w_ran_phi-self.tuning_harmonic_phi)) 
      
    teo_g_phi_pw_peak = np.pi/2
    teo_xi_phi_pw_peak = self.variance*self.teo_xi_pw_phi(1,self.sigma_phi)
    
    teo_h_phi_pw_peak = self.B**2*teo_g_phi_pw_peak +(1-self.B)**2*teo_xi_phi_pw_peak
    self.teo_h_phi_pw = (1-self.B)**2*self.variance*self.teo_xi_pw_phi(self.w_ran_phi,self.sigma_phi)
    self.teo_h_phi_pw[peak_phi_idx]=teo_h_phi_pw_peak
    
    teo_v_phi_pw_peak = self.A_pop*teo_h_phi_pw_peak                     # both signal and noise are amplified by A_pop
    self.teo_v_phi_pw = (1-self.B)**2*self.variance*self.teo_xi_pw_phi(self.w_ran_phi,self.sigma_phi)
    self.teo_v_phi_pw[peak_phi_idx]=teo_v_phi_pw_peak
    
    teo_k_phi_pw_peak = self.A_pop
    self.teo_k_phi_pw = np.ones_like(self.w_ran_phi)
    self.teo_k_phi_pw[peak_phi_idx]=teo_k_phi_pw_peak
    
    # theoretical filter in phase
    self.k_phi=(np.sqrt(self.A_pop)-1)*np.cos(self.phi_ran)/np.pi
    self.k_phi[0]+=1/self.dphi
    
    
    # Space domain 
    #-----------------------------------------------
    
    self.w_ran_x=np.arange(0,9,0.01)
    peak_x_idx=np.argmin(abs(self.w_ran_x-self.tuning_harmonic_x)) 
     
    teo_g_x_pw_peak = self.L/4
    teo_xi_x_pw_peak = self.variance*self.teo_xi_pw_x(self.L*self.f,self.sigma_x)
      
    teo_h_x_pw_peak = self.B**2*teo_g_x_pw_peak+(1-self.B)**2*teo_xi_x_pw_peak
    self.teo_h_x_pw = (1-self.B)**2*self.variance*self.teo_xi_pw_x(self.w_ran_x,self.sigma_x)
    self. teo_h_x_pw[peak_x_idx]=teo_h_x_pw_peak
    
    teo_v_x_pw_peak = self.B**2*self.A_pop*teo_g_x_pw_peak+(1-self.B)**2*self.A_noise*teo_xi_x_pw_peak   # noise is amplified by A_noise, signal by A_pop
    self.teo_v_x_pw = (1-self.B)**2*self.variance*self.teo_xi_pw_x(self.w_ran_x,self.sigma_x)*self.A_noise
    self.teo_v_x_pw[peak_x_idx]=teo_v_x_pw_peak
    
    teo_k_x_pw_peak = teo_v_x_pw_peak/teo_h_x_pw_peak
    self.teo_k_x_pw = np.ones_like(self.w_ran_x)*self.A_noise
    self.teo_k_x_pw[peak_x_idx] = teo_k_x_pw_peak
    
    # theoretical filter in space
    self.k_x=(np.sqrt(self.A_cell)-np.sqrt(self.A_noise))*np.cos(2*np.pi*self.f*self.x_ran)*2./self.L
    self.k_x[0]+=1/self.dx
    
    
    print '------------------'  
    print 'B:%.2f' %self.B
    print 'sigma_phi: %.4f'% self.sigma_phi
    print 'sigma_x: %.4f'% self.sigma_x
    print
    print 'A_cell_est: %.2f'% self.A_cell_est_mat[plot_B_idx,plot_sigma_phi_idx,plot_sigma_x_idx]
    print 'A_noise_est: %.2f' % self.A_noise_est_mat[plot_B_idx,plot_sigma_phi_idx,plot_sigma_x_idx]
    print 'A_noise_est_pw: %.2f' % self.A_noise_est_pw_mat[plot_B_idx,plot_sigma_phi_idx,plot_sigma_x_idx]
    print
    print 'A_cell_teo: %.2f' % self.A_cell
    print 'A_noise_teo: %.2f' % self.A_noise
    print '------------------'  

