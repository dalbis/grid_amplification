# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 09:33:42 2016

@author: dalbis
"""

from numpy.fft import fft2,fftshift,ifft2,fftfreq
from numpy.random import randn,rand,seed,permutation,randint
import numpy as np
import os
from time import clock
from scipy.optimize import minimize_scalar
import gridlib as gl
import simlib as sl


def get_grid_T_sigma(grid_T,C=5):
  """
  Numerically computes a sigma for the distribution of grid scales within a module.
  This is done such that the correlation structure within each module is preserved. see ffamp2d_scalegradient
  The parameter C is a global scaling factor.
  """
  
  # the method is based on approximate iso-correlation lines of the form  T*C/(C+T) and T*C/(C-T) where C is a parameter 
  # that correspond to a given correlation value
  # The idea is to find a circle tangent to two-isocorrelation lines

  # distance between the upper iso-line and the center scale point  
  dist = lambda x : (x-grid_T)**2+(C*x/(C-x) -grid_T )**2
  res=minimize_scalar(dist,bounds=(0,grid_T))
  
  # tangent point
  x_tang=res.x
  y_tang=C*x_tang/(C-x_tang)
  
  # the circle radius is the distance between such tangent point and the center scale of the module
  radius=np.sqrt((y_tang-grid_T)**2+(x_tang-grid_T)**2)
  
  # we take as sigma radius/3
  sigma=radius/3.
  
  #print 'Module: center= %.3f sigma=%.3f'%(grid_T,sigma)scatter_sigma_phi
  
  return sigma

  


def get_grid_peak_pos(n,L,grid_T,grid_angle,grid_T_vect,grid_angle_vect,zero_phase=False):
  """
  computes the position of the grid peaks of a regular grid. Note that grid peaks
  one priod outside the boundaries are also included to be able sto still have fraction 
  of fields with centers outside the environment
  """
  
  # unit vectors of the grid lattice (one per grid)
  U1=grid_T_vect*np.squeeze(np.stack([np.sin(2*np.pi/3+grid_angle_vect), -np.cos(2*np.pi/3+grid_angle_vect)],axis=1))
  U2=grid_T_vect*np.squeeze(np.stack([-np.sin(grid_angle_vect), np.cos(grid_angle_vect)],axis=1))

  # grid phases  
  phases=gl.get_phases(n,grid_T,grid_angle)
  
  # number of bumps (grid periods) in the environmnet
  num_periods=np.ceil(L/grid_T_vect.min())

  # tiling the unit cell for the required number of periods
  periods=np.arange(-num_periods,num_periods+1,1)
  P1,P2=np.meshgrid(periods,periods)  # periods x periods

  # compute grid peaks
  phase0_peaks=P1[np.newaxis,:,:]*U1[:,:,np.newaxis,np.newaxis]+P2[np.newaxis,:,:]*U2[:,:,np.newaxis,np.newaxis] # N x 2 x periods x periods

  phase0_peaks=phase0_peaks.reshape(n**2,2,len(P1)**2)

  if zero_phase is True:
    grid_peaks=phase0_peaks
  else:
    grid_peaks=phase0_peaks-phases[:,:,np.newaxis] # N x 2 X num_peaks



  return grid_peaks



def gen_noisy_grids(L,nx,dx,n,grid_T,grid_angle,
                    grid_T_vect,grid_angle_vect,
                    jitter_variance,jitter_sigma_phi,
                    input_mean,
                    scale_field_size_ratios,
                    zero_phase):
  
  """
  L : side-length of the arena
  nx: number of space samples per side
  dx: resolution of spatial sampling
  n: square root of the number of cells 
  grid_T: mean grid period
  grid_angle: mean grid orientation
  grid_T_vect: noisy grid period (one per cell)
  grid_angle_vect: noisy grid orientation (one per cell)
  jitter_variance: variance of the grid-field jitter (in space)
  jitter_sigma_phi: standard deviation of the jitter across neurons
  input_mean: mean input
  scale_field_size_ratios: n**2 X 1 array: ratios between the grid scale and the size of a grid field (one per grid in case they need to be different)
  zero_phase: generate all grids with zero phases
  """
  
  # Gaussian modeling a grid peak
  g_fun = lambda p,sigma: np.exp(-np.sum(p**2,2)/(2*sigma**2))
            
  # consider larger field for peaks in field but with center outside
  margin=grid_T_vect.max()

  XX,YY=np.mgrid[-L/2-margin:L/2+margin:dx,-L/2-margin:L/2+margin:dx]    

  # generate perfect grids  
  larger_pos=np.array([np.ravel(XX), np.ravel(YY)]).T   
                                                  
  nx_large=int(np.sqrt(larger_pos.shape[0]))
  inputs_large=np.zeros((nx_large**2,n**2))

  # get grid peak positions
  grid_peak_pos=get_grid_peak_pos(n,L,grid_T,grid_angle,grid_T_vect,grid_angle_vect,zero_phase)  # N x 2 x num_peaks
  num_peaks=grid_peak_pos.shape[2]
  
  # need to jitter the grid peaks
  if jitter_variance>0:
    peaks_jitter=np.random.randn(n**2,2,num_peaks)  # N x 2 x num_peaks (variance 1)
    
    if jitter_sigma_phi>0:
      
        phases=gl.get_phases(n,1,0,return_u12=False)
  
        filt_phi=np.exp(-np.sum(phases**2,1)/(2*jitter_sigma_phi**2))
        filt_phi/=filt_phi.sum()
  
        # Filter noise in phase
        filt_phi=filt_phi.reshape(n,n)
        filt_phi_ft=fft2(filt_phi)  
        peaks_jitter_ft = fft2(peaks_jitter.reshape(n,n,2,num_peaks),axes=[0,1])
        jitter_filt_ft=np.multiply(peaks_jitter_ft,filt_phi_ft[:,:,np.newaxis,np.newaxis])
        peaks_jitter = np.real(ifft2(jitter_filt_ft,axes=[0,1]))     
        peaks_jitter=peaks_jitter.reshape(n**2,2,num_peaks)       # N x 2 x num_peaks (modified variance)
       
    # rescale to the desired variance
    peaks_jitter/=peaks_jitter.std()
    peaks_jitter*=np.sqrt(jitter_variance)
            
    # add the jittered
    noisy_peak_pos=grid_peak_pos+peaks_jitter
    
  # grid peaks are not jittered  
  else:
    noisy_peak_pos=grid_peak_pos
    
  # add a gaussian for each noisy peak center      
  for grid_idx in xrange(n**2):        
    grid_noisy_peak=noisy_peak_pos[grid_idx,:,:].squeeze().T           # num_peaks x 2
    P0=larger_pos[np.newaxis,:,:]-grid_noisy_peak[:,np.newaxis,:]      # num_peaks x num_pixels x 2
    inputs_large[:,grid_idx]=g_fun(P0,grid_T_vect[grid_idx]*scale_field_size_ratios[grid_idx]).astype(np.float32).sum(axis=0)
  
  # crop out the out the outer margin, we retain an inner square of size (self.nx, self.nx)
  margin_nx=np.int((nx_large-nx)/2.)
  inputs_unfolded=inputs_large.reshape(nx_large,nx_large,n**2)
  inputs=inputs_unfolded[margin_nx:margin_nx+nx,margin_nx:margin_nx+nx]     
  inputs=inputs.reshape(nx**2,n**2) 
  
  # shift down and clip (to mimic perfect grid inputs)
  inputs-=0.5
  inputs=inputs.clip(min=0)
  inputs=inputs/inputs.mean(axis=0)*input_mean
  
  return inputs,grid_peak_pos,noisy_peak_pos,grid_T_vect
    
  


def rhombus_mises(phases,sigma):
  """
  Returns a 2D von Mises on a rhombus with 60 degrees angle.
  The function is notmalized such that its peak in zero is one
  """
  
  # wave vector of the cosine waves in the von Mises    
  def k_vect(n):
    return np.array([np.cos(n*np.pi/3),np.sin(n*np.pi/3)])
  
  sum_cos=0
  for n in xrange(3):    
    sum_cos+=np.cos(np.dot(phases,k_vect(n)))#+np.cos(np.dot(phases,k_vect(2)))
    
  return np.exp((sum_cos-3)/sigma**2) 
    
    

def gen_corr_noise_2d(L,nx,n,noise_variance,noise_sigma_phi,noise_sigma_x,filter_x=True,filter_phi=True,mises=True):
  """
  Generates correlated noise in 2D. Noise can be correlated across two orthogonal directions:
  space (autocorrelation length is noise_sigma_x), and phase (autocorrelation length is noise_sigma_phi).
  The noise variance is given and is kept fixed, i.e., the autocorrelation in zero is equal to the variance.
  The 2D phase is defined such that the unit rhombus has side-length 2*pi in analogy the 1D model      
  
  The noise autocorrelation is a 2D Gaussian in space/phase. We assume periodicity in a square [0,L]^2 (torus) and
  in a rhombus [0,2pi]^2 (twisted torus with 60 degrees angle)
  To achieve this, we first generate white noise the specified noise_variance (scaled by dx and dphi),
  then we introduce correlations in Fourier domain.
  
  Specifically (repeat procedure for space and phase correlations):
    - compute the spectrum of the wanted noise autocorrelation numerically or theoretically (shall be real and positive)
    - compute the spectrum of the filter (square root of the autocorrelation spectrum)
    - multiply the spectrum of the noise by the spectrum of the filter
    - take the inverse fourier transform of the result
    
  The end result shall be that the variance of the noise stays fixed even for large noise correlation lengths.
  In the  limit of very large noise correlation lengths the autocorrlation of the noise shall be flat. 
  
  """
  assert(noise_sigma_phi>0 and noise_sigma_x>0)
  
  
  # generate uncorrelated Gaussian noise (note the scaling by dx and d_phi)
  xi=np.sqrt(noise_variance)*randn(nx**2,n**2)


  if filter_x is True:
    
    #----------------------    FILTER NOISE IN SPACE   -------------------------
    
    # scale white noise by square root of sampling interval      
    dx=float(L**2)/nx**2
    xi/=np.sqrt(dx)
  
    # generate space samples     
    X,Y=np.mgrid[-L/2:L/2:np.sqrt(dx),-L/2:L/2:np.sqrt(dx)]
    pos=np.array([np.ravel(X), np.ravel(Y)]).T

    # autocorrelation in space   
    if mises is True:      
      # 2d orthogonal von mises in space (two orthogonal wave vectors)
      autocorr_x=np.exp((np.cos(2*np.pi*pos[:,0]/L)+np.cos(2*np.pi*pos[:,1]/L)-2)/noise_sigma_x**2)     
      
    else:
      # 2d Gaussian in space
      autocorr_x=np.exp(-np.sum(pos**2,1)/(2*noise_sigma_x**2))  
    autocorr_x=autocorr_x.reshape(nx,nx)
    
    # spectrum of the autocorrelation (this assumes periodic bounderies outside [0,L]^2)
    # note that we take only the real part (imaginary parts shall be small and due to discrete sampling)
    autocorr_x_ft=(fft2(fftshift(autocorr_x))*dx).real
  
    # spectrum of the filter that we will use to introduce correlations in the noise  
    # note that we take only positive values (negative values shall be small and due to discrete sampling)
    autocorr_x_ft[autocorr_x_ft<0]=0
    filt_x_ft=np.sqrt(autocorr_x_ft)
   
    # compute spectrum of the noise in space and multiply by the spectrum of the filter in space
    xi_ft = fft2(xi.reshape(nx,nx,n**2),axes=[0,1])*dx
    xi_filt_ft=np.multiply(xi_ft,filt_x_ft[:,:,np.newaxis])
      
    # transfrom back to space domain
    xi = fftshift(np.real(ifft2(xi_filt_ft,axes=[0,1])),axes=[0,1])/dx
    xi=xi.reshape(nx**2,n**2)

  if filter_phi is True:
    
  
    #----------------------    FILTER NOISE IN PHASE   -------------------------

    # scale white noise by square root of sampling interval      
    dphi=(2*np.pi)**2*np.sqrt(3)/2./n**2
    xi/=np.sqrt(dphi)
    
    gp=gl.GridProps(n,2*np.pi,0.)
      
    # autocorrelation in phase  
    if mises is True:      
      # 2D mises on twisted torus (sum of three waves that are 60 degrees apart)
      autocorr_phi=rhombus_mises(gp.phases,noise_sigma_phi)
    else:
      # 2D circulary symmetric Gaussian on rhombus             
      autocorr_phi=np.exp(- ( gp.phases[:,0]**2+ gp.phases[:,1]**2) /(2*noise_sigma_phi**2))
    
    # note that we retain only the real part (imaginary part shall be small and due to sampling)
    autocorr_phi_ft=gl.fourier_on_lattice(2*np.pi,gp.u1_rec,gp.u2_rec,gp.phases,autocorr_phi).real
    
    # note that we retain only positive values (negative values shall be small and due to sampling)
    autocorr_phi_ft[autocorr_phi_ft<0]=0
    
    # spectrum of the filter that we will use to introduce correlations in the noise  
    filt_phi_ft=np.sqrt(autocorr_phi_ft)
      
    # compute spectrum of the noise in phase and multiply by the spectrum of the filter in phase (lattice)
    xi_ft=gl.fourier_on_lattice(2*np.pi,gp.u1_rec,gp.u2_rec,gp.phases,xi.T)
    xi_filt_ft=np.multiply(xi_ft,filt_phi_ft[:,np.newaxis])
          
    # transfrom back to phase domain
    xi=gl.inverse_fourier_on_lattice(2*np.pi,gp.u1_rec,gp.u2_rec,gp.phases,xi_filt_ft)
    xi=xi.reshape(nx**2,n**2)
      
  
  # explicitly normalize to fixed variance (not strictly required but could give cleaner results)
  xi_norm=xi/xi.std()*np.sqrt(noise_variance)

  return xi_norm
    
    

  
def add_noise_and_normalize(pos,L,nx,n,
                            noise_sigma_x,
                            noise_sigma_phi,
                            input_mean,
                            signal_weight,
                            grids,
                            ):
  
  """
  This function adds a blanket of spatial noise to perfect grids.
  The noise has zero mean and variance equal to the grids variance.
  Noise correlation length in space is controlled by the parameter noise_sigma_x.
  Noise correlation length across neurons is controlled by the parameter noise_sigma_phi.
  This noise blanket is then added to the signal and the result clipped to positive values
  
  Note that the mean of each spatial map can vary across neurons in case of strongly correlated noise across space.
  Similarly, the mean population input can vary across locations in case of strongly correlated noise across neurons.
  Also note that if the noise is fully correlated both across space and across neurons, multiple realizations 
  are needed to compute the expected mean and variance. 
  
  Numerically, to ensure that the noise variance does not change after adding correlations we need to normalize by the square root
  of the filter's autocorellation value in 0 (peak). This is because the autocorrelation of a filtered stochastic process
  is the autocorrelation of the input multiplied by the autocorrelation of the filter. The autocorrelation is zero gives the variance
  of the stochastic process.
  
  """
  
  # clip and fix the mean of the grids
  grids=grids.clip(min=0)
  B=input_mean/grids.mean(axis=0)
  grids*=B
  
  # compute the grids variance which, this will scale the strength of the noise
  grids_variance=grids.var()


  if signal_weight < 1.:
    
  
    xi=gen_corr_noise_2d(L,nx,n,grids_variance,noise_sigma_phi,noise_sigma_x)

    # combine noise and grids. Because the noise is zero mean and we apply a static non-linearity,
    # the total input mean decreases with smaller signal weights. To compensate for this effect, we fit a constant c
    # such that the total input mean stays roughly constant after the non-linearity is applied
    c=0
    it=0
    while (it<=200):
      inputs=(grids*signal_weight+xi*(1-signal_weight)+c).clip(min=0)
      if inputs.mean()>=input_mean:
        break
      c+=0.1
      it+=1
      
    assert(it<200),'ERROR: Could not find a suitable normalization constant!'  
    
  else:
    inputs=grids
    xi=np.zeros_like(grids)


  inputs_flat=np.ascontiguousarray(inputs,dtype=np.float32)
  noise_flat=np.ascontiguousarray(xi,dtype=np.float32)
  grids_flat=np.ascontiguousarray(grids,dtype=np.float32)
        
  return inputs_flat,grids_flat,noise_flat
    
                          


class InputType:
  
  INPUT_GAU_GRID='GaussianGrid' 
  INPUT_GAU_GRID_PLUS_NOISE='GaussianGridPlusNoise' 
  INPUT_GAU_NOISY_CENTERS='GaussianNoisyCenters'  
  INPUT_GAU_RANDOM_CENTERS='GaussianRandomCenters'

  # sum of positive and negativa Gaussians: shifted to have positive rates and rescaled to have fixed mean
  INPUT_GAU_MIX='MixOfGau'             

  # sum of positive and negativa Gaussians: shifted to have fixed mean (positive and negative rates!!)
  INPUT_GAU_MIX_NEG='MixOfGauNeg'      

  # sum of positiva Gaussians with variable amps rescaled to fixed mean
  INPUT_GAU_MIX_POS='MixOfGauPos'      

  # sum of positiva Gaussians with fixed amps rescaled to fixed mean
  INPUT_GAU_MIX_POS_FIXAMP='MixOfGauPosFixAmp' 
  
  INPUT_RAND='Random'
  INPUT_RAND_CORR='RandomCorrelated'
  
  INPUT_BVC='BVC'
  
  # grids plus addditive noise
  INPUT_NOISY_GRID='NoisyGrid'  
  INPUT_NOISY_GRID_UNIFORM_ANGLES='NoisyGridUniformAngles'  
  
  # noisy grids with multiple modules
  INPUT_NOISY_GRID_MODULES='NoisyGridModules'  
  
  # noisy grids with a gradient of scales
  INPUT_NOISY_GRID_SCALEGRADIENT='NoisyGridScaleGradient'
  
  # noisy grids with a gradient of scales and orientations
  INPUT_NOISY_GRID_SCALE_ANGLE_GRADIENT='NoisyGridScaleAngleGradient'

  # nosy grids with a gradient of field-size ratios
  INPUT_NOISY_GRID_FIELD_SIZE_GRADIENT='NoisyGridsFieldSizeGradient'
    
  
class SpatialInputs(object):
  
  results_path=os.path.join(sl.get_results_path(),'spatial_inputs')
  
  basic_key_params=['inputs_type','n','nx','L','periodic_inputs',
                    'input_mean','sigma','same_fixed_norm','fixed_norm']
  
  key_params_map={
  
    InputType.INPUT_GAU_GRID: basic_key_params+['shift_and_clip','outside_ratio'],
    InputType.INPUT_GAU_GRID_PLUS_NOISE: basic_key_params+['shift_and_clip','outside_ratio','inputs_seed',
                                                           'signal_weight','noise_sigma_x','noise_sigma_phi'],

    InputType.INPUT_GAU_NOISY_CENTERS: basic_key_params+['shift_and_clip','centers_std','inputs_seed','outside_ratio'],
    InputType.INPUT_GAU_RANDOM_CENTERS: basic_key_params+['shift_and_clip','inputs_seed','outside_ratio'],

    InputType.INPUT_GAU_MIX: basic_key_params+['num_gau_mix','inputs_seed'],  
    InputType.INPUT_GAU_MIX_NEG: basic_key_params+['num_gau_mix','inputs_seed'],  
    InputType.INPUT_GAU_MIX_POS: basic_key_params+['num_gau_mix','inputs_seed'],  
    InputType.INPUT_GAU_MIX_POS_FIXAMP: basic_key_params+['num_gau_mix','inputs_seed'],  
    InputType.INPUT_RAND: basic_key_params+['inputs_seed'],        
    InputType.INPUT_RAND_CORR: basic_key_params+['inputs_seed'],

    InputType.INPUT_BVC: basic_key_params+['sigma_rad_0','beta','sigma_ang'],



    InputType.INPUT_NOISY_GRID: [ 'inputs_type','n','nx','L',
                                  'grid_T','grid_angle','grid_T_sigma', 'grid_angle_sigma',
                                  'input_mean','jitter_variance','jitter_sigma_phi',
                                  'inputs_seed',
                                  'signal_weight','noise_sigma_x','noise_sigma_phi',
                                  'same_fixed_norm','fixed_norm','zero_phase','scale_field_size_ratio'],
                                  
    InputType.INPUT_NOISY_GRID_UNIFORM_ANGLES:['inputs_type','n','nx','L',
                                  'grid_T','grid_T_sigma',
                                  'input_mean','jitter_variance','jitter_sigma_phi',
                                  'inputs_seed',
                                  'signal_weight','noise_sigma_x','noise_sigma_phi',
                                  'same_fixed_norm','fixed_norm','zero_phase','scale_field_size_ratio'],


    InputType.INPUT_NOISY_GRID_MODULES:
                                [ 'inputs_type','nx','L',
                                  'num_modules','module_n',
                                  'base_grid_T','grid_angle','scale_ratio','grid_T_sigmas','grid_angle_sigmas',
                                  'input_mean','jitter_variance','jitter_sigma_phi',
                                  'inputs_seed',
                                  'signal_weight','noise_sigma_x','noise_sigma_phi',
                                  'same_fixed_norm','fixed_norm',
                                  'uniform_random_angle',
                                  'random_angle_per_module',
                                  'normalize_boundary_effects',
                                  'zero_phase','scale_field_size_ratio'
                                 ],
                                 
    # n**2 grids with same phase and orientation but scales evenly distributed between grid_T_min, grid_T_max
    InputType.INPUT_NOISY_GRID_SCALEGRADIENT: [ 'inputs_type','n','nx','L',
                                  'grid_T','grid_angle',
                                  'input_mean','jitter_variance', 'jitter_sigma_phi',
                                  'inputs_seed',
                                  'grid_T_min','grid_T_max',
                                  'signal_weight','noise_sigma_x','noise_sigma_phi',
                                  'same_fixed_norm','fixed_norm','zero_phase','scale_field_size_ratio'],

    # n**2 grids with same phase and orientation but scales evenly distributed between grid_T_min, grid_T_max
    InputType.INPUT_NOISY_GRID_FIELD_SIZE_GRADIENT: [ 'inputs_type','n','nx','L',
                                  'grid_T','grid_angle','grid_T_sigma', 'grid_angle_sigma',
                                  'input_mean','jitter_variance', 'jitter_sigma_phi',
                                  'inputs_seed',
                                  'field_size_ratio_min','field_size_ratio_max',
                                  'signal_weight','noise_sigma_x','noise_sigma_phi',
                                  'same_fixed_norm','fixed_norm','zero_phase'],

    # n**2 grids with same phase but scales evenly distributed between grid_T_min, grid_T_max and orientation evenly distributed between -pi/6 and pi/6
    InputType.INPUT_NOISY_GRID_SCALE_ANGLE_GRADIENT: [ 'inputs_type','n','nx','L',
                                  'grid_T','grid_angle',
                                  'input_mean','jitter_variance','jitter_sigma_phi',
                                  'inputs_seed',
                                  'grid_T_min','grid_T_max','signal_weight',
                                  'noise_sigma_x','noise_sigma_phi',
                                  'same_fixed_norm','fixed_norm','zero_phase','scale_field_size_ratio'],

                                       
  }

  @staticmethod
  def get_key_params(paramMap):
    
    if paramMap.has_key('tap_inputs'):
      return SpatialInputs.key_params_map[paramMap['inputs_type']]+['tap_inputs',
      'tap_border_size','tap_border_type']
      
    else:
      return SpatialInputs.key_params_map[paramMap['inputs_type']]
    
    
    
  @staticmethod  
  def get_data_path(paramMap):
    return os.path.join(SpatialInputs.results_path,SpatialInputs.get_id(paramMap)+'_data.npz')
    
    
  def __init__(self,paramMap,do_print=False,comp_gridness_score=False,comp_tuning_index=True,force_gen=False,keys_to_load=[]):

    keyParamMap={}
    
    # import parameters
    for param in SpatialInputs.get_key_params(paramMap):
      keyParamMap[param]=paramMap[param]
      setattr(self,param,paramMap[param])
      #print param,paramMap[param]

    if self.inputs_type==InputType.INPUT_NOISY_GRID_MODULES:
      assert(len(self.module_n)==self.num_modules)
      self.N=(np.array(self.module_n)**2).sum()
      
    else:
      self.N=self.n**2
        
      
    self.str_id=sl.gen_string_id(keyParamMap)
    
    #print self.str_id
    
    self.hash_id=sl.gen_hash_id(self.str_id)

    
    self.paramsPath=os.path.join(self.results_path,self.hash_id+'_log.txt')
    self.dataPath=os.path.join(self.results_path,self.hash_id+'_data.npz')   



    # generate and save data    
    if force_gen or not os.path.exists(self.dataPath):
      print 'Input data non present, generating...'
      self.gen_data(do_print,comp_gridness_score,comp_tuning_index)

    # load data 
    self.load_data(do_print,keys_to_load)



        
  def load_data(self,do_print,keys_to_load=[]):
    """
    Loads data from disk
    """
    
    if do_print:
      print
      print 'Loading input data, Id = %s'%self.hash_id
    
    data= np.load(self.dataPath,mmap_mode='r',allow_pickle=True)
    
    loaded_keys=[]
    
    if len(keys_to_load)==0:
      for k,v in data.items():
        setattr(self,k,v)
        loaded_keys.append(k)
    else: 
      for k in keys_to_load:
        setattr(self,k,data[k])
        loaded_keys.append(k)

     
    if do_print:
      print 'Loaded variables: '+' '.join(loaded_keys)
    


  def gen_data(self,do_print=False,comp_gridness_score=False,comp_tuning_index=True):
    """
    Generates input data and saves it to disk
    """
  
    if do_print:  
      print
      print 'Generating inputs data, id = %s'%self.hash_id
      
      print 'Computing gridness score: %s'%str(comp_gridness_score)
      print 'Computing tuning index: %s'%str(comp_tuning_index)
          
    # set seed
    if self.inputs_type not in (InputType.INPUT_GAU_GRID, InputType.INPUT_BVC)  :
      seed(self.inputs_seed)
    

    
    self.dx=self.L/self.nx
    self.X,self.Y=np.mgrid[-self.L/2:self.L/2:self.dx,-self.L/2:self.L/2:self.dx]
    self.pos=np.array([np.ravel(self.X), np.ravel(self.Y)]).T
    
    toSaveMap={'inputs_type':self.inputs_type}

    if hasattr(self,'sigma'):
      # gaussian peak
      self.amp=self.input_mean*self.L**2/(2*np.pi*self.sigma**2)


    # noisy grids 
    if self.inputs_type in (InputType.INPUT_NOISY_GRID,
                            InputType.INPUT_NOISY_GRID_UNIFORM_ANGLES,
                            InputType.INPUT_NOISY_GRID_SCALEGRADIENT,
                            InputType.INPUT_NOISY_GRID_SCALE_ANGLE_GRADIENT,
                            InputType.INPUT_NOISY_GRID_FIELD_SIZE_GRADIENT):
      
      self.gen_inputs_noisy_grids()
      toSaveMap.update({'inputs_flat':self.inputs_flat,
                        'noise_flat':self.noise_flat,
                        'input_mean':self.input_mean,
                        'grid_T_vect':self.grid_T_vect,
                        'grid_angle_vect':self.grid_angle_vect,
                        'grid_peak_pos':self.grid_peak_pos,
                        'noisy_peak_pos':self.noisy_peak_pos})
    

    if self.inputs_type==InputType.INPUT_NOISY_GRID_FIELD_SIZE_GRADIENT:
      toSaveMap.update({'scale_field_size_ratio_vect':self.scale_field_size_ratio_vect})
      
    # noisy grids with modules                        
    if self.inputs_type==InputType.INPUT_NOISY_GRID_MODULES:



      self.gen_inputs_noisy_grids_modules()
      toSaveMap.update({'inputs_flat':self.inputs_flat,
                        'input_mean':self.input_mean,
                        'grid_T_vect':self.grid_T_vect,
                        'grid_angle_vect':self.grid_angle_vect,

                        'grid_peak_pos_module_0':self.grid_peak_pos_per_module[0],
                        'grid_peak_pos_module_1':self.grid_peak_pos_per_module[1],
                        'grid_peak_pos_module_2':self.grid_peak_pos_per_module[2],
                        'grid_peak_pos_module_3':self.grid_peak_pos_per_module[3],
                        'grid_T_sigmas':self.grid_T_sigmas
                        }
                       )
       

    # gaussian inputs
    elif self.inputs_type in (InputType.INPUT_GAU_GRID,InputType.INPUT_GAU_NOISY_CENTERS, InputType.INPUT_GAU_RANDOM_CENTERS):
      self.gen_inputs_gaussian()
      toSaveMap.update({'inputs_flat':self.inputs_flat,'amp':self.amp,'centers':self.centers})

    elif self.inputs_type in (InputType.INPUT_GAU_GRID_PLUS_NOISE,):
      self.gen_inputs_gaussian()
      toSaveMap.update({'inputs_flat':self.inputs_flat,'signal_flat':self.signal_flat,'noise_flat':self.noise_flat,'centers':self.centers})
      
    # mixture of gaussians
    elif self.inputs_type in (InputType.INPUT_GAU_MIX,InputType.INPUT_GAU_MIX_NEG, InputType.INPUT_GAU_MIX_POS,InputType.INPUT_GAU_MIX_POS_FIXAMP,InputType.INPUT_RAND,InputType.INPUT_RAND_CORR):
      self.gen_inputs_random()
      toSaveMap.update({'inputs_flat':self.inputs_flat,'input_scalings':self.input_scalings,'random_amps':self.random_amps})
            
    # boundary vector cells        
    elif self.inputs_type == InputType.INPUT_BVC:
      self.gen_inputs_bvc()
      toSaveMap.update({'inputs_flat':self.inputs_flat})

    # compute spectra
    self.comp_input_spectra()
    spectraMap={'in_freqs':self.in_freqs,'in_mean_dft':self.in_mean_dft,'in_mean_amp':self.in_mean_amp,
    'in_pw_profiles':self.in_pw_profiles,'in_mean_pw_profile':self.in_mean_pw_profile,
    'in_amp_profiles':self.in_amp_profiles,'in_mean_amp_profile':self.in_mean_amp_profile}     
    toSaveMap.update(spectraMap)

    # compute gridness scores    
    if comp_gridness_score is True:
      self.comp_gridness_scores()
      scoresMap={'in_scores':self.in_scores,'in_spacings':self.in_spacings,
      'in_angles':self.in_angles,'in_phases':self.in_phases}    
      toSaveMap.update(scoresMap)

    # compute grid tuning indexes
    if comp_tuning_index is True:
      self.comp_tuning_indexes()
      tuningMap={'grid_tuning_in':self.grid_tuning_in}    
      toSaveMap.update(tuningMap)

     
    # save
    sl.ensureParentDir(self.dataPath)
    np.savez(self.dataPath,**toSaveMap)      
    
    
    
  def comp_input_spectra(self):
    """
    Compute input power spectra
    """
    
    self.nx=int(self.nx)

    
    inputs_mat=self.inputs_flat.reshape(self.nx,self.nx,self.N)

    in_allfreqs = fftshift(fftfreq(self.nx,d=float(self.L)/self.nx))
    self.in_freqs=in_allfreqs[self.nx/2:]
    
    in_dft_flat=fftshift(fft2(inputs_mat,axes=[0,1]),axes=[0,1])*(float(self.L)/self.nx)**2

    in_pw=abs(in_dft_flat)**2
    in_amp=abs(in_dft_flat)
      
    self.in_mean_dft=np.mean(in_dft_flat,axis=2)
    self.in_mean_amp=np.mean(in_amp,axis=2)
    
    self.in_amp_profiles=gl.dft2d_profiles(in_amp)
    self.in_pw_profiles=gl.dft2d_profiles(in_pw)
    self.in_mean_amp_profile=np.mean(self.in_amp_profiles,axis=0)
    self.in_mean_pw_profile=np.mean(self.in_pw_profiles,axis=0)
    
    
  def comp_gridness_scores(self):

    self.in_scores,self.in_spacings,self.in_angles,self.in_phases=gl.compute_scores_evo(
    self.inputs_flat,self.nx,self.L,num_steps=10) #50

  def comp_tuning_indexes(self):
    self.grid_tuning_in=gl.comp_grid_tuning_index(self.L,self.nx,self.inputs_flat)
    
    
  def normalize_to_fixed_norm(self,fixed_norm):
    for cell_idx in xrange(self.N):
      grid=self.inputs_flat[:,cell_idx]
      norm=np.sqrt((grid**2).mean())
      grid*=fixed_norm/norm     
      self.inputs_flat[:,cell_idx]=grid

 

  def gen_inputs_noisy_grids_modules(self):
    """
    Noisy jittered grids with modular scales
    """

            
    seed(self.inputs_seed)
          

    self.grid_peak_pos_per_module=[]
    self.noisy_peak_pos_per_module=[]
    
    # iterate for each module
    for module_idx in xrange(self.num_modules):

      # each module has a different preferred angle (chosen at random)
      if self.random_angle_per_module is True:
        grid_angle=np.random.uniform(low=-np.pi/6,high=np.pi/6)
      else:
        grid_angle=self.grid_angle
      
      # get parameters of the current module
      grid_T=self.base_grid_T*self.scale_ratio**module_idx      
      n=self.module_n[module_idx]      
      grid_T_sigma=self.grid_T_sigmas[module_idx]
      grid_angle_sigma=self.grid_angle_sigmas[module_idx]
            
      # create a vector of grid spacings within a module
      grid_T_vect=np.ones((n**2,1))*grid_T+grid_T_sigma*np.random.randn(n**2,1)
      
      # create a vector of grid angles within a module
      if self.uniform_random_angle is True:
        grid_angle_vect=np.random.uniform(low=-np.pi/6,high=np.pi/6,size=(n**2,1))
      else:
        grid_angle_vect=np.ones((n**2,1))*grid_angle+grid_angle_sigma*np.random.randn(n**2,1)
    
      # generate regular grids for the current module
      grids,grid_peak_pos,noisy_peak_pos,grid_T_vect=gen_noisy_grids(self.L,self.nx,self.dx,
                                                         n,grid_T,grid_angle,
                                                         grid_T_vect,grid_angle_vect,
                                                         self.jitter_variance,self.jitter_sigma_phi,
                                                         self.input_mean,
                                                         self.scale_field_size_ratio*np.ones_like(n**2),self.zero_phase)                
      # add noise 
      inputs_flat,grids_flat,noise_flat=add_noise_and_normalize(
          self.pos,self.L,self.nx,n,self.noise_sigma_x,self.noise_sigma_phi,
          self.input_mean,self.signal_weight,grids)

      # save interesting variables
      self.grid_peak_pos_per_module.append(grid_peak_pos)
      self.noisy_peak_pos_per_module.append(noisy_peak_pos)
                  
      if module_idx==0:
        self.inputs_flat=inputs_flat
        self.grid_T_vect=grid_T_vect
        self.grid_angle_vect=grid_angle_vect
        
      else:
        self.inputs_flat=np.hstack([self.inputs_flat,inputs_flat])
        self.grid_T_vect=np.vstack([self.grid_T_vect,grid_T_vect])       
        self.grid_angle_vect=np.vstack([self.grid_angle_vect,grid_angle_vect])       
        
    # === end of for loop iterating over modules ===        
        
    # normalize to fixed norm        
    if self.same_fixed_norm is True:
      self.normalize_to_fixed_norm(self.fixed_norm)
      
    # normalize boundary effects  
    if self.normalize_boundary_effects is True:
      
      module_offset=0
      for module_idx in xrange(self.num_modules):

        n=self.module_n[module_idx]      
        module_idxs=np.arange(n**2)+module_offset

        mean_input=self.inputs_flat[:,module_idxs].mean(axis=1)
        diff=mean_input-mean_input.min()
  
        self.inputs_flat[:,module_idxs]+=diff.max()-diff[:,np.newaxis]
        
        avg_mean_input=self.inputs_flat[:,module_idxs].mean(axis=1).mean()
        print avg_mean_input
        
        self.inputs_flat[:,module_idxs]+=(self.input_mean+0.5-avg_mean_input)  
        
        module_offset+=n**2


                                  
          
      
  def gen_inputs_noisy_grids(self):   
    """
    Generates regular grids and jitters the fields location
    """
    
    seed(self.inputs_seed)
    
    
    # standard case in which grid scale is normally distributed around grid_T with a certain sigma
    if (self.inputs_type == InputType.INPUT_NOISY_GRID or self.inputs_type ==InputType.INPUT_NOISY_GRID_FIELD_SIZE_GRADIENT):
      grid_T_vect=np.ones((self.n**2,1))*self.grid_T+self.grid_T_sigma*np.random.randn(self.n**2,1)
      grid_angle_vect=np.ones((self.n**2,1))*self.grid_angle+self.grid_angle_sigma*np.random.randn(self.n**2,1)
            
    # case in which grid scale is evenly spaced within a certain interval   
    elif self.inputs_type == InputType.INPUT_NOISY_GRID_SCALEGRADIENT:
      
      assert(self.zero_phase is True)
      grid_T_vect=np.linspace(self.grid_T_min,self.grid_T_max,self.n**2)[:,np.newaxis]
      grid_angle_vect=np.ones((self.n**2,1))*self.grid_angle
      
    elif self.inputs_type ==InputType.INPUT_NOISY_GRID_UNIFORM_ANGLES:
      grid_T_vect=np.ones((self.n**2,1))*self.grid_T+self.grid_T_sigma*np.random.randn(self.n**2,1)
      grid_angle_vect=np.random.uniform(low=-np.pi/6.,high=np.pi/6,size=(self.n**2,1))
      self.grid_angle=0.

    # case in which both grid scale and grid_orientation are evenly distributed
    elif self.inputs_type == InputType.INPUT_NOISY_GRID_SCALE_ANGLE_GRADIENT:
      
      assert(self.zero_phase is True)
      grid_T_ran=np.linspace(self.grid_T_min,self.grid_T_max,self.n)
      grid_angle_ran=np.linspace(-np.pi/6.,np.pi/6.,self.n)
    
      grid_angle_mat,grid_T_mat=np.meshgrid(grid_angle_ran,grid_T_ran)
      
      grid_T_vect=grid_T_mat.ravel()[:,np.newaxis]
      grid_angle_vect=grid_angle_mat.ravel()[:,np.newaxis]
      
    # generate a gradient of field-size ratios   
    if self.inputs_type == InputType.INPUT_NOISY_GRID_FIELD_SIZE_GRADIENT:
      self.scale_field_size_ratio_vect=np.linspace(self.field_size_ratio_min,self.field_size_ratio_max,self.n**2)
    # the field-size ratio is constant for every cell  
    else:
      self.scale_field_size_ratio_vect=np.ones(self.n**2)*self.scale_field_size_ratio
      
      
      
    inputs,grid_peak_pos,noisy_peak_pos,grid_T_vect=gen_noisy_grids(self.L,self.nx,self.dx,self.n,
                                                                    self.grid_T,self.grid_angle,
                                                                    grid_T_vect,grid_angle_vect,
                                                                    self.jitter_variance,self.jitter_sigma_phi,
                                                                    self.input_mean,
                                                                    self.scale_field_size_ratio_vect,
                                                                    self.zero_phase)
    
    self.inputs_flat,self.grids_flat,self.noise_flat=add_noise_and_normalize(self.pos,self.L,self.nx,self.n,
                                                                   self.noise_sigma_x,
                                                                   self.noise_sigma_phi,
                                                                   self.input_mean,self.signal_weight,inputs)
  
    if self.same_fixed_norm is True:
      self.normalize_to_fixed_norm(self.fixed_norm)
      
    self.inputs_flat=np.ascontiguousarray(self.inputs_flat, dtype=np.float32)
    self.grid_peak_pos=grid_peak_pos        # N x 2 x num_peaks
    self.noisy_peak_pos=noisy_peak_pos      # N x 2 x num_peaks
    self.grid_T_vect=grid_T_vect
    self.grid_angle_vect=grid_angle_vect
    
  
  

  def gen_inputs_bvc(self):
    """
    Generates Boundary Vector Cell inputs
    """
   
        
    d_ran=np.linspace(0.1,self.L/2.,num=self.n,endpoint=False)  
    phi_ran=np.linspace(0,2*np.pi,num=self.n,endpoint=False) 
    
    # standard deviation of the gaussian as a function of distance
    sigma_rad = lambda d: (d/self.beta+1)*self.sigma_rad_0
    
    # boundary vector field, i.e., the blob
    bvf= lambda p_dist,p_ang,d,phi: np.exp(-(p_dist-d)**2/(2*sigma_rad(d)**2))/(np.sqrt(2*np.pi)*sigma_rad(d)) *\
                                    np.exp(-(np.remainder((p_ang-phi),2*np.pi)-np.pi)**2/(2*self.sigma_ang**2))/(np.sqrt(2*np.pi)*self.sigma_ang)    
    
    # position of the walls
    east_wall=np.where(self.pos[:,0]==self.X.min())[0]
    west_wall=np.where(self.pos[:,0]==self.X.max())[0]
    north_wall=np.where(self.pos[:,1]==self.Y.max())[0]
    south_wall=np.where(self.pos[:,1]==self.Y.min())[0]
    wall_pos=np.hstack([east_wall,west_wall,north_wall,south_wall])
    num_walls=4
    
    pos_shift=self.pos[np.newaxis,:,:]
    p_wall_shift=self.pos[wall_pos,:][:,np.newaxis,:]-pos_shift
    p_wall_shift=p_wall_shift.reshape(self.nx*num_walls*self.nx**2,2)
    
    p_wall_dist=np.sqrt(np.sum(p_wall_shift**2,axis=1))
    p_wall_ang=np.arctan2(p_wall_shift[:,1],p_wall_shift[:,0])
    
    #p_dist=sqrt(np.sum(self.pos**2,axis=1))
    #p_ang=arctan2(self.pos[:,1],self.pos[:,0])
    

    self.inputs_flat=np.zeros((self.nx**2,self.N),dtype=np.float32)
    #self.blobs_flat=zeros((self.nx**2,self.N),dtype=float32)

    start_clock=clock()
    idx=0
    for d in d_ran:
      for phi in phi_ran:
        sl.print_progress(idx,self.N,start_clock=start_clock)
        self.inputs_flat[:,idx]=np.mean(bvf(p_wall_dist,p_wall_ang,d,phi).reshape(self.nx*num_walls,self.nx**2),axis=0)
        #self.blobs_flat[:,idx]=bvf(p_dist,p_ang,d,phi)
        idx+=1
        
    # scale to fixed mean
    self.input_scalings=self.input_mean/np.mean(self.inputs_flat,axis=0)    
    self.inputs_flat*=self.input_scalings

      
  def gen_inputs_random(self):
    """
    Generate inputs from low-pass filtered noise
    """
    
    #dx=self.L/self.nx
    
    g_fun = lambda p:self.amp*np.exp(-np.sum(p**2,1)/(2*self.sigma**2))

    # white noise 

    # fixed amplitude
    if self.inputs_type == InputType.INPUT_GAU_MIX_POS_FIXAMP:
      wn=np.ones((self.nx,self.nx,self.N)).astype(np.float32)
    
    # only positive values
    if self.inputs_type in (InputType.INPUT_GAU_MIX_POS,InputType.INPUT_RAND,InputType.INPUT_RAND_CORR):
      #wn=abs(randn(self.nx,self.nx,self.N).astype(np.float32))
      wn=np.random.uniform(size=(self.nx,self.nx,self.N)).astype(np.float32)
      
    # positive and negative
    elif self.inputs_type in (InputType.INPUT_GAU_MIX,InputType.INPUT_GAU_MIX_NEG):
      wn=randn(self.nx,self.nx,self.N).astype(np.float32)

    if self.inputs_type in (InputType.INPUT_GAU_MIX,InputType.INPUT_GAU_MIX_NEG,
                            InputType.INPUT_GAU_MIX_POS,InputType.INPUT_GAU_MIX_POS_FIXAMP):

      assert(self.num_gau_mix<=self.nx**2)
      if self.num_gau_mix<self.nx**2:

        # set N-num_gau_mix elements to zero          
        wn_flat=wn.reshape(self.nx**2,self.N)

        self.random_amps=[]
        for i in xrange(self.N):        
          idxs_to_zero=permutation(self.nx**2)[:self.nx**2-self.num_gau_mix]
          non_zero_idxs=np.setdiff1d(np.arange(self.nx**2),idxs_to_zero)
          wn_flat[idxs_to_zero,i]=0
          self.random_amps.append(wn_flat[non_zero_idxs,i].tolist())
          
        wn=wn_flat.reshape(self.nx,self.nx,self.N).astype(np.float32)
                
    # convolution             
    wn_dft=fft2(wn,axes=[0,1]).astype(np.complex64)
    gx=g_fun(self.pos).reshape(self.nx,self.nx)
    filt_x=np.real(fftshift(ifft2(fft2(gx)[:,:,np.newaxis]*wn_dft,axes=[0,1]),axes=[0,1])).astype(np.float32)
    filt_xu=filt_x
    
    # add Gaussian correlation across inputs    
    if self.inputs_type==InputType.INPUT_RAND_CORR:
      
      self.get_inputs_centers()
      gu=g_fun(self.centers).reshape(self.n,self.n)
      
      filt_x_dftu=fft2(filt_x.reshape(self.nx*self.nx,self.n,self.n),axes=[1,2]).astype(np.complex64)
      filt_xu=np.real(fftshift(ifft2(fft2(gu)[np.newaxis,:,:]*filt_x_dftu,axes=[1,2]),axes=[1,2])).astype(np.float32)
      
    # normalization
    filt_xu_flat=filt_xu.reshape(self.nx*self.nx,self.N)

    # adjust baseline by shifting the minimum at zero
    if self.inputs_type in (InputType.INPUT_GAU_MIX,InputType.INPUT_RAND,InputType.INPUT_RAND_CORR):
      filt_xu_flat-=np.amin(filt_xu_flat,axis=0)

    # shift to fixed mean    
    if self.inputs_type == InputType.INPUT_GAU_MIX_NEG:
      filt_xu_flat-=np.mean(filt_xu_flat,axis=0) 
      filt_xu_flat+=self.input_mean
      self.input_scalings=np.ones(self.N)
      
    else:
      # scale to fixed mean
      self.input_scalings=self.input_mean/np.mean(filt_xu_flat,axis=0)    
      filt_xu_flat*=self.input_scalings

    self.inputs_flat=np.ascontiguousarray(filt_xu_flat,dtype=np.float32)
    
    if self.same_fixed_norm is True:
      self.normalize_to_fixed_norm(self.fixed_norm)
            
      
      
      
  def gen_inputs_gaussian(self):
    """
    Generate Gaussian inputs
    """
    
    g_fun = lambda p: np.exp(-np.sum(p**2,2)/(2*self.sigma**2))#*self.amp   

    # get input centers      
    self.get_inputs_centers()
    
    # gaussian input
    P0=self.pos[np.newaxis,:,:]-self.centers[:,np.newaxis,:]
    inputs=g_fun(P0).astype(np.float32)

      
    # add periodic boundaries    
    if self.periodic_inputs is True:
      for idx,center_shifted in enumerate(([0.,self.L], [0.,-self.L], [self.L,0.], [-self.L,0.], [-self.L,-self.L], [-self.L,self.L], [self.L,self.L], [self.L,-self.L])):
        P=P0+np.array(center_shifted)
        inputs+=g_fun(P).astype(np.float32)
        
        
    if self.shift_and_clip is True:
      # shift down and clip (to mimic perfect grid inputs)
      inputs-=0.5
      inputs=inputs.clip(min=0)
      
    if self.inputs_type==InputType.INPUT_GAU_GRID_PLUS_NOISE:
      # add noise 
      self.inputs_flat,self.signal_flat,self.noise_flat=add_noise_and_normalize(self.pos,self.L,self.nx,self.n,
                                                                   self.noise_sigma_x,
                                                                   self.noise_sigma_phi,
                                                                   self.input_mean,self.signal_weight,inputs.T)
    else:
      # normalize to fixed mean
      inputs=inputs/inputs.mean(axis=0)*self.input_mean                
      self.inputs_flat=np.ascontiguousarray(inputs.T.reshape(self.nx*self.nx,self.N), dtype=np.float32)


    if self.same_fixed_norm is True:
      self.normalize_to_fixed_norm(self.fixed_norm)
                    
      
      
      
  def get_inputs_centers(self):
    """
    Computes input centers
    """
    
    if self.inputs_type==InputType.INPUT_GAU_RANDOM_CENTERS:
  
      # randomly distributed gaussian centers      
      SSX=(rand(self.n,self.n)-0.5)*self.L*self.outside_ratio
      SSY=(rand(self.n,self.n)-0.5)*self.L*self.outside_ratio
      self.centers= np.array([np.ravel(SSX), np.ravel(SSY)]).T
      
    else:

      # sample gaussian centers on a regular grid
      if self.periodic_inputs is True:
        ran,step=np.linspace(-self.L/2.,self.L/2.,self.n,endpoint=False,retstep=True)      
      else:
        
        ran,step=np.linspace(-self.L/2*self.outside_ratio,self.L/2*self.outside_ratio,self.n,endpoint=False,retstep=True)
        ran=ran+step/2.
        
      SSX,SSY = np.meshgrid(ran,ran)
      self.centers= np.array([np.ravel(SSX), np.ravel(SSY)]).T
    
      if self.inputs_type==InputType.INPUT_GAU_NOISY_CENTERS:
        NX=randn(self.n,self.n)*self.centers_std
        NY=randn(self.n,self.n)*self.centers_std
        
        self.centers+= np.array([np.ravel(NX), np.ravel(NY)]).T

  def __plot_sample_with_modules(self,num_samples=3):
    
    assert(self.inputs_type==InputType.INPUT_NOISY_GRID_MODULES)
    
    import pylab as pl
    import plotlib as pp

    pl.figure(figsize=(8,6))

    module_offset=0
    plot_idx=1
    for module_idx in xrange(self.num_modules):
    
        module_N=self.module_n[module_idx]**2
        
        module_idxs=np.arange(module_N)+module_offset
        for cell_idx in xrange(num_samples):   
          grid_map=self.inputs_flat[:,module_idxs[cell_idx]].reshape(self.nx,self.nx).T
          pl.subplot(self.num_modules,num_samples,plot_idx,aspect='equal')
          pl.pcolormesh(grid_map,
          rasterized=True)
          pp.noframe()
          plot_idx+=1
          
        module_offset+=module_N
    

  def plot_space_phase_sample(self,num_neurons=9,num_pos=9,vmin=None,vmax=None,plot_noise=False):
    
    assert(self.inputs_type==InputType.INPUT_NOISY_GRID)
    
    import gridlib as gl
    import plotlib as pp
    import pylab as pl

    if plot_noise is True:
      signal=self.noise_flat
    else:
      signal=self.inputs_flat
      
    gp=gl.GridProps(self.n,1.,0.)
    
    dx=self.L/self.nx

    X,Y=np.mgrid[-self.L/2:self.L/2:dx,-self.L/2:self.L/2:dx]
    
    # get all positions
    p0=np.array([0.,-.75])
    all_pos=[]
    p=p0.copy()
    for pos_cont in xrange(num_pos):
      p=p0+pos_cont*np.array([0.0,0.15])
      all_pos.append(p)
    
    # get all neuron indexes
    all_phase_idxs=[]
    step=1/4.
    for i in (0,1,2):
      for j in (0,1,2):
        phase=(i-1)*step*gp.u1+(j-1)*step*gp.u2
        phase_idx=gl.get_pos_idx(phase,gp.phases)
        all_phase_idxs.append(phase_idx)
        
    #all_neuron_idxs = (np.arange(num_neurons)-num_neurons)+zero_phase_idx
    
    
    pl.figure(figsize=(18,10))
    
    plot_idx=1
    
    if vmin is None:
      vmin = signal.min()
      
    if vmax is None:
      vmax=signal.max()
    
    for phase_idx in all_phase_idxs:
      
      pl.subplot(2,num_neurons,plot_idx,aspect='equal')
      pl.pcolormesh(X,Y,signal[:,phase_idx].reshape(self.nx,self.nx),vmin=vmin,vmax=vmax)
      #pl.colorbar()
      
      for pos_cont in xrange(num_pos):
        pl.plot(all_pos[pos_cont][0],all_pos[pos_cont][1],mfc=[0,0,0,0.],marker='o',mec='k',mew=1.5)
        
        
    
      pl.title('min: %.2f\nmax: %.2f\nmean:%.2f'%(signal[:,phase_idx].min(),
                                                  signal[:,phase_idx].max(),
                                                  signal[:,phase_idx].mean()),fontsize=6)
      pp.noframe()
      plot_idx+=1
      
    
    for pos_cont in xrange(num_pos):
    
      pl.subplot(2,num_pos,plot_idx,aspect='equal')
          
      x_pos_idx=pp.plot_population_activity(signal.T,self.L,all_pos[pos_cont],vmin=vmin,vmax=vmax)
      
      
      for phase_idx in all_phase_idxs:
        pl.plot(gp.phases[phase_idx,0],gp.phases[phase_idx,1],mfc=[0,0,0,0.],marker='o',mec='k',mew=1.5)
      
      pl.title('min: %.2f\nmax: %.2f\nmean:%.2f'%(signal[x_pos_idx,:].min(),
                                                  signal[x_pos_idx,:].max(),
                                                  signal[x_pos_idx,:].mean()),fontsize=6)
      
      pp.noframe()
      plot_idx+=1  
  


  def plot_scale_angle_dists(self):
    
    import pylab as pl
    import plotlib as pp
      
    assert (hasattr(self,'grid_T_vect') and hasattr(self,'grid_angle_vect'))    
    
    pl.figure()
    pl.subplots_adjust(left=0.2,bottom=0.2,wspace=0.3)
    pl.subplot(121)
    h,e=np.histogram(self.grid_T_vect,bins=25,range=[self.grid_T-5*self.grid_T_sigma,self.grid_T+5*self.grid_T_sigma])
    pl.bar(e[1:],h*1.0/h.sum(),width=(e[1]-e[0]),color='k',edgecolor='k')
    pp.custom_axes()
    pl.xlim([.4,.8])
    pl.xlabel('Grid scale [m]')  
    
    pl.subplot(122)
    h,e=np.histogram(self.grid_angle_vect*180/np.pi,bins=25,range=[-5,5])
    pl.bar(e[1:],h*1.0/h.sum(),width=(e[1]-e[0]),color='k',edgecolor='k')
    
    pp.custom_axes()
    pl.xlabel('Grid angle [deg]')  


  def plot_sample(self,random=False,num_samples=16,input_idxs=None, title=False,plot_colorbar=False,plot_peak_pos=False):
    
      
    if self.inputs_type==InputType.INPUT_NOISY_GRID_MODULES:
      
      # sample plotting for modules
      self.__plot_sample_with_modules(num_samples)
      
    else:
      
      # standard sample plotting
      import pylab as pl
      from plotlib import noframe,colorbar
      from numpy import var,floor,ceil,arange,sqrt
      
      dx=self.L/self.nx
      X,Y=np.mgrid[-self.L/2:self.L/2:dx,-self.L/2:self.L/2:dx]


      sparseness=self.comp_sparseness()
  
      if input_idxs is None:
        if random is True:
          input_idxs=randint(0,self.N,num_samples)
        else:
          input_idxs=arange(num_samples)
      else:
        num_samples=len(input_idxs)
        
      nsx=int(ceil(sqrt(num_samples)))
      nsy=int(floor(sqrt(num_samples)))
      pl.figure(figsize=(8,8))
      for idx,input_idx in enumerate(input_idxs):
        pl.subplot(nsx,nsy,idx+1,aspect='equal')
        noframe()
        pl.pcolormesh(X,Y,self.inputs_flat[:,input_idx].reshape(self.nx,self.nx),cmap='jet',rasterized=True)
        pl.xlim(-self.L/2,self.L/2)        
        pl.ylim(-self.L/2,self.L/2)        
        
        if plot_peak_pos is True:
          #pl.plot(self.grid_peak_pos[input_idx,0,:],self.grid_peak_pos[input_idx,1,:],'.k')
          pl.plot(self.noisy_peak_pos[input_idx,0,:],self.noisy_peak_pos[input_idx,1,:],'.w')
          
        
        if plot_colorbar:
          colorbar()
        if title:
          pl.title('m:%.3f v:%.2e s:%.2e'%(np.mean(self.inputs_flat[:,input_idx]),var(self.inputs_flat[:,input_idx]),sparseness[input_idx]),fontsize=14)
      
      
  def plot_scores(self):
    import pylab as pl
    from plotlib import custom_axes
    from numpy import median 
    from plotlib import MaxNLocator
    
    pl.figure(figsize=(2.8,1.8))
    pl.subplots_adjust(bottom=0.3,left=0.3)
    pl.hist(self.in_scores,bins=50,color='k')
    pl.axvline(median(self.in_scores),color='r')
    custom_axes()
    pl.xlabel('Input gridness score',fontsize=11)
    pl.ylabel('Number of neurons',fontsize=11)
    pl.xlim([-0.5,2])
    #pl.ylim([0,60])
    pl.gca().yaxis.set_major_locator(MaxNLocator(3))
    pl.gca().xaxis.set_major_locator(MaxNLocator(3))
    pl.title('median=%.3f'%median(self.in_scores))
      
  def get_overlap_mean(self,th=None):
    if th is None:
      th=self.amp/2.
    return np.mean(np.sum(self.inputs_flat>th,axis=1))


  def get_overlap_var(self,th=None):
    if th is None:
      th=self.amp/2.
    return np.var(np.sum(self.inputs_flat>th,axis=1))
    

  def get_overlap_cv(self,th=None):
    if th is None:
      th=self.amp/2.
    return np.sqrt(self.get_overlap_var(th))/self.get_overlap_mean(th)
    
  def get_num_overlp(self,th=None):  
    if th is None:
      th=self.amp/2.
    return np.sum(self.inputs_flat>th,axis=1).reshape(self.nx,self.nx)
    
  def plot_num_overlap(self,th=None):
    if th is None:
      th=self.amp/2.
    
    import pylab as pl
    import plotlib as pp

    pl.figure()
    pl.subplot(111,aspect='equal')
    pl.pcolormesh(np.sum(self.inputs_flat>th,axis=1).reshape(self.nx,self.nx))
    pp.noframe()
    pp.colorbar()    
      
  def get_var_of_mean_input(self)  :
    input_mean=self.inputs_flat.mean(axis=1).reshape(self.nx,self.nx)        
    return input_mean.var() 
      
  def get_cv_of_mean_input(self)  :
    input_mean=self.inputs_flat.mean(axis=1).reshape(self.nx,self.nx)        
    return np.sqrt(input_mean.var())/input_mean.mean() 
    
  def plot_mean_input(self) :
    import pylab as pl
    input_mean=self.inputs_flat.mean(axis=1).reshape(self.nx,self.nx)        
    self.dx=self.L/self.nx
    X,Y=np.mgrid[-self.L/2:self.L/2:self.dx,-self.L/2:self.L/2:self.dx]
  
    pl.figure()
    pl.subplot(111,aspect='equal')
    pl.pcolormesh(X,Y,input_mean)
    pl.title('var: %e'%input_mean.var())
    if self.inputs_type in (InputType.INPUT_GAU_GRID,
                            InputType.INPUT_GAU_NOISY_CENTERS,
                            InputType.INPUT_GAU_RANDOM_CENTERS):
      pl.plot(self.centers[:,0],self.centers[:,1],'.k')
    pl.colorbar()
    
  def get_boundary_diff(self):
    input_mean_beff=self.inputs_flat.mean(axis=1).reshape(self.nx,self.nx)        
    bound_diff=self.input_mean-input_mean_beff
    return bound_diff
  
  def plot_boundary_diff(self):
    import pylab as pl
    pl.figure()
    pl.subplot(111,aspect='equal')
    pl.pcolormesh(self.get_boundary_diff())
    pl.colorbar()
    
  def plot_corrected_total_input(self) :
    import pylab as pl
    input_mean=self.inputs_flat.mean(axis=1).reshape(self.nx,self.nx)        
    self.dx=self.L/self.nx
    X,Y=np.mgrid[-self.L/2:self.L/2:self.dx,-self.L/2:self.L/2:self.dx]
  
    pl.figure()
    pl.subplot(111,aspect='equal')
    pl.pcolormesh(X,Y,input_mean+self.get_boundary_diff())
    if self.inputs_type==InputType.INPUT_GAU_GRID:
      pl.plot(self.centers[:,0],self.centers[:,1],'.k')
    pl.colorbar()
    
  def comp_sparseness(self):
    return np.sum(self.inputs_flat,axis=0)**2/(self.nx**2*np.sum(self.inputs_flat**2,axis=0))
  
  def comp_mean(self):
    return np.mean(self.inputs_flat,axis=0)

  def comp_var(self):
    return np.var(self.inputs_flat,axis=0)
    
  def print_input_stats(self):
    print
    print 'N = %d' %self.n**2
    print 'L = %d' %self.L
    print 'InputType = %s'%self.inputs_type
    if self.inputs_type == InputType.INPUT_GAU_MIX_POS:
      print 'Num Gau Mix = %d'%self.num_gau_mix
    print    
    print 'Single-input mean = %.3f'%np.mean(self.comp_mean())
    print 'Single-input variance = %.3f'%np.mean(self.comp_var())
    print 'Single-input sparseness = %.3f'%np.mean(self.comp_sparseness())
    print 
    print 'Mean-input variance = %.e'%np.var(np.mean(self.inputs_flat,axis=1))

if __name__ == '__main__':
  
  # one field per input map
  param_map={'inputs_type': InputType.INPUT_GAU_MIX_POS,'n':5,'nx':100,'L':2.,
             'periodic_inputs':True,'input_mean':5,'sigma':0.15,
             'same_fixed_norm':False,'fixed_norm':0.,'num_gau_mix':1,'inputs_seed':0}

  inputs=SpatialInputs(param_map)
  inputs.plot_sample()
  