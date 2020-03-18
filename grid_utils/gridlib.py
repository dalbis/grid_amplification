import numpy as np
from numpy.fft import fft2,fftfreq,fftshift
from numpy.random import rand
from scipy.ndimage import rotate
from scipy.signal import fftconvolve
from scipy.stats.stats import pearsonr
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.measurements import label
from scipy.ndimage import gaussian_filter

from simlib import print_progress
import time

class GridProps(object): 
  """
  Utility class to store the properties of a grid pattern
  """
  def __init__(self,n,grid_T,grid_angle):
    
    self.n=n
    self.N=n**2
    self.grid_T=grid_T
    self.grid_angle=grid_angle
    
    self.R_T,self.u1,self.u2,self.u1_rec,self.u2_rec=get_phase_lattice(self.grid_T,self.grid_angle)    
    self.phases=get_phases(self.n,self.grid_T,self.grid_angle,return_u12=False)
    
def get_trans_index(W,ret_W_rolled=False):
  """
  Computes an index that quantifies how much a matrix is translation invariant.
  For a perfectly translation invariant matrix the index approaches 1. 
  This is calculated by applying a circular shift to each row of the matrix with offset equal to the
  row number. For a perfectly translation-invariant matrix, the result is a matrix where all rows are equal.
  We then quantify translation invariance by computing the mean pearson's correlation coefficient between
  each row and the mean row of the matrix. 
  
  """

  import scipy
  
  W_rolled=np.zeros_like(W)
  
  for row_idx in xrange(W.shape[0]):
    W_rolled[row_idx,:]=np.roll(W[row_idx,:],-row_idx)
  
  W_mean_row=W_rolled.mean(axis=0)
  
  row_corrs=np.zeros(W.shape[0])  
  for row_idx in xrange(W.shape[0]):
    row_corrs[row_idx]=scipy.stats.pearsonr(W_rolled[row_idx,:],W_mean_row)[0]
  
    trans_index=row_corrs.mean()
  
  if ret_W_rolled:
    return trans_index,W_rolled
  else:
    return trans_index



def get_recurrent_matrix_tuning_index(W,gp):

  """
  W: recurrent weight matrix
  gp: instance of GridProps
  
  This function takes as imput a recurrent weight matrix that connects grids with similar phases.
  It computes how strong is the phase tuning by comparing the mean of the first harmonic as compared to the mean.
  Note that because the phase space is a rhombus we use a Fourier transform with non-orthogonal unit vectors.
  For each row of the matrix we compute the 2D spectrum of the weights in rhomobidal space and than average the 
  amplitudes at the six first harmonics ([-1,-1],[1,1],[0,1],[0,-1],[1,0],[-1,0] )
 
  """
  
  mean_amps=np.zeros(gp.N)
  for phase_idx in xrange(gp.N):
    mean_amps[phase_idx]=get_conn_to_one_neuron_tuning_index(W[phase_idx],gp)
  return mean_amps.mean()

def get_conn_to_one_neuron_tuning_index(W_one_neuron,gp):
  """
  Compute the recurrent connectivity tuning index for the input connections to one neuron
  """
  max_harmonic=2
  HX,HY,ft=fourier_on_lattice(gp.grid_T,gp.u1_rec,gp.u2_rec,gp.phases,W_one_neuron,max_harmonic=max_harmonic,return_harmonics=True)
  FT=ft.reshape(2*max_harmonic,2*max_harmonic)
  
  # first harmonic
  H1=  np.bitwise_and(HX==0,np.abs(HY)==1)+\
       np.bitwise_and(np.abs(HX)==1,HY==0)+\
       np.bitwise_and(HX==-1,HY==-1)+\
       np.bitwise_and(HX==1,HY==1)
  
  # baseline
  H0=np.bitwise_and(np.abs(HX)==0,np.abs(HY)==0)
  
  return (np.abs(FT[H1])/np.abs(FT[H0])).mean()


  
def get_reciprocal_rhombus_unit_vectors(u1,u2):
  """
  Get the unit vector of the reciprocal rhombus in Fourier space
  """
  U = np.vstack([u1, u2]).T
  U_rec = 2*np.pi*(np.linalg.inv(U)).T
  
  # unit vectors of the reciprocal lattice
  u1_rec = U_rec[:,0]
  u2_rec = U_rec[:,1]
  
  return u1_rec,u2_rec


def get_periodic_dist_on_rhombus(n,ref_phase,phases,u1,u2):
  """
  The function returns the periodic distance on rhombus between the ref_phase and all other phases
  The result is an array of n**2 elements, i.e., one distance measure for each phase in the rhombus
  """
  
  # we have to compute nine symmetries to account for 2d periodicity
  dist_mat=np.zeros((n**2,9))

  count=0
  for xp in (-1,0,1):
    for yp in (-1,0,1):
      shift=xp*u1+yp*u2
      dist_mat[:,count]=np.sqrt(((phases-ref_phase+shift)**2).sum(axis=1))
      count+=1
  
  dist=dist_mat.min(axis=1)
  
  return dist



def get_corr_distance_fun(corr,n,u1,u2,phases):
  
  """
  Returns the radial profile (correlatio VS distance function) for each row of the given correlation matrix
  """

  print 'rows: %d'%corr.shape[0]
  
  all_dist=[]  
  all_profiles=[]
  for row_idx in  xrange(corr.shape[0]):
    ref_phase=phases[row_idx,:]
    dist=get_periodic_dist_on_rhombus(n,ref_phase,phases,u1,u2)
  
    dist, inverse,counts = np.unique(dist, return_counts=True, return_inverse=True)
    

    profile =np. bincount(inverse, corr[row_idx,:].squeeze())
    profile/=counts
    
    all_dist.append(dist)
    all_profiles.append(profile)
    
  return all_dist,all_profiles


def find_place_fields(ratemap,max_th=0.3,min_size=9,ret_labels=False):
  """
  Segments place fields from a rate map
  max_th: threshold relative to the maximal firing rate
  min_size: minimum field size in pixels
  """
    
  raw_labels,raw_num_fields=label(ratemap>ratemap.max()*max_th)
  labels=np.zeros_like(ratemap)
  label_idx=1
  
  for i in range(raw_num_fields):
    label_mask=raw_labels==(i+1)
    
    if label_mask.sum()>min_size:
      labels[label_mask]=label_idx
      label_idx+=1
  
      
  num_fields=labels.max()
  
  if ret_labels:  
    return labels,num_fields
  else:
    return num_fields

  

def get_phases(n,grid_T,grid_angle,return_u12=False):
  """
  Samples grid phases evenly on a rhombus with side-length grid_T and orientation grid_angle
  If return_u12=True the function returns also the unit vectors of the rhombus
  """
  
  # unit vectors of the direct lattice
  u1 = grid_T*np.array([np.sin(2*np.pi/3+grid_angle), -np.cos(2*np.pi/3+grid_angle)])
  u2 = grid_T*np.array([-np.sin(grid_angle), np.cos(grid_angle)])
    
  # phase samples
  ran = np.array([np.arange(-n/2.,n/2.)/n]).T
  u1_phases = np.array([u1])*ran
  u2_phases = np.array([u2])*ran
    
  X1,X2=np.meshgrid(u1_phases[:,0],u2_phases[:,0])
  Y1,Y2=np.meshgrid(u1_phases[:,1],u2_phases[:,1])
  X,Y=X1+X2,Y1+Y2

  if return_u12 is True:     
    return u1,u2,np.array([np.ravel(X), np.ravel(Y)]).T
  else:
    return np.array([np.ravel(X), np.ravel(Y)]).T
  
  
def get_pos_idx(p,pos):
  """
  Return the index of the closest position to 'p' found in the array of positions 'pos'
  """  
  p_dist=((np.array(p)-pos)**2).sum(axis=1)  
  p_idx = np.argmin(p_dist)
  return p_idx
  
def get_angle_amps(num_dft,freq_idx,nx):
  """
  Compute the amplitudes of modes of a given frequency (freq_idx) but different angles
  num_dft: DFT of a grid pattern
  freq_idx: Frequency at which the modes need to be computed
  nx: sampling interval in space
  """
  
  # all radii and all angles  
  yr, xr = np.indices((nx,nx))
  all_r =  np.around( np.sqrt((xr - nx/2)**2 + (yr - nx/2)**2))
  all_ang =  np.arctan2(yr-nx/2,xr-nx/2)*180/ np.pi+180
  all_ang[all_ang==360]=0
  
  # flat dfts
  allr_flat=all_r.reshape(nx**2)
  all_ang_flat = all_ang.reshape(nx**2)
  num_dft_flat=num_dft.reshape(nx**2,num_dft.shape[2])

  # indexes
  idxs= np.arange(nx**2)[allr_flat==freq_idx]
  uidxs=idxs[0:len(idxs)/2]
  
  # take as zero the angle of the fastest growing mode
  max_ang=all_ang_flat[ np.argmax(num_dft_flat[:,-1])]
  angles=np.remainder(all_ang_flat-max_ang,180)[uidxs]

  # amplitudes
  amps=  [ np.squeeze(num_dft_flat[idx,:]) for idx in uidxs]
  
  return angles,amps
  

def get_grid_params(J,L,nx,num_steps=50,return_cx=False):
  """
  Estimates parameters of a grid pattern, i.e., gridness core, grid spacing,
  grid angle,and grid phase
  
  J: input grid pattern 
  L: side-length of the environment
  nx: number of space samples
  num_steps: number of iteration steps to compute the gridness score
  return_cx: if True returns also the autocorrelation matrix
  """
  
  dx=L/nx
  X,Y=np.mgrid[-L/2:L/2:dx,-L/2:L/2:dx]
  pos=np.array([np.ravel(X), np.ravel(Y)])
  
  if return_cx is True:
    score,best_outr,angle,spacing,cx=gridness(J,L/nx,
                                           computeAngle=True,doPlot=False,
                                           num_steps=num_steps,return_cx=True)   
  else:
    score,best_outr,angle,spacing=gridness(J,L/nx,
                                           computeAngle=True,doPlot=False,
                                           num_steps=num_steps,return_cx=False)   

  if spacing is not np.NaN and angle is not np.NaN:                                         
    ref_grid=simple_grid_fun(pos,grid_T=spacing,
                             angle=-angle,phase=[0, 0]).reshape(nx,nx)
    phase=get_grid_phase(J,ref_grid,L/nx,doPlot=False,use_crosscorr=True)
  else:
    phase=np.NaN

  if return_cx is True:
    return score, spacing,angle,phase,cx
  else:    
    return score, spacing,angle,phase


  
def compute_scores_evo(J_vect,n,L,num_steps=50):
  """
  Computes gridness scores for a matrix at different time points
  J_vect = N x num_snaps
  """
  
  num_snaps=J_vect.shape[1]
  assert(J_vect.shape[0]==n**2)
  start_clock=time.time()
  best_score=-1      
  
  scores=np.zeros(num_snaps)
  spacings=np.zeros(num_snaps)
  angles=np.zeros(num_snaps)
  phases=np.zeros((2,num_snaps))
  
  for snap_idx in xrange(num_snaps):
    print_progress(snap_idx,num_snaps,start_clock=start_clock)

    J=J_vect[:,snap_idx]
    
    score,spacing,angle,phase= get_grid_params(J.reshape(n,n),L,n,num_steps=num_steps)

    best_score=max(best_score,score)
    scores[snap_idx]=score
    spacings[snap_idx]=spacing
    angles[snap_idx]=angle
    phases[:,snap_idx]=phase
    
  score_string='final_score: %.2f    best_score: %.2f    mean_score: %.2f\n'%(score,best_score,np.mean(scores))  
  print score_string

  return scores,spacings,angles,phases      

        
def dft2d_num(M_evo,L,n,nozero=True):
  """
  Computes the 2D DFT of a n x n x time_samples matrix wrt the first two dimensions.
  The DC component is set to zero
  """
  
  assert(len(M_evo.shape)==3)
  assert(M_evo.shape[0]==M_evo.shape[1])

  allfreqs = fftshift(fftfreq(n,d=L/n))
  freqs=allfreqs[n/2:]
  M_dft_evo=fftshift(abs(fft2(M_evo,axes=[0,1])),axes=[0,1])
  if nozero is True:
    M_dft_evo[n/2,n/2,:]=0
  return M_dft_evo,freqs,allfreqs



def dft2d_teo(J0_dft,eigs,time,n):
  """
  Compute the theoretical DFT solution of a linear dynamical system
  given the eigenvalues and the initial condition
  """
  N=n**2
  teo_dft=J0_dft.reshape(N,1)*np.exp(time[np.newaxis,:]*eigs.reshape(N,1))
  teo_dft=teo_dft.reshape(n,n,len(time))
  teo_dft[n/2,n/2]=0

  return teo_dft

  
  
def radial_profile(data,norm=False):
  """
  Compute radial profile of a 2D function sampled on a square domain,
  assumes the function is centered in the middle of the square

  # TEST:
  #  
  #  ran=np.arange(-1.01,1.01,0.01)
  #  SSX,SSY = meshgrid(ran,ran)
  #  
  #  T=np.exp(-(SSX**2+SSY**2))
  #  P=radial_profile(T,norm=True)
  #  
  #  pl.figure()
  #  pl.subplot(111,aspect='equal')
  #  pl.pcolormesh(SSX,SSY,T)
  #  custom_axes()
  #  colorbar()
  #  
  #  pl.figure()
  #  pl.plot(ran[101:],P)
  
  """
  
  assert(len(data.shape)==2)
  assert(data.shape[0]==data.shape[1])


  center=np.array(data.shape)/2
  yr, xr = np.indices((data.shape))
  r = np.around(np.sqrt((xr - center[0])**2 + (yr - center[1])**2))
  r = r.astype(int)

  profile =np.bincount(r.ravel(), data.ravel())
  
  if norm is True:
    nr = np.bincount(r.ravel())
    profile/=nr
    
  profile=profile[:len(data)/2]    
  
  return profile
  

def dft2d_profiles(M_dft_evo):
  """
  Computes DFT 2D radial profiles at different time points
  """
  
  assert(len(M_dft_evo.shape)==3)
  assert(M_dft_evo.shape[0]==M_dft_evo.shape[1])

  num_snaps=M_dft_evo.shape[2]
  profiles = np.array([radial_profile(abs(M_dft_evo[:,:,idx]),norm=True) for idx in xrange(num_snaps)])
  return profiles


def gridness_evo(M,dx,num_steps=50):
  """
  Compute gridness evolution
  M: matrix nx X nx X num_steps (or num_cells)
  """
  scores=[]
  spacings=[]
  assert(len(M.shape)==3)
  num_snaps=M.shape[2]
  print 'Computing scores...'
  for idx in xrange(num_snaps):
    
    score,best_outr,orientation,spacing=gridness(M[:,:,idx],dx,computeAngle=False,doPlot=False,num_steps=num_steps)
    scores.append(score)
    spacings.append(spacing)
    print_progress(idx,num_snaps)
  return scores,spacings
        
def unique_rows(a):
  """
  Removes duplicates rows from a matrix
  """
  a = np.ascontiguousnp.array(a)
  unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
  return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
  
  
def detect_peaks(cx,size=2,do_plot=False):
  """
  Takes an image and detect the peaks usingthe local maximum filter.
  Returns a boolean mask of the peaks (i.e. 1 when
  the pixel's value is the neighborhood maximum, 0 otherwise)
  """

  # smooth the autocorrelation for noise reduction
  cx_smooth=gaussian_filter(cx, sigma=3)

  # define an size-connected neighborhood
  neighborhood = generate_binary_structure(size,size)
  
  #apply the local maximum filter; all pixel of maximal value in their neighborhood are set to 1
  local_max = maximum_filter(cx_smooth, footprint=neighborhood)==cx_smooth
  #local_max is a mask that contains the peaks we are looking for, but also the background. In order to isolate the peaks we must remove the background from the mask.

  #we create the mask of the background
  background = (cx_smooth==0)

  #a little technicality: we must erode the background in order to successfully subtract it form local_max, otherwise a line will 
  # appear along the background border (artifact of the local maximum filter)
  eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

  #we obtain the final mask, containing only peaks, by removing the background from the local_max mask
  detected_peaks = local_max.astype(int) - eroded_background.astype(int)

  if do_plot is True:
    import pylab as pl
    
    pl.figure(figsize=(10,3))
    pl.subplots_adjust(wspace=0.3)
    pl.subplot(131,aspect='equal')
    pl.pcolormesh(cx_smooth)
    pl.title('Smoothed autocorrelation')
    pl.subplot(132,aspect='equal')
    pl.pcolormesh(local_max)
    pl.title('Local maxima')
    pl.subplot(133,aspect='equal')
    pl.pcolormesh(background)
    pl.title('Background')

  return detected_peaks

def detect_six_closest_peaks(cx,doPlot=False):
  """
  Detects the six peaks closest to the center of the autocorrelogram
  cx: autocorrelogram matrix
  """

  # indexes to cut a circle in the auto-correlation matrix
  SX,SY = np.meshgrid(range(cx.shape[0]),range(cx.shape[1]))
  
  if np.remainder(cx.shape[0],2)==1:
    tile_center = np.array([[(cx.shape[0]+1)/2-1, (cx.shape[1]+1)/2-1]]).T
  else:
    tile_center = np.array([[(cx.shape[0])/2, (cx.shape[1])/2]]).T
  

  peaks=detect_peaks(cx)
  peaks_xy = np.array([SX[peaks==1],SY[peaks==1]])
  
  if peaks_xy.shape[1]<6:
    print 'Warning: less than 6 peaks found!!!'
    return np.zeros((2,6)),tile_center
    
  else:
    peaks_dist = np.sqrt(sum((peaks_xy-tile_center)**2,0))
    
    sort_idxs = np.argsort(peaks_dist)
    
    peaks_dist=peaks_dist[sort_idxs]
    peaks_xy=peaks_xy[:,sort_idxs]
  
    # filter out center and peaks too close to the center (duplicates)
    to_retain_idxs=peaks_dist>2
    
    
    if sum(to_retain_idxs)<6:
      print 'Warning: less than 6 peaks to retain!!!'
      return np.zeros((2,6)),tile_center
      
    else:      
      peaks_dist=peaks_dist[to_retain_idxs]
      peaks_xy=peaks_xy[:,to_retain_idxs]   
      idxs=np.arange(6)
    
    
      if doPlot is True:
        import pylab as pl  
        pl.figure()
        pl.subplot(111,aspect='equal')
        pl.pcolormesh(cx)
        pl.scatter(peaks_xy[0,idxs]+0.5,peaks_xy[1,idxs]+0.5)
        pl.scatter(tile_center[0]+0.5,tile_center[1]+0.5)
        
      return peaks_xy[:,idxs],tile_center

    

def comp_psi_scores(L,nx,r_maps):
  """
  Similar to Simon's score for firing rate maps
  """
  
  if len(r_maps.shape)==1:
    r_maps=r_maps[:,np.newaxis]

  assert(r_maps.shape[0]==nx**2)

  all_psi=[]  
  for cell_idx in xrange(r_maps.shape[1]):
    
    r_map=r_maps[:,cell_idx]
    cx=norm_autocorr(r_map.reshape(nx,nx))
  
  
    peaks,center = detect_six_closest_peaks(cx,False)           # get six closest peaks
  
    #print '%d peaks detected: '%peaks.shape[1]
    
    cent_peaks=peaks-center                               # center them
    peak_dists = np.sqrt(sum(cent_peaks**2,0))               # peak distances
    norm_peaks = cent_peaks/peak_dists                    # normalize to unit norm
    
    angles = np.arccos(norm_peaks[0,:])  # calculate angle
    
    psi_M=np.zeros(5)
    for M_idx,M in enumerate([2,3,4,5,6]):
      psi_M[M_idx]=np.abs(np.mean(np.exp(1j*M*angles)))
     
    if np.argmax(psi_M)==4:
      psi=psi_M[4]
    else:
      psi=0.
    #print angles*180/np.pi
    
    all_psi.append(psi)
  
  return np.array(all_psi)
        

  
def get_grid_phase(x,x0,ds,doPlot=False,use_crosscorr=True):
  """
  Return the grid phase relative to a given reference gridness
  x: grid for which the phase has to be estimated
  x0: reference grid
  """
  if use_crosscorr is True:
    cx=norm_crosscorr(x,x0,type='full')
    # cutout the central part
    n=(cx.shape[0]+1)/2
    cx=cx[n-1-n/2:n-1+n/2,n-1-n/2:n-1+n/2]    
  else:
    cx=x
    n=cx.shape[0]    
  
  L=n*ds
  SX,SY,tiles=get_tiles(L,ds)
  
  peaks=detect_peaks(cx)

  if peaks.sum()>0:
    peaks_xy = np.array([SX[peaks==1],SY[peaks==1]])
    peaks_dist = sum(peaks_xy**2,0)
    
    idxs = np.argmin(peaks_dist)
  
    if doPlot is True:
      import pylab as pl  
      pl.pcolormesh(SX,SY,cx)
      pl.scatter(peaks_xy[0,idxs],peaks_xy[1,idxs])
      pl.plot(0,0,'.g')
  
    phase = peaks_xy[:,idxs]
  else:
    phase=np.NaN
    
  return phase


def get_grid_spacing_and_orientation(cx,ds,doPlot=False,compute_angle=True,ax=None):
  """
  Returns the grid orientation given the autocorrelogram
  ds: space discretization step
  cx: autocorrelogram matrix
  :returns: an angle in radiants
  """

  peaks,center = detect_six_closest_peaks(cx)           # get six closest peaks
  
  #print '%d peaks detected: '%peaks.shape[1]
  
  cent_peaks=peaks-center                               # center them
  #print cent_peaks
  
  peak_dists = np.sqrt(sum(cent_peaks**2,0))               # peak distances
  norm_peaks = cent_peaks/peak_dists                    # normalize to unit norm

  norm_peak_1quad_idxs=np.bitwise_and(norm_peaks[1,:]>0,norm_peaks[0,:]>0)          # indexes of the peaks in the first quadrant x>0 and y>0
  spacing=np.mean(peak_dists)*ds
  
  #print 'get_grid_spacing_and_orientation spacing=%.2f'%spacing
  
  angle=np.NaN

  if compute_angle is True:
    # if we have at least one peak in the first quadrant
    if any(norm_peak_1quad_idxs) == True: 
      norm_peaks_1quad=norm_peaks[:,norm_peak_1quad_idxs]                          # normalized coordinates of the peaks in the first quadrant
      norm_orientation_peak_idx=np.argmin(norm_peaks_1quad[1,:])                   # index of the peak with minumum y 
      norm_orientation_peak=norm_peaks_1quad[:,norm_orientation_peak_idx]          # normalized coordinates of the peak with minimum y
      
      peaks_1quad = peaks[:,norm_peak_1quad_idxs]                                 # coordinates of the peaks in the first quadrant
      orientation_peak=peaks_1quad[:,norm_orientation_peak_idx]                   # coordinates of the peak with minimum y 
  
      angle = np.arccos(norm_orientation_peak[0])  # calculate angle
      
      if angle <0:
        angle=angle+np.pi/3
  
      if doPlot is True:
        import pylab as pl  
        if ax is None:
          pl.figure()
          pl.subplot(111,aspect='equal')

        pl.pcolormesh(cx/(cx[center[0],center[1]]),vmax=1.,cmap='binary',rasterized=True)
        pl.plot([center[0]+.5,orientation_peak[0]+.5],[center[1]+.5,orientation_peak[1]+.5],'-y',linewidth=2)

        for i in xrange(6):
          pl.scatter(peaks[0,i]+0.5,peaks[1,i]+0.5,c='r')
          
        pl.scatter(center[0]+0.5,center[1]+0.5,c='r')
        hlen=cx.shape[0]/3.
        pl.xlim([center[0]-hlen,center[0]+hlen])
        pl.ylim([center[1]-hlen,center[1]+hlen])
    else:
      pass
      #print "no peaks in the first quadrant"

  return angle,spacing


def fr_fun(h,gain=.1,th=0,sat=1,type='arctan'):
  """
  Threshold-saturation firing rate function 
  h: input
  sat: saturation level
  gain: gain
  th: threshold
  """
  if type == 'arctan':
    return sat*2/np.pi*np.arctan(gain*(h-th))*0.5*(np.sign(h-th) + 1)
  elif type == 'sigmoid':
    return sat*1/(1+np.exp(-gain*(h-th)))
  elif type == 'rectified':
    return h*0.5*(np.sign(h-th) + 1)
  elif type=='linear':
    return h



def pf_fun(pos,center=np.array([0,0]),sigma=0.05,amp=1):
  """
  Gaussian place-field input function
  pos: position
  center: center
  sigma: place field width
  amp: maximal amplitude
  """
  
  # multiple positions one center
  if len(pos.shape)>1 and len(center.shape)==1:
    center = np.array([center]).T
  # one position multiple centers
  if len(pos.shape)==1 and len(center.shape)>1:
   pos = np.array([pos]).T
  return np.exp(-sum((pos-center)**2,0)/(2*sigma**2))*amp



def simple_grid_fun(pos,grid_T,angle=0,phase=[0, 0],waves=[0,1,2]):
  """
  Another function for a grid with a simpler mathematical description
  """
  assert(  grid_T is not np.NaN 
         and angle is not np.NaN
         and phase is not np.NaN)
  
  alpha=np.array([np.pi*i/3+angle for i in waves])
  k=4*np.pi/(np.sqrt(3)*grid_T)*np.array([np.cos(alpha),np.sin(alpha)]).T
  if len(pos.shape)>1:
    phase = np.array([phase]).T
  
  return sum(np.cos(np.dot(k,pos+phase)),0)
  

def norm_crosscorr(x,y,type='full',pearson=True):
  """
  Normalized cross-correlogram
  """
  n = fftconvolve(np.ones(x.shape),np.ones(x.shape),type)
  cx=np.divide(fftconvolve(np.flipud(np.fliplr(x)),y,type),n)
  if pearson is True:
    return (cx-x.mean()**2)/x.var()
  else:
    return cx
    
def norm_autocorr(x,type='full',pearson=True):
  """
  Normalized autocorrelation, we divide about the amount of overlap which is given by the autoconvolution of a matrix of ones
  """
  x0 = x-x.mean()
  #return fftconvolve(flipud(fliplr(x)),x,type)
  n = fftconvolve(np.ones(x0.shape),np.ones(x0.shape),type)
  cx=np.divide(fftconvolve(np.flipud(np.fliplr(x0)),x0,type),n)
  if pearson is True:
    return cx/x.var()
  else:
    return cx

def comp_score(cx,idxs,min_diff=False):
  """
  Calculates the gridness score for an autocorrelation pattern and a given array of indexes for elements to retain.
  For the final gridness score the elements shall be outside an inner radius around the central peak and inside an outer radius
  containing the six closest peaks
  cx: autocorrelogram
  idxs: array of indexes for the elements to retain
  """
  deg_ran = [60, 120, 30, 90, 150]   # angles for the gridness score    
  c = np.zeros(len(deg_ran))         # correlation for each rotation angle
  cx_in = cx[idxs[0,:],idxs[1,:]]    # elements of the autocorellation pattern to retain

  # calculate correlation for the five angles
  for deg_idx in range(len(deg_ran)):
    rot = rotate(cx,deg_ran[deg_idx],reshape=False)
    rot_in = rot[idxs[0,:],idxs[1,:]]
    c[deg_idx]=pearsonr(cx_in,rot_in)[0]

  # gridness score by taking the minimum difference
  if min_diff is True:    
    score=c[0:2].min()-c[2:].max()
  # gridness score by taking tha difference of the means  
  else:
    score=np.mean(c[0:2])-np.mean(c[2:]) 
  return score

def get_score_corr_angle(cx,idxs):
  """
  Compute the Pearnons's correlation of all rotation angles 
  """
  
  deg_ran = np.arange(0,180,1)      # angles for the gridness score    
  c = np.zeros(len(deg_ran))         # correlation for each rotation angle
  cx_in = cx[idxs[0,:],idxs[1,:]]    # elements of the autocorellation pattern to retain

  # calculate correlation for the five angles
  for deg_idx in range(len(deg_ran)):
    rot = rotate(cx,deg_ran[deg_idx],reshape=False)
    rot_in = rot[idxs[0,:],idxs[1,:]]
    c[deg_idx]=pearsonr(cx_in,rot_in)[0]

  return deg_ran,c
  



def gridness(x,ds,doPlot=False,computeAngle=False,num_steps=20,
             score_th_for_orientation=0.3,axes=None,cx=None,pearson=True,return_cx=False,min_diff=False):

  if cx is None:
    cx = norm_autocorr(x,pearson=pearson)                                         # compute the normalized autocorrelation of the pattern

  # compute the radial profile and the inner radius of the ring
  profile=radial_profile(cx,norm=False)
  inrad=np.argwhere(profile<0)[0]

  # compute grid spacing
  angle,spacing=get_grid_spacing_and_orientation(cx,ds,doPlot=False,compute_angle=False)
               
  # indexes to cut a circle in the auto-correlation matrix
  SX,SY = np.meshgrid(range(cx.shape[0]),range(cx.shape[1]))
  tiles= np.array([np.ravel(SX), np.ravel(SY)])
  if np.remainder(cx.shape[0],2)==1:
    tile_center = np.array([[(cx.shape[0]+1)/2-1, (cx.shape[1]+1)/2-1]]).T
  else:
    tile_center = np.array([[(cx.shape[0])/2, (cx.shape[1])/2]]).T
  tiles_dist = np.sqrt(sum((tiles-tile_center)**2,0))
    
  # minimal and maximal outer radii
  max_outr=np.ceil(spacing*2/ds)
  min_outr=np.floor(spacing*0.5/ds)


  outr_ran = np.arange(min_outr,max_outr,max_outr/num_steps)    # range of outer radii for the gridness score
  best_score = -2                                               # best gridness score
  best_outr = min_outr                                          # best radius    
  
  # loop over increasing radii and retain the best score
  for outr_idx in range(len(outr_ran)):
  
    # compute score for the current outer radius
    idxs=tiles[:,np.bitwise_and(tiles_dist>inrad,tiles_dist<outr_ran[outr_idx])]
    score = comp_score(cx,idxs,min_diff=True)
  
    # retain best score
    if score > best_score:
      best_score = score
      best_outr = outr_ran[outr_idx]
  
  
  # plot if requested
  if doPlot is True:
    import pylab as pl    
    import plotlib as pp
    ax=pl.gca() if axes is None else axes
    pl.sca(ax)
    pl.figure(figsize=(8,6))
    
    ax_grid = pl.GridSpec(2, 2, wspace=0.4, hspace=0.3)
    
    #pl.subplots_adjust(wspace=0.5)
    
    ax=pl.subplot(ax_grid[0,0])
    pl.axis('equal')
    pl.pcolormesh(x.T,rasterized=True)
    pp.colorbar()
    pp.noframe()
    
    ax=pl.subplot(ax_grid[0,1])
    pl.axis('equal')
    pl.pcolormesh(cx,rasterized=True)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.set_frame_on(False)
    theta_ran = np.arange(0,2*np.pi,0.1)
    pl.plot(best_outr*np.cos(theta_ran)+tile_center[0],best_outr*np.sin(theta_ran)+tile_center[1],'w')
    pl.plot(max_outr*np.cos(theta_ran)+tile_center[0],max_outr*np.sin(theta_ran)+tile_center[1],'k')
    pl.plot((spacing/ds)*np.cos(theta_ran)+tile_center[0],(spacing/ds)*np.sin(theta_ran)+tile_center[1],'g')
    pl.plot(inrad*np.cos(theta_ran)+tile_center[0],inrad*np.sin(theta_ran)+tile_center[1],'w')
    pl.text(10,10,'%.2f'%best_score, color='black',fontsize=10, weight='bold',bbox={'facecolor':'white'})
    pp.colorbar()
    
    ax=pl.subplot(ax_grid[1,:])
    deg_ran,c = get_score_corr_angle(cx,idxs)
    pl.plot(deg_ran,c,'-k')
    pl.xlabel('Angle')
    pl.ylabel('Correlation')
    pp.custom_axes()
    pl.axhline(1)
    pl.ylim(-1,1)
    pl.title('max= %.2f min=%.2f'%(c.max(),c.min()))
  
  # calculate angle if there we pass a threshold for the gridness
  angle=np.NaN
  if computeAngle is True and best_score > score_th_for_orientation:
    angle,spacing = get_grid_spacing_and_orientation(cx,ds,compute_angle=True)
  
  if return_cx  is True:
    return best_score,best_outr,angle,spacing,cx
  else:
    return best_score,best_outr,angle,spacing

  
               
  
  
#def gridness_old(x,ds,doPlot=False,computeAngle=False,num_steps=20,
#             score_th_for_orientation=0.3,axes=None,cx=None,pearson=True,return_cx=False,min_diff=False):
#  """
#  Computes the gridness score of a pattern
#  x: pattern 
#  doPolt: plots the autocorrelogram and the gridness score
#  """
#  if cx is None:
#    cx = norm_autocorr(x,pearson=pearson)                                         # compute the normalized autocorrelation of the pattern
#
#
#  # compute grid spacing and orientation    
#  angle,spacing=get_grid_spacing_and_orientation(cx,ds,doPlot=False,compute_angle=False)
#
#  max_outr=np.ceil(spacing*2.5/ds)
#  min_outr=np.floor(spacing*0.7/ds)
#  outr_ran = np.arange(min_outr,max_outr,max_outr/num_steps)    # range of outer radii for the gridness score
#  best_score = -2                                               # best gridness score
#  best_outr = min_outr                                          # best radius
#
#  # indexes to cut a circle in the auto-correlation matrix
#  SX,SY = np.meshgrid(range(cx.shape[0]),range(cx.shape[1]))
#  tiles= np.array([np.ravel(SX), np.ravel(SY)])
#  if np.remainder(cx.shape[0],2)==1:
#    tile_center = np.array([[(cx.shape[0]+1)/2-1, (cx.shape[1]+1)/2-1]]).T
#  else:
#    tile_center = np.array([[(cx.shape[0])/2, (cx.shape[1])/2]]).T
#  tiles_dist = np.sqrt(sum((tiles-tile_center)**2,0))
#
#  # loop over increasing radii and retain the best score
#  for outr_idx in range(len(outr_ran)):
#
#    # compute score for the current outer radius
#    idxs=tiles[:,tiles_dist<outr_ran[outr_idx]]
#    score = comp_score(cx,idxs,min_diff=min_diff)
#
#    # retain best score
#    if score > best_score:
#      best_score = score
#      best_outr = outr_ran[outr_idx]
#
#  # take as inner radius half of the outer radius and recompute the score
#  in_r = best_outr/2
#  idxs= tiles[:,np.logical_and(tiles_dist>in_r,tiles_dist<best_outr)]
#  best_score = comp_score(cx,idxs,min_diff=min_diff)
#
#  # plot if requested
#  if doPlot is True:
#    import pylab as pl    
#    import plotlib as pp
#    #ax=pl.gca() if axes is None else axes
#    #pl.sca(ax)
#    pl.figure(figsize=(8,6))
#    
#    ax_grid = pl.GridSpec(2, 2, wspace=0.4, hspace=0.3)
#    
#    #pl.subplots_adjust(wspace=0.5)
#    
#    ax=pl.subplot(ax_grid[0,0])
#    pl.axis('equal')
#    pl.pcolormesh(x.T,rasterized=True)
#    pp.colorbar()
#    pp.noframe()
#    
#    ax=pl.subplot(ax_grid[0,1])
#    pl.axis('equal')
#    pl.pcolormesh(cx,rasterized=True)
#    ax.axes.get_yaxis().set_visible(False)
#    ax.axes.get_xaxis().set_visible(False)
#    ax.set_frame_on(False)
#    theta_ran = np.arange(0,2*np.pi,0.1)
#    pl.plot(best_outr*np.cos(theta_ran)+tile_center[0],best_outr*np.sin(theta_ran)+tile_center[1],'w')
#    pl.plot(max_outr*np.cos(theta_ran)+tile_center[0],max_outr*np.sin(theta_ran)+tile_center[1],'k')
#    pl.plot((spacing/ds)*np.cos(theta_ran)+tile_center[0],(spacing/ds)*np.sin(theta_ran)+tile_center[1],'g')
#    pl.plot(in_r*np.cos(theta_ran)+tile_center[0],in_r*np.sin(theta_ran)+tile_center[1],'w')
#    pl.text(10,10,'%.2f'%best_score, color='black',fontsize=10, weight='bold',bbox={'facecolor':'white'})
#    pp.colorbar()
#    
#    ax=pl.subplot(ax_grid[1,:])
#    deg_ran,c = get_score_corr_angle(cx,idxs)
#    pl.plot(deg_ran,c,'-k')
#    pl.xlabel('Angle')
#    pl.ylabel('Correlation')
#    pp.custom_axes()
#    pl.axhline(1)
#    pl.ylim(-1,1)
#    pl.title('max= %.2f min=%.2f'%(c.max(),c.min()))
#    
#    
#  # calculate angle if there we pass a threshold for the gridness
#  angle=np.NaN
#  if computeAngle is True and best_score > score_th_for_orientation:
#    angle,spacing = get_grid_spacing_and_orientation(cx,ds,compute_angle=True)
#  
#  if return_cx  is True:
#    return best_score,best_outr,angle,spacing,cx
#  else:
#    return best_score,best_outr,angle,spacing

def get_tiles(L,ds):
  """
  Returns the positions of the vertices of a square grid of side length L.
  The parameter ds indicates the grid spacing.
  """
  SX,SY = np.meshgrid(np.arange(-L/2.,L/2.,ds),np.arange(-L/2.,L/2.,ds))
  tiles= np.array([np.ravel(SX), np.ravel(SY)])
  return SX,SY,tiles
  
def get_tiles_int(L,num_samp=200):
  """
  Returns the positions of the vertices of a square grid of side length L.
  The parameter ds indicates the grid spacing.
  """
  samples=np.arange(-num_samp/2,num_samp/2)/float(num_samp)*L
  SX,SY = np.meshgrid(samples,samples)
  tiles= np.array([np.ravel(SX), np.ravel(SY)])
  return SX,SY,tiles
  
def divide_triangle(parent_triangle,grid_vertices,level=0,max_level=3,prec=9):
  """
  Recursively tassellates an equilateral triangles. 
  parent_triangle: input list of the vertices of the triangle to tasselate
  grid_vertices: output set of the vertices of the tassellated grid
  level: current level of the recursion
  max_level: desired level of recursive tassellation  
  The algorithm works like this. The triangle is divided in four equilateral
  child triangles by taking the midpoints of its edges. The central child triangle
  has vertices given by the thee midpoints, which are computed by the function
  get_child_triangle. Than the vertices of the other three sibling triangles are
  computed by the function get_sibling_triangles. After this subdivision,
  the function is called recursively for generated child triangle.
  """
  
  # the two main functions of the algorithm  
  get_child_triangle = lambda parent_triangle: [ tuple(np.around(0.5*(np.array(parent_triangle[i])+np.array(parent_triangle[i-1])),6)) for i in range(3) ]
  get_sibling_triangles = lambda parent_triangle,child_triangle: [ [ parent_triangle[p-3], child_triangle[p-3], child_triangle[p-2] ] for p in range(3)]

  child_triangle=get_child_triangle(parent_triangle)   # get the central child triangle
  [grid_vertices.add(v) for v in child_triangle if v not in grid_vertices]       # add it to the final set of vertices
  child_triangles=[child_triangle]

  if level<max_level:  
    child_triangles+= get_sibling_triangles(parent_triangle,child_triangle)
    for new_parent_triangle in child_triangles:
      divide_triangle(new_parent_triangle,grid_vertices,level+1,max_level,prec)
  else:
    return 
  
 
 
def get_all_phases(freq,angle,num_phases=100):
  """
  Returns a set of phases evenly distributed within the whole phase space
  """
  # the elementary phase space is an hexagon   
  side=np.sqrt(3)/(3*freq)
  hexagon=get_hexagon(side,angle)

  # first we get the phases uniformly spaced on a parallelogram
  axes=(0,1)
  phases=get_phases_on_pgram(freq,angle,num_phases,axes=axes)
  
  # then we shift these phases by +-lshift in the direction of the largest
  # diagonal of the parralelogram
  lshift=np.sqrt(3)/(6*freq)
  alpha1=angle+np.pi/6+axes[0]*np.pi/3
  alpha2=angle+np.pi/6+axes[1]*np.pi/3
  shift= lshift*(np.array([np.cos(alpha1)+np.cos(alpha2),np.sin(alpha1)+np.sin(alpha2)]))
  phases_shift1=phases+np.array([shift])
  phases_shift2=phases-np.array([shift])
  
  # we stack the three set of phases obtained
  all_phases=np.vstack((phases,phases_shift1,phases_shift2))
  
  # we discard the phases outside the elementary phase space  
  idxs=np.points_inside_poly(all_phases,hexagon)
  all_phases=all_phases[idxs,:]
  return all_phases
  
  
  
def get_hexagon(side,angle):
  """
  Returns the vertices of an hexagon of a given side length and oriented according
  to a given angle. The first and the last vertices are the same (this is to have)
  a closed line whan plotting the hexagon.
  """
  verts=np.zeros((7,2))
  for i in range(7):
    alpha=angle+np.pi/6+i*np.pi/3
    verts[i,0]=side*np.cos(alpha)
    verts[i,1]=side*np.sin(alpha)
  return verts
  
def get_rhombus(side,angle=np.pi/6):
  """
  Returns the vertices of a rhombus with edges oriented 60 degrees apart. 
  The first and the last vertices are the same (this is to have)
  a closed line whan plotting the polygon.
  """
  verts=np.zeros((5,2))
  verts[1,:]=side*np.array([np.cos(angle),np.sin(angle)])
  verts[3,:]=side*np.array([np.cos(angle+np.pi/3),np.sin(angle+np.pi/3)])
  verts[2,:]=verts[1,:]+verts[3,:]
  center=(verts[1,:]+verts[3,:])/2
  verts-=center
  return verts
  
def get_simple_hexagon(side,angle):
  """
  Same as get_hexagon but without the pi/6 offset in the orientation
  """
  verts=np.zeros((7,2))
  for i in range(7):
    alpha=angle+i*np.pi/3
    verts[i,0]=side*np.cos(alpha)
    verts[i,1]=side*np.sin(alpha)
  return verts
  
  
  
def get_phases_on_pgram(freq,angle,num_phases=36,axes=(0,1)):
  """
  Returns a set of phases uniformely sampled within a parallelogram.
  The parallelogram is the space spanned by two vectors oriented as two
  of the three grid axes and having length equal to double the period of the 
  cosine waves that form the grid.
  """
  # period of the cosines of the grid with the given parameter
  l=np.sqrt(3)/(2*freq)
  dl =l/(np.sqrt(num_phases)/2)
  ran=np.arange(-l,l,dl)+dl/2
  
  # the angles of the two axes    
  alpha1=angle+np.pi/6+axes[0]*np.pi/3
  alpha2=angle+np.pi/6+axes[1]*np.pi/3
   
  # points on the first axis
  x_phases1=np.cos(alpha1)*ran
  y_phases1=np.sin(alpha1)*ran

  # points on the first axis
  x_phases2=np.cos(alpha2)*ran
  y_phases2=np.sin(alpha2)*ran

  # points spanned by the two axes  
  X1,X2=np.meshgrid(x_phases1,x_phases2)
  Y1,Y2=np.meshgrid(y_phases1,y_phases2)
  X,Y=X1+X2,Y1+Y2
  phases = np.array([np.ravel(X), np.ravel(Y)]).T
  return phases   
    


def get_phases_on_axes(freq,angle,num_phases=60,axes=(0,1,2)):
  """
  Returns a set of phases such that the sum of grids with these phases is 
  flat, i.e., all grids cancel out. This is obtained by sampling phases on
  three lines with a length that is the double of the cosine
  period. The three lines are 60 degrees apart and are tilted by 90 degrees
  with respect to the original grid angle.
  """  
  # period of the cosines of the grid with the given parameter
  l=np.sqrt(3)/(2*freq)

  phases_per_axis = num_phases/len(axes)
  dl =l/phases_per_axis
  ran=np.arange(-l,l,dl)+dl/2

  x_phases = np.array([]) 
  y_phases = np.array([])   

  for i in axes:
    x_phases=np.concatenate((x_phases,np.cos(angle+np.pi/6+i*np.pi/3)*ran))
    y_phases=np.concatenate((y_phases,np.sin(angle+np.pi/6+i*np.pi/3)*ran))
    
  phases = np.array((x_phases,y_phases)).T
  return phases
  
  
################################
#### 2D GRIDS AND LATTICE ######
################################


  
def fourier_on_lattice(side,p1_rec,p2_rec,samples,signal,max_harmonic=None,return_harmonics=False):
  """
  Fourier series on a Bravais lattice whose unit vectors are 60 degrees apart
  
  inputs:
  --------
  
  side: side-length of the rhomboidal primary cell of the direct lattice
  p1_rec: primary vector of the reciprocal lattice
  p2_rec: primary vector of the reciprocal lattice
  samples: space samples in the lattice
  signal: signal of which the Fourier transform should be taken. SHAPE: n**2 X num_signals
  max_harmonic: maximum number of harmonic to compute (limit for faster computation)
  
  return_harmonics: if True the harmonics matrices are returned too
  
  output:
  -------

  F: Fourier transform on lattice. SHAPE: (2*max_harmonic)**2 x num_signals
     e.g. if max_harmonic is 2: we have 16 x num_signals, so harmonics go from -max_harmonic to max_harmonic-1

  
  """
  
  if max_harmonic is None:
    max_harmonic=np.int(np.sqrt(len(samples))/2)
    
  s1 = np.dot(samples,p1_rec)
  s2 = np.dot(samples,p2_rec)
  s12 = np.array([s1,s2])
  
  k_ran = np.arange(-max_harmonic,max_harmonic)  
  A,B = np.meshgrid(k_ran,k_ran)
  ab= np.array([np.ravel(A), np.ravel(B)]).T


  F= np.dot(np.exp(-1j*np.dot(ab,s12)),signal)
  
        
  # normalize by multiplying by the area of a rhombus with side-length dphi     
  V=side*side*np.sqrt(3)/2.
  F=F*V/len(samples)
  
  if return_harmonics:
    return A,B,F
  else:
    return F        


def inverse_fourier_on_lattice(side,p1_rec,p2_rec,samples,F):
  """
  Fourier series on a Bravais lattice whose unit vectors are 60 degrees apart
  
  input:
  -------
  
  side: side-length of the rhomboidal primary cell of the direct lattice
  p1_rec: primary vector of the reciprocal lattice
  p2_rec: primary vector of the reciprocal lattice
  samples: space samples in the lattice
  F: signal of which the inverse Fourier transform should be taken  SHAPE: n**2 X num_signals
  
  output:
  -------
  signal: inverse Fourier transfrom of F, SHAPE: n**2 X num_signals
  
  """
  
  max_harmonic =int((np.sqrt(F.shape[0]))/2 )
  
  
  k_ran = np.arange(-max_harmonic,max_harmonic)  

  s1 = np.dot(samples,p1_rec)
  s2 = np.dot(samples,p2_rec)
  
  s12 = np.array([s1,s2])
  A,B = np.meshgrid(k_ran,k_ran)
  ab= np.array([np.ravel(A), np.ravel(B)]).T
  
 
  signal= np.dot(F.T,np.exp(1j*np.dot(ab,s12)))
      
  # normalize      
  V=side*side*np.sqrt(3)/2
  signal=np.real(signal)/V
  
  return signal

def power_on_lattice(side,p1_rec,p2_rec,samples,signal,max_harmonic=None):  
  F=fourier_on_lattice(side,p1_rec,p2_rec,samples,signal,max_harmonic=max_harmonic)
  pw=(F*F.conjugate())/(side**2*np.sqrt(3)/2)
  return pw

def autocorr_on_lattice(side,p1_rec,p2_rec,samples,signal):
  pw=power_on_lattice(side,p1_rec,p2_rec,samples,signal)
  autocorr=inverse_fourier_on_lattice(side,p1_rec,p2_rec,samples,pw)
  return autocorr


def get_mean_pw_on_phase_rhombus(n,nx,L,signal,max_harmonic=10):

  #  signal: signal of which the Fourier transform should be taken. SHAPE: n**2 X num_signals

  gp=GridProps(n,2*np.pi,0)

  # compute noise power
  pw_phi=power_on_lattice(2*np.pi,gp.u1_rec,gp.u2_rec,gp.phases,signal,max_harmonic=max_harmonic).mean(axis=1)
  pw=pw_phi.real.reshape(max_harmonic*2,max_harmonic*2)
  hran=np.arange(2*max_harmonic)-max_harmonic
  
  return hran,pw


def get_mean_pw_on_space_rhombus(nx,L,signal,max_harmonic=10):
  """
  Computes the mean power estimated on a triangular lattice in space
  """
  hran,F=get_spectrum_on_space_rhombus(nx,L,signal,0.,max_harmonic=max_harmonic)
  pw=np.abs(F)**2
  pw_mean=pw.mean(axis=2)
  return hran,pw_mean
  
def get_spectrum_on_space_rhombus(nx,L,signal,grid_angle,max_harmonic=10,do_plot=False,swap_xy=False):
  """
  This compute the mean Fourier spectrum on a lattice for a given firing rate pattern (signal) sampled on a square grid
  The first step is to crop out a rhombus from the square grid where the signal is sampled.
  Afterwards we proceed using the method fourier_on_lattice
  
  """
  
  from matplotlib.path import Path
  
  if len(signal.shape)==1:
    signal=signal[:,np.newaxis]
    
  # space samples
  dx=float(L)/nx
  X,Y=np.mgrid[-L/2.:L/2.:dx,-L/2.:L/2.:dx]
  pos=np.array([np.ravel(X), np.ravel(Y)]).T
  
  # cut out a rhombus in space
  gp_L=GridProps(nx,L,grid_angle)
  R_L=gp_L.R_T
  path=Path(R_L)    
  idxs = path.contains_points(pos)
  pos_romb=pos[idxs,:]
  
  if swap_xy is True:
    # swap x and y if needed
    signal_mat=signal.reshape(nx,nx,signal.shape[1])
    signal_mat=signal_mat.transpose((1,0,2))
    signal=signal_mat.reshape(nx**2,signal.shape[1])
  
  # compute Fourier spectrum on lattice 
  F=fourier_on_lattice(L,gp_L.u1_rec,gp_L.u2_rec,pos_romb,signal[idxs,:],max_harmonic=max_harmonic)

  F=F.reshape(max_harmonic*2,max_harmonic*2,signal.shape[1])
  hran=np.arange(2*max_harmonic)-max_harmonic
  
  if do_plot is True:
    
    # plot the first input
    plot_amp=np.squeeze(np.abs(F[:,:,0]))
    #plot_amp[max_harmonic,max_harmonic]=0
  
    import pylab as pl
    import plotlib as pp
    pl.figure(figsize=(10,5))
    pl.subplots_adjust(wspace=0.3)
    
    pl.subplot(121,aspect='equal')
    pp.plot_on_rhombus(gp_L.R_T,L,0,len(signal[idxs,0]),pos_romb,signal[idxs,0])       
    pl.title('Pattern cropped on lattice')  
    
    pl.subplot(122,aspect='equal')
    pl.pcolormesh(plot_amp)
    pl.title('Fourier amplitude')  
    pp.colorbar()  
  
  return hran,F

def get_tuning_harmonic_masks(grid_harmonic,hran):
  """
  Compute binary masks to select 2D harmonics for Acell and Anoise
  """

  #grid_harmonic=int(L/grid_T)
  HX,HY=np.meshgrid(hran,hran)
    
  tuning_mask=np.zeros_like(HX).astype(bool)
  tuning_mask[(np.abs(HX)==grid_harmonic) & (HY==0)]=True
  tuning_mask[(np.abs(HY)==grid_harmonic) & (HX==0)]=True
  tuning_mask[(HX==grid_harmonic) & (HY==grid_harmonic)]=True
  tuning_mask[(HX==-grid_harmonic) & (HY==-grid_harmonic)]=True
  
  return tuning_mask


def get_phase_lattice(T,grid_angle):  
  """
  Returns the phase-space rhombus (R_T) and the unit vectors of the direct 
  (u_1 and u_2) and reciprocal (u1_rec and u2_rec) lattices of the phase space.  
  """
  
  # phase space
  R_T=get_rhombus(T,np.pi/6+grid_angle)
  
  u1 = T*np.array([np.cos(np.pi/6+grid_angle), np.sin(np.pi/6+grid_angle)])
  u2 = T*np.array([np.cos(grid_angle+np.pi/2), np.sin(grid_angle+np.pi/2)])
  
  U = np.vstack([u1, u2]).T
  U_rec = 2*np.pi*(np.linalg.inv(U)).T
  
  # unit vectors of the reciprocal lattice, note the scaling by 2pi
  u1_rec = U_rec[:,0]
  u2_rec = U_rec[:,1]

  return R_T,u1,u2,u1_rec,u2_rec

def get_phase_samples(n,u1,u2):
  """
  Sample phases in a lattice with unit vectors u1 and u2
  """

  # phase samples
  ran = np.arange(-n/2.,n/2.)/n
  u1_phases = np.array([u1])*ran[:,np.newaxis]
  u2_phases = np.array([u2])*ran[:,np.newaxis]
  
  X1,X2=np.meshgrid(u1_phases[:,0],u2_phases[:,0])
  Y1,Y2=np.meshgrid(u1_phases[:,1],u2_phases[:,1])
  X,Y=X1+X2,Y1+Y2
  phases = np.array([np.ravel(X), np.ravel(Y)]).T
  return phases

    
def get_space_samples(nx,L):

  """
  Sample space in a square lattice of side-length L
  """
  
  # space samples
  ran = np.arange(-nx/2.,nx/2.)/nx
  
  # ortogonal unit vectors for space
  v1=(L)*np.array([0,1])
  v2=(L)*np.array([1,0])
  
  v1_pos = np.array([v1])*ran[:,np.newaxis]
  v2_pos = np.array([v2])*ran[:,np.newaxis]
  
  
  X1,X2=np.meshgrid(v1_pos[:,0],v2_pos[:,0])
  Y1,Y2=np.meshgrid(v1_pos[:,1],v2_pos[:,1])
  X,Y=X1+X2,Y1+Y2
  pos = np.array([np.ravel(X), np.ravel(Y)]).T
  
  return pos



def get_square_signal(N,NX,pos,phases,T):
  """
  Computes a square grid as the sum of two waves
  """
  
  N=len(phases)
  NX=len(pos)

  angles=np.array([np.pi/2*i for i in np.arange(2)])
  k=2*np.pi/T*np.array([np.cos(angles),np.sin(angles)]).T
  
  pos_x = pos[:,0]
  pos_y = pos[:,1]
  
  phases_x = phases[:,0]
  phases_y = phases[:,1]
  
  pp_x = pos_x[np.newaxis,:]+phases_x[:,np.newaxis]
  pp_y = pos_y[np.newaxis,:]+phases_y[:,np.newaxis]
  
  g=np.zeros((N,NX))
  
  for i in range(2):
    g+=np.cos(k[i,0]*pp_x+k[i,1]*pp_y)  
  return g  


def get_grid_signal(N,NX,pos,phases,T,grid_angle):
  """
  Computes a triangular grid as the sum of three waves
  """
  
  N=len(phases)
  NX=len(pos)
  T_cos = T/2*np.sqrt(3) 
  
  angles=np.array([np.pi*i/3+grid_angle for i in np.arange(3)])
  k=2*np.pi/T_cos*np.array([np.cos(angles),np.sin(angles)]).T
  
  pos_x = pos[:,0]
  pos_y = pos[:,1]
  
  phases_x = phases[:,0]
  phases_y = phases[:,1]
  
  pp_x = pos_x[np.newaxis,:]+phases_x[:,np.newaxis]
  pp_y = pos_y[np.newaxis,:]+phases_y[:,np.newaxis]
  
  g=np.zeros((N,NX))
  
  for i in range(3):
    g+=np.cos(k[i,0]*pp_x+k[i,1]*pp_y)  
  return g  

 
  
def clipped_zoom(img, zoom_factor, **kwargs):
  """
  Zooms into an image by keeping the number of pixels constant
  """

  import warnings
  from scipy.ndimage import zoom
  h, w = img.shape[:2]

  # For multichannel images we don't want to apply the zoom factor to the RGB
  # dimension, so instead we create a tuple of zoom factors, one per array
  # dimension, with 1's for any trailing dimensions after the width and height.
  zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

  # Zooming out
  if zoom_factor < 1:

      # Bounding box of the zoomed-out image within the output array
      zh = int(np.round(h * zoom_factor))
      zw = int(np.round(w * zoom_factor))
      top = (h - zh) // 2
      left = (w - zw) // 2

      # Zero-padding
      out = np.zeros_like(img)
      
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

  # Zooming in
  elif zoom_factor > 1:

      # Bounding box of the zoomed-in region within the input array
      zh = int(np.ceil(h / zoom_factor))
      zw = int(np.ceil(w / zoom_factor))
      top = (h - zh) // 2
      left = (w - zw) // 2
      
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")        
        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
      
      # `out` might still be slightly larger than `img` due to rounding, so
      # trim off any extra pixels at the edges
      trim_top = ((out.shape[0] - h) // 2)
      trim_left = ((out.shape[1] - w) // 2)
      out = out[trim_top:trim_top+h, trim_left:trim_left+w]

  # If zoom_factor == 1, just return the input array
  else:
      out = img
  return out
    

def normalize_grid(L,nx,r_map,target_harmonic,est_angle,est_spacing,do_plot=False,verbose=False):
  """
  Normalize grid to fixed spacing and orientation in order to compute the grid tuning index.
  We normalize the grid to a spacing of L/target_harmonic where L is the arena length and target_harmonic is an integer
  The default orientation is zero angle (aligned to vertical border)
  
  L: side length of the arena
  nx: number of samples per dimension
  r_map: one firing rate map in the arena
  target_harmonic: an integer we normalize to a spacing of L/target_harmonic 
  """
  
  assert(type(target_harmonic)==int and target_harmonic>1 )
  
  # make sure the data is 2D
  r_map=r_map.reshape(nx,nx)
  
  # estimate spacing and orientation
  #cx=norm_autocorr(r_map)  
  #st_angle,est_spacing=get_grid_spacing_and_orientation(cx,float(L)/nx,doPlot=False)  
  #print 'est_angle: %.2f est_spacing: %.2f'%(est_angle*180/np.pi,est_spacing)
  
  # rescale and rotate to normalized space  
  zoom_factor=(float(L)/target_harmonic)/est_spacing
  rotation_angle=np.remainder(est_angle*180/np.pi,60)
  if rotation_angle>30.:
    rotation_angle=rotation_angle-60.
  
  if verbose:
    print 'zoom: %.2f rotation: %.2f '%(zoom_factor, rotation_angle)
  r_map_rescaled=clipped_zoom(r_map,zoom_factor) 
  
  assert np.all(r_map_rescaled.shape==(nx,nx))
  r_map_norm=rotate(r_map_rescaled, rotation_angle,reshape=False)
  
  # plotting  
  if do_plot is True:
    import pylab as pl  
    import plotlib as pp
    
    pl.figure(figsize=(12,5))
    pl.subplots_adjust(wspace=0.3)

    pl.subplot(131,aspect='equal')
    pl.pcolormesh(r_map.T)  
    pp.colorbar()
    pl.title('Original')
    
    pl.subplot(132,aspect='equal')
    pl.pcolormesh(r_map_rescaled.T)    
    pp.colorbar()               
    pl.title('Rescaled')
    
    pl.subplot(133,aspect='equal')
    pl.pcolormesh(r_map_norm.T)
    pp.colorbar()
    pl.title('Rescaled and Rotated')
    
  return r_map_norm  




def comp_grid_tuning_index(L,nx,r_maps,do_plot=False,verbose=False,warnings=False,return_tuning_harmonics=False):
  """
  Compute grid tuning index for a batch of firing rate maps
  L: side length of the arena
  nx: number of samples per dimension
  r_maps: nx**2 X N firing rate maps
  """
  
  assert(r_maps.min()>=0)
  
  max_harmonic=10
  
  if len(r_maps.shape)==1:
    r_maps=r_maps[:,np.newaxis]

  num_cells=r_maps.shape[1]
  assert(r_maps.shape[0]==nx**2)
    
  dx=float(L)/nx

  r_maps_norm=np.zeros_like(r_maps)  
  target_harmonics=np.zeros(num_cells)
  
  for cell_idx in xrange(num_cells):
    if verbose:
      print 'Normalizing %d/%d'%(cell_idx,num_cells)
    
    plot_curr_cell=do_plot & (cell_idx==0)
    
    r_map=r_maps[:,cell_idx]

   
    # compute autocorrelation and estimate grid spacing and orientation  
    cx=norm_autocorr(r_map.reshape(nx,nx))
    est_angle,est_spacing=get_grid_spacing_and_orientation(cx,dx,doPlot=plot_curr_cell)
    if np.isnan(est_angle):
      est_angle=0.
      if warnings:
        print 'Cannot estimate angle for cell_idx=%d, pattern will not be rotated'%cell_idx
    
    if  not np.isnan(est_spacing):
      target_harmonic=int(round(L/est_spacing))
      
      if verbose:
        print 'target_harmonic=%d for  cell_idx=%d'%(target_harmonic,cell_idx)
      
      if target_harmonic>1:
        # normalize firing rate map to scale L/target_harmonic and angle zero
        r_map_norm = normalize_grid(L,nx,r_map,target_harmonic,est_angle,est_spacing,do_plot=plot_curr_cell,verbose=verbose)
      else:
        if warnings:
          print 'WARNING: Target harmonic smaller than 2 for cell_idx=%d'%cell_idx
        target_harmonic=np.nan
        r_map_norm=np.zeros(nx**2)
        
    else:
      if warnings:
        print 'WARNING: Cannot estimate spacing for cell_idx=%d, setting grid-tuning index to 0'%cell_idx
      target_harmonic=np.nan
      r_map_norm=np.zeros(nx**2)

    # save normalized map and target harmonic for this pattern
    r_maps_norm[:,cell_idx]=r_map_norm.reshape(nx**2)
    target_harmonics[cell_idx]=target_harmonic
    
    
  # compute the Fourier amplitude of all the patterns on space rhombus
  hran,F=get_spectrum_on_space_rhombus(nx,L,r_maps_norm,0.,max_harmonic,do_plot=do_plot)
  F_amp=np.abs(F)
  
  grid_tuning_indexes=np.zeros(num_cells)
  
  for cell_idx in xrange(num_cells):
    if not np.isnan(target_harmonics[cell_idx]):
      # compute mean power at the target harmonic and at DC
      tuning_mask=get_tuning_harmonic_masks(target_harmonics[cell_idx],hran)
      tuning_amp=F_amp[tuning_mask,cell_idx].mean(axis=0)    
      dc_value=F_amp[max_harmonic,max_harmonic,cell_idx] #the DC is at position max_harmonic 
  
      # compute the tuning index
      grid_tuning_indexes[cell_idx]=tuning_amp/dc_value
  
#  
#  # compute the Fourier amplitude of all the patterns on space rhombus
#  hran,F=get_spectrum_on_space_rhombus(nx,L,r_maps_norm,0.,max_harmonic,do_plot=do_plot)
#  F_amp=np.abs(F)
#  
#  # compute mean power at the target harmonic and at DC
#  tuning_mask=get_tuning_harmonic_masks(L,float(L)/target_harmonic,hran)
#  tuning_amp=F_amp[tuning_mask,:].mean(axis=0)    
#  dc_values=F_amp[max_harmonic,max_harmonic,:] #why maxharmonic here?
#  
#  # compute the tuning index
#  grid_tuning_indexes=tuning_amp/dc_values
#  
  
  
  assert np.all(np.isfinite(grid_tuning_indexes))
  
  if return_tuning_harmonics:
    return target_harmonics,grid_tuning_indexes
  else:
    return grid_tuning_indexes
  

  
#################
#### TESTING ####
#################

def get_test_grids_spacing_range(L,nx,num_grids,spacing_min,spacing_max,zero_phases=True,grid_angle=0):
  

  spacings=np.linspace(spacing_min,spacing_max,num_grids)
  norm_phases = np.zeros((2,num_grids)).T


  SX,SY,tiles=get_tiles(L,float(L)/nx)
  
  grids=np.zeros((nx,nx,num_grids))
  
  for idx,spacing in enumerate(spacings):

    max_grid_phase_x = 2.0*spacing
    max_grid_phase_y = spacing*np.sqrt(3)


    phase=norm_phases[idx,:]
    phase[0]*=max_grid_phase_x
    phase[1]*=max_grid_phase_y

  
    grid=simple_grid_fun(tiles,spacing,angle=grid_angle,phase=phase)
    grids[:,:,idx]=grid.reshape(nx,nx)
  return grids,spacings



def get_test_grids_all_angles(L,nx,num_grids,grid_spacing,zero_phases=True):
  
  ang_range=np.arange(0,np.pi/3,np.pi/3/num_grids)
  max_grid_phase_x = 2.0*grid_spacing
  max_grid_phase_y = grid_spacing*np.sqrt(3)
  phases = np.zeros((2,num_grids)).T
  phases[:,0] = rand(num_grids)*max_grid_phase_x
  phases[:,1] = rand(num_grids)*max_grid_phase_y

  SX,SY,tiles=get_tiles(L,float(L)/nx)
  
  grids=np.zeros((nx,nx,num_grids))
  
  for idx,ang in enumerate(ang_range):
    
    grid=simple_grid_fun(tiles,grid_spacing,angle=ang,phase=phases[idx,:])
    grids[:,:,idx]=grid.reshape(nx,nx)
  return grids,ang_range  
    
  
def test_orientation_and_spacing_detection():
  """
  A function to test grid orientation detection
  """
  import pylab as pl  

  L=2
  nx=100
  num_grids=25
  grid_spacing=.5
  
  grids,angles=get_test_grids_all_angles(L,nx,num_grids,grid_spacing,zero_phases=True)

  
  pl.figure(figsize=(10,10))

  for idx in xrange(num_grids):
    ax=pl.subplot(5,5,idx+1,aspect='equal')
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.set_frame_on(False)
    grid=grids[:,:,idx]
    cx=norm_autocorr(grid)
    pl.axis('equal')    
    est_angle,est_spacing=get_grid_spacing_and_orientation(cx,float(L)/nx,doPlot=True,ax=pl.gca())

    ang_deg=angles[idx]*360/(2*np.pi)
    est_angle_deg=np.remainder((est_angle*360/(2*np.pi))-30,60)
    pl.text(20,20,'Real Ang.: %.2f\nEst Ang.: %.2f'%(ang_deg,est_angle_deg),fontsize=9,color='k',weight='bold',bbox={'facecolor':'white','edgecolor':'white'})
    pl.text(20,80,'Real Sp.: %.2f\nEst Sp.: %.2f'%(grid_spacing,est_spacing),fontsize=9,color='k',weight='bold',bbox={'facecolor':'white'})
  pl.subplots_adjust(hspace=0.1,wspace=0.1,left=0.05,right=0.95,top=0.95,bottom=0.05)

 
 


 