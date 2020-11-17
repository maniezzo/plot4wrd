import numpy as np
import pandas as pd

from pykalman import KalmanFilter


class Preproc:
   width = 1# rollong window width
   
   # contructor
   def __init__(self,width=1):
      self.width = width
      
   # kalman filtering
   def kalman(self,ds):
      ds = ds.fillna(method='ffill') # fill in NaN's
      measurements = ds.to_numpy()
      kf = KalmanFilter(transition_matrices=[1],
                     observation_matrices=[1],
                     initial_state_mean=measurements[0],
                     initial_state_covariance=1,
                     observation_covariance=10,
                     transition_covariance=1) 
      state_means, state_covariances = kf.filter(measurements)
      state_std = np.sqrt(state_covariances[:,0]) # in case I needed them
      return state_means
   
   # exponential moving average
   def expMA(self,ds):
      return ds.ewm(span=self.width,adjust=False).mean() # exponential MA
   
   # centered moving average
   def rollCMA(self,ds):
      ds1 = ds.rolling(center=True,window=self.width).mean() # rolling MA
      ds1[0:int(self.width/2)] = ds[0:int(self.width/2)] # fill in
      ds1[int(self.width/2):len(ds)] = ds[int(self.width/2):len(ds)] # fill in
      return ds1
   


