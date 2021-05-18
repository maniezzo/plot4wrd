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

   # rolling outlier detection
   def median_filter(self,num_std = 3):
      def _median_filter(self,x):
         _median = np.median(x)
         _std = np.std(x)
         s = x[-1]
         return s if s >= _median - num_std * _std and s <= _median + num_std * _std else np.nan
      return _median_filter

   # drop outliers (further than 1.5 iqr)   
   def drop_outliers(self,df, field_name, cutoff = 1.5):
      iqr = (np.percentile(df[field_name], 75) - np.percentile(df[field_name], 25))
      distance = cutoff * iqr
      df.drop(df[df[field_name] > distance + np.percentile(df[field_name], 75)].index, inplace=True)
      df.drop(df[df[field_name] < np.percentile(df[field_name], 25) - distance].index, inplace=True)

   # clip outliers (5 and 95 percentile)   
   def clip_outliers(self, df, minPercentile = 0.05, maxPercentile = 0.95):
      df_list = list(df)
      for _ in range(len(df.columns)):
         df[df_list[_]] = df[df_list[_]].clip((df[df_list[_]].quantile(minPercentile)),(df[df_list[_]].quantile(maxPercentile)))
