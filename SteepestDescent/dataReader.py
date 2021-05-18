import pandas as pd, numpy as np
import dbManager, math
import preprocessor

# num_std: quanti a sx e a dx del mediano
# accetta solo dati +- num_std*stdev dal mediano
def median_filter(num_std=3):
    def _median_filter(x):
        _median = np.median(x)
        _std = np.std(x)
        s = x[-1]
        return s if s >= _median - num_std * _std and s <= _median + num_std * _std else _median #np.nan
    return _median_filter

# legge dataframe da database
def parse_table_data(dbtable):
   global df
   Db = dbManager.SqLiteDB("sqlite", dbpath='..//', dbname='jan2021data.sqlite')
   query = "SELECT timestamp as time, tagID, pos_x as x, pos_y as y, pos_z as z, quality,n_dist, anchorID_1, dist_1, anchorID_2, dist_2, anchorID_3, dist_3, anchorID_4, dist_4, anchorID_5, dist_5, ax, ay, az, gx, gy, gz, mx, my, mz FROM "+dbtable
   df = Db.get_all_data(table=dbtable,query=query)
   df['time'] = pd.to_datetime(df['time'], format="%M:%S.%f")
   df['time'] = df['time'] + pd.DateOffset(years=121, month=1, day=21, hour=12)    
   df.replace([None], np.nan, inplace=True)
   
   prep = preprocessor.Preproc()
   window=7
   for attr in ['dist_1','dist_2','dist_3','dist_4','dist_5']:
      print(attr)
      temp = df[attr].iloc[0:window]
      # apply: ndarray input if raw=True or a Series if raw=False
      df[attr]=df[attr].rolling(window).apply(median_filter(num_std=3), raw=True)
      df[attr].iloc[0:window] = temp
      df[attr]=df[attr].fillna(method='bfill') # backward fill

   #get ordered anchor data
   define_anchor_columns()
   init_df()                          
   return 
    
# from db to distances
def getDistances(d0):
   global df
   newcols  = ['A1','A2','A3','A4','A5']

   # da una pagina
   from math import log10
   MHz= 6000 # raw_input('MHz FREQUENCY (2417, 5200, ...) : ')
   MHz=int(MHz)
   dBm= 23 # raw_input('dBm TRANSMITTER POWER (23, 63, ...) : ')
   dBm=int(dBm)
   FSPL = 27.55 # Free-Space Path Loss adapted avarage constant for home WiFI routers and following units
   m = 10 ** (( FSPL - (20 * log10(MHz)) + abs(dBm) ) / 20 )
   m=round(m,2)
   dist = m

   initdist = d0[0,:]  
   icol = 0
   lmbda = 0.125
   esp = (df[newcols[icol]]-df[newcols[icol]][0])/(10*2)#-20* np.log10( (4*3.14*initdist[icol]/lmbda) ))/(10*2) 
   dist = initdist[icol]*np.power( 10, esp )

   # formula banale, d0*(p0/pr)^1/n
   n=0.25
   icol=0
   d1 = d0[0,icol]*(df[newcols[icol]][0]/df[newcols[icol]])**(1/n)

   d = pd.DataFrame()
   for icol in range(len(newcols)) :
      d[newcols[icol]] = d0[0,icol]*(df[newcols[icol]]/df[newcols[icol]][0])**(1/n)
      #esp = (df[newcols[icol]]-df[newcols[icol]][0])/20
      #esp.fillna(value=np.nan, inplace=True)
      #d[newcols[icol]] = initdist[icol]*np.power( 10, esp )
      d[newcols[icol]].clip(lower=0,upper=15,inplace=True)
      d[newcols[icol]] =  d[newcols[icol]].rolling(3,min_periods=1).mean()
   return d

# makes sure of types and index
def init_df():
   global df
   df = df.replace('-', np.nan)
   df.time = df.time.astype('datetime64[ns]')
   df.x = df.x.astype(float)
   df.y = df.y.astype(float)
   df.z = df.z.astype(float)
   df.quality = df.quality.astype(float)
   # set time ad index of time series
   df.set_index('time', inplace=True)
   df.index = df.index.floor('10ms') # keep time down to centiseconds

   # fake distance
   df['x'] = df[['x','y']].mean(axis=1)
   df['distance'] = df[['x','y']].mean(axis=1)
   return 

# nuove colonne, una per ancora
def define_anchor_columns():
   global df
   anchcols = ['anchorID_1','anchorID_2','anchorID_3','anchorID_4','anchorID_5']
   anchnames= ["bleA2","bleA3","bleA4","bleA5","bleA6"]
   newcols  = ['A1','A2','A3','A4','A5']

   df["A1"] = df.loc[df['anchorID_1']=="bleA2"].loc[:,['dist_1']]
   df["A2"] = df.loc[df['anchorID_2']=="bleA3"].loc[:,['dist_2']]
   df["A3"] = df.loc[df['anchorID_3']=="bleA4"].loc[:,['dist_3']]
   df["A4"] = df.loc[df['anchorID_4']=="bleA5"].loc[:,['dist_4']]
   df["A5"] = df.loc[df['anchorID_5']=="bleA6"].loc[:,['dist_5']]

   for inew in range(0,len(newcols)):
      for icol in range(0,len(anchcols)):
         colid = list(df.columns).index(anchcols[icol])
         for i in range(len(df.index)):
            if(pd.notnull(df[anchcols[icol]][i])):
               if(df[anchcols[icol]][i]==anchnames[inew]):
                  df[newcols[inew]][i]=df.iloc[i,colid+1]
   return

# calcola distanze fra ancore e punti fissi
def fixedpdist(anchornames,fixedpnames,anchorcoord,fixedpcoords):
   d0 = np.zeros((len(fixedpnames),len(anchornames)))
   for ian in range(len(anchornames)):
      for ipn in range(len(fixedpnames)):
         acoord = anchorcoord[anchornames[ian]]
         pcoord = fixedpcoords[fixedpnames[ipn]]
         d0[ipn,ian] = math.sqrt((acoord[0]-pcoord[0])**2 +
                        (acoord[1]-pcoord[1])**2
                        ) # +(acoord[2]-pcoord[2])**2 )
   return d0

