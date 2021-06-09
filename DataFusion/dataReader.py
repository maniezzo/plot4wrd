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
def parse_table_data():
   global df
   Db = dbManager.SqLiteDB("sqlite", dbpath='..//', dbname='jan2021data.sqlite')
   query = "SELECT timestamp as time, tagID, pos_x as x, pos_y as y, pos_z as z, quality,n_dist, anchorID_1, dist_1, anchorID_2, dist_2, anchorID_3, dist_3, anchorID_4, dist_4, anchorID_5, dist_5, ax, ay, az, gx, gy, gz, mx, my, mz FROM fusion_EG_1"
   df = Db.get_all_data(table="fusion_EG_1",query=query)
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
  
def parse_csv_data(path):
   df_data = pd.DataFrame(pd.read_csv('..//'+path,
                                       sep=',',
                                       names=['id','time','tagID','x','y','z','quality'],
                                       header=None))
   df_data = df_data.drop(columns=['id'])
   df_data = df_data.iloc[1:]
   df_data['x'] = pd.to_numeric(df_data['x'],errors='coerce')
   df_data['y'] = pd.to_numeric(df_data['y'],errors='coerce')
   df_data['z'] = pd.to_numeric(df_data['z'],errors='coerce')
   df_data['quality'] = pd.to_numeric(df_data['quality'],errors='coerce')
   return df_data
    
# legge dataframe da database
def parse_table_data2(anchorcoord):
   global df
   
   cv = parse_csv_data('Simulation//CV_CV1_210609.csv')
   df_cv = pd.DataFrame(columns=['time','x','y'])
   df_cv['time'] = cv['time']
   df_cv['x'] = cv['x']
   df_cv['y'] = cv['y']
   df_cv = df_cv.set_index('time')
   real = parse_csv_data('Simulation//UWB_NoNOISE_210609.csv')
   df_real = pd.DataFrame(columns=['time','x','y'])
   df_real['time'] = real['time']
   df_real['x'] = real['x']
   df_real['y'] = real['y']
   df_real = df_real.set_index('time')
   
   df = pd.DataFrame(columns=['time', 'tagID', 'x', 'y', 'z', 'quality','n_dist', 'anchorID_1', 'dist_1', 'anchorID_2', 'dist_2', 'anchorID_3', 'dist_3', 'anchorID_4', 'dist_4', 'anchorID_5', 'dist_5', 'anchorID_6', 'dist_6'])
   df['x'] = 0
   df['y'] = 0
   df['z'] = 0
   df['n_dist'] = 4
   
   anchornames = list(anchorcoord.keys())
   for i in range(len(anchornames)):
       df_antenna = parse_csv_data('Simulation//UWB_'+anchornames[i]+'_210609.csv')
       df['time'] = df_antenna['time']
       df['tagID'] = df_antenna['tagID']
       df['quality'] = df_antenna['quality']
       df['anchorID_'+str(i+1)] = anchornames[i]
       df['dist_'+str(i+1)] = np.sqrt((anchorcoord[anchornames[i]][0] - df_antenna['x'])**2 + (anchorcoord[anchornames[i]][1] - df_antenna['y'])**2)
    
   df = df.set_index('time')
   prep = preprocessor.Preproc()
   window=7
   # for attr in ['dist_1','dist_2','dist_3','dist_4','dist_5']:
   for attr in ['dist_1','dist_2','dist_3','dist_4','dist_5','dist_6']:
      print(attr)
      temp = df[attr].iloc[0:window]
      # apply: ndarray input if raw=True or a Series if raw=False
      df[attr]=df[attr].rolling(window).apply(median_filter(num_std=3), raw=True)
      df[attr].iloc[0:window] = temp
      df[attr]=df[attr].fillna(method='bfill') # backward fill

   #get ordered anchor data
   define_anchor_columns()
   # init_df()                          
   return df_cv, df_real

# from db to distances
def getDistances(d0):
   global df
   newcols  = ['CB1D','8418','198A','D20C','9028','CC90']

   # da una pagina
   from math import log10
   # MHz= 6000 # raw_input('MHz FREQUENCY (2417, 5200, ...) : ')
   # MHz=int(MHz)
   # dBm= 23 # raw_input('dBm TRANSMITTER POWER (23, 63, ...) : ')
   # dBm=int(dBm)
   # FSPL = 27.55 # Free-Space Path Loss adapted avarage constant for home WiFI routers and following units
   # m = 10 ** (( FSPL - (20 * log10(MHz)) + abs(dBm) ) / 20 )
   # m=round(m,2)
   # dist = m

   # initdist = d0[0,:]  
   # icol = 0
   # lmbda = 0.125
   # esp = (df[newcols[icol]]-df[newcols[icol]][0])/(10*2)#-20* np.log10( (4*3.14*initdist[icol]/lmbda) ))/(10*2) 
   # dist = initdist[icol]*np.power( 10, esp )

   # # formula banale, d0*(p0/pr)^1/n
   # n=0.25
   # icol=0
   # d1 = d0[0,icol]*(df[newcols[icol]][0]/df[newcols[icol]])**(1/n)

   d = pd.DataFrame()
   for icol in range(len(newcols)) :
       d[newcols[icol]] = df[newcols[icol]]
      # d[newcols[icol]] = d0[0,icol]*(df[newcols[icol]]/df[newcols[icol]][0])**(1/n)
      # #esp = (df[newcols[icol]]-df[newcols[icol]][0])/20
      # #esp.fillna(value=np.nan, inplace=True)
      # #d[newcols[icol]] = initdist[icol]*np.power( 10, esp )
      # d[newcols[icol]].clip(lower=0,upper=15,inplace=True)
      # d[newcols[icol]] =  d[newcols[icol]].rolling(3,min_periods=1).mean()
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
   anchcols = ['anchorID_1','anchorID_2','anchorID_3','anchorID_4','anchorID_5','anchorID_6']
   anchnames= ['CB1D','8418','198A','D20C','9028','CC90']#["bleA2","bleA3","bleA4","bleA5"]#,"bleA6"]
   newcols  = ['CB1D','8418','198A','D20C','9028','CC90'] #['A1','A2','A3','A4','A5']

   df["CB1D"] = df.loc[df['anchorID_1']=="CB1D"].loc[:,['dist_1']]
   df["8418"] = df.loc[df['anchorID_2']=="8418"].loc[:,['dist_2']]
   df["198A"] = df.loc[df['anchorID_3']=="198A"].loc[:,['dist_3']]
   df["D20C"] = df.loc[df['anchorID_4']=="D20C"].loc[:,['dist_4']]
   df["9028"] = df.loc[df['anchorID_5']=="9028"].loc[:,['dist_5']]
   df["CC90"] = df.loc[df['anchorID_6']=="CC90"].loc[:,['dist_6']]

   for inew in range(0,len(newcols)):
      for icol in range(0,len(anchcols)):
         colid = list(df.columns).index(anchcols[icol])
         for i in range(1,len(df.index)):
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

