import numpy as np
import pandas as pd, os
import matplotlib.pyplot as plt
import sys     # for comand line options
import animate # needs conda install fsspec
import preprocessor as prep
import graddesc as gd
import dataReader as dR
import trilateration as tri
from keras.models import Sequential
from keras.layers import Dense
from keras import Input

def compute_windows(nparray, npast=1):
    dataX, dataY = [], [] # window and value
    for i in range(len(nparray)-1):
        a = nparray[i,:]
        dataX.append(a[:npast])
        dataY.append(a[npast:])
    return np.array(dataX), np.array(dataY)

def plotStanza(xfig,yfig,fig=None):
   if fig is None:
      fig, ax = plt.subplots(figsize=(xfig,yfig))
   ax = plt.gca()
   img = plt.imread("../stanza.png")
   ax.imshow(img,zorder=0, extent=[0, xfig, 0, yfig])
   
def plotPath(x,y):
   fig, ax = plt.subplots(figsize=(xfig,yfig))
   ax.scatter(y, x, c="blue", label="UWB", edgecolors='none')
   plotStanza(xfig,yfig,fig)
   plt.show()
   
def mlp(dataset):
    # train - test sets
   cutpoint = int(len(dataset) * 0.7) # 70% train, 30% test
   train, test = dataset[:cutpoint], dataset[cutpoint:]
   print("Len train={0}, len test={1}".format(len(train), len(test)))
   
   npast = 4
   trainX, trainY = compute_windows(train, npast)
   testX, testY = compute_windows(test, npast) # should get also the last npred of train
   
   # Multilayer Perceptron model - parte che effettivamente costruisce la rete
   model = Sequential()
   n_hidden = 3
   n_output = 2 # 4 neuroni di input, 3 hidden e 2 output
   model.add(Dense(n_hidden, input_dim=npast, activation='relu')) # hidden neurons, 1 layer
   model.add(Dense(n_output)) # output neurons
   model.compile(loss='mean_squared_error', optimizer='adam')
   model.fit(trainX, trainY, epochs=300, batch_size=13, verbose=2) # batch_size divisor of len(trainX)
   
   # Model performance
   trainScore = model.evaluate(trainX, trainY, verbose=2)
   print('Score on train: MSE = {0:0.2f} '.format(trainScore))
   testScore = model.evaluate(testX, testY, verbose=2)
   print('Score on test: MSE = {0:0.2f} '.format(testScore))

    
   prediction = model.predict(np.append(trainX, testX, axis=0)) # predictions
   # testForecast = model.predict(testX) # forecast

   plotPath(prediction[:,0], prediction[:,1])
   return model

# Program entry point
if __name__ == "__main__":
   print("init")
   # change working directory to script path
   abspath = os.path.abspath(__file__)
   dname = os.path.dirname(abspath)
   os.chdir(dname)

   source_path=""
   if len(sys.argv)>1:
      source_path=sys.argv[1]
      dbtable=sys.argv[2]
   else:
      source_path='..//'
      dbtable='fusion_EG_1'

   anchorcoord = {
      "CB1D": (2.5, 0, 0),
      "8418": (2.5, 16.9, 0),
      "198A": (4.95, 12.1 , 0),
      "D20C": (0, 4.6, 0),
      "9028": (4.95, 4.6, 0),
      "CC90": (0, 12.1, 0)
   }
   anchornames = list(anchorcoord.keys())

   fixedpcoords = {
      "Pos0": (2.50, 4.20, 1,15),
      "Pos1": (2.50, 6.66, 1,15),
      "Pos2": (3.95, 6.66, 1,15),
      "Pos3": (2.50, 9.06, 1,15),
      "Pos4": (1.00, 9.06, 1,15),
      "Pos5": (2.50, 13.00, 1,15),
      "Pos6": (1.00, 13.00, 1,15)
   }
   fixedpnames = list(fixedpcoords.keys())

   d0 = dR.fixedpdist(anchornames,fixedpnames,anchorcoord,fixedpcoords)

   df = pd.DataFrame()
   df_cv = pd.DataFrame()
   df_real = pd.DataFrame()
   df_cv, df_real = dR.parse_table_data2(anchorcoord)
   #df.to_csv("df.csv")
   d = dR.getDistances(d0)

   #dfmerged = pd.merge_ordered(df,df,on="time",suffixes=("_1","_2"), fill_method="ffill")
   fig = plt.figure(figsize=(6,4))
   ax = plt.gca()
   d.plot(kind='line',ax=ax)
   plt.show()

   # trilateration
   T = tri.Trilat()
   i=0
   #p = np.zeros(2)
   points   = pd.DataFrame(columns=['time','x','y'])
   antennas = pd.DataFrame(columns=['x','y'])
   for i in range(len(anchornames)):
      antennas = antennas.append({"x":anchorcoord[anchornames[i]][0],
                                  "y":anchorcoord[anchornames[i]][1]},
                                  ignore_index=True)
   for i in range(len(d)):  
      p = T.getPointCoord(d.iloc[i],antennas)
      if p is not None:
         points = points.append({"time":d.index[i],"x":p[0],"y":p[1]},ignore_index=True)
      else:
         points = points.append({"time":d.index[i],"x":0,"y":0},ignore_index=True) # tanto per metterci qualcosa

   points = points.set_index('time')
   numpoints = len(df)

   xfig = 16.9       # room length
   yfig = 5          # room width
   
   df_uwb_cv = pd.concat([points['x'], points['y'], df_cv['x'], df_cv['y'], df_real['x'], df_real['y']], axis=1, keys=['x_uwb', 'y_uwb', 'x_cv', 'y_cv','x','y'])
   
   #df_uwb_cv = df_uwb_cv.fillna(value=None, method='backfill', axis=None, limit=None, downcast=None)
   # df_uwb_cv = df_uwb_cv.interpolate(method='linear')
   #df_uwb_cv = df_uwb_cv.ffill()  
   # Plot uwb path
   plotPath(df_uwb_cv.x_uwb, df_uwb_cv.y_uwb)
   #plot cv path
   plotPath(df_uwb_cv.x_cv, df_uwb_cv.y_cv)
   # Plot real path
   plotPath(df_uwb_cv.x, df_uwb_cv.y)
   
   df_uwb_cv.x_cv = df_uwb_cv.apply(
     lambda row: row['x_uwb'] if np.isnan(row['x_cv']) else row['x_cv'], 
     axis=1
   )
   df_uwb_cv.y_cv = df_uwb_cv.apply(
     lambda row: row['y_uwb'] if np.isnan(row['y_cv']) else row['y_cv'], 
     axis=1
   )
   # df_uwb_cv = df_uwb_cv.interpolate(method='linear')
   plotPath(df_uwb_cv.x_cv, df_uwb_cv.y_cv)
   # Plot real path
   
   dataset = df_uwb_cv.values
   
   mlp(dataset)
   
   print("end")
