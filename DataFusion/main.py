import numpy as np
import pandas as pd, os
import matplotlib.pyplot as plt
import sys     # for comand line options
import animate # needs conda install fsspec
import preprocessor as prep
import graddesc as gd
import dataReader as dR
import trilateration as tri

def plotStanza(xfig,yfig,fig=None):
   if fig is None:
      fig, ax = plt.subplots(figsize=(xfig,yfig))
   ax = plt.gca()
   img = plt.imread("../stanza.png")
   ax.imshow(img,zorder=0, extent=[0, xfig, 0, yfig])

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
   dR.parse_table_data2(anchorcoord)
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
   points   = pd.DataFrame(columns=['x','y'])
   antennas = pd.DataFrame(columns=['x','y'])
   for i in range(len(anchornames)):
      antennas = antennas.append({"x":anchorcoord[anchornames[i]][0],
                                  "y":anchorcoord[anchornames[i]][1]},
                                  ignore_index=True)
   for i in range(len(d)-1):  
      p = T.getPointCoord(d.iloc[i],antennas)
      if p is not None:
         points = points.append({"x":p[0],"y":p[1]},ignore_index=True)
      else:
         points = points.append({"x":0,"y":0},ignore_index=True) # tanto per metterci qualcosa

   numpoints = len(df)
   # inverto x con y per fare la figura larga
   #d.plot(kind='line', color='red', use_index=True, ax=ax)
   #y = (5-dfmerged.x_1)
   #x = dfmerged.y_1
   xfig = 16.9       # room length
   yfig = 5          # room width
   #a = animate.AnimatedScatter(numpoints,x,y,xfig,yfig)
   # a = animate.AnimatedScatter(len(points),points.y,points.x,xfig,yfig)
   
   #dfmerged = pd.merge_ordered(df,df,on="time",suffixes=("_1","_2"), fill_method="ffill")
   fig, ax = plt.subplots(figsize=(xfig,yfig))
   #ax.invert_yaxis()
   ax.scatter(points.y, points.x, c="blue", label="points", edgecolors='none')

   # circle0 = plt.Circle((antennas.iloc[0].y, antennas.iloc[0].x), d.iloc[1,0], color='r', fill=False)
   # circle1 = plt.Circle((antennas.iloc[1].y, antennas.iloc[1].x), d.iloc[1,1], color='r', fill=False)
   # circle2 = plt.Circle((antennas.iloc[2].y, antennas.iloc[2].x), d.iloc[1,2], color='r', fill=False)
   # circle3 = plt.Circle((antennas.iloc[3].y, antennas.iloc[3].x), d.iloc[1,3], color='r', fill=False)
   # #circle4 = plt.Circle((antennas.iloc[4].y, 5-antennas.iloc[4].x), d.iloc[1,4], color='r', fill=False)
   # ax.add_artist(circle0)
   # ax.add_artist(circle1)
   # ax.add_artist(circle2)
   # ax.add_artist(circle3)
   # #ax.add_artist(circle4)

   plotStanza(xfig,yfig,fig)
   plt.show()

   print("end")
