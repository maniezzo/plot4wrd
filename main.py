import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from dbManager import SqLiteDB
from preprocessor import Preproc
from trilateration import Trilat
from graddesc import GradientDescent

def main():
   
   dbms = SqLiteDB(SqLiteDB.SQLITE,dbpath='..//data//july 7//', dbname='july7data.sqlite')
    # C.1_2TAG-4ANC F.1_2TAG-4ANC F.2_2TAG-4ANC C.1_2TAG-4ANC C.1_2TAG-4ANC
   dbms.TABLE = "F.1_2TAG-4ANC"
   query = "SELECT Time, D20C, CC90, \"9028\", \"198A\" FROM '{TABLENAME}'".format(TABLENAME=dbms.TABLE)
   df = dbms.get_all_data(query=query,table=dbms.TABLE)
   # Remove PK_UID column
   df.drop(df.columns[[0]], axis = 1, inplace = True) 
   print(df.head)
   
   # gca stands for 'get current axis'
   ax = plt.gca()
   df.plot(kind='line',y='D20C',ax=ax)
   df.plot(kind='line',y='CC90', color='red', ax=ax)
   df.plot(kind='line',y='9028', color='green', ax=ax)
   df.plot(kind='line',y='198A', color='black', ax=ax)
   plt.show()
   
   width = 10
   prep = Preproc(width=width)
   grad = GradientDescent()
   grad.verbose = verbose

   for column in df.columns[1:5]:
      name = df[column].name
      kname = 'k'+name
      rname = 'r'+name
      print(name)

      df[kname] = prep.kalman(df[name])
      df[rname] =  prep.rollCMA(df[kname]) # rolling centered MA
      
      # gca stands for 'get current axis'
      #ax = plt.gca()
      #df.plot(kind='line',y=name,ax=ax)
      #df.plot(kind='line',y=kname, color='green', ax=ax)
      #df.plot(kind='line',y=rname, color='red', ax=ax)
      #plt.show()
   
   # gca stands for 'get current axis', if a fig is already there
   ax = plt.gca()
   df.plot(kind='line',y='rD20C',ax=ax)
   df.plot(kind='line',y='rCC90', color='red', ax=ax)
   df.plot(kind='line',y='r9028', color='green', ax=ax)
   df.plot(kind='line',y='r198A', color='black', ax=ax)
   plt.show()
   
   columns = ["rD20C", "rCC90", "r9028", "r198A"]
   t = 0 # time
   
   xlist=[]
   ylist=[]
   
   while (t < len(df)):
      dist = df[columns].iloc[t]
      imax = np.argmax(dist)
      # removing most distant anchor
      dist3 = dist.drop(dist.index[imax])
      anchor3 = anchor.drop(anchor.index[imax])
      trlat = Trilat()
      p0 = trlat.getPointCoord(dist3,anchor3) # first position estimate
      xp=p0[0]
      yp=p0[1]
      
      if(t==0):
         fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
         # (or fr the existing figure)
         # fig = plt.gcf()
         # ax = fig.gca()
         circle0 = plt.Circle((anchor.iloc[0].x, anchor.iloc[0].y), dist[0], color='r', fill=False)
         circle1 = plt.Circle((anchor.iloc[1].x, anchor.iloc[1].y), dist[1], color='r', fill=False)
         circle2 = plt.Circle((anchor.iloc[2].x, anchor.iloc[2].y), dist[2], color='r', fill=False)
         circle3 = plt.Circle((anchor.iloc[3].x, anchor.iloc[3].y), dist[3], color='r', fill=False)
         ax.add_artist(circle0)
         ax.add_artist(circle1)
         ax.add_artist(circle2)
         ax.add_artist(circle3)
         ax.plot(anchor.x,anchor.y,'ro')
         fig.show()
      
   #while (t < len(df)):
      dist = df[columns].iloc[t]
      iter_cost = []
      niter = 50
      lr = 0.0005
      dist,xp,yp = grad.gradDescWithAntennas(anchor,xp,yp,lr,niter,dist,iter_cost)
   
      # Final results
      if(verbose >= 1):
         print('Iter {0} Final distance matrix: {1}'.format(t,dist))
         print('Iter {0} xp:{1} yp:{2}'.format(t,xp,yp))
   
      # this is plotted at each iteration
      if(verbose == 2):
         plt.scatter(anchor.x,anchor.y)
         plt.scatter(xp,yp)
         plt.legend(('Initial positions','Anchor positions','Final positions'))
         plt.title('Position history')
         plt.show()
   
         # Plot of the error function
         plt.figure()
         plt.plot(range(len(iter_cost)),iter_cost)
         plt.xlabel('Iter')
         plt.ylabel('Cost')
         plt.show()
      
      xlist.append(xp)
      ylist.append(yp)
      t+=1

   fig = plt.figure(1,figsize=(9,6))
   ax = fig.add_subplot(111)
   ax.set_title('Position trace')
   ax.set_ylabel('Y', color='red')
   ax.set_xlabel('X')
   ax.plot(anchor.x,anchor.y,'ro')
   ax.scatter(xlist, ylist, c="g", alpha=0.5, label="Final positions")
   ax.legend()
   fig.show()
   plt.show()


# Program entry point
if __name__ == "__main__":
    print("init")
    columns = ["x", "y"]
    index  = ["D20C", "CC90", "9028", "198A"]
    anchor = pd.DataFrame(index=index, columns=columns)
    anchor.loc['D20C'].x=0; anchor.loc['D20C'].y=0; 
    anchor.loc['CC90'].x=0; anchor.loc['CC90'].y=5; 
    anchor.loc['9028'].x=4; anchor.loc['9028'].y=0; 
    anchor.loc['198A'].x=4; anchor.loc['198A'].y=5; 
    verbose = 1
    main()
    print("end")
    