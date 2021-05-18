import numpy as np, pandas as pd
import matplotlib.pyplot as plt

class GradientDescent:
   
   EPS = 0.001
   verbose = 0
   
   def __inti__(self):
      ptint("Activating optimizer")
   
   def guessXYbyPCA(self,X):
      from sklearn.decomposition import PCA
      pca = PCA(n_components=2)
      XYguess = pca.fit_transform(X)
      return XYguess
   
   # matrix of distances among beacons
   def gradDesc(self,x_prime,y_prime,lr,n_iterations,orig_dist,iter_cost):
      for iter in range(n_iterations):
         if(verbose==2):
            plt.scatter(x_prime,y_prime)   # save to see the path of the updated positions
         d_prime = np.zeros((len(x_prime),len((x_prime))))
   
         # Current distances
         for i in range(len(x_prime)):
            for j in range(i, len(x_prime)):
               d_prime[i, j] = np.sqrt(pow((x_prime[i] - x_prime[j]),2) + pow((y_prime[i] - y_prime[j]),2))
   
      # integration of the trasposed upper triangular matrix, symmetric
         d_prime = d_prime + d_prime.T
         delta_x = np.ones(len(x_prime))
         delta_y = np.ones(len(y_prime))
         cost = np.sum((orig_dist-d_prime)**2)  # SSD is cost
         
         if(cost > EPS):
            # Gradient of the cost
            for k in range(len(delta_x)):
               temp_x  = 0
               temp_x1 = 0
               temp_y  = 0
               temp_y1 = 0
               for j in range(len(delta_x)):
                  if j != k:
                     temp_x += -2 * (orig_dist[k, j] - d_prime[k, j])\
                           / d_prime[k, j] * (x_prime[k] - x_prime[j])
                     temp_y += -2 * (orig_dist[k, j] - d_prime[k, j])\
                           / d_prime[k, j]* (y_prime[k] - y_prime[j])
               for i in range(len(delta_x)):
                  if i != k:
                     temp_x1 += 2 * (orig_dist[i, k] - d_prime[i, k]) \
                           / d_prime[i, k] * (x_prime[i] - x_prime[k])
                     temp_y1 += 2 * (orig_dist[i, k] - d_prime[i,k]) \
                           / d_prime[i, k] * (y_prime[i] - y_prime[k])
               temp_x += temp_x1
               temp_y += temp_y1
      
               delta_x[k] = temp_x  # dE/dxi
               delta_y[k] = temp_y  # dE/dyi
      
            # Update the position
            x_prime = x_prime - lr*delta_x
            y_prime = y_prime - lr*delta_y
      
            # Print the cost at every epoch
            if (iter % 100 == 0) and (self.verbose >= 1):
               print('Iteration {0} cost {1}'.format(iter,cost))
               iter_cost.append(cost)
         
      return d_prime,x_prime,y_prime
   
   # works on a matrix of distances among beacons
   def gradDescWithAntennas(self,anchor,xp,yp,lr,niter,d0,iter_cost):
      d_prime = np.zeros(len(d0))               # just for dimensioning
      for iter in range(niter):
         if(self.verbose == 2):
            plt.scatter(xp,yp)   # save to see the path of the updated positions
   
         # Current distances
         for i in range(len(anchor)):
            d_prime[i] = np.sqrt(pow((xp - anchor.iloc[i].x),2) + pow((yp - anchor.iloc[i].y),2))
   
         delta_x = np.ones(len(anchor))
         delta_y = np.ones(len(anchor))
         cost = np.sum((d0-d_prime)**2)  # sum square deviations
   
         if(cost > self.EPS):
            # Cost gradient 
            temp_x  = 0
            temp_y  = 0
            for j in range(len(delta_x)):
               temp_x += -2 * (d0[j] - d_prime[j]) / d_prime[j] * xp
               temp_y += -2 * (d0[j] - d_prime[j]) / d_prime[j] * yp
      
            delta_x = temp_x  # dE/dxp
            delta_y = temp_y  # dE/dyp
      
            # Position update 
            xp = xp - lr*delta_x
            yp = yp - lr*delta_y
      
            # Printout of the cost
            if (iter % 100 == 0) and (self.verbose >= 1):
               print('Iteration {0} cost {1}'.format(iter,cost))
               iter_cost.append(cost)
      return d_prime,xp,yp
