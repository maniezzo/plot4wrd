import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import datareader as dr
from graddesc import gradDesc,guessXYbyPCA
import trilateration as trl

if __name__ == "__main__":
	lr = 0.0005
	n_iterations = 1000
	
	filepath = "../data/june10/"
	dfd,m,q = dr.readDistConv(filepath)
	longdist = 10
	filepath += "case_01.csv"
	df,orig_dist = dr.readGrayRecords(filepath,longdist,m,q)
	
	emitters = dfd.mac[dfd.color=="GRAY"]
	receivers = dfd.mac[dfd.color=="BLUE"]
	antennas = dfd[['x','y']][dfd.color=="BLUE"]
	antennas.reset_index(drop=True, inplace=True) # reindexing from 0
	XYguess = np.empty(shape=[0, 2]) # init empty 2 col array
	for j in range(0,len(emitters)): # emitters are on columns
		pointDist = orig_dist[:,j]
		point = trl.getPointCoord(pointDist,antennas)
		XYguess = np.append(XYguess, [[point[0],point[1]]], axis=0)
	# here reads from known positions
	#x,y,x_prime,y_prime,orig_dist = dr.read_data()
	
	# XYguess = guessXYbyPCA(orig_dist)
	x_prime=XYguess[:,0]
	y_prime=XYguess[:,1]

	plt.figure(figsize=(9,6))
	plt.scatter(x_prime,y_prime)
	plt.scatter(antennas.x,antennas.y)
	plt.legend(('Detected positions','Anchor positions'))
	plt.title('Detected device positioning')
	plt.show()

	iter_cost = []
	plt.figure(figsize=(9,6))
	plt.scatter(x_prime,y_prime)
	d_prime,x_prime,y_prime = gradDesc(x_prime,y_prime,lr,n_iterations,orig_dist,iter_cost)

	# Final results
	print('Final distance matrix: \n', d_prime)
	print('Updated x_prime is:', x_prime)
	print('Updated y_prime is:', y_prime)

	plt.scatter(antennas.x,antennas.y)
	plt.scatter(x_prime,y_prime)
	plt.legend(('Initial positions','Anchor positions','Final positions'))
	plt.title('Position history')
	plt.show()

	# Plot of the error function
	plt.figure()
	plt.plot(range(len(iter_cost)),iter_cost)
	plt.xlabel('Iter')
	plt.ylabel('Cost')
	plt.show()
