import pandas as pd
import numpy as np

def readDistConv(filepath):
	filename="case01signals.csv"
	thetas = {}
	if(filename=="case01signals.csv"):
		dfd = pd.read_csv(filepath+"case01signals.csv")
		m = np.array([[0,-14.8,-10],
		[-22.545455, 0,-6.428571],
		[-10.954545,-8.24359,0]])
			
		q = np.array([[0,-674.6,-413.372549],
		 [-1102.363636, 0,-180.392857],
		 [-450.272727,-303.474359,0]])
		return dfd,m,q
	dfd = pd.read_csv(filepath+"db2meters.csv")
	#thetas = a = [[0 for x in range(2)] for y in range(dfd.shape[0])]
	for i in range(0,dfd.shape[0]):
		x1,y1 = 1,dfd.iloc[i,3]
		x2,y2 = 4,dfd.iloc[i,6]
		m = (float(y2-y1))/(x2-x1)
		b = (y2 - (m * x2))
		thetas[dfd.iloc[i].mac] = [m,b]

	return dfd,thetas

def readGrayRecords(filepath,longdist,m,q):
	df = pd.read_csv(filepath)
	df = df[df.emitter != 'ERROR'] # elimina alcuni errori di lettura
	df = df[df.type == 'GREY']     # tiene solo le distanze fra beacon
	df.drop_duplicates(subset=None, keep="first", inplace=True) # i duplicati non servono
	receivers = df.receiver.unique()
	emitters  = df.emitter.unique()
	dist = np.zeros(shape=(len(receivers),len(emitters))) # init matrice distanze
	for i in range(0,len(receivers)):
		for j in range(0,len(emitters)):
			if receivers[i]==emitters[j]:
				continue
			dftemp = df.loc[(df.emitter == emitters[j]) & (df.receiver == receivers[i])]
			mean = dftemp["distance dBm"].mean()
			print("receiver {0} emitter {1} mean {2}".format(receivers[i],emitters[j],mean))
			#irec = list(thetas.keys()).index(receivers[i])
			#jemt = list(thetas.keys()).index(emitters[j])
			dist[i,j] = mean
	db2meters(dist,emitters,receivers,m,q,longdist)
	return df,dist

def readBlueRecords(filepath,longdist,thetas):
	df = pd.read_csv(filepath)
	df = df[df.emitter != 'ERROR'] # elimina alcuni errori di lettura
	df = df[df.type == 'BLUE']     # tiene solo i distanze da ancore
	df.drop_duplicates(subset=None, keep="first", inplace=True) # i duplicati non servono
	receivers = df.receiver.unique()
	emitters  = df.emitter.unique()
	dist = np.zeros(shape=(len(receivers),len(emitters))) # init matrice distanze
	for i in range(0,len(receivers)):
		for j in range(0,len(emitters)):
			if receivers[i]==emitters[j]:
				continue
			dftemp = df.loc[(df.emitter == emitters[j]) & (df.receiver == receivers[i])]
			mean = dftemp["distance dBm"].mean()
			print("receiver {0} emitter {1} mean {2}".format(receivers[i],emitters[j],mean))
			#irec = list(thetas.keys()).index(receivers[i])
			#jemt = list(thetas.keys()).index(emitters[j])
			dist[i,j] = -mean
	db2meters(dist,emitters,receivers,thetas,longdist)
	return df,dist

def db2meters(dist,emitters,receivers,m,q,longdist):
	n = len(receivers)
	for ii in range(0,n):
		for jj in range(0,len(emitters)):
			#m = thetas[receivers[i]][0]
			#q = thetas[receivers[i]][1]
			i = receivers[ii]
			j = emitters[jj]
			if dist[i,j] == 0: continue
			if np.isnan(dist[i,j]): dist[i,j] = longdist
			else:
				#dist[i,j] = (dist[i,j]-q[i,j])/m[i,j]
				dist[i,j] = m[i,j]*dist[i,j]+q[i,j]
	return dist

def read_data():
	df = pd.read_csv("distances.csv")
	x = df.x.values
	y = df.y.values

	# Target distance matrix
	fread = True
	if(not(fread)):
		orig_dist = np.array([[0.00, 49.27, 35.04, 76.27, 92.51, 17.30, 44.86, 47.72, 79.46, 67.08],
						   [54.50, 0.00, 33.91, 16.48, 38.86, 38.22, 20.79, 13.45, 80.07, 25.43],
						   [31.05, 30.13, 0.00, 58.02, 58.65, 10.92, 22.60, 24.72, 69.20, 57.73],
						   [73.37, 20.39, 52.90, 0.00, 64.15, 59.11, 43.34, 39.60, 96.44, 16.14],
						   [87.64, 53.42, 59.12, 58.47, 0.00, 69.18, 35.18, 44.50, 51.01, 79.21],
						   [13.60, 44.85, 15.44, 57.82, 67.67, 0.00, 32.13, 27.88, 73.64, 51.29],
						   [47.09, 23.06, 26.59, 42.23, 42.57, 36.59, 0.00, 10.62, 53.46, 55.67],
						   [47.85, 20.36, 16.27, 45.36, 40.82, 36.13, 17.25, 0.00, 65.07, 45.55],
						   [87.56, 82.62, 61.13, 96.60, 57.68, 75.59, 58.37, 66.84, 0.00, 111.29],
						   [67.87, 28.91, 53.75, 10.64, 65.76, 58.46, 58.42, 44.88, 110.38, 0.00]])

		# Initial positions
		x_prime = np.array([78.00925, 24.56355, 67.51808, 0.641486,42.90634,69.58149,51.03748,39.73699,93.42763,13.02554])
		y_prime = np.array([4.333213,37.67453,36.18686,36.62602,93.51352,10.83147,50.7646,33.99816,75.83722,13.10436])
	else:
		x_prime = df.xprime.values
		y_prime = df.yprime.values
		orig_dist = df.iloc[:,5:15].to_numpy()
	return x,y,x_prime,y_prime,orig_dist
