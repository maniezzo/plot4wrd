import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys     # for comand line options
import animate  # needs conda install fsspec
import preprocessor as prep
import graddesc as gd
import dataReader as dR
import trilateration as tri
import sympy as sym
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def plotStanza(xfig, yfig, fig=None):
    if fig is None:
        fig, ax = plt.subplots(figsize=(xfig, yfig))
    ax = plt.gca()
    img = plt.imread("../stanza.png")
    ax.imshow(img, zorder=0, extent=[0, xfig, 0, yfig])


def getAnchorName(num):
    names = {
        0: "A1",
        1: "A2",
        2: "A3",
        3: "A4",
        4: "A5"
    }
    return names.get(num, "nan")


def adjusteX(all_x):
    return [5-x for x in all_x]
    
def showInterpolation(list_x, list_y, num_points):
    f1 = interp1d(list_x, list_y, kind = 'linear')
    ynew = np.linspace(list_x[0], list_x[len(list_x)-1], num_points)
    plt.plot(list_x, list_y, 'o', ynew, f1(ynew), 'w-')
    
def interpolationGC():
    range_p0 = range(43)        #0-42
    range_p1 = range(52,91)     #52-90
    range_p3 = range(111,151)   #111-150
    range_p5 = range(190,238)   #190-237

    
def interpolationEG():
    range_p0 = range(33)    #0-32
    range_p1 = range(53,83) #50-92 oppure 53-92
    range_p2 = range(93,141)
    
  
#Transform all distances in alfa
def getMeasurementsWeights():
    global d, xfig #xfig = Max distance
    alfas = pd.DataFrame()
    cols  = ['A1','A2','A3','A4','A5']
    for icol in range(len(cols)):
        alfas[cols[icol]] = (1-(d[cols[icol]]/xfig))*0.0001
    return alfas

# Media degli alfa per antenna all'interno del range temporale
def getAlfasInRange(rangeT):
    global alfas
    rangeAlfas = alfas[rangeT[0]:rangeT[len(rangeT)-1]]
    return rangeAlfas.mean(axis = 0, skipna = True) 

# Calcolo delle singole sommatorie più interne del calcolo del gradiente
def getXYGradient(sj, range_t):
    global d, antennas
    d_new = d[range_t[0]:range_t[len(range_t)-1]]
    cols  = ['A1','A2','A3','A4','A5']
    resX = pd.DataFrame()
    resY = pd.DataFrame()
    res = pd.DataFrame(index=cols, columns=['x','y'])
    for icol in range(len(cols)):
        resX[cols[icol]] = -4 * (antennas.iloc[icol].x - sj['x']) * ((antennas.iloc[icol].x - sj['x'])**2 + (antennas.iloc[icol].y - sj['y'])**2 - d_new[cols[icol]]**2)
        resY[cols[icol]] = -4 * (antennas.iloc[icol].x - sj['y']) * ((antennas.iloc[icol].x - sj['x'])**2 + (antennas.iloc[icol].y - sj['y'])**2 - d_new[cols[icol]]**2)
    
    res.x = resX.sum().values
    res.y = resY.sum().values
    return res.T, resX, resY#.sum() #Sommo tutto (primo sum per colonna, secondo tutte le colonne tra loro)

# Calcolo del gradiente
def getGradient(sj, range_t):
    alfas = getAlfasInRange(range_t)
    x = 0
    y = 0
    gradientXY = getXYGradient(sj, range_t)[0]
    for i in alfas.index:
        x = x + alfas[i] * gradientXY[i].x
        y = y + alfas[i] * gradientXY[i].y
    
    return {'x': x, 'y': y}    

# Calcolo dell'errore di sj
def getError(sj, range_t):
    global d
    alfas = getAlfasInRange(range_t)
    d_new = d[range_t[0]:range_t[len(range_t)-1]]
    cols  = ['A1','A2','A3','A4','A5']
    res = pd.DataFrame()
    for icol in range(len(cols)):
        res[cols[icol]] = ((antennas.iloc[icol].x - sj['x'])**2 + (antennas.iloc[icol].y - sj['y'])**2 - d_new[cols[icol]]**2)**2
    errors = res.sum().values
    error = 0
    for i in range(len(alfas)):
        error = error + alfas[i] * errors[i]
    return error

# Implementazione dello steepest descent
def steepestDescent(sj, beta, range_t):
    Max_k1 = 1000
    Max_k2 = 500
    gamma1 = 0.9
    gamma2 = 1.1
    sj_new = {'x': 0, 'y': 0} 
    
    step3 = True
    step5 = True
    
    #Step2
    k1 = 0
    
    while step3:
        step5 = True
        #Step3
        gradient = getGradient(sj, range_t)
        k2 = 0
    
        #Step4
        #(abs(gradient['x']) < 0.021843564706984107 and abs(gradient['y'] < 0.1733691628329498)) or 
        if (k1 > Max_k1): #da usare anche la soglia quando si avranno gradienti sensati
            return sj, beta
        else:
            while step5:
                #Step5
                sj_new['x'] = sj['x'] - beta*gradient['x']
                sj_new['y'] = sj['y'] - beta*gradient['y']
                k2 = k2 + 1
                #Step6
                sj_error = getError(sj, range_t)
                sj_new_error = getError(sj_new, range_t)
                if (sj_new_error < sj_error): 
                    sj = sj_new
                    print(sj)
                    beta = beta * gamma2
                    k1 = k1 + 1
                    #Torna a step3
                    step5 = False
                else:
                    if k2 < Max_k2:
                        beta = beta * gamma1
                        #Torna a step 5
                    else:
                        #step5 = False
                        return sj, beta
                    
    return sj, beta
    
# Program entry point
if __name__ == "__main__":
    print("init")
    # change working directory to script path
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    source_path = ""
    if len(sys.argv) > 1:
        source_path = sys.argv[1]
        dbtable = sys.argv[2]
    else:
        source_path = '..//'
        dbtable = 'fusion_EG_1'

    anchorcoord = {
        "bleA2": (4.97, 4.87, 2.9),
        "bleA3": (0, 5.56, 2.97),
        "bleA4": (4.97, 11.74, 2.97),
        "bleA5": (0, 12.57, 2.96),
        "bleA6": (2.51, 16.7, 2.71)
    }
    anchornames = list(anchorcoord.keys())

    fixedpcoords = {
        "Pos0": (2.50, 4.20, 1, 15),
        "Pos1": (2.50, 6.66, 1, 15),
        "Pos2": (3.95, 6.66, 1, 15),
        "Pos3": (2.50, 9.06, 1, 15),
        "Pos4": (1.00, 9.06, 1, 15),
        "Pos5": (2.50, 13.00, 1, 15),
        "Pos6": (1.00, 13.00, 1, 15)
    }
    fixedpnames = list(fixedpcoords.keys())

    d0 = dR.fixedpdist(anchornames, fixedpnames, anchorcoord, fixedpcoords)

    df = pd.DataFrame()
    dR.parse_table_data(dbtable)
    # df.to_csv("df.csv")
    d = dR.getDistances(d0)
    # d['roundTime'] = d.index.floor('1s')
    # meanD = d.groupby('roundTime').mean()

    # #dfmerged = pd.merge_ordered(df,df,on="time",suffixes=("_1","_2"), fill_method="ffill")
    # fig = plt.figure(figsize=(6, 4))
    # ax = plt.gca()
    # meanD.plot(kind='line', ax=ax)
    # plt.show()
    # # fff=meanD['A1'].interpolate()
    # # fff.plot(kind='line', ax=ax)
    # # plt.show()

    # trilateration
    T = tri.Trilat()
    i = 0
    #p = np.zeros(2)
    points = pd.DataFrame(columns=['x', 'y'])
    antennas = pd.DataFrame(columns=['x', 'y'])
    for i in range(len(anchornames)):
        antennas = antennas.append({"x": anchorcoord[anchornames[i]][0],
                                    "y": anchorcoord[anchornames[i]][1]},
                                    ignore_index=True)
    for i in range(len(d)):
        p = T.getPointCoord(d.iloc[i], antennas)
        if p is not None:
            points = points.append({"x": p[0], "y": p[1]}, ignore_index=True)
        else:
            # tanto per metterci qualcosa
            points = points.append({"x": 0, "y": 0}, ignore_index=True)

    numpoints = len(df)

    xfig = 16.5       # room length
    yfig = 5          # room width
    fig, ax = plt.subplots(figsize=(xfig, yfig))

    range_p0 = range(0,6)
    range_p1 = range(6,21)
    range_p2 = range(21,31)
    alfas = getMeasurementsWeights()
    # alfas_in_range = getAlfasInRange(range(0,33))
    sj = {"x": 3,"y": 2}
    
    #Decommentare sotto se si vuole vedere lo steepest in azione. NB con gradienti erronei non funziona
    # ax.scatter(sj['y'], sj['x'], c="red", label="points", edgecolors='none')
    # sj, beta = steepestDescent(sj, 1.1, range_p0)
    # ax.scatter(sj['y'], sj['x'], c="white", label="points", edgecolors='none')
    # sj, beta = steepestDescent(sj, beta, range_p1)
    # ax.scatter(sj['y'], sj['x'], c="white", label="points", edgecolors='none')
    # sj, beta = steepestDescent(sj, beta, range_p2)
    # ax.scatter(sj['y'], sj['x'], c="blue", label="points", edgecolors='none')
    
    #Risultati dei vari alfa e dei gradienti e dell'errore nel range 0-33
    mediaAlfas = getAlfasInRange(range(0,33))
    
    singleGradient, singleGradientX, singleGradientY = getXYGradient(sj, range(0,33))
    gradient = getGradient(sj, range(0,33))
    error = getError(sj, range(0,33))

    plotStanza(xfig, yfig, fig)
    plt.show()

    print("end")
