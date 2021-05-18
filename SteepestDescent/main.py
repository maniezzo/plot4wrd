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
import itertools


def getMeasureError(device, antenna, d):
    return ((antenna.x - device.x)**2 + (antenna.y - device.y)**2 - d**2)

def getAllMeasureErrors(devices, antenna, d):
    errors = []
    for i in range(len(devices)):
        errors = np.append(errors, getMeasureError(devices.iloc[i], antenna, d))
    return errors

def getMeasureMinError(devices, antenna, d):
    errors = []
    for i in range(len(devices)):
        errors = np.append(errors, getMeasureError(devices.iloc[i], antenna, d))
    return np.min(errors)

def getMinError(devicesPos, antenna, d):
    best = devicesPos.iloc[0]
    min_err = getMeasureError(devicesPos.iloc[0], antenna, d)
    for i in range(len(devicesPos)):
        device_pos = devicesPos.iloc[i]
        new_err = getMeasureError(device_pos, antenna, d)
        if 0 <= new_err < min_err:
            min_err = new_err
            best = device_pos
    # return minErr
    return best, min_err

def getOverallError(measure_range):
    global antennas, d
    single_antenna_measure_errors = []
    for i in range(len(antennas)):
        print(antennas.iloc[i].x)
        all_measure_errors = []
        for j in measure_range:
            all_pos_xy = getPosXY(antennas.iloc[i], d.iloc[j].iloc[i])
            #all_measure_errors = np.append(all_measure_errors, getMeasureMinError(all_pos_xy, antennas.iloc[i], d.iloc[j].iloc[i]))
            all_measure_errors = np.append(all_measure_errors, getAllMeasureErrors(all_pos_xy, antennas.iloc[i], d.iloc[j].iloc[i]))
        single_antenna_measure_errors = np.append(
            single_antenna_measure_errors,
            np.sum(np.square(all_measure_errors))
            )
    return single_antenna_measure_errors


def getXYbyDistancesMean(measure_range):
    global antennas, d
    somma = 0
    cont = 0
    for i in range(len(antennas)):
        for j in measure_range:
            somma = somma + d.iloc[j].iloc[i]
            cont = cont + 1
    media = somma/cont  
    return getPosXY(antennas.iloc[0], media)
    

def getPosXY(antenna, d):
    x, y = sym.symbols('x,y', real=True)
    eq = sym.Eq((antenna.x - x)**2 + (antenna.y - y)**2, d**2)
    sol = sym.solve((eq), x)
    y_range = np.arange(4, 14)
    res = pd.DataFrame(columns=['x', 'y'])
    all_x = []
    all_y = []
    for i in range(len(sol)):
        fy = sym.lambdify(y, sol[i], "numpy")
        arrx = fy(y_range)
        all_x = np.append(all_x, arrx)
        all_y = np.append(all_y, y_range)

    res = pd.DataFrame({'x': all_x, 'y': all_y})
    res.dropna(inplace=True)
    # drop values with x > 5
    return res[res.x < 5]


def getDeviceXY_old2(antenna, d):
   # print("init get device")
    x, y = sym.symbols('x,y', real=True)
    #eq = Eq((4.97 - x)**2 + (4.87 - y)**2, 2.86**2)

    expr = (antenna.x - x)**2 + (antenna.y - y)**2
    f = sym.lambdify([x, y], expr, "numpy")
    # f(2.11, 4.87) == 2.86**2
    # round(f(2.11, 4.87),2) == round(2.86**2,2)
    l_x = np.arange(2, 3, 0.01)
    l_y = np.arange(3.5, 5, 0.01)
    res = pd.DataFrame(columns=['x', 'y'])
    d = d**2
    minRange = d-0.01
    maxRange = d+0.01
    for i in range(len(l_x)):
        for j in range(len(l_y)):
            if minRange < f(l_x[i], l_y[j]) < maxRange:
                res = res.append({"x": l_x[i], "y": l_y[j]}, ignore_index=True)
   # print("end get device")
    return res


def getDeviceXY_old(antenna, d):
    x, y = sym.symbols('x,y', real=True)
    eq = sym.Eq((antenna.x - x)**2 + (antenna.y - y)**2, d**2)
    sol = sym.solve((eq), (x, y))
    ra = np.arange(4, 8)
    all_y = []
    for i in range(len(sol)):
        f = sym.lambdify(y, sol[i][0], "numpy")
        all_y = list(set(all_y) | set(f(ra)[~np.isnan(f(ra))]))

    #all_y = [round(num, 5) for num in all_y]
    df = pd.DataFrame(columns=['x', 'y'])
    for i in range(len(sol)):
        for elem in all_y:
            if 0 < elem < 16:
                res = sol[i][0].evalf(subs={y: elem})
                if isinstance(res, sym.Float):
                    if 0 < res < 5:
                        df = df.append(
                            {"x": float(res), "y": elem}, ignore_index=True)
    return df


def systemTrilat(antennasaaaaaa, d):
    global antennas
    x, y = sym.symbols('x,y', real=True)
    a = sym.Eq((antennas.iloc[0].x - x)**2 +
               (antennas.iloc[0].y - y)**2, d[0]**2)
    b = sym.Eq((antennas.iloc[1].x - x)**2 +
               (antennas.iloc[1].y - y)**2, d[1]**2)
    c = sym.Eq((antennas.iloc[2].x - x)**2 +
               (antennas.iloc[2].y - y)**2, d[2]**2)
    #d = sym.Eq((antennas.iloc[3].x - x)**2 + (antennas.iloc[3].y - y)**2, d[3]**2)
    #e = sym.Eq((anchor5.x - x)**2 + (anchor5.y - y)**2, d[4]**2)
    return sym.solve((a, b, c), (x, y))  # ,manual=True)


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
    dR.parse_table_data()
    # df.to_csv("df.csv")
    d = dR.getDistances(d0)

    #dfmerged = pd.merge_ordered(df,df,on="time",suffixes=("_1","_2"), fill_method="ffill")
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    d.plot(kind='line', ax=ax)
    plt.show()

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
    # inverto x con y per fare la figura larga
    #d.plot(kind='line', color='red', use_index=True, ax=ax)
    #y = (5-dfmerged.x_1)
    #x = dfmerged.y_1
    xfig = 16.5       # room length
    yfig = 5          # room width
    #a = animate.AnimatedScatter(numpoints,x,y,xfig,yfig)

    #dfmerged = pd.merge_ordered(df,df,on="time",suffixes=("_1","_2"), fill_method="ffill")
    fig, ax = plt.subplots(figsize=(xfig, yfig))
    # ax.invert_yaxis()
    #ax.scatter(points.y, 5-points.x, c="blue", label="points", edgecolors='none')

    getOverallError(range(10))
    
    # from scipy.interpolate import interp1d
    
    # x = np.linspace(0, 10, num=11, endpoint=True)
    # y = np.cos(-x**2/9.0)
    # f = interp1d(x, y)
  
    # xnew = np.linspace(0, 10, num=41, endpoint=True)
    # import matplotlib.pyplot as plt
    # plt.plot(x, y, 'o', xnew, f(xnew), '-')
    # plt.legend(['data', 'linear'], loc='best')
    # plt.show()
  

    #ax.scatter(prova.y, prova.x, c="white", label="points", edgecolors='none')
    #ax.scatter(bestDeviceXYA1.y, bestDeviceXYA1.x, c="blue", label="points", edgecolors='none')

    plotStanza(xfig, yfig, fig)
    plt.show()

    print("end")
