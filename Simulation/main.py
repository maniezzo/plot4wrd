import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys     # for comand line options
import animate # needs conda install fsspec
from datetime import datetime, timedelta
import noisegenerator as ng


def plotStanza(xfig, yfig, anchor, df, fig=None):
    if fig is None:
        fig, ax = plt.subplots(figsize=(xfig, yfig))
    ax = plt.gca()
    img = plt.imread("../stanza.png")
    ax.scatter(df['y'], df['x'], c="orange", edgecolors='none')
    if not anchor.empty:
        ax.plot(anchor.y, anchor.x, 'rD', markersize=12)
        ax.annotate(anchor.name, (anchor.y, anchor.x), xytext=(anchor.y+0.25, anchor.x), fontsize=18)
    ax.imshow(img, zorder=0, extent=[0, xfig, 0, yfig])
    plt.show()
    
 
# simulation of staying still in one position for a certain period (in ms)
def stay(position, milliseconds, noise_qnt='', noise_dir='', noise_canc=(0,0)):
    global path_df, simulationtime
    period = milliseconds#/100
    # if len(path_df.time) == 0 :
    #     date_rng = pd.date_range(str(datetime.today().strftime('%d/%m/%Y')).split()[0]+' 00:00:00.000000', periods=period, freq='100ms')
    # else:
    
    date_rng = pd.date_range(simulationtime, periods=period, freq='100ms')
    # for i in range(period):
    #     simulationtime = simulationtime + timedelta(milliseconds=100)
    df = pd.DataFrame(columns=['time', 'tagID', 'x', 'y', 'z', 'quality'])
    df.time = date_rng
    df.tagID = 'tag1'
    simulationtime = df.time.iloc[len(df.time)-1] + timedelta(milliseconds=100)
    
    noise_x, noise_y = ng.generateNoise(period, noise_qnt, noise_dir)
    
    df.x = position.x + noise_x
    df.y = position.y + noise_y
    df.z = position.z
    df.quality = 0
    dfupdate = pd.DataFrame()
    if noise_canc[0] < noise_canc[1]:
        # dfupdate = df.sample(np.random.randint(round(len(df)*(noise_canc[0]*0.01)),round(len(df)*(noise_canc[1]*0.01))))
        # dfupdate = df.sample(np.random.randint((len(df)*(noise_canc[0]*0.01)),(len(df)*(noise_canc[1]*0.01))))
        dfupdate = df.sample(round(np.random.uniform((len(df)*(noise_canc[0]*0.01)),(len(df)*(noise_canc[1]*0.01)))))
        # if 0 in dfupdate.index.values:
        #     dfupdate=dfupdate.drop(0)
        #df = df.drop(dfupdate.index)
        
    elif noise_canc == (100,100):
        dfupdate = df.sample(len(df))
        #df = df.drop(dfupdate.index)
    df.loc[dfupdate.index.values, ['x','y','z','quality']] = np.nan
    path_df=path_df.append(df, ignore_index=True)

# generator of positions from start to stop posizion
def generatePositions(start, stop, period):
    positions = np.array([start] * period)

    # generate positions in range
    for i in range(1,period):
        positions[i] = positions[i-1]+(stop-start)/period
    positions = positions + (stop-start)/100
    return positions

    
# simulation of the movement from the start position to the stop position within the time range period (in ms)
def move(from_pos, to_pos, milliseconds, noise_qnt='', noise_dir='', noise_canc=(0,0)):
    global path_df, simulationtime
    period = milliseconds
    # if len(path_df.time) == 0 :
    #     date_rng = pd.date_range(str(datetime.today().strftime('%d/%m/%Y')).split()[0]+' 00:00:00.000000', periods=period, freq='100ms')
    # else:
    #     date_rng = pd.date_range(path_df.time.iloc[len(path_df.time)-1] + timedelta(milliseconds=100), periods=period, freq='100ms')
    
    date_rng = pd.date_range(simulationtime, periods=period, freq='100ms')
    # for i in range(period):
    #     simulationtime = simulationtime + timedelta(milliseconds=100)
    
    df = pd.DataFrame(columns=['time', 'tagID', 'x', 'y', 'z', 'quality'])
      
    df.time = date_rng
    simulationtime = df.time.iloc[len(df.time)-1] + timedelta(milliseconds=100)
    df.tagID = 'tag1'
    df.quality = 0
    
    noise_x, noise_y = ng.generateNoise(period, noise_qnt, noise_dir)
    
    if from_pos.x == to_pos.x:
        df.x = from_pos.x + noise_x
        df.y = generatePositions(from_pos.y, to_pos.y, period) + noise_y
    else:
        if 'left' in noise_dir or 'right' in noise_dir:
            from_pos_noise, to_pos_noise = ng.generateObliqueDistortion(noise_dir, noise_qnt)
            df.y = generatePositions(from_pos.y + from_pos_noise, to_pos.y + to_pos_noise, period) + noise_y
        else:
            df.y = from_pos.y + noise_y
        df.x = generatePositions(from_pos.x, to_pos.x, period) + noise_x
        
    df.z = 0
    dfupdate = pd.DataFrame()
    if noise_canc[0] < noise_canc[1]:
        # dfupdate = df.sample(np.random.randint(round(len(df)*(noise_canc[0]*0.01)),round(len(df)*(noise_canc[1]*0.01))))
        # dfupdate = df.sample(np.random.randint((len(df)*(noise_canc[0]*0.01)),(len(df)*(noise_canc[1]*0.01))))
        dfupdate = df.sample(round(np.random.uniform((len(df)*(noise_canc[0]*0.01)),(len(df)*(noise_canc[1]*0.01)))))
        # if 0 in dfupdate.index.values:
        #     dfupdate=dfupdate.drop(0)
        # df = df.drop(dfupdate.index)
    elif noise_canc == (100,100):
        dfupdate = df.sample(len(df))
        # df = df.drop(dfupdate.index)
    df.loc[dfupdate.index.values, ['x','y','z','quality']] = np.nan
    path_df=path_df.append(df, ignore_index=True)
    
def moveCV(from_pos, to_pos, milliseconds, noise_qnt, noise_dir):
    global path_df
    move(from_pos, to_pos, milliseconds, noise_qnt, noise_dir)
    # dfupdate = path_df.sample(np.random.randint(round(len(path_df)*0.15),round(len(path_df)*0.25)))
    # # if 0 in dfupdate.index.values:
    # #     dfupdate.drop(1)
    # path_df = path_df.drop(dfupdate.index)
 
# creates path of simulation with no noise
def createPathNoNoise(animation):
    stay(positions.loc['Pos0'], 20)
    move(positions.loc['Pos0'], positions.loc['Pos1'], 60)
    stay(positions.loc['Pos1'], 20)
    move(positions.loc['Pos1'], positions.loc['Pos2'], 30)
    stay(positions.loc['Pos2'], 20)
    move(positions.loc['Pos2'], positions.loc['Pos1'], 30)
    move(positions.loc['Pos1'], positions.loc['Pos3'], 60)
    stay(positions.loc['Pos3'], 20)
    move(positions.loc['Pos3'], positions.loc['Pos4'], 30)
    stay(positions.loc['Pos4'], 20)
    move(positions.loc['Pos4'], positions.loc['Pos3'], 30)
    move(positions.loc['Pos3'], positions.loc['Pos5'], 120)
    stay(positions.loc['Pos5'], 20)
    move(positions.loc['Pos5'], positions.loc['Pos6'], 30)
    stay(positions.loc['Pos6'], 20)
    move(positions.loc['Pos6'], positions.loc['Pos5'], 30)
    move(positions.loc['Pos5'], positions.loc['Pos0'], 240)
    
    path_df['x'] = 5-path_df['x']
    fig, ax = plt.subplots(figsize=(xfig, yfig))
    plotStanza(xfig, yfig, pd.DataFrame(), path_df, fig)
    path_df.to_csv('UWB_NoNOISE_'+str(datetime.today().strftime('%y%m%d')).split()[0]+'.csv', encoding='utf-8')
    
    if animation:
        a = animate.AnimatedScatter(len(path_df),path_df,xfig,yfig,pd.DataFrame())

# creates path of simulation with noise: pov CB1D antenna
def createCB1DPath(animation):
    stay(positions.loc['Pos0'], 20, 'low')
    move(positions.loc['Pos0'], positions.loc['Pos1'], 60, 'low')
    stay(positions.loc['Pos1'], 20, 'low')
    move(positions.loc['Pos1'], positions.loc['Pos2'], 30, 'low2medium', 'left')
    stay(positions.loc['Pos2'], 20, 'medium', 'left')
    move(positions.loc['Pos2'], positions.loc['Pos1'], 30, 'medium2low', 'left')
    move(positions.loc['Pos1'], positions.loc['Pos3'], 60, 'medium')
    stay(positions.loc['Pos3'], 20, 'medium')
    move(positions.loc['Pos3'], positions.loc['Pos4'], 30, 'medium', 'top-left')
    stay(positions.loc['Pos4'], 20, 'medium', 'top-left')
    move(positions.loc['Pos4'], positions.loc['Pos3'], 30, 'mediumR', 'top-left')
    move(positions.loc['Pos3'], positions.loc['Pos5'], 120, 'medium')
    stay(positions.loc['Pos5'], 20, 'hight')
    move(positions.loc['Pos5'], positions.loc['Pos6'], 30, 'hight', 'top-left')
    stay(positions.loc['Pos6'], 20, 'hight', 'top-left')
    move(positions.loc['Pos6'], positions.loc['Pos5'], 30, 'hightR', 'top-left')
    move(positions.loc['Pos5'], positions.loc['Pos3'], 120, 'hight2medium')
    move(positions.loc['Pos3'], positions.loc['Pos1'], 60, 'medium')
    move(positions.loc['Pos1'], positions.loc['Pos0'], 60, 'low')
    
    pathCB1D_df = path_df
    pathCB1D_df['x'] = 5-path_df['x']
    fig, ax = plt.subplots(figsize=(xfig, yfig))
    plotStanza(xfig, yfig, anchors_df.loc['CB1D'], pathCB1D_df, fig)
    pathCB1D_df.to_csv('UWB_CB1D_'+str(datetime.today().strftime('%y%m%d')).split()[0]+'.csv', encoding='utf-8')
    
    if animation:
        a = animate.AnimatedScatter(len(pathCB1D_df),pathCB1D_df,xfig,yfig,anchors_df.loc['CB1D'])
 
# creates path of simulation with noise: pov 8418 antenna
def create8418Path(animation):
    stay(positions.loc['Pos0'], 20, 'hight')
    move(positions.loc['Pos0'], positions.loc['Pos1'], 60, 'hight')
    stay(positions.loc['Pos1'], 20, 'hight')
    move(positions.loc['Pos1'], positions.loc['Pos2'], 30, 'hight', 'right')
    stay(positions.loc['Pos2'], 20, 'hight', 'right')
    move(positions.loc['Pos2'], positions.loc['Pos1'], 30, 'hight', 'right')
    move(positions.loc['Pos1'], positions.loc['Pos3'], 60, 'hight2medium')
    stay(positions.loc['Pos3'], 20, 'medium')
    move(positions.loc['Pos3'], positions.loc['Pos4'], 30, 'medium', 'top-right')
    stay(positions.loc['Pos4'], 20, 'medium', 'top-right')
    move(positions.loc['Pos4'], positions.loc['Pos3'], 30, 'mediumR', 'top-right')
    move(positions.loc['Pos3'], positions.loc['Pos5'], 120, 'low')
    stay(positions.loc['Pos5'], 20, 'low')
    move(positions.loc['Pos5'], positions.loc['Pos6'], 30, 'low')
    stay(positions.loc['Pos6'], 20, 'low')
    move(positions.loc['Pos6'], positions.loc['Pos5'], 30, 'low')
    move(positions.loc['Pos5'], positions.loc['Pos3'], 120, 'low')
    move(positions.loc['Pos3'], positions.loc['Pos1'], 60, 'medium2hight')
    move(positions.loc['Pos1'], positions.loc['Pos0'], 60, 'hight')
    
    path8418_df = path_df
    path8418_df['x'] = 5-path_df['x']
    fig, ax = plt.subplots(figsize=(xfig, yfig))
    plotStanza(xfig, yfig, anchors_df.loc['8418'], path8418_df, fig)
    path8418_df.to_csv('UWB_8418_'+str(datetime.today().strftime('%y%m%d')).split()[0]+'.csv', encoding='utf-8')
    
    if animation:
        a = animate.AnimatedScatter(len(path8418_df),path8418_df,xfig,yfig,anchors_df.loc['8418'])

# creates path of simulation with noise: pov D20C antenna    
def createD20CPath(animation):
    stay(positions.loc['Pos0'], 20, 'low', 'top')
    move(positions.loc['Pos0'], positions.loc['Pos1'], 60, 'low', 'top')
    stay(positions.loc['Pos1'], 20, 'low')
    move(positions.loc['Pos1'], positions.loc['Pos2'], 30, 'low')
    stay(positions.loc['Pos2'], 20, 'low')
    move(positions.loc['Pos2'], positions.loc['Pos1'], 30, 'low')
    move(positions.loc['Pos1'], positions.loc['Pos3'], 60, 'low2medium', 'bottom')
    stay(positions.loc['Pos3'], 20, 'medium', 'bottom-left')
    move(positions.loc['Pos3'], positions.loc['Pos4'], 30, 'medium', 'bottom')
    stay(positions.loc['Pos4'], 20, 'medium', 'bottom')
    move(positions.loc['Pos4'], positions.loc['Pos3'], 30, 'mediumR', 'bottom')
    move(positions.loc['Pos3'], positions.loc['Pos5'], 120, 'medium2hight', 'bottom')
    stay(positions.loc['Pos5'], 20, 'hight', 'bottom')
    move(positions.loc['Pos5'], positions.loc['Pos6'], 30, 'hight', 'bottom-right')
    stay(positions.loc['Pos6'], 20, 'hight', 'bottom-right')
    move(positions.loc['Pos6'], positions.loc['Pos5'], 30, 'hight', 'bottom-right')
    move(positions.loc['Pos5'], positions.loc['Pos3'], 120, 'hight2medium', 'bottom-right')
    move(positions.loc['Pos3'], positions.loc['Pos1'], 60, 'medium2low', 'bottom')
    move(positions.loc['Pos1'], positions.loc['Pos0'], 60, 'low', 'top')
    
    pathD20C_df = path_df
    pathD20C_df['x'] = 5-path_df['x']
    fig, ax = plt.subplots(figsize=(xfig, yfig))
    plotStanza(xfig, yfig, anchors_df.loc['D20C'], pathD20C_df, fig)
    pathD20C_df.to_csv('UWB_D20C_'+str(datetime.today().strftime('%y%m%d')).split()[0]+'.csv', encoding='utf-8')
    
    if animation:
        a = animate.AnimatedScatter(len(pathD20C_df),pathD20C_df,xfig,yfig,anchors_df.loc['D20C'])
 
# creates path of simulation with noise: pov 198A antenna
def create198APath(animation):
    stay(positions.loc['Pos0'], 20, 'hight')
    move(positions.loc['Pos0'], positions.loc['Pos1'], 60, 'hight')
    stay(positions.loc['Pos1'], 20, 'hight')
    move(positions.loc['Pos1'], positions.loc['Pos2'], 30, 'hight', 'right')
    stay(positions.loc['Pos2'], 20, 'hight', 'right')
    move(positions.loc['Pos2'], positions.loc['Pos1'], 30, 'hight', 'right')
    move(positions.loc['Pos1'], positions.loc['Pos3'], 60, 'hight2medium')
    stay(positions.loc['Pos3'], 20, 'medium')
    move(positions.loc['Pos3'], positions.loc['Pos4'], 30, 'medium2low')
    stay(positions.loc['Pos4'], 20, 'low', 'right')
    move(positions.loc['Pos4'], positions.loc['Pos3'], 30, 'low2medium', 'right')
    move(positions.loc['Pos3'], positions.loc['Pos5'], 120, 'medium2low')
    stay(positions.loc['Pos5'], 20, 'low')
    move(positions.loc['Pos5'], positions.loc['Pos6'], 30, 'low')
    stay(positions.loc['Pos6'], 20, 'low')
    move(positions.loc['Pos6'], positions.loc['Pos5'], 30, 'low')
    move(positions.loc['Pos5'], positions.loc['Pos3'], 120, 'low2medium')
    move(positions.loc['Pos3'], positions.loc['Pos1'], 60, 'medium2hight')
    move(positions.loc['Pos1'], positions.loc['Pos0'], 60, 'hight')
    
    path198A_df = path_df
    path198A_df['x'] = 5-path_df['x']
    fig, ax = plt.subplots(figsize=(xfig, yfig))
    plotStanza(xfig, yfig, anchors_df.loc['198A'], path198A_df, fig)
    path198A_df.to_csv('UWB_198A_'+str(datetime.today().strftime('%y%m%d')).split()[0]+'.csv', encoding='utf-8')
    
    if animation:
        a = animate.AnimatedScatter(len(path198A_df),path198A_df,xfig,yfig,anchors_df.loc['198A'])

def create9028Path(animation):
    stay(positions.loc['Pos0'], 20, 'low')
    move(positions.loc['Pos0'], positions.loc['Pos1'], 60, 'low')
    stay(positions.loc['Pos1'], 20, 'low')
    move(positions.loc['Pos1'], positions.loc['Pos2'], 30, 'low2medium')
    stay(positions.loc['Pos2'], 20, 'medium')
    move(positions.loc['Pos2'], positions.loc['Pos1'], 30, 'medium2low')
    move(positions.loc['Pos1'], positions.loc['Pos3'], 60, 'low2medium')
    stay(positions.loc['Pos3'], 20, 'medium')
    move(positions.loc['Pos3'], positions.loc['Pos4'], 30, 'medium')
    stay(positions.loc['Pos4'], 20, 'medium')
    move(positions.loc['Pos4'], positions.loc['Pos3'], 30, 'mediumR')
    move(positions.loc['Pos3'], positions.loc['Pos5'], 120, 'medium2hight')
    stay(positions.loc['Pos5'], 20, 'hight')
    move(positions.loc['Pos5'], positions.loc['Pos6'], 30, 'hight', 'top-left')
    stay(positions.loc['Pos6'], 20, 'hight', 'top-left')
    move(positions.loc['Pos6'], positions.loc['Pos5'], 30, 'hightR', 'top-left')
    move(positions.loc['Pos5'], positions.loc['Pos3'], 120, 'hight2medium')
    move(positions.loc['Pos3'], positions.loc['Pos1'], 60, 'medium2low')
    move(positions.loc['Pos1'], positions.loc['Pos0'], 60, 'low')
    
    path9028_df = path_df
    path9028_df['x'] = 5-path_df['x']
    fig, ax = plt.subplots(figsize=(xfig, yfig))
    plotStanza(xfig, yfig, anchors_df.loc['9028'], path9028_df, fig)
    path9028_df.to_csv('UWB_9028_'+str(datetime.today().strftime('%y%m%d')).split()[0]+'.csv', encoding='utf-8')
    
    if animation:
        a = animate.AnimatedScatter(len(path9028_df),path9028_df,xfig,yfig,anchors_df.loc['9028'])
 
def createCC90Path(animation):
    stay(positions.loc['Pos0'], 20, 'hight')
    move(positions.loc['Pos0'], positions.loc['Pos1'], 60, 'hight')
    stay(positions.loc['Pos1'], 20, 'hight')
    move(positions.loc['Pos1'], positions.loc['Pos2'], 30, 'hight2medium')
    stay(positions.loc['Pos2'], 20, 'medium')
    move(positions.loc['Pos2'], positions.loc['Pos1'], 30, 'medium2hight')
    move(positions.loc['Pos1'], positions.loc['Pos3'], 60, 'hight2medium')
    stay(positions.loc['Pos3'], 20, 'low')
    move(positions.loc['Pos3'], positions.loc['Pos4'], 30, 'low2medium')
    stay(positions.loc['Pos4'], 20, 'medium')
    move(positions.loc['Pos4'], positions.loc['Pos3'], 30, 'medium2low')
    move(positions.loc['Pos3'], positions.loc['Pos5'], 120, 'low')
    stay(positions.loc['Pos5'], 20, 'low')
    move(positions.loc['Pos5'], positions.loc['Pos6'], 30, 'low2medium')
    stay(positions.loc['Pos6'], 20, 'medium')
    move(positions.loc['Pos6'], positions.loc['Pos5'], 30, 'medium2low')
    move(positions.loc['Pos5'], positions.loc['Pos3'], 120, 'low')
    move(positions.loc['Pos3'], positions.loc['Pos1'], 60, 'medium2hight')
    move(positions.loc['Pos1'], positions.loc['Pos0'], 60, 'hight')
    
    pathCC90_df = path_df
    pathCC90_df['x'] = 5-path_df['x']
    fig, ax = plt.subplots(figsize=(xfig, yfig))
    plotStanza(xfig, yfig, anchors_df.loc['CC90'], pathCC90_df, fig)
    pathCC90_df.to_csv('UWB_CC90_'+str(datetime.today().strftime('%y%m%d')).split()[0]+'.csv', encoding='utf-8')
    
    if animation:
        a = animate.AnimatedScatter(len(pathCC90_df),pathCC90_df,xfig,yfig,anchors_df.loc['CC90'])

def getMiddlePath(p0, p1, divider=(1/2)):
    x = min(p0.x, p1.x) + (max(p0.x, p1.x) - min(p0.x, p1.x))*(1-divider)
    data = {'x': x, 'y': p1.y, 'z': p1.z}
    return pd.Series(data=data, index=['x', 'y', 'z'])
        
def createCV1Path(animation):
    stay(positions.loc['Pos0'], 20, 'medium')
    move(positions.loc['Pos0'], positions.loc['Pos1'], 60, 'hight2medium',noise_canc=(55,75))
    stay(positions.loc['Pos1'], 20, 'medium')
    move(positions.loc['Pos1'], positions.loc['Pos2'], 30, 'hight',noise_canc=(90,95))
    stay(positions.loc['Pos2'], 20, 'hight')
    move(positions.loc['Pos2'], positions.loc['Pos1'], 30, 'hight',noise_canc=(90,95))
    move(positions.loc['Pos1'], positions.loc['Pos3'], 60, 'low',noise_canc=(55,65))
    stay(positions.loc['Pos3'], 20, 'low')
    
    # x = min(positions.loc['Pos4'].x, positions.loc['Pos3'].x) + (max(positions.loc['Pos4'].x, positions.loc['Pos3'].x) - min(positions.loc['Pos4'].x, positions.loc['Pos3'].x))/2
    # data = {'x': x, 'y': positions.loc['Pos3'].y, 'z': positions.loc['Pos3'].z}
    # pos = pd.Series(data=data, index=['x', 'y', 'z'])
    pos = getMiddlePath(positions.loc['Pos4'],positions.loc['Pos3'])
    
    
    move(positions.loc['Pos3'], pos, 15, 'medium',noise_canc=(0,0))
    move(pos, positions.loc['Pos4'], 15, 'low','left',noise_canc=(0,0))
    stay(positions.loc['Pos4'], 20, 'medium','')
    move(positions.loc['Pos4'], pos, 15, 'medium',noise_canc=(0,0))
    move(pos, positions.loc['Pos3'], 15, 'medium',noise_canc=(0,0))
    move(positions.loc['Pos3'], positions.loc['Pos5'], 120, 'low',noise_canc=(0,0))
    stay(positions.loc['Pos5'], 20, 'low')
    move(positions.loc['Pos5'], positions.loc['Pos6'], 30, 'low',noise_canc=(0,0))
    stay(positions.loc['Pos6'], 20, 'low')
    move(positions.loc['Pos6'], positions.loc['Pos5'], 30, 'low',noise_canc=(0,0))
    move(positions.loc['Pos5'], positions.loc['Pos3'], 120, 'low',noise_canc=(0,0))
    move(positions.loc['Pos3'], positions.loc['Pos1'], 60, 'low',noise_canc=(55,65))
    move(positions.loc['Pos1'], positions.loc['Pos0'], 60, 'medium',noise_canc=(55,75))
    
    pathCV1_df = path_df
    pathCV1_df['x'] = 5-path_df['x']
    # dfupdate = path_df.sample(np.random.randint(round(len(pathCV1_df)*0.65),round(len(pathCV1_df)*0.75)))
    # if 0 in dfupdate.index.values:
    #     dfupdate=dfupdate.drop(0)
    # pathCV1_df = pathCV1_df.drop(dfupdate.index)
    fig, ax = plt.subplots(figsize=(xfig, yfig))
    plotStanza(xfig, yfig, cameras_df.loc['CV1'], pathCV1_df, fig)
    pathCV1_df.to_csv('CV_CV1_'+str(datetime.today().strftime('%y%m%d')).split()[0]+'.csv', encoding='utf-8')
    
    if animation:
        a = animate.AnimatedScatter(len(pathCV1_df),pathCV1_df,xfig,yfig,cameras_df.loc['CV1'])
        
def createCV2Path(animation):
    stay(positions.loc['Pos0'], 20, 'medium',noise_canc=(60,65))
    move(positions.loc['Pos0'], positions.loc['Pos1'], 60, 'medium2low',noise_canc=(40,65))
    stay(positions.loc['Pos1'], 20, 'low',noise_canc=(40,65))
    
    
    move(positions.loc['Pos1'], positions.loc['Pos2'], 30, 'low2medium',noise_canc=(50,60))
    stay(positions.loc['Pos2'], 20, 'medium',noise_canc=(50,70))
    move(positions.loc['Pos2'], positions.loc['Pos1'], 30, 'medium2low',noise_canc=(60,65))
    move(positions.loc['Pos1'], positions.loc['Pos3'], 60, 'low2medium',noise_canc=(50,60))
    stay(positions.loc['Pos3'], 20, 'medium',noise_canc=(80,90))
  
    pos = getMiddlePath(positions.loc['Pos4'],positions.loc['Pos3'],1/3)
    move(positions.loc['Pos3'], pos, 10, 'low', 'left',noise_canc=(40,55))
    move(pos, positions.loc['Pos4'], 20, 'low','left',noise_canc=(90,95))
    stay(positions.loc['Pos4'], 20, 'medium','top-left',noise_canc=(85,90))
    move(positions.loc['Pos4'], pos, 20, 'low',noise_canc=(90,95))
    move(pos, positions.loc['Pos3'], 10, 'low', 'left',noise_canc=(30,45))
    
    # move(positions.loc['Pos3'], positions.loc['Pos4'], 30, 'medium',noise_canc=(100,100))
    # stay(positions.loc['Pos4'], 20, 'medium','',noise_canc=(100,100))
    # move(positions.loc['Pos4'], positions.loc['Pos3'], 30, 'medium',noise_canc=(100,100))
    
    
    move(positions.loc['Pos3'], positions.loc['Pos5'], 120, 'medium','left',noise_canc=(70,85))
    stay(positions.loc['Pos5'], 20, 'low','left',noise_canc=(60,75))
    
    pos = getMiddlePath(positions.loc['Pos5'],positions.loc['Pos6'],1/3)
    move(positions.loc['Pos5'], pos, 10, 'medium', 'top-left')
    move(pos, positions.loc['Pos6'], 20, 'medium', 'top-right',noise_canc=(70,80))
    stay(positions.loc['Pos6'], 20, 'medium','top',noise_canc=(80,85))
    move(positions.loc['Pos6'], pos, 20, 'medium', 'top-right',noise_canc=(70,85))
    move(pos, positions.loc['Pos5'], 10, 'medium', 'top-left', noise_canc=(60,70))
    move(positions.loc['Pos5'], positions.loc['Pos3'], 120, 'medium',noise_canc=(80,85))
    move(positions.loc['Pos3'], positions.loc['Pos1'], 60, 'medium2low',noise_canc=(80,85))
    move(positions.loc['Pos1'], positions.loc['Pos0'], 60, 'low2medium',noise_canc=(55,75))
    
    pathCV2_df = path_df
    pathCV2_df['x'] = 5-path_df['x']
    fig, ax = plt.subplots(figsize=(xfig, yfig))
    plotStanza(xfig, yfig, cameras_df.loc['CV1'], pathCV2_df, fig)
    pathCV2_df.to_csv('CV_CV2_'+str(datetime.today().strftime('%y%m%d')).split()[0]+'.csv', encoding='utf-8')
    
    if animation:
        a = animate.AnimatedScatter(len(pathCV2_df),pathCV2_df,xfig,yfig,cameras_df.loc['CV1'])
        
def main():  
    global anchors_df, cameras_df, positions, path_df, pathCB1D_df, path8418_df, pathD20C_df, path9028_df, pathCC90_df, pathCV1_df, xfig, yfig, simulationtime
    
    xfig = 16.9      # room length
    yfig = 5          # room width
    
    
    anchors_df = pd.DataFrame(np.array([[0, 12.1], [0, 4.6], [2.5, 0], [4.95, 4.6], [4.95, 12.1], [2.5, 16.9]]),
                              index=['CC90','D20C','CB1D','9028','198A','8418'],
                              columns=['x', 'y'])
    
    cameras_df = pd.DataFrame(np.array([[2.35, 11.5], [2.35, 6.38]]),
                              index=['CV0','CV1'],
                              columns=['x', 'y'])
    
    positions = pd.DataFrame(np.array([[2.5, 4.2, 0], [2.50, 6.66, 0], [3.95, 6.66, 0], [2.50, 9.06, 0], [1.00, 9.06, 0], [2.50, 13.00, 0], [1.00, 13.00, 0]]),
                             columns=['x', 'y', 'z'],
                             index=['Pos0','Pos1','Pos2','Pos3','Pos4','Pos5','Pos6'])
    
    # simulationtime = datetime.strptime(str((datetime.today().strftime('%d/%m/%Y')).split()[0]+' 00:00:00.000000'),"%d/%m/%Y %H:%M:%S.%f")
    # path_df = pd.DataFrame(columns=['time', 'tagID', 'x', 'y', 'z', 'quality'])
    # createPathNoNoise(False)
    # simulationtime = datetime.strptime(str((datetime.today().strftime('%d/%m/%Y')).split()[0]+' 00:00:00.000000'),"%d/%m/%Y %H:%M:%S.%f")
    # path_df = pd.DataFrame(columns=['time', 'tagID', 'x', 'y', 'z', 'quality'])
    # createCB1DPath(False)
    # simulationtime = datetime.strptime(str((datetime.today().strftime('%d/%m/%Y')).split()[0]+' 00:00:00.000000'),"%d/%m/%Y %H:%M:%S.%f")
    # path_df = pd.DataFrame(columns=['time', 'tagID', 'x', 'y', 'z', 'quality'])
    # create8418Path(False)
    # simulationtime = datetime.strptime(str((datetime.today().strftime('%d/%m/%Y')).split()[0]+' 00:00:00.000000'),"%d/%m/%Y %H:%M:%S.%f")
    # path_df = pd.DataFrame(columns=['time', 'tagID', 'x', 'y', 'z', 'quality'])
    # createD20CPath(False)
    # simulationtime = datetime.strptime(str((datetime.today().strftime('%d/%m/%Y')).split()[0]+' 00:00:00.000000'),"%d/%m/%Y %H:%M:%S.%f")
    # path_df = pd.DataFrame(columns=['time', 'tagID', 'x', 'y', 'z', 'quality'])
    # create198APath(False)  
    # simulationtime = datetime.strptime(str((datetime.today().strftime('%d/%m/%Y')).split()[0]+' 00:00:00.000000'),"%d/%m/%Y %H:%M:%S.%f")
    # path_df = pd.DataFrame(columns=['time', 'tagID', 'x', 'y', 'z', 'quality'])
    # create9028Path(False)
    # simulationtime = datetime.strptime(str((datetime.today().strftime('%d/%m/%Y')).split()[0]+' 00:00:00.000000'),"%d/%m/%Y %H:%M:%S.%f")
    # path_df = pd.DataFrame(columns=['time', 'tagID', 'x', 'y', 'z', 'quality'])
    # createCC90Path(False)
    # simulationtime = datetime.strptime(str((datetime.today().strftime('%d/%m/%Y')).split()[0]+' 00:00:00.000000'),"%d/%m/%Y %H:%M:%S.%f")
    # path_df = pd.DataFrame(columns=['time', 'tagID', 'x', 'y', 'z', 'quality'])
    # createCV1Path(False)
    simulationtime = datetime.strptime(str((datetime.today().strftime('%d/%m/%Y')).split()[0]+' 00:00:00.000000'),"%d/%m/%Y %H:%M:%S.%f")
    path_df = pd.DataFrame(columns=['time', 'tagID', 'x', 'y', 'z', 'quality'])
    createCV2Path(False)
    
# Program entry point
if __name__ == "__main__":
    print("init")
    source_path=""
    cv_file_name=""
    uwb_file_name=""
    
    if len(sys.argv)>1:
       source_path=sys.argv[1]
       cv_file_name=sys.argv[2]
       uwb_file_name=sys.argv[3]
    else:
        source_path='c://AAAToBackup//progetti//4wrd//data//20201116//Test1//'
        cv_file_name='CV_L2_201116.csv'
        uwb_file_name='UWB_L2_201116.csv'
        # source_path='c://AAAToBackup//progetti//4wrd//data//20201117//201117_Test1_EG//'
        # cv_file_name='CV_L2_201117.csv'
        # uwb_file_name='UWB_L2_201117.csv'
       # source_path='c://AAAToBackup//progetti//4wrd//data//20201117//201117_Test2_GC//'
       # cv_file_name='CV_L2_201117.csv'
       # uwb_file_name='UWB_L2_201117.csv'

    main()

    print("end")
