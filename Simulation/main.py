import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys     # for comand line options
import animate # needs conda install fsspec
from datetime import datetime, timedelta


def plotStanza(xfig, yfig, fig=None):
    if fig is None:
        fig, ax = plt.subplots(figsize=(xfig, yfig))
    ax = plt.gca()
    img = plt.imread("../stanza.png")
    ax.imshow(img, zorder=0, extent=[0, xfig, 0, yfig])

 
# simulation of staying still in one position for a certain period (in ms)
def stay(position, milliseconds):
    global path_df
    period = milliseconds#/100
    if len(path_df.time) == 0 :
        date_rng = pd.date_range('1/1/2021 00:00:00.000000', periods=period, freq='100ms')
    else:
        date_rng = pd.date_range(path_df.time.iloc[len(path_df.time)-1] + timedelta(milliseconds=100), periods=period, freq='100ms')
    df = pd.DataFrame(columns=['time', 'tagID', 'x', 'y', 'z', 'quality'])
    df.time = date_rng
    df.tagID = 'tag1'
    df.x = position.x
    df.y = position.y
    df.z = position.z
    df.quality = 0
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
def move(from_pos, to_pos, milliseconds):
    global path_df
    period = milliseconds
    if len(path_df.time) == 0 :
        date_rng = pd.date_range('1/1/2021 00:00:00.000000', periods=period, freq='100ms')
    else:
        date_rng = pd.date_range(path_df.time.iloc[len(path_df.time)-1] + timedelta(milliseconds=100), periods=period, freq='100ms')
    df = pd.DataFrame(columns=['time', 'tagID', 'x', 'y', 'z', 'quality'])
      
    df.time = date_rng
    df.tagID = 'tag1'
    df.quality = 0
    
    if from_pos.x == to_pos.x:
        df.x = from_pos.x
        df.y = generatePositions(from_pos.y, to_pos.y, period)
    else:
        df.y = from_pos.y
        df.x = generatePositions(from_pos.x, to_pos.x, period)
        
    df.z = 0
    path_df=path_df.append(df, ignore_index=True)
    
def createPath():
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
    
def main():  
    global anchors_df, positions, path_df
    
    anchors_df = pd.DataFrame(np.array([[0, 12.1], [0, 4.6], [2.5, 0], [4.95, 4.6], [4.95, 12.1], [2.5, 16.9]]),
                              index=['CC90','D20C','CB1D','9028','198A','8418'],
                              columns=['x', 'y'])
    

    path_df = pd.DataFrame(columns=['time', 'tagID', 'x', 'y', 'z', 'quality'])
    positions = pd.DataFrame(np.array([[2.5, 4.2, 0], [2.50, 6.66, 0], [3.95, 6.66, 0], [2.50, 9.06, 0], [1.00, 9.06, 0], [2.50, 13.00, 0], [1.00, 13.00, 0]]),
                             columns=['x', 'y', 'z'],
                             index=['Pos0','Pos1','Pos2','Pos3','Pos4','Pos5','Pos6'])
    
    createPath()
    
    # graphical correction of coordinates
    path_df['x'] = 5-path_df['x']
    
    xfig = 16.9      # room length
    yfig = 5          # room width
    fig, ax = plt.subplots(figsize=(xfig, yfig))
    ax.scatter(path_df['y'], path_df['x'], c="orange", edgecolors='none')
    
    # a = animate.AnimatedScatter(len(path_df),path_df,xfig,yfig)
    plotStanza(xfig, yfig, fig)
    plt.show()

    
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
