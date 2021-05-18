import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys     # for comand line options
import animate # needs conda install fsspec

def init_df(df):
    df = df.replace('-', np.nan)
    df.time = df.time.astype('datetime64[ns]')
    df.x = df.x.astype(float)
    df.y = df.y.astype(float)
    df.z = df.z.astype(float)
    df.quality = df.quality.astype(float)
    # set time ad index of time series
    df.set_index('time', inplace=True)
    # fake distance
    df['distance'] = df[['x','y']].mean(axis=1)
    return df

def parse_cv_data():
    df = pd.DataFrame(pd.read_csv(source_path + cv_file_name,
                                  sep=',',
                                  names=['time', 'tagID', 'x', 'y', 'z', 'quality'],
                                  header=None))
    return init_df(df)

def parse_uwb_data():
    # keep only the lines that contain the 0) key string
    oldFormat = False
    if(oldFormat):
    # takes data by row
       df = pd.DataFrame(pd.read_csv(source_path + uwb_file_name,
                                     sep='\n',
                                     names=['time'],
                                     header=None))
       df = df[df['time'].str.contains('0\)')==True]
       # parse the various data contained in the key lines
       df['time'] = df['time'].map(lambda x: x.replace('[','|')
                                           .replace(']','')
                                           .replace('0)','')
                                           .replace(',x0D','')
                                           .replace('   ','|')
                                           .replace(',','|')
                                           .replace('|','',1))
       df[['time', 'tagID', 'x', 'y', 'z', 'quality']] = df.time.str.split('|', expand=True)
    else:
       df = pd.DataFrame(pd.read_csv(source_path + uwb_file_name,
                                     sep=',',
                                     names=['time', 'tagID', 'x', 'y', 'z', 'quality','nix'],
                                  header=None))
       df = df.drop(columns=['nix'])
    return init_df(df)

def main():    
    cv_df = parse_cv_data()
    cv_df.index = cv_df.index.floor('10ms') # keep time down to centiseconds
    uwb_df = parse_uwb_data()
    uwb_df.index = uwb_df.index.floor('10ms') # keep time down to centliseconds
    dfmerged = pd.merge_ordered(cv_df,uwb_df,on="time",suffixes=("_1","_2"), fill_method="ffill")
    ax = plt.gca()
    cv_df.plot(kind='line', y='distance', color='red', use_index=True, label='CV', ax=ax)
    uwb_df.plot(kind='line', y='distance', color='blue', use_index=True, label='UWB', ax=ax)
 
    # Prova fusione
    df = pd.concat([cv_df, uwb_df])
    df = df.sort_index()
    
    df.plot(kind='line', y='distance', color='red', use_index=True, label='Distances')

    numpoints = len(dfmerged)
    # inverto x con y per fare la figura larga
    y = (5-dfmerged.x_1)
    x = dfmerged.y_1
    xfig = 16.5       # room length
    yfig = 5          # room width
    #a = animate.AnimatedScatter(numpoints,x,y,xfig,yfig)
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

    main()
    print("end")
