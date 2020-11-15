import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys # for comand line options
import animate

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
    # takes data by row
    df = pd.DataFrame(pd.read_csv(source_path + uwb_file_name,
                                  sep='\n',
                                  names=['time'],
                                  header=None))
    # keep only the lines that contain the 0) key string
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
    return init_df(df)

def main():    
    cv_df = parse_cv_data()
    uwb_df = parse_uwb_data()
    
    ax = plt.gca()
    cv_df.plot(kind='line', y='distance', color='red', use_index=True, label='CV', ax=ax)
    uwb_df.plot(kind='line', y='distance', color='blue', use_index=True, label='UWB', ax=ax)
 
    """
    # Prova fusione
    df = pd.concat([cv_df, uwb_df])
    df = df.sort_index()
    
    df.plot(kind='line', y='distance', color='red', use_index=True, label='Distances')
    """
    
    numpoints = len(cv_df)
    # inverto x con y per fare la figura larga
    y = cv_df.x
    x = cv_df.y
    a = animate.AnimatedScatter(numpoints,x,y)
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
       source_path='c://AAAToBackup//progetti//4wrd//data//20201022//201022_Test2//'
       cv_file_name='CV_L2_201022.txt'
       uwb_file_name='UWB_L2_201022.txt'

    main()
    print("end")
