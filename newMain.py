import pandas as pd
import matplotlib.pyplot as plt
import sys # for comand line options

def main():
    cv_df = pd.DataFrame(pd.read_csv(source_path + cv_file_name,
                                     sep=',',
                                     names=["Time", "TagID", "x", "y", "z", "Quality"],
                                     header=None))

    uwb_df = pd.DataFrame(pd.read_csv(source_path + uwb_file_name,
                                      sep='\n',
                                      names=["Time"],
                                      header=None))
    
    uwb_df = uwb_df[uwb_df['Time'].str.contains('0\)')==True]
    uwb_df['Time'] = uwb_df['Time'].map(lambda x: x.replace('[','|')
                                        .replace(']','')
                                        .replace('0)','')
                                        .replace(',x0D','')
                                        .replace('   ','|')
                                        .replace(',','|')
                                        .replace('|','',1))
    uwb_df[['Time', 'TagID', 'x', 'y', 'z', 'Quality']] = uwb_df.Time.str.split('|',expand=True)
    
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
       source_path='..//DataSet//20201022//201022_Test2//'
       cv_file_name='CV_L2_201022.txt'
       uwb_file_name='UWB_L2_201022.txt'
       
          
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