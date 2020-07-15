import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey

class SqLiteDB:
    # Main DB Connection Ref Obj
    db_engine = None
	 
    # Class members
    SQLITE                  = 'sqlite'

    # Table Names
    TABLE           = 'users'

    # http://docs.sqlalchemy.org/en/latest/core/engines.html
    DB_ENGINE = {
        SQLITE: 'sqlite:///{DB}'
    }

	 # constructor
    def __init__(self, dbtype, username='', password='', dbpath='', dbname=''):
        dbtype = dbtype.lower()
        if dbpath != '':
           dbname = dbpath+dbname
           
        if dbtype in self.DB_ENGINE.keys():
            engine_url = self.DB_ENGINE[dbtype].format(DB=dbname)
            self.db_engine = create_engine(engine_url)
            print(self.db_engine)
        else:
            print("DBType is not found in DB_ENGINE")
    
    # passes the query on to the provider
    def execute_query(self, query=''):
        if query == '' : return
        print (query)
        with self.db_engine.connect() as connection:
            try:
                connection.execute(query)
            except Exception as e:
                print(e)
                
    # get all table data
    def get_all_data(self, table='', query=''):
        query = query if query != '' else "SELECT * FROM '{}';".format(table)
        #print(query)
        with self.db_engine.connect() as connection:
            try:
                result = connection.execute(query)
            except Exception as e:
                print(e)
            else:
                for row in result:
                    print(row) # print(row[0], row[1], row[2])
                result.close()
        print("\n")
        table_df = pd.read_sql_table(
           table,
           con=self.db_engine
           )
        return table_df
            