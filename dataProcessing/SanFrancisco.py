from datetime import datetime
import pandas as pd
from Common import *


class SanFrancisco:
    @staticmethod
    def sf_crime():
        df = pd.read_csv("/Users/fernandacarrillo/Documents/SDSU_FALL_2023/BDA_594_Fall_2023/Crime_alytics/San_Francisco_Crime.csv")
        print(df.head(3))
        print(dir(Common))
        metadata_df = Common.get_metadata(df)
        print(metadata_df)

        start_date=datetime.strptime('2018-01-01', '%Y-%m-%d').date()
        end_date=datetime.strptime('2023-12-31', '%Y-%m-%d').date()

        filter_df=Common.categorise_date(df,'ARREST DATE','%m/%d/%Y %I:%M:%S %p','arrest')
        filter_df=filter_df[(filter_df['arrest_date'] >= start_date) & (filter_df['arrest_date'] <= end_date)]
        return
    


pd.set_option("display.max_rows", 100000)
pd.set_option("display.max_columns", 1000)
pd.set_option("display.expand_frame_repr", True)
pd.set_option('display.width', 1000)
print("start execution")
SanFrancisco.sf_crime()
print("end execution")
# Assuming the script and the CSV file are in the same directory


    # create a DataFrame from the dictionary
    
    # sort the DataFrame by the number of missing values in descending order


# Display the metadata DataFrame explicitly


