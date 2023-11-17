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
        return
    



print("start execution")
SanFrancisco.sf_crime()
print("end execution")
# Assuming the script and the CSV file are in the same directory


    # create a DataFrame from the dictionary
    
    # sort the DataFrame by the number of missing values in descending order


# Display the metadata DataFrame explicitly


