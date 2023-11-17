import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class Common:
    @staticmethod
    def get_metadata(df):
        col_names=df.columns # retreive column names
        dtypes=df.dtypes # variable types
        num_missing=df.isna().sum() # missing values total
        #each coll
        metadata_dict= {"var_name":col_names,
                       "var_types":dtypes,
                       "num_missing": num_missing}
        metadata_df=pd.DataFrame.from_dict(metadata_dict)
        return metadata_df.sort_values("num_missing",ascending=False) # sort by missing val

    @staticmethod
    def categorise_date(dataframe,date_column,date_format,prefix=''):
        def categorise_time(date_row):
            hour = int(date_row[:2]) # date_row.hour
            if hour > 6 and hour <=12:
                return 'Morning'
            elif hour > 12 and hour <=18:
                return 'Afternoon'
            elif hour > 18 and hour <=24:
                return 'Evening'
            else:
                return 'Night'

        column_name_prefix = '' if prefix=='' else prefix+'_'
        dataframe[column_name_prefix+'date'] = pd.to_datetime(dataframe[date_column], format=date_format).dt.date
        dataframe[column_name_prefix+'time'] = pd.to_datetime(dataframe[date_column], format=date_format).dt.strftime('%H:%M:%S')
        # morning(6am-12pm), afternoon(12pm-6pm), evening(6pm-12pm) and night(12pm-6am)
        dataframe[column_name_prefix+'time_c'] = dataframe[column_name_prefix+'time'].apply(categorise_time)
        dataframe[column_name_prefix+'hour'] = dataframe[column_name_prefix+'time'].str[:2]
        return dataframe

    @staticmethod
    def model_design(categorical_list, model_func):
        cat_pipe = Pipeline(
            steps = [
                ("one_hot_encode", OneHotEncoder(handle_unknown="ignore"))
            ]
        )
        pipeline = ColumnTransformer(
            transformers=[
                ("cat", cat_pipe, categorical_list)
            ]
        )
        model = Pipeline(
            steps=[
                ("prepossing", pipeline),
                ("random_forest", model_func)
            ]
        )
        return model
