import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


class Common:
    @staticmethod
    def get_metadata(df):
        col_names = df.columns  # retreive column names
        dtypes = df.dtypes  # variable types
        num_missing = df.isna().sum()  # missing values total
        # each coll
        metadata_dict = {"var_name": col_names,
                         "var_types": dtypes,
                         "num_missing": num_missing}
        metadata_df = pd.DataFrame.from_dict(metadata_dict)
        return metadata_df.sort_values("num_missing", ascending=False)  # sort by missing val

    @staticmethod
    def categorise_date(dataframe, date_column, date_format, prefix=''):
        def categorise_time(date_row):
            hour = int(date_row[:2])  # date_row.hour
            if 6 < hour <= 12:
                return 'Morning'
            elif 12 < hour <= 18:
                return 'Afternoon'
            elif 18 < hour <= 24:
                return 'Evening'
            else:
                return 'Night'

        column_name_prefix = '' if prefix == '' else prefix + '_'
        dataframe[column_name_prefix + 'date'] = pd.to_datetime(dataframe[date_column], format=date_format).dt.date
        dataframe[column_name_prefix + 'time'] = pd.to_datetime(dataframe[date_column], format=date_format).dt.strftime(
            '%H:%M:%S')
        # morning(6am-12pm), afternoon(12pm-6pm), evening(6pm-12pm) and night(12pm-6am)
        dataframe[column_name_prefix + 'time_c'] = dataframe[column_name_prefix + 'time'].apply(categorise_time)
        dataframe[column_name_prefix + 'hour'] = dataframe[column_name_prefix + 'time'].str[:2]
        return dataframe

    @staticmethod
    def feature_str(dataframe, column):
        dataframe[column] = dataframe[column].astype(str)
        return dataframe

    @staticmethod
    def eda(dataframe, x_list=[], y_val=''):
        tg = len(x_list)
        i = 1
        fig = plt.figure(figsize=(30, 10 * tg))
        print(Common.get_metadata(dataframe))
        for x_val in x_list:
            print("Analysis for feature: ", x_val)
            print("Total NULL Values in attribute: ", dataframe[x_val].isnull().sum())
            print("Total Unique Attributes:", dataframe[x_val].nunique(), "\nValues: ", dataframe[x_val].unique())
            contingency_table = pd.crosstab(dataframe[x_val], dataframe[y_val])
            print(f"contingency table between {x_val} and {y_val}: ")
            print(contingency_table)
            chi2, p, _, _ = chi2_contingency(contingency_table)
            print(f"Chi-Square Value: {chi2}\nP-value: {p}")
            ax1 = fig.add_subplot(tg, 3, i)
            sns.countplot(data=dataframe, x=x_val, ax=ax1)
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
            ax2 = fig.add_subplot(tg, 3, i + 1)
            sns.countplot(data=dataframe, x=x_val, hue=y_val, ax=ax2)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
            ax3 = fig.add_subplot(tg, 3, i + 2)
            dataframe[x_val].value_counts().plot(kind='pie', autopct='%.2f', ax=ax3)
            i += 3
        plt.show()

    @staticmethod
    def model_design(categorical_list, model_func):
        cat_pipe = Pipeline(
            steps=[
                ("one_hot_encode", OneHotEncoder(handle_unknown="ignore"))
            ]
        )
        pipeline = ColumnTransformer(
            transformers=[
                ("categorical_encoding", cat_pipe, categorical_list)
            ]
        )
        model = Pipeline(
            steps=[
                ("preprocessing", pipeline),
                ("def_model", model_func)
            ]
        )
        return model
