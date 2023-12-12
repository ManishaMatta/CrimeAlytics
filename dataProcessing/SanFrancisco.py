import datetime as datetime
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from Common import *
import pickle

crime_category = {
    'VIOLENT CRIMES': ['ASSAULT', 'ROBBERY', 'KIDNAPPING', 'SEX OFFENSES (FORCIBLE)'],
    'PROPERTY CRIMES': ['BURGLARY', 'LARCENY/THEFT', 'VEHICLE THEFT', 'ARSON', 'VANDALISM', 'STOLEN PROPERTY'],
    'DRUG AND SUBSTANCE-RELATED CRIMES': ['DRUG/NARCOTIC', 'DRUNKENNESS', 'DRIVING UNDER THE INFLUENCE'],
    'PUBLIC ORDER CRIMES': ['DISORDERLY CONDUCT', 'TRESPASS', 'LOITERING', 'LIQUOR LAWS', 'GAMBLING', 'PROSTITUTION'],
    'WHITE-COLLAR CRIMES': ['FRAUD', 'FORGERY/COUNTERFEITING', 'EMBEZZLEMENT', 'BAD CHECKS', 'EXTORTION', 'BRIBERY'],
    'MISCELLANEOUS CRIMES': ['WARRANTS', 'OTHER OFFENSES', 'SUSPICIOUS OCCURRENCE', 'NON-CRIMINAL', 'MISSING PERSON', 'RECOVERED VEHICLE', 'SUICIDE'],
    'SEX OFFENSES AND RELATED CRIMES': ['SEX OFFENSES (NON-FORCIBLE)', 'PORNOGRAPHY/OBSCENE MATERIAL'],
    'WEAPONS-RELATED CRIMES': ['WEAPON LAWS'],
    'SECONDARY CODES': ['SECONDARY CODES', 'SECOND OFFENSES (NOT SPECIFIED)']
}
mapping_dict = {
    1: 'WESTERN ADDITION',
    2: 'WEST OF TWIN PEAKS',
    3: 'VISITACION VALLEY',
    4: 'TWIN PEAKS',
    5: 'SOUTH OF MARKET',
    6: 'TREASURE ISLAND',
    7: 'PRESIDIO HEIGHTS',
    8: 'PRESIDIO',
    9: 'POTRERO HILL',
    10: 'PORTOLA',
    11: 'PACIFIC HEIGHTS',
    12: 'OUTER RICHMOND',
    13: 'OUTER MISSION',
    14: 'SUNSET/PARKSIDE',
    15: 'OCEANVIEW/MERCED/INGLESIDE',
    16: 'NORTH BEACH',
    17: 'NOE VALLEY',
    18: 'LONE MOUNTAIN/USF',
    19: 'LINCOLN PARK',
    20: 'SEACLIFF',
    21: 'NOB HILL',
    22: 'MISSION BAY',
    23: 'MISSION',
    24: 'RUSSIAN HILL',
    25: 'MARINA',
    26: 'LAKESHORE',
    27: 'TENDERLOIN',
    28: 'MCLAREN PARK',
    29: 'JAPANTOWN',
    30: 'INNER SUNSET',
    31: 'HAYES VALLEY',
    32: 'HAIGHT ASHBURY',
    33: 'GOLDEN GATE PARK',
    34: 'INNER RICHMOND',
    35: 'GLEN PARK',
    36: 'FINANCIAL DISTRICT/SOUTH BEACH',
    37: 'EXCELSIOR',
    38: 'CHINATOWN',
    39: 'CASTRO/UPPER MARKET',
    40: 'BERNAL HEIGHTS',
    41: 'BAYVIEW HUNTERS POINT'
}


class SanFrancisco:
    @staticmethod
    def sf_crime():
        crime_df = pd.read_csv("/Users/fernandacarrillo/Documents/SDSU_FALL_2023/BDA_594_Fall_2023/Crime_alytics/San_Francisco_Crime.csv")
        #crime_df = crime_df[(crime_df['incident_year'] >= 2018) & (crime_df['Year'] <= 2023)]

        #Drop missing neighborhood values
        crime_df.dropna(subset=['Analysis Neighborhoods 2 2'], inplace=True)

        # Apply the mapping to create a new column 'Neighborhood_Name'
        crime_df ['Neighborhood_Name'] = crime_df ['Analysis Neighborhoods 2 2'].map(mapping_dict)


        #The 'crime_time' in this line is an arbitrary label or tag that is being used to identify the new column created during the function call.
        #Concatenate and format time and date and pass in to function
        #crime_df['crime_datetime'] = pd.to_datetime(crime_df['Date'] + ' ' + crime_df['Time'], format='%m/%d/%Y %H:%M').dt.strftime('%I:%M:%S %p')

        # Assuming 'crime_df' is your DataFrame
        crime_df['Time'] = pd.to_datetime(crime_df['Time'], format='%H:%M')

        #Categorizing day into weekday or weekend
        crime_df['Day_Category'] = crime_df['DayOfWeek'].apply(lambda x: 'weekday' if x in ['Monday', 'Tuesday', 'Wednesday', 'Thursday'] else 'weekend')

        # Now, you can extract the hour and categorize it into time categories
        crime_df['Hour'] = crime_df['Time'].dt.hour


        # Define a function to categorize the hour into time categories
        def categorize_time(hour):
            if 6 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 18:
                return 'afternoon'
            elif 18 <= hour < 24:
                return 'evening'
            else:
                return 'night'

        # Apply the function to create the 'Time_Category' column
        crime_df['Time_Category'] = crime_df['Hour'].apply(categorize_time)

        # Print the DataFrame to check the results
        # Assuming 'crime_df' is your DataFrame

        #crime_filter_df = Common.categorise_date(crime_df, 'crime_datetime', '%m/%d/%Y %I:%M:%S %p', 'crime_time')


        #Maps categories and creates the column 'mapped category'
        #crime_df['Mapped_Category'] = crime_df['Category'].apply(lambda x: next((key for key, value in crime_category.items() if x in value), None))
        # Assuming 'crime_df' is your DataFrame
        crime_df['Mapped_Category'] = crime_df['Category'].apply(lambda x: next((key for key, value in crime_category.items() if x in value), None))

        # Display the unique values in the new 'Mapped_Category' column
        #print(crime_df['Mapped_Category'].unique())


        # Assuming sf_df is the DataFrame containing your data, new column boolean resolved or unresolved
        crime_df['Resolution_Category'] = crime_df['Resolution'].apply(lambda x: 'resolved' if x not in ['NONE', 'PSYCHOPATHIC CASE', 'UNFOUNDED'] else 'unresolved')

        #Categorize DayOfWeek into weekdays and weekends, add it to the model

        # crime_df

        #filter_df=Common.categorise_date(df,'ARREST DATE','%m/%d/%Y %I:%M:%S %p','arrest')
        #filter_df=filter_df[(filter_df['arrest_date'] >= start_date) & (filter_df['arrest_date'] <= end_date)]
        #Common.eda(crime_df, ['Mapped_Category', 'Day_Category','Time_Category', 'Neighborhood_Name' ], 'Resolution_Category')

        X = crime_df.loc[:, ['Mapped_Category', 'Day_Category','Time_Category', 'Neighborhood_Name']]
        y = crime_df.loc[:, "Resolution_Category"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777)

        #CRIME MODEL

        crime_model = Common.model_design(['Mapped_Category', 'Day_Category','Time_Category', 'Neighborhood_Name'],
                                          LogisticRegression())

        #crime_model = Common.model_design(['Mapped_Category', 'Day_Category','Time_Category', 'Neighborhood_Name'],
        #RandomForestClassifier())

        print(crime_model)
        crime_model.fit(X_train, y_train)
        crime_predictions = crime_model.predict(X_test)
        print(crime_predictions)
        crime_accuracy = accuracy_score(y_test, crime_predictions)
        print(f"Accuracy: ", round(crime_accuracy * 100, 2), "%")  # 88.39 %
        print(classification_report(y_test, crime_predictions))

        pickle.dump(crime_model , open('/Users/fernandacarrillo/Documents/SDSU_FALL_2023/BDA_594_Fall_2023/CrimeAlytics/resources/sanfrancisco-lrmodel.pk1' , 'wb'))
