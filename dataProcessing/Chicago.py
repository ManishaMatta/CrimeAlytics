from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from Common import *
import pickle

crime_distribution = {
    'BATTERY': 'Violent Crimes',
    'ASSAULT': 'Violent Crimes',
    'HOMICIDE': 'Violent Crimes',
    'ROBBERY': 'Violent Crimes',
    'KIDNAPPING': 'Violent Crimes',
    'STALKING': 'Violent Crimes',
    'INTERFERENCE WITH PUBLIC OFFICER': 'Violent Crimes',
    'INTIMIDATION': 'Violent Crimes',
    'HUMAN TRAFFICKING': 'Violent Crimes',
    'THEFT': 'Property Crimes',
    'BURGLARY': 'Property Crimes',
    'MOTOR VEHICLE THEFT': 'Property Crimes',
    'CRIMINAL DAMAGE': 'Property Crimes',
    'ARSON': 'Property Crimes',
    'NARCOTICS': 'Drug-Related Crimes',
    'OTHER NARCOTIC VIOLATION': 'Drug-Related Crimes',
    'WEAPONS VIOLATION': 'Weapons and Violations',
    'CONCEALED CARRY LICENSE VIOLATION': 'Weapons and Violations',
    'CRIM SEXUAL ASSAULT': 'Sexual Offenses',
    'SEX OFFENSE': 'Sexual Offenses',
    'CRIMINAL SEXUAL ASSAULT': 'Sexual Offenses',
    'DECEPTIVE PRACTICE': 'Public Order Crimes',
    'LIQUOR LAW VIOLATION': 'Public Order Crimes',
    'CRIMINAL TRESPASS': 'Public Order Crimes',
    'PUBLIC PEACE VIOLATION': 'Public Order Crimes',
    'PROSTITUTION': 'Public Order Crimes',
    'GAMBLING': 'Public Order Crimes',
    'OBSCENITY': 'Public Order Crimes',
    'RITUALISM': 'Public Order Crimes',
    'PUBLIC INDECENCY': 'Public Order Crimes',
    'NON - CRIMINAL': 'Public Order Crimes',
    'NON-CRIMINAL': 'Public Order Crimes',
    'NON-CRIMINAL (SUBJECT SPECIFIED)': 'Public Order Crimes',
    'OFFENSE INVOLVING CHILDREN': 'Crimes Involving Children',
    'OTHER OFFENSE': 'Other Offense'
}

location_distribution = {
    'APARTMENT': 'Residential Areas',
    'RESIDENCE': 'Residential Areas',
    'RESIDENCE - PORCH / HALLWAY': 'Residential Areas',
    'RESIDENCE PORCH/HALLWAY': 'Residential Areas',
    'RESIDENCE-GARAGE': 'Residential Areas',
    'RESIDENTIAL YARD (FRONT/BACK)': 'Residential Areas',
    'RESIDENCE - YARD (FRONT / BACK)': 'Residential Areas',
    'RESIDENCE - GARAGE': 'Residential Areas',
    'DRIVEWAY': 'Residential Areas',
    'DRIVEWAY - RESIDENTIAL': 'Residential Areas',
    'CHA APARTMENT': 'Residential Areas',
    'CHA ELEVATOR': 'Residential Areas',
    'CHA GROUNDS': 'Residential Areas',
    'CHA HALLWAY': 'Residential Areas',
    'CHA HALLWAY / STAIRWELL / ELEVATOR': 'Residential Areas',
    'CHA HALLWAY/STAIRWELL/ELEVATOR': 'Residential Areas',
    'CHA LOBBY': 'Residential Areas',
    'CHA PARKING LOT': 'Residential Areas',
    'CHA PARKING LOT / GROUNDS': 'Residential Areas',
    'CHA PARKING LOT/GROUNDS': 'Residential Areas',
    'CHA PLAY LOT': 'Residential Areas',
    'CHA STAIRWELL': 'Residential Areas',
    'HALLWAY': 'Residential Areas',
    'HOUSE': 'Residential Areas',
    'BASEMENT': 'Residential Areas',
    'PORCH': 'Residential Areas',
    'POOL ROOM': 'Residential Areas',
    'YARD': 'Residential Areas',
    'ROOF': 'Residential Areas',
    'STAIRWELL': 'Residential Areas',
    'GARAGE': 'Residential Areas',
    'ELEVATOR': 'Residential Areas',
    'AIRPORT VENDING ESTABLISHMENT': 'Airport Areas',
    'AIRPORT TERMINAL LOWER LEVEL - SECURE AREA': 'Airport Areas',
    'AIRPORT TERMINAL UPPER LEVEL - SECURE AREA': 'Airport Areas',
    'AIRPORT BUILDING NON-TERMINAL - SECURE AREA': 'Airport Areas',
    'AIRPORT/AIRCRAFT': 'Airport Areas',
    'AIRPORT EXTERIOR - NON-SECURE AREA': 'Airport Areas',
    'AIRPORT EXTERIOR - SECURE AREA': 'Airport Areas',
    'AIRPORT TRANSPORTATION SYSTEM (ATS)': 'Airport Areas',
    'AIRCRAFT': 'Airport Areas',
    'AIRPORT TERMINAL UPPER LEVEL - NON-SECURE AREA': 'Airport Areas',
    'AIRPORT TERMINAL LOWER LEVEL - NON-SECURE AREA': 'Airport Areas',
    'AIRPORT PARKING LOT': 'Airport Areas',
    'AIRPORT BUILDING NON-TERMINAL - NON-SECURE AREA': 'Airport Areas',
    'AIRPORT TERMINAL MEZZANINE - NON-SECURE AREA': 'Airport Areas',
    'SCHOOL, PUBLIC, BUILDING': 'Educational Areas',
    'SCHOOL, PRIVATE, BUILDING': 'Educational Areas',
    'SCHOOL - PUBLIC GROUNDS': 'Educational Areas',
    'SCHOOL, PRIVATE, GROUNDS': 'Educational Areas',
    'COLLEGE / UNIVERSITY - RESIDENCE HALL': 'Educational Areas',
    'COLLEGE / UNIVERSITY - GROUNDS': 'Educational Areas',
    'SCHOOL - PUBLIC BUILDING': 'Educational Areas',
    'SCHOOL, PUBLIC, GROUNDS': 'Educational Areas',
    'SCHOOL - PRIVATE GROUNDS': 'Educational Areas',
    'SCHOOL - PRIVATE BUILDING': 'Educational Areas',
    'PUBLIC GRAMMAR SCHOOL': 'Educational Areas',
    'SCHOOL YARD': 'Educational Areas',
    'COLLEGE/UNIVERSITY GROUNDS': 'Educational Areas',
    'COLLEGE/UNIVERSITY RESIDENCE HALL': 'Educational Areas',
    'POLICE FACILITY/VEH PARKING LOT': 'Police Areas',
    'POLICE FACILITY / VEHICLE PARKING LOT': 'Police Areas',
    'POLICE FACILITY': 'Police Areas',
    'JAIL / LOCK-UP FACILITY': 'Police Areas',
    'CTA "L" PLATFORM': 'CTA',
    'CTA "L" TRAIN': 'CTA',
    'CTA BUS': 'CTA',
    'CTA BUS STOP': 'CTA',
    'CTA GARAGE / OTHER PROPERTY': 'CTA',
    'CTA PARKING LOT / GARAGE / OTHER PROPERTY': 'CTA',
    'CTA PLATFORM': 'CTA',
    'CTA PROPERTY': 'CTA',
    'CTA STATION': 'CTA',
    'CTA SUBWAY STATION': 'CTA',
    'CTA TRACKS - RIGHT OF WAY': 'CTA',
    'CTA TRAIN': 'CTA',
    'PARKING LOT / GARAGE (NON RESIDENTIAL)': 'Parking',
    'PARKING LOT/GARAGE(NON.RESID.)': 'Parking',
    'PARKING LOT': 'Parking',
    'VACANT LOT': 'Empty Spaces',
    'VACANT LOT / LAND': 'Empty Spaces',
    'VACANT LOT/LAND': 'Empty Spaces',
    'ABANDONED BUILDING': 'Empty Spaces',
    'AUTO': 'Vehicles',
    'AUTO / BOAT / RV DEALERSHIP': 'Vehicles',
    'OTHER COMMERCIAL TRANSPORTATION': 'Vehicles',
    'OTHER RAILROAD PROP / TRAIN DEPOT': 'Vehicles',
    'OTHER RAILROAD PROPERTY / TRAIN DEPOT': 'Vehicles',
    'RAILROAD PROPERTY': 'Vehicles',
    'TAXICAB': 'Vehicles',
    'TRAILER': 'Vehicles',
    'TRUCK': 'Vehicles',
    'VEHICLE - COMMERCIAL': 'Vehicles',
    'VEHICLE - COMMERCIAL: ENTERTAINMENT / PARTY BUS': 'Vehicles',
    'VEHICLE - COMMERCIAL: TROLLEY BUS': 'Vehicles',
    'VEHICLE - DELIVERY TRUCK': 'Vehicles',
    'VEHICLE - OTHER RIDE SHARE SERVICE (E.G., UBER, LYFT)': 'Vehicles',
    'VEHICLE - OTHER RIDE SHARE SERVICE (LYFT, UBER, ETC.)': 'Vehicles',
    'VEHICLE NON-COMMERCIAL': 'Vehicles',
    'VEHICLE-COMMERCIAL': 'Vehicles',
    'VEHICLE-COMMERCIAL - ENTERTAINMENT/PARTY BUS': 'Vehicles',
    'VEHICLE-COMMERCIAL - TROLLEY BUS': 'Vehicles',
    'STREET': 'Roads',
    'SIDEWALK': 'Roads',
    'ALLEY': 'Roads',
    'BRIDGE': 'Roads',
    'GANGWAY': 'Roads',
    'EXPRESSWAY EMBANKMENT': 'Highway',
    'HIGHWAY / EXPRESSWAY': 'Highway',
    'HIGHWAY/EXPRESSWAY': 'Highway',
    'GOVERNMENT BUILDING / PROPERTY': 'Government Buildings',
    'GOVERNMENT BUILDING/PROPERTY': 'Government Buildings',
    'GOVERNMENT BUILDING': 'Government Buildings',
    'FEDERAL BUILDING': 'Government Buildings',
    'ANIMAL HOSPITAL': 'Hospital & Nursing Home',
    'HOSPITAL': 'Hospital & Nursing Home',
    'HOSPITAL BUILDING / GROUNDS': 'Hospital & Nursing Home',
    'HOSPITAL BUILDING/GROUNDS': 'Hospital & Nursing Home',
    'NURSING / RETIREMENT HOME': 'Hospital & Nursing Home',
    'NURSING HOME': 'Hospital & Nursing Home',
    'NURSING HOME/RETIREMENT HOME': 'Hospital & Nursing Home',
    'MEDICAL / DENTAL OFFICE': 'Hospital & Nursing Home',
    'MEDICAL/DENTAL OFFICE': 'Hospital & Nursing Home',
    'CURRENCY EXCHANGE': 'Bank',
    'BANK': 'Bank',
    'ATM (AUTOMATIC TELLER MACHINE)': 'ATM',
    'CREDIT UNION': 'Bank',
    'COIN OPERATED MACHINE': 'Bank',
    'SAVINGS AND LOAN': 'Bank',
    'APPLIANCE STORE': 'Stores and Office Workspace',
    'CLEANING STORE': 'Stores and Office Workspace',
    'COMMERCIAL / BUSINESS OFFICE': 'Stores and Office Workspace',
    'CONVENIENCE STORE': 'Stores and Office Workspace',
    'LIQUOR STORE': 'Stores and Office Workspace',
    'DEPARTMENT STORE': 'Stores and Office Workspace',
    'DRUG STORE': 'Stores and Office Workspace',
    'GARAGE/AUTO REPAIR': 'Stores and Office Workspace',
    'GROCERY FOOD STORE': 'Stores and Office Workspace',
    'NEWSSTAND': 'Stores and Office Workspace',
    'OFFICE': 'Stores and Office Workspace',
    'RETAIL STORE': 'Stores and Office Workspace',
    'SMALL RETAIL STORE': 'Stores and Office Workspace',
    'PAWN SHOP': 'Stores and Office Workspace',
    'WAREHOUSE': 'Stores and Office Workspace',
    'GAS STATION': 'Gas Station',
    'GAS STATION DRIVE/PROP.': 'Gas Station',
    'FIRE STATION': 'Fire Station',
    'CONSTRUCTION SITE': 'Construction Site',
    'FACTORY / MANUFACTURING BUILDING': 'Factory Areas',
    'FACTORY/MANUFACTURING BUILDING': 'Factory Areas',
    'CLUB': 'Commercial Spaces',
    'BANQUET HALL': 'Commercial Spaces',
    'BARBER SHOP/BEAUTY SALON': 'Commercial Spaces',
    'BARBERSHOP': 'Commercial Spaces',
    'CAR WASH': 'Commercial Spaces',
    'CEMETARY': 'Commercial Spaces',
    'CHURCH / SYNAGOGUE / PLACE OF WORSHIP': 'Commercial Spaces',
    'CHURCH/SYNAGOGUE/PLACE OF WORSHIP': 'Commercial Spaces',
    'DAY CARE CENTER': 'Commercial Spaces',
    'LIBRARY': 'Commercial Spaces',
    'MOVIE HOUSE / THEATER': 'Commercial Spaces',
    'MOVIE HOUSE/THEATER': 'Commercial Spaces',
    'SPORTS ARENA / STADIUM': 'Commercial Spaces',
    'SPORTS ARENA/STADIUM': 'Commercial Spaces',
    'ATHLETIC CLUB': 'Commercial Spaces',
    'BOWLING ALLEY': 'Commercial Spaces',
    'YMCA': 'Commercial Spaces',
    'BAR OR TAVERN': 'Restaurants',
    'HOTEL': 'Restaurants',
    'HOTEL / MOTEL': 'Restaurants',
    'HOTEL/MOTEL': 'Restaurants',
    'MOTEL': 'Restaurants',
    'TAVERN': 'Restaurants',
    'TAVERN / LIQUOR STORE': 'Restaurants',
    'TAVERN/LIQUOR STORE': 'Restaurants',
    'RESTAURANT': 'Restaurants',
    'BEACH': 'Outdoor Spaces',
    'BOAT / WATERCRAFT': 'Outdoor Spaces',
    'BOAT/WATERCRAFT': 'Outdoor Spaces',
    'LAKE': 'Outdoor Spaces',
    'LAKEFRONT / WATERFRONT / RIVERBANK': 'Outdoor Spaces',
    'LAKEFRONT/WATERFRONT/RIVERBANK': 'Outdoor Spaces',
    'RIVER BANK': 'Outdoor Spaces',
    'FOREST PRESERVE': 'Outdoor Spaces',
    'FARM': 'Outdoor Spaces',
    'HORSE STABLE': 'Outdoor Spaces',
    'KENNEL': 'Outdoor Spaces',
    'PARK PROPERTY': 'Outdoor Spaces',
    'VESTIBULE': 'Outdoor Spaces',
    'WOODED AREA': 'Outdoor Spaces',
    'OTHER': 'Other',
    'OTHER (SPECIFY)': 'Other'
}

class Chicago:

    @staticmethod
    def chicago_crime():
        crime_df = pd.read_csv(
            "/Users/Manisha/Documents/MS/SDSU/course/BDA-594/finalProject/datasets/Crimes_-_2001_to_Present.csv")
        crime_df = crime_df[(crime_df['Year'] >= 2018) & (crime_df['Year'] <= 2023)]
        crime_df = Common.feature_str(crime_df, 'Ward')
        crime_filter_df = Common.categorise_date(crime_df, 'Date', '%m/%d/%Y %I:%M:%S %p', 'crime')
        # start_date = datetime.strptime('2018-01-01', '%Y-%m-%d').date()
        # end_date = datetime.strptime('2023-12-31', '%Y-%m-%d').date()
        # crime_filter_df = crime_filter_df[(crime_filter_df['crime_date'] >= start_date) & (crime_filter_df['crime_date'] <= end_date)]
        # print(crime_filter_df.head(3))
        print(Common.get_metadata(crime_df))

        crime_df['crime_type'] = crime_df['Primary Type'].str.strip().map(crime_distribution).fillna('Other Offense')
        crime_df['location_desc'] = crime_df['Location Description'].str.strip().map(location_distribution).fillna('Other')
        # print(Common.get_metadata(crime_df))
        Common.eda(crime_df, ['crime_type', 'location_desc', 'crime_time_c', 'Ward'], 'Arrest')
        # print(Common.get_metadata(crime_filter_df[['Primary Type', 'Location Description', 'crime_time_c', 'Arrest']]))
        # @todo - ignoring records with NA for now
        # crime_na_df = crime_filter_df.dropna()
        # print("crime data dataset size", crime_df.shape, crime_filter_df.shape, crime_na_df.shape)
        X = crime_df.loc[:, ['crime_type', 'location_desc', 'crime_time_c', 'Ward']]
        y = crime_df.loc[:, "Arrest"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777)
        # print(X_train.shape, " : ", y_train.shape)
        # print(X_test.shape, " : ", y_test.shape)
        # print(Common.get_metadata(crime_na_df[['Primary Type', 'Location Description', 'crime_time_c', 'Arrest']]))

        ccrime_model = Common.model_design(['crime_type', 'location_desc', 'crime_time_c', 'Ward'],
                                           RandomForestClassifier())

        print(ccrime_model)
        ccrime_model.fit(X_train, y_train)
        ccrime_predictions = ccrime_model.predict(X_test)
        print(ccrime_predictions)
        ccrime_accuracy = accuracy_score(y_test, ccrime_predictions)
        print(f"Accuracy: ", round(ccrime_accuracy * 100, 2), "%")  # 88.39 %
        print(classification_report(y_test, ccrime_predictions))

        pickle.dump(ccrime_model, open('/resources/model/chicago-rfmodel.pk1', 'wb'))

        # loaded_ccrime_model = pickle.load(open('/Users/Manisha/Documents/MS/SDSU/course/BDA-594/finalProject/project/CrimeAlytics/resources/model/chicago-rfmodel.pk1' , 'rb'))
        # loaded_model_accuracy = loaded_ccrime_model.score(X_test, y_test)
        # print("Loaded Model Accuracy:" , loaded_model_accuracy * 100, "%")


# @TODO add to main function
# print("start execution")
# Chicago.chicago_crime()
# print("end execution")


# python /Users/Manisha/Documents/MS/SDSU/course/BDA-594/finalProject/project/CrimeAlytics/dataProcessing/Chicago.py

