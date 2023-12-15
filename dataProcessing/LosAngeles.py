from datetime import datetime
import pandas as pd
from Common import *
from xgboost import XGBClassifier
import pickle

area = ['Central', 'Rampart', 'Southwest', 'Hollenbeck', 'Harbor', 'Hollywood', 'Wilshire', 'West LA', 'Van Nuys', 'West Valley', 'Northeast', '77th Street', 'Newton', 'Pacific', 'N Hollywood', 'Foothill', 'Devonshire', 'Southeast', 'Mission', 'Olympic', 'Topanga']
crimeType = ['GRAND THEFT', 'ASSAULT', 'THEFT', 'VANDALISM', 'IDENTITY THEFT', 'SERIOUS ASSAULT', 'THREATS', 'WEAPON ASSAULT', 'TRESPASSING', 'VIOLATION', 'OTHERS', 'SEX OFFENDERS', 'CRIMES AGAINST CHILDREN', 'HOMICIDE', 'KIDNAPPING', 'FINANCIAL CRIME']
premise = ['STREET', 'DOMESTIC', 'PARKING/GARAGE', 'BUSINESS AREA', 'VEHICLE', 'OTHER', 'PUBLIC PLACE', 'ENTERTAINMENT ZONE', 'BANK', 'EDUCATIONAL AREA', 'MEDICAL CENTERS', 'DOMESTIC TEMPORARY', 'ROADS']
weaponsUsed = ['BODILY FORCE', 'UNKNOWN WEAPON', 'NOT USED', 'FIREARM', 'SHARP OBJECTS', 'OTHER', 'BLUNT OBJECT']
victimRace = ['Hispanic', 'Asian', 'White', 'Black', 'American Indian/Alaskan', 'Other US', 'Other Race', 'Do not Wish to disclose/Not Reported']

class LosAngeles:
    @staticmethod
    def test():
        return
def la_crime(la_dataframe):
        Base_Data = pd.read_pickle("/mount/src/crimealytics/resources/model/LA_BaseData.pkl")
        age = la_dataframe['Age'].iloc[0]
        date_format = '%m/%d/%Y'
        lag_val = (datetime.strptime(la_dataframe['date_rptd'].iloc[0], date_format) - datetime.strptime(la_dataframe['date_inc'].iloc[0], date_format)).days
        date_occr_val = datetime.strptime(la_dataframe['date_inc'].iloc[0], date_format).day
        mon_occur_var ="Month of Occurance_" +str(datetime.strptime(la_dataframe['date_inc'].iloc[0], date_format).month)
        area_map = {'Central':1,'Rampart':2,'Southwest':3,'Hollenbeck':4,'Harbor':5,
                    'Hollywood':6,'Wilshire':7,'West LA':8,'Van Nuys':9,'West Valley':10,'Northeast':11,
                    '77th Street':12,'Newton':13,'Pacific':14,'N Hollywood':15,'Foothill':16,'Devonshire':17,
                    'Southeast':18,'Mission':19,'Olympic':20,'Topanga':21,}
        area_var = "AREA_" + str(area_map[la_dataframe['AREA'].iloc[0]])
        tmp = la_dataframe['Crime Type'].iloc[0]
        tmp = tmp.replace("ASSAULT", "ASSULT")
        grp_var = "GROUP_"+ tmp
        prm_var = "PREMSE_"+ str(la_dataframe['PREMISE'].iloc[0])
        gen_var = "Gender_"+ str(la_dataframe['Gender'].iloc[0])
        race_map = {'Hispanic': 'Race_H', 'Asian': 'Race_Asian', 'White': 'Race_W', 'Black':'Race_B', 'Other Race':'Race_O', 'Do not Wish to disclose/Not Reported':'Race_X', 'American Indian/Alaskan':'Race_I', 'Other US':'Race_Other US'}
        race_var = race_map[la_dataframe['Race'].iloc[0]]
        wep_var = "WEAPON_"+ str(la_dataframe['Weapons Used'].iloc[0])
        tmp = la_dataframe['time_occr'].iloc[0]
        tmp= tmp.replace("Afternoon","Morning")
        tmp= tmp.replace("Evening","Night")
        time_var = "TIME CAT_"+ tmp
        main_dict = {'Age': la_dataframe['Age'].iloc[0],
                     'Lag_in_Report': lag_val,
                     'Date of Occurance' : date_occr_val,
                     area_var : 1,
                     mon_occur_var : 1,
                     grp_var : 1,
                     prm_var : 1,
                     gen_var : 1,
                     race_var : 1,
                     wep_var : 1,
                     time_var : 1}
        df_main = pd.DataFrame([main_dict], columns=['Age','Lag_in_Report','Date of Occurance',area_var,mon_occur_var,grp_var,prm_var,gen_var,race_var,wep_var,time_var])
        data_final = pd.concat([Base_Data, df_main]).fillna(0)
        with open('/mount/src/crimealytics/resources/model/LA_best_xgb_model.pkl_f1', 'rb') as model_file:
            loaded_xgb_model = pickle.load(model_file)
        y_pred=loaded_xgb_model.predict(data_final)
        y_prob =loaded_xgb_model.predict_proba(data_final)
        pred_proba = round(y_prob[:, 1][0]*100, 2)
        if y_pred[0] == 0:
            return (f"UNRESOLVED : Resolution Probability {pred_proba} %")
        else:
            return (f"RESOLVED : Resolution Probability {pred_proba} %")


# my_dict = {  'Time of Occurence': ['Morning','Afternoon','Evening' , 'Night', 'Missing'],
#              ,
#              'Date of Incidence' : '12/12/2023',
#                  'Date Reported' : '12/13/2023',
# 'Vicitim Age' : 30,
# 'Victim Sex' : ['M','F','X']
# '
# }

# print(LosAngeles.la(pd.DataFrame([{'time_occr': 'Morning', 'AREA': 'Central', 'Crime Type': 'ASSAULT', 'PREMISE': 'DOMESTIC', 'Weapons Used': 'UNKNOWN WEAPON', 'date_inc': '12/12/2023', 'date_rptd': '12/13/2023', 'Age': 30, 'Gender': 'M', 'Race': 'Asian'}], columns=['time_occr', 'AREA', 'Crime Type', 'PREMISE', 'Weapons Used','date_inc', 'date_rptd', 'Age', 'Gender', 'Race'])))


# my_dict = {
# [radio] : 'Time of Occurence': ['Morning','Afternoon','Evening' , 'Night', 'Missing'],
# [drop down] :'AREA': ['Central','Rampart','Southwest','Hollenbeck','Harbor','Hollywood','Wilshire','West LA','Van Nuys','West Valley',
#                       'Northeast','77th Street','Newton','Pacific','N Hollywood','Foothill','Devonshire','Southeast','Mission','Olympic',
#                       'Topanga'],
# [drop down ] 'Crime Type': ['GRAND THEFT','ASSAULT','THEFT','VANDALISM','IDENTITY THEFT','SERIOUS ASSAULT','THREATS','WEAPON ASSAULT',
#                             'TRESPASSING','VIOLATION','OTHERS','SEX OFFENDERS','CRIMES AGAINST CHILDREN','HOMICIDE','KIDNAPPING',
#                             'FINANCIAL CRIME'],
# [drop down] 'Premise' : ['STREET','DOMESTIC','PARKING/GARAGE','BUSINESS AREA','VEHICLE','OTHER','PUBLIC PLACE','ENTERTAINMENT ZONE',
#                           'BANK','EDUCATIONAL AREA','MEDICAL CENTERS','DOMESTIC TEMPORARY','ROADS'],
#              'Weapons Used': ['BODILY FORCE','UNKNOWN WEAPON','NOT USED','FIREARM','SHARP OBJECTS','OTHER','BLUNT OBJECT'],
#   [text] 'Date of Incidence' :
#   [text]   'Date Reported' :
# [st.number_input] 'Vicitim Age' :
# [radio] 'Victim Sex' : ['M','F','X']
# [dropdown/radio] 'Vicitim Race' : ['Hispanic','Asian','White','Black','American Indian/Alaskan','Other US','Other Race','Do not Wish to disclose/Not Reported']
# }




