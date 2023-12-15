# streamlit run /Users/Manisha/Documents/MS/SDSU/course/BDA-594/finalProject/project/CrimeAlytics/dataProcessing/mainApp.py
# ?city=la : https://crimealytics-bda594.streamlit.app/?city=la
# ?city=sf : https://crimealytics-bda594.streamlit.app/?city=sf
# ?city=ch : https://crimealytics-bda594.streamlit.app/?city=ch
# https://crimealytics-bda594.streamlit.app/
# crimealytics ∙ main ∙ dataProcessing/mainApp.py



import os
import streamlit as st
import pandas as pd
import pickle


st.set_page_config(page_title='CrimeAlytics', layout='wide', page_icon='logo2.png', initial_sidebar_state='auto')
st.title('CrimeAlytics', anchor="CrimeAlytics", help='https://github.com/ManishaMatta/CrimeAlytics')
st.caption("BDA-594 Course")
caption_text = 'BDA-594 Project - CrimeAlytics [Group 3]'

param = st.experimental_get_query_params().get("city", [""])[0]

city_option = ''
if param == 'la':
    city_option='Los Angeles'
elif param == 'sf':
    city_option = 'San Francisco'
else :
    city_option = 'Chicago'

cities=['Chicago', 'Los Angeles',  'San Francisco']
add_selectbox = st.sidebar.selectbox('Focussed Cities', tuple(cities), index=cities.index(city_option))

if add_selectbox == 'Chicago':

    from Chicago import crime_distribution,location_distribution
    left_column, right_column = st.columns(2)
    with right_column:
        crime_desc_chosen = st.selectbox('Crime Type',tuple(set(crime_distribution.values())))
    with left_column:
        crime_loc_chosen = st.selectbox('Crime Location', tuple(set(location_distribution.values())))
    ward_dtl = round(st.slider('Ward',1.0,50.0),0)
    crime_time_option = st.radio('Time of Crime', ('Morning','Afternoon','Evening','Night'))
    button_val = st.button('Predict')
    if button_val:
        chicago_path = os.path.join("/mount/src/crimealytics/resources/model", "chicago-rfmodel.pk1")
        loaded_ccrime_model = pickle.load(open(chicago_path, 'rb'))
        x_value = pd.DataFrame([{'crime_type': crime_desc_chosen, 'location_desc': crime_loc_chosen, 'crime_time_c': crime_time_option, 'Ward': str(ward_dtl)}], columns=['crime_type', 'location_desc', 'crime_time_c', 'Ward'])
        cc_predict_value = loaded_ccrime_model.predict(x_value[['crime_type', 'location_desc', 'crime_time_c', 'Ward']].dropna())
        if cc_predict_value:
            st.subheader(f'_RESOLVED_')
        else:
            st.subheader(f'_UNRESOLVED_')

elif add_selectbox == 'Los Angeles':
    from LosAngeles import la_crime, area, crimeType, premise, weaponsUsed, victimRace
    left_column, right_column = st.columns(2)
    with right_column:
        area_chosen = st.selectbox('Crime Location', tuple(set(area)))
    with left_column:
        premise_chosen = st.selectbox('Location Premise', tuple(set(premise)))
    left_column, middle_column, right_column = st.columns(3)
    with right_column:
        crime_desc_chosen = st.selectbox('Crime Type', tuple(set(crimeType)))
    with middle_column:
        weapons_chosen = st.selectbox('Weapons Used', tuple(set(weaponsUsed)))
    with left_column:
        race_chosen = st.selectbox('Victim Race', tuple(set(victimRace)))
    left_column, right_column = st.columns(2)
    with left_column:
        crime_time_option = st.radio('Time of Crime', ('Morning', 'Afternoon', 'Evening', 'Night', 'Missing'))
    with right_column:
        incident_dt = st.text_input('Date of Incident', 'MM/DD/YYYY')
        reported_dt = st.text_input('Incident Reported Date', 'MM/DD/YYYY')
    left_column, right_column = st.columns(2)
    with left_column:
        sex_option = st.radio('Victim Sex', ('M', 'F', 'X'))
    with right_column:
        victim_age = st.number_input('Victim Age', value=None, step=1)
    button_val = st.button('Predict')
    if button_val:
        x_value = pd.DataFrame([{'time_occr': crime_time_option, 'AREA': area_chosen, 'Crime Type': crime_desc_chosen, 'PREMISE': premise_chosen, 'Weapons Used': weapons_chosen, 'date_inc': incident_dt, 'date_rptd': reported_dt, 'Age': int(victim_age), 'Gender': sex_option, 'Race': race_chosen}], columns=['time_occr', 'AREA', 'Crime Type', 'PREMISE', 'Weapons Used','date_inc', 'date_rptd', 'Age', 'Gender', 'Race'])
        cc_predict_value = la_crime(x_value.dropna())
        st.subheader(f'_{str(cc_predict_value)}_')

else:
    from SanFrancisco import mapping_dict,crime_category
    left_column, right_column = st.columns(2)
    with right_column:
        crime_desc_chosen = st.selectbox('Crime Type',tuple(set(crime_category.keys())))
    with left_column:
        crime_loc_chosen = st.selectbox('Crime Location', tuple(set(mapping_dict.values())))
    left_column, right_column = st.columns(2)
    with right_column:
        crime_time_option = st.radio('Time of Crime', ('Morning','Afternoon','Evening','Night'))
    with left_column:
        crime_day_option = st.radio('Day of Crime', ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'weekend'))
    button_val = st.button('Predict')
    if button_val:
        loaded_ccrime_model = pickle.load(open('/mount/src/crimealytics/resources/model/SF-LRmodel.pk1', 'rb'))
        x_value = pd.DataFrame([{'Mapped_Category': crime_desc_chosen, 'Day_Category': crime_day_option, 'Time_Category': crime_time_option.lower(),'Neighborhood_Name':crime_loc_chosen}], columns=['Mapped_Category', 'Day_Category','Time_Category', 'Neighborhood_Name'])
        cc_predict_value = loaded_ccrime_model.predict(x_value.dropna())
        st.write("predicted value:", str(cc_predict_value))
        st.subheader(f'_{str(cc_predict_value)}_')

