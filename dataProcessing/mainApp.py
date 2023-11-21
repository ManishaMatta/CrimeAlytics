# streamlit run /Users/Manisha/Documents/MS/SDSU/course/BDA-594/finalProject/project/CrimeAlytics/websiteDesign/mainApp.py
# python -m streamlit run your_script.py
# streamlit run https://raw.githubusercontent.com/streamlit/demo-uber-nyc-pickups/master/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import time
import pickle

st.title('CrimeAlytics - Execution')

add_selectbox = st.sidebar.selectbox(
    'Interested City',
    ('Chicago', 'Los Angeles',  'San Francisco')
)
if add_selectbox == 'Chicago':

    from Chicago import crime_distribution,location_distribution
    left_column, right_column = st.columns(2)
    with right_column:
        crime_desc_chosen = st.radio(
            'Crime Type', tuple(set(crime_distribution.values())))
    with left_column:
        crime_loc_chosen = st.radio(
            'Crime Location', tuple(set(location_distribution.values())))
    ward_dtl = round(st.slider('Ward',1.0,50.0),0)
    crime_time_option = st.selectbox('Time of Crime',('Morning','Afternoon','Evening','Night'))
    button_val = st.button('Predict')
    st.write(button_val)
    if button_val:
        st.write("********************************************************")
        # @TODO make the pickel file work!!
        st.write(add_selectbox," : ",ward_dtl,crime_loc_chosen,crime_desc_chosen,crime_time_option)
        loaded_ccrime_model = pickle.load(open('/Users/Manisha/Documents/MS/SDSU/course/BDA-594/finalProject/project/CrimeAlytics/resources/output/chicago-rfmodel.pk1' , 'rb'))
        # x_value = pd.DataFrame([{'crime_type': crime_desc_chosen, 'location_desc': crime_loc_chosen, 'crime_time_c': crime_time_option, 'Ward': ward_dtl}], columns=['crime_type', 'location_desc', 'crime_time_c', 'Ward'])
        x_value = pd.DataFrame([{'crime_type': 'Weapons and Violations', 'location_desc': 'Parking', 'crime_time_c': 'Morning', 'Ward': 12.0}], columns=['crime_type', 'location_desc', 'crime_time_c', 'Ward'])
        cc_predict_value = loaded_ccrime_model.predict(x_value[['crime_type', 'location_desc', 'crime_time_c', 'Ward']].dropna())
        st.write("predicted value: ", cc_predict_value)
        st.write("********************************************************")
# elif add_selectbox=='Los Angeles':
#     LA
# else:
#     SF

# @st.cache_data
# def long_running_function(param1, param2):
#     return
# st.cache_resource is the recommended way to cache global resources like ML models or database connections – unserializable objects that you don’t want to load multiple times. Using it, you can share these resources across all reruns and sessions of an app without copying or duplication. Note that any mutations to the cached return value directly mutate the object in the cache (more details below).



# 'Starting a long computation...'
# # Add a placeholder
# latest_iteration = st.empty()
# bar = st.progress(0)
# for i in range(100):
#     # Update the progress bar with each iteration.
#     latest_iteration.text(f'Iteration {i+1}')
#     bar.progress(i + 1)
#     time.sleep(0.1)
# '...and now we\'re done!'


# @TODO MAke the model execute
# @TODO add background picture
# @TODO add code for LA and SF

# page_bg_img = '''
# <style>
# body {
# background-image: url("/Users/Manisha/Desktop/Screenshot\ 2023-11-11\ at\ 11.47.12\ AM.png");
# background-size: cover;
# }
# </style>
# '''
#
# st.markdown(page_bg_img, unsafe_allow_html=True)

# st.markdown(
#     """
#     <style>
#     .reportview-container {
#         background: url("/Users/Manisha/Desktop/Screenshot\ 2023-11-11\ at\ 11.47.12\ AM.png")
#     }
#    .sidebar .sidebar-content {
#         background: url("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/US_map_-_states_and_capitals.png/640px-US_map_-_states_and_capitals.png")
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

########################################################################################

# df = pd.DataFrame({
#     'first column': [1, 2, 3, 4],
#     'second column': [10, 20, 30, 40]
# })
# df
#
# st.write("Here's our first attempt at using data to create a table:")
# st.write(pd.DataFrame({
#     'first column': [1, 2, 3, 4],
#     'second column': [10, 20, 30, 40]
# }))
#
# dataframe = np.random.randn(10, 20)
# st.dataframe(dataframe)
#
# dataframe = pd.DataFrame(
#     np.random.randn(10, 20),
#     columns=('col %d' % i for i in range(20)))
# st.dataframe(dataframe.style.highlight_max(axis=0))
#
# dataframe = pd.DataFrame(
#     np.random.randn(10, 20),
#     columns=('col %d' % i for i in range(20)))
# st.table(dataframe)
#
# chart_data = pd.DataFrame(
#     np.random.randn(20, 3),
#     columns=['a', 'b', 'c'])
# st.line_chart(chart_data)
#
# map_data = pd.DataFrame(
#     np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
#     columns=['lat', 'lon'])
# st.map(map_data)
#
#
# x = st.slider('x')  # 👈 this is a widget
# st.write(x, 'squared is', x * x)
#
# st.text_input("Your name", key="name")
# # You can access the value at any point with:
# st.session_state.name
#
# if st.checkbox('Show dataframe'):
#     chart_data = pd.DataFrame(
#         np.random.randn(20, 3),
#         columns=['a', 'b', 'c'])
#     chart_data
#
#
# df = pd.DataFrame({
#     'first column': [1, 2, 3, 4],
#     'second column': [10, 20, 30, 40]
# })
# option = st.selectbox(
#     'Which number do you like best?',
#     df['first column'])
# 'You selected: ', option
#
#
# # Add a selectbox to the sidebar:
# # add_selectbox = st.sidebar.selectbox(
# #     'How would you like to be contacted?',
# #     ('Email', 'Home phone', 'Mobile phone')
# # )
# # # Add a slider to the sidebar:
# # add_slider = st.sidebar.slider(
# #     'Select a range of values',
# #     0.0, 100.0, (25.0, 75.0)
# # )
#
# left_column, right_column = st.columns(2)
# # You can use a column just like st.sidebar:
# left_column.button('Press me!')
# # Or even better, call Streamlit functions inside a "with" block:
# with right_column:
#     chosen = st.radio(
#         'Sorting hat',
#         ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
#     st.write(f"You are in {chosen} house!")
#
#
# 'Starting a long computation...'
# # Add a placeholder
# latest_iteration = st.empty()
# bar = st.progress(0)
# for i in range(100):
#     # Update the progress bar with each iteration.
#     latest_iteration.text(f'Iteration {i+1}')
#     bar.progress(i + 1)
#     time.sleep(0.1)
# '...and now we\'re done!'
#
#
# @st.cache_data
# def long_running_function(param1, param2):
#     return
# st.cache_resource is the recommended way to cache global resources like ML models or database connections – unserializable objects that you don’t want to load multiple times. Using it, you can share these resources across all reruns and sessions of an app without copying or duplication. Note that any mutations to the cached return value directly mutate the object in the cache (more details below).


########################################################################################