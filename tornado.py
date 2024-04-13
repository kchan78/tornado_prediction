# App to predict tornado class
# Using two pre-trained ML models in Streamlit

# Import libraries
import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import precision_score, recall_score, f1_score
from math import sqrt
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt # Matplotlib

# Package to implement ML Models
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor # Decision Tree
from xgboost import XGBRegressor, XGBClassifier  # XG Boost

# Package for data partitioning
from sklearn.model_selection import train_test_split

# Time packages
import time

# Ignore Deprecation Warnings
import warnings
warnings.filterwarnings('ignore')

# %matplotlib inline



### Intro to the app ###
st.title('Tornado Injury Prediction: A Machine Learning App') 

# Display the image
st.image('tornado_image.gif', width = 700)

st.subheader('Utilize our advanced Machine Learning application to predict whether or not a tornado will cause injuries.') 

st.write('Upload a CSV file or use the following form to get started')


### Reading all of the pickled models ###
# Decision Tree
dt_pickle = open('dt_tornados.pickle', 'rb') 
clf_dt = pickle.load(dt_pickle) 
dt_pickle.close() 

# XGBoost
xg_pickle = open('xg_tornados.pickle', 'rb') 
clf_xg = pickle.load(xg_pickle) 
xg_pickle.close() 

# Unique Mappings
output_pickle = open('output_tornados.pickle', 'rb')
unique_mappings = pickle.load(output_pickle)
output_pickle.close()

# Create color mappings for predictions
def color_cells(val):
    if val == 'No Injuries':
        return 'background-color: rgba(50, 205, 50, 0.75)' #limegreen
    elif val == 'A Few Injuries':
        return 'background-color: rgba(255, 255, 0, 0.75)' #yellow
    elif val == 'Many Injuries':
        return 'background-color: rgba(255, 165, 0, 0.75)' #orange
    else:
        return ''
    

### Load in dataset ###

# Import Data from years 2018, 2021, 2022, 2023
storm_locations_2023_df = pd.read_csv('StormEvents_locations-ftp_v1.0_d2023_c20231017.csv')
storm_locations_2023_df.head()
storm_details_2023_df = pd.read_csv('StormEvents_details-ftp_v1.0_d2023_c20231017.csv')
storm_details_2023_df.head()

storm_2023_df = storm_locations_2023_df.merge(storm_details_2023_df, how="left", left_on='EVENT_ID', right_on='EVENT_ID')

storm_details_2022_df = pd.read_csv('StormEvents_details-ftp_v1.0_d2022_c20231116.csv')
storm_locations_2022_df = pd.read_csv('StormEvents_locations-ftp_v1.0_d2022_c20231116.csv')
storm_2022_df = storm_locations_2022_df.merge(storm_details_2022_df, how="left", left_on='EVENT_ID', right_on='EVENT_ID')

storm_locations_2021_df = pd.read_csv('StormEvents_locations-ftp_v1.0_d2021_c20231017.csv')
storm_details_2021_df = pd.read_csv('StormEvents_details-ftp_v1.0_d2021_c20231017.csv')
storm_2021_df = storm_locations_2021_df.merge(storm_details_2021_df, how="left", left_on='EVENT_ID', right_on='EVENT_ID')

storm_locations_2018_df = pd.read_csv('StormEvents_locations-ftp_v1.0_d2018_c20230616.csv')
storm_details_2018_df = pd.read_csv('StormEvents_details-ftp_v1.0_d2018_c20230616.csv')
storm_2018_df = storm_locations_2018_df.merge(storm_details_2018_df, how="left", left_on='EVENT_ID', right_on='EVENT_ID')

# subset to tornados only
tornado_2023_df = storm_2023_df[storm_2023_df["EVENT_TYPE"] == "Tornado"]
tornado_2023_df = tornado_2023_df.drop_duplicates(subset=['EVENT_ID'])

tornado_2022_df = storm_2022_df[storm_2022_df["EVENT_TYPE"] == "Tornado"]
tornado_2022_df = tornado_2022_df.drop_duplicates(subset=['EVENT_ID'])

tornado_2021_df = storm_2021_df[storm_2021_df["EVENT_TYPE"] == "Tornado"]
tornado_2021_df = tornado_2021_df.drop_duplicates(subset=['EVENT_ID'])

tornado_2018_df = storm_2018_df[storm_2018_df["EVENT_TYPE"] == "Tornado"]
tornado_2018_df = tornado_2018_df.drop_duplicates(subset=['EVENT_ID'])

# Add all datsets together
tornado_df = pd.concat([tornado_2023_df, tornado_2022_df, tornado_2021_df, tornado_2018_df])

# Clean dataset
# Convert date_time column to datetime object
tornado_df['BEGIN_DATE_TIME'] = tornado_df['BEGIN_DATE_TIME'].apply(pd.to_datetime)
tornado_df['END_DATE_TIME'] = tornado_df['END_DATE_TIME'].apply(pd.to_datetime)
tornado_df['TIME_ELAPSED'] = (tornado_df['END_DATE_TIME'] - tornado_df['BEGIN_DATE_TIME']).dt.total_seconds() / 60

# find range of tornado from lat & lon
tornado_df['VERTICAL_MOVEMENT'] = abs(tornado_df['BEGIN_LAT'] - tornado_df['END_LAT'])
tornado_df['HORIZONTAL_MOVEMENT'] = abs(tornado_df['BEGIN_LON'] - tornado_df['END_LON'])

# change property and crop damage column from string to numeric
def convert_to_numeric(value):
  value = str(value).lower()
  if 'k' in value:
      return float(value.replace('k', '')) * 1000
  elif 'm' in value:
      return float(value.replace('m', '')) * 1000000
  else:
      return float(value)

# Apply the conversion function to the column
tornado_df['DAMAGE_PROPERTY'] = tornado_df['DAMAGE_PROPERTY'].apply(convert_to_numeric)
tornado_df['DAMAGE_CROPS'] = tornado_df['DAMAGE_CROPS'].apply(convert_to_numeric)

# Making injuries into a classifier
tornado_df['INJURIES_BINARY'] = tornado_df.apply(lambda x: 'Injuries' if x['INJURIES_DIRECT'] > 0 else 'No Injuries', axis = 1)

tornado_df['INJURIES_CLASS'] = tornado_df['INJURIES_DIRECT'].apply(lambda x: 0 if x == 0 else (1 if 1 <= x <= 5 else 2))


# Tornado Injuries Prediction Dataset
df_injuries = tornado_df[['BEGIN_DAY', 'BEGIN_TIME', 'STATE',  'YEAR',
       'MONTH_NAME', 'CZ_TYPE', 'CZ_NAME', 'WFO',
       'INJURIES_CLASS', 
  #   'DAMAGE_PROPERTY', 'DAMAGE_CROPS', 
       'SOURCE',
  #     'TOR_F_SCALE', 
       'TOR_LENGTH', 'TOR_WIDTH', 
       'BEGIN_LAT', 'BEGIN_LON', 'END_LAT', 'END_LON', 'TIME_ELAPSED',
       'VERTICAL_MOVEMENT', 'HORIZONTAL_MOVEMENT']]


# drop NA values
df_injuries.dropna(inplace=True)


# Output column for prediction
output = df_injuries['INJURIES_CLASS'] 

# df_injuries = df_injuries.drop('INJURIES_CLASS', axis=1)


X = df_injuries.drop(columns = ['INJURIES_CLASS'])
y = df_injuries['INJURIES_CLASS']

# encode categorical variables
cat_var = ['STATE', 'MONTH_NAME', 'CZ_TYPE', 'CZ_NAME', 'WFO','SOURCE']#, 'TOR_F_SCALE']
X_encoded = pd.get_dummies(X, columns = cat_var)

# split train/test dataset
train_X, test_X, train_y, test_y = train_test_split(X_encoded, y, test_size = 0.3, random_state = 1)


### Creating Model Metrics Dataframe ###
pred1 = clf_dt.predict(test_X)
pred2 = clf_xg.predict(test_X)

# Calculate metrics
precision_1 = precision_score(test_y, pred1, average='weighted')
recall_1 = recall_score(test_y, pred1, average='weighted')
f1_score_1 = f1_score(test_y, pred1, average='weighted')

precision_2 = precision_score(test_y, pred2, average='weighted')
recall_2 = recall_score(test_y, pred2, average='weighted')
f1_score_2 = f1_score(test_y, pred2, average='weighted')

# Create dataframe
data = {
    'ML Model': ['Decision Tree', 'XGBoost'],
    'precision': [precision_1, precision_2],
    'recall': [recall_1, recall_2],
    'f1-score': [f1_score_1, f1_score_2]
}

models_df = pd.DataFrame(data)

# Identify the row with the lowest and highest R^2 values
min_row = models_df[models_df['precision'] == models_df['precision'].min()]
max_row = models_df[models_df['precision'] == models_df['precision'].max()]

# Create a style DataFrame with background colors for the entire row
models_df = models_df.style.apply(lambda row: ['background-color: orange' if row.name in min_row.index else '' for col in row], axis=1)
models_df = models_df.apply(lambda row: ['background-color: green' if row.name in max_row.index else '' for col in row], axis=1)


st.write('To ensure optimal results, please ensure that your data strictly adheres to the specified format outlined below:')
st.dataframe(X.head())



### Ask for user CSV or inputs ###

# Option 1: Asking users to input their data as a file
tornado_file = st.file_uploader('Upload your own tornado data')


### Form for Users ###
if tornado_file is None:
    # Asking users to input their data using a form
    with st.form('user_inputs'): 

        # All user inputs
        begin_day = st.number_input('Start day of tornado', min_value=0.0, max_value=31.0)

        begin_time = st.number_input('Start time of tornado (such that 11:58 pm is entered as 2358)', min_value=0.0, max_value=2359.0)

        state = st.selectbox('Select the state', options=df_injuries["STATE"].unique())

        year = st.selectbox('Select the year', options=df_injuries["YEAR"].unique())

        month_name = st.selectbox('Select the month', options=df_injuries["MONTH_NAME"].unique())

        cz_type = st.selectbox('Indicates whether the event happened in a (C) County/Parish, (Z) NWS Public Forecast Zone or (M) Marine', options=df_injuries["CZ_TYPE"].unique())

        cz_name = st.selectbox('Select the name of the county', options=df_injuries["CZ_NAME"].unique())

        wfo = st.selectbox('Select the National Weather Service Forecast Office’s area of responsibility (County Warning Area) in which the event occurred.', options=df_injuries["WFO"].unique())

        source = st.selectbox('Select the source of the tornado report', options=df_injuries["SOURCE"].unique())

        tor_length = st.number_input('Length of tornado while on the ground in miles', min_value=0.0)

        tor_width = st.number_input('Width of tornado while on the ground in feet', min_value=0.0)

        begin_lat = st.number_input('Starting latitude', min_value=0.0000, max_value=90.0000) 

        begin_lon = st.number_input('Starting longitude', min_value=-180.0000, max_value=0.0000)

        end_lat = st.number_input('Ending latitude', min_value=0.0000, max_value=90.0000) 

        end_lon = st.number_input('Ending longitude', min_value=-180.0000, max_value=0.0000)

        time_elapsed = st.number_input('Elapsed time of tornado', min_value=0.0)

        model = st.selectbox('Select Machine Learning Model for Prediction', options=['Decision Tree', 'XGBoost']) 
  
        # Print Model Metrics Dataframe
        st.dataframe(models_df)

        # Submit button
        st.form_submit_button() 

    ### Dummy Variables for Categorical ###
    if model == 'Decision Tree': 
        model_selection = clf_dt 
    elif model == 'XGBoost': 
        model_selection = clf_xg

    original = X_encoded.copy()
    user_encoded_df = original.tail(1)


    ### Prediction for User Input ###
    # Using predict() with new data provided by the user
    new_prediction = model_selection.predict(user_encoded_df) 
    prediction_class = unique_mappings[new_prediction][0]

    new_prediction_prob = model_selection.predict_proba(user_encoded_df)

    # Show the predicted traffic volume on the app
    st.write('{} Prediction: We predict your tornado will have {} with {:.0%} probability.'.format(model, prediction_class, new_prediction_prob.max())) 

else:
    ### Upload a file ###
    with st.form('user_model_inputs'):
        model = st.selectbox('Select Machine Learning Model for Prediction', options=['Decision Tree', 'XGBoost']) 
  
        # Print Model Metrics Dataframe
        st.dataframe(models_df)

        # Submit button
        st.form_submit_button() 

    ### Dummy Variables for Categorical ###
    if model == 'Decision Tree': 
        model_selection = clf_dt 
    elif model == 'XGBoost': 
        model_selection = clf_xg

    # Loading data
    user_df = pd.read_csv(tornado_file) # User provided data
    original_df = df_injuries # Original data to create ML model

    # Dropping null values
    user_df = user_df.dropna() 
    original_df = original_df.dropna() 

    # Remove output (species) and year columns from original data
    original_df = original_df.drop(columns = ['INJURIES_CLASS'])

    # Ensure the order of columns in user data is in the same order as that of original data
    user_df = user_df[original_df.columns]

    # Concatenate two dataframes together along rows (axis = 0)
    combined_df = pd.concat([original_df, user_df], axis = 0)

    # Number of rows in original dataframe
    original_rows = original_df.shape[0]

    # Create dummies for the combined dataframe
    combined_df_encoded = pd.get_dummies(combined_df)

    # Split data into original and user dataframes using row index
    original_df_encoded = combined_df_encoded[:original_rows]
    user_df_encoded = combined_df_encoded[original_rows:]

    # Predictions for user data
    user_pred = model_selection.predict(user_df_encoded)

    # Predicted class
    user_pred_class = unique_mappings[user_pred]

    # Adding predicted species to user dataframe
    user_df['Predicted Injuries'] = user_pred_class

    # Prediction Probabilities
    user_pred_prob = model_selection.predict_proba(user_df_encoded)
    # Storing the maximum prob. (prob. of predicted species) in a new column
    user_df['Predicted Injuries Probability'] = user_pred_prob.max(axis = 1)

    # Apply the styling to the DataFrame
    result = user_df.style.applymap(color_cells, subset=['Predicted Injuries'])


    # Show the predicted species on the app
    st.subheader("Predicting Your Tornado's Injuries")
    st.dataframe(result)



# Showing additional items
st.subheader("Prediction Performance")
tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

if model == 'Decision Tree': 
   model_prefix = 'dt'
elif model == 'XGBoost': 
   model_prefix = 'xg'

with tab1:
  st.image(model_prefix + '_feature_imp.png')
with tab2:
  st.image(model_prefix + '_class_matrix.png')
with tab3:
    df = pd.read_csv(model_prefix + '_class_report.csv', index_col=0)
    st.dataframe(df)

