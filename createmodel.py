import json
import pandas as pd
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np

path_load = Path("resources") / "road_analysis_cutted-data" / "roadloadwholemonth.json"
path_weath = Path("resources") / "weather-data" / "cutted-weather-data-en.json"
#loading data about about road load
with open(path_load, 'r') as f:
    traffic_data = json.load(f) #get all info about json file

traffic_records = []
for entry in traffic_data['summaries']:
    date = entry['dateRangeName'].split()[-1]  # extract the day of August
    hour = entry['timeSetName'].split()[1] # extract an hour range (00-01)
    hour_start = int(hour.split('-')[0]) # take initial hour with leading nulls and turn it into number
    
    record = {
        'date': f"2024-08-{date.zfill(2)}", #date is string cause we will use it only for merging data from 2 jsons
        'hour': hour_start,
        'averageSampleSize': entry['averageSampleSize'], #take an average amount of transport on segment of MKAD
        'maxaveragesamplesize': max([x['averageSampleSize'] for x in traffic_data['summaries']]) #x - every element 
        #in list "summaries"
    }
    traffic_records.append(record) #add every formed record-dict (data for one hour) into the list

traffic_df = pd.DataFrame(traffic_records) #convert list of dicts in table, where keys in dicts - names of collumns
#and values of these keys - data in strings
traffic_df['load'] = traffic_df['averageSampleSize'] / traffic_df['maxaveragesamplesize'] *100
#add a new column - load, and fill every string of the column using mathcing data of other columns of the string

#weather datas
with open(path_weath, 'r') as f:
    weather_data = json.load(f)

weather_records = []
for day in weather_data['forecast']['forecastday']: #go through every elem-day in these parts
    for hour in day['hour']: #go through every elem-hour in field 'hour' of each day
        time = pd.to_datetime(hour['time']) #turn data from datetime field "time" to structured datetime in pandas
        record = {
            'date': time.strftime('%Y-%m-%d'), #to extract from that date in fromat we need
            'hour': time.hour, #and time as well (in format 0-23) to make it coincide with format from traffic_df
            'temp_c': hour['temp_c'],
            'will_it_rain': hour['will_it_rain'],
            'condition': hour['condition']['text'], #general text data about weather
            'humidity': hour['humidity'],
            'wind_kph': hour['wind_kph'],
            'cloud': hour['cloud']
        }
        weather_records.append(record)

weather_df = pd.DataFrame(weather_records) #as well turn into table of weather

#merge 2 tables (weather and load) into single one with date and hour paramethers 
df = pd.merge(traffic_df, weather_df, on=['date', 'hour'])


conditions = pd.get_dummies(df['condition']) #creates a table with unique column for every string-value in 
#column 'condition' from main table. if this text-weather-condition is indicated at a particular string at main table
#we set 1 in the according string of collumn of our weather-condition in the new table 'conditions', else - set 0
conditions = conditions.rename(columns={
    'Clear': 'Ясно',
    'Cloudy': 'Облачно', 
    'Heavy rain at times': 'Временами сильный дождь',
    'Moderate or heavy rain shower': 'Умеренный или сильный ливневый дождь',
    'Moderate rain at times': 'Временами умеренный дождь', 
    'Overcast': 'Пасмурно', 
    'Partly cloudy': 'Переменная облачность', 
    'Patchy rain possible': 'Местами дождь', 
    'Sunny': 'Солнечно'
})
df = pd.concat([df, conditions], axis=1) #unite two tables horizontally by columns (axis=1)
df.drop('condition', axis=1, ) # axis=1 tells we should delete the column with such name, not a string
#inplace=True make the func change the existing table instead of creating the new one

#preparing data for ml
features = ['hour', 'temp_c', 'will_it_rain', 'humidity', 'wind_kph', 'cloud'] + list(conditions.columns) 
#to a list also add the list of columns-conditons which quanity and names could be different in theory
X = df[features] #on what we will base our prediction
y = df['load'] #what we gonna to predict

# print(df.head(10), len(list(conditions.columns)), list(conditions.columns))
# print(df[['date', 'hour', 'load']]) #testing

#normalizing of X data - all number-values are turned into value from 0 to 1 
#to make neuronet work with them correctly not giving to ones paramethers more value than to others
#if they are measured in different metrics
scaler = MinMaxScaler() #create scaler object 
X_scaled = scaler.fit_transform(X) #for every value in column x_scaled=(x-x_min)/(x_max-x_min)
#after we change scaler object such a way that it would remmber our min/max data in every column for fruther use
#with help of fit() and after normalize all data using calculated min/max with help of transform()

pickle.dump(scaler, open("scaler.pkl", "wb"))
#save configured scaler with min/maxes for every column of data X in file scaler.pkl w - writing mode
#b - writing in binary format

#dividing data on training and test, random_state remmebers a random dividing for making the operation repeatable
#test_size gives what percent of whole data will become test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=40)

def right_preds_percent(Model, x_test, tlrance):
    #acceptable error in percentage
    tolerance = tlrance

    y_pred = Model.predict(x_test).flatten() #get the predictions of model on test data - X_test and turnes it into
    #usual array for further more handy work
    #flatten turns massive from 2d (which neuro returns cause have 1 output neuron) to 1d 
    y_true = y_test.values  #get real values for X_test

    correct_predictions = np.abs(y_pred - y_true) <= tolerance#find all y_pred values which difference in percentaпe
    #with y_true less or equal to tolerance, return array with True for appropriate predictions and False for others
    accuracy_percentage = np.mean(correct_predictions) * 100 #calculate an average value of array 
    #(True - 1, false - 0) and turnes this value into percent format

    return f"Percentage of right predicts of load (error ≤{tolerance}%): {accuracy_percentage:.2f}%"

if __name__ == "__main__":
    # creating and training model
    input_features = X_train.shape[1] #shape give us (amount of strings, amount of columns)
    #we take amount of columns in our table of dataset for training for further givinig to neuronet amount of params
#and after here input_shape=(input_features,) we convey to model size for every vector-object given on input
#submit every object as vector from x numbers
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_features,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(1)
    ])

    optim = Adam()
    model.compile(optimizer=optim, loss='mse', metrics=['mae'])
    
    #education of model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, 
                    validation_data=(X_test, y_test), verbose=1)
    
    #saving the model for further use
    model.save('road_load_model.keras')

    #final estimating of model on test set of data and showing average erorrs of all values of set
    loss, mae = model.evaluate(X_test, y_test)#testing neuronet on test data and get loss function and metrics
    # which reflect the erorr of calculated data compared with y_test using that for validation

    print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}") #mae shows an average amount of percent on which neuronet make
    #a mistake while trying to predict the load of mkad road by paramethes of weather and time, that it gets

    print(right_preds_percent(model,X_test,10))


    
