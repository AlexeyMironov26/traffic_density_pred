from tensorflow.keras.models import load_model
import pickle
import pandas as pd
from createmodel import X_test, y_test, right_preds_percent

#import scaler created on training data with min/max values for every column
with open("scaler.pkl", "rb") as f:  #rb -reading in binary mode
    scaler = pickle.load(f)

#import created model
model = load_model('road_load_model.keras')

def predict_traffic(weather_data):

    WEATHER_CONDITIONS = [
        'Ясно',
        'Облачно',
        'Временами сильный дождь',
        'Умеренный или сильный ливневый дождь',
        'Временами умеренный дождь',
        'Пасмурно',
        'Переменная облачность',
        'Местами дождь',
        'Солнечно'
    ] #right order of wether conditions
    
    
    weather_encoding = {
        condition: [1 if weather_data['condition'] == condition else 0]
        for condition in WEATHER_CONDITIONS
    } #fill our dict of weather conditions obtained in weather_data from our client's request
    #if name of condition obtained from client (weather data is at flaskapp.py) is matches with one of our condition
    #from weather_conditions, then set 1 in front of that condition in our dict, in front of all the rest conditions
    #  in our dict set 0
    #condition:[1] - we need [] for correct creating of dataframe table by pd because those are categorical data
    # (for letting pandas to determine the length of columns) 

    #gather all parameters in the right order taking the info from weather_data (weather data is at flaskapp.py)
    input_data = {
        'hour': [weather_data['hour']],
        'temp_c': [weather_data['temp_c']],
        'will_it_rain': [weather_data['will_it_rain']],
        'humidity': [weather_data['humidity']],
        'wind_kph': [weather_data['wind_kph']],
        'cloud': [weather_data['cloud']],
        **weather_encoding  #unpack dict - condition:[0/1]
    }
    
    #create general right order of all input parameters for further specifying it as [columns_order] after
    #pd.DataFrame to create a table exactly with the same order, which was during learning of model
    #(extremely important for getting a proper prediction)
    columns_order = [ 
        'hour', 'temp_c', 'will_it_rain', 'humidity', 'wind_kph', 'cloud',
        *WEATHER_CONDITIONS  #unpack the right order of conditions
    ]

    X = pd.DataFrame(input_data)[columns_order]#must be 2d format
#and order of columns in data should be the same, which was on stage of training
    
    X_scaled = scaler.transform(X)#transform obtained data (1 string in table) with help 
#of min/max values for every column
    prediction = model.predict(X_scaled).flatten()[0]
    return round(prediction, 2) #round prediction for 2 signs after ,




#final estimating of model on test set of data and showing average erorrs of all values of set
loss, mae = model.evaluate(X_test, y_test)#testing neuronet on test data
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}") #mae shows an average amount of percent on which neuronet make
#a mistake while trying to predict the load of mkad road by parameteres of weather and time, that it gets

print(right_preds_percent(model, X_test, 10)) #testing on percentage of right predictions on dataset