from flask import Flask, render_template, request, jsonify
from usemodel import predict_traffic
import pandas as pd

app = Flask(__name__) #create a flask app, where __name__ pointing on a current python file

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        #all values in request.html are strigs -> sometimes we need int() or float()
        #after pushing the button of type "submit" in index.html we get post-request from <form method="POST">
        #form is sended by html automaticly to url, on which the page is loaded (url of this flask app)
        #in .html on title of form we can specify the endpoint of our flaskapp to where send the request 
        # like action="/", using request.form[] we get value by a field "name" in an obtained form
        weather_data = {
            "hour": int(request.form["hour"]),
            "temp_c": float(request.form["temp_c"]),
            "will_it_rain": int(request.form["will_it_rain"]),
            "humidity": int(request.form["humidity"]),
            "wind_kph": float(request.form["wind_kph"]),
            "cloud": int(request.form["cloud"]),
            "condition": request.form["condition"],
        } #with help of request.form take information about all parameteres obtained in body of post-request 
        #(written ar index.html )
        
        #predict traffic density
        traffic_load = predict_traffic(weather_data)
        
        #after calculation a prediction according to weather_data render_template find file with specified name
        #(index.html) at folder "templates" and finds there placeholder {{prediction}} to paste there a calculated
        #value after applying this function to html file the page reloading with new prediction 
        return render_template("index.html", prediction=traffic_load)
    
    #when client opens our site for the first time browser automatically sends get request on root endpoint /
    #of our flask server and then this command performs taking templates/index.html and giving it to browser
    return render_template("index.html", prediction=None) #if request method not post (get)

if __name__ == "__main__": #for avoiding run during running file, where current file is imported
    app.run(debug=True) #debug - autoreload after changes and more detailed showing of errors