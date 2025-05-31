I used at this project version of python 3.12.6 (for better compatibility install )
1. You can create virtual environment in derictory of the project (if you don't see already folder "venv") before 
with "python -m venv venv"
To activate virtual environment enter "venv\Scripts\activate" (for cmd ), 
After activation you can install necessary dependencies for correct working of app with command
"pip install -r requirementes.txt" 
2. to run the server go into directory of this project at cmd/powershell or bash and input a command
"python flaskapp.py" 
3. TO quit virtual environment enter in cmd "deactivate"
4. You can make your server available on the internet istalling ngrok. 
if ngrock is installed, you can run the flask app (python flaskapp.py) and after enter "ngrok http http://127.0.0.1:5000"
 (after http must be address where yor server is runnin) and you'll get internet link of your site from ngrock