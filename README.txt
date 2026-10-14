TÜMOSAN SMARTFUEL — HOW TO RUN (WINDOWS)

1. Extract the entire ZIP into a folder, e.g. Desktop\Tumosan_SmartFuel_App.
2. Open Command Prompt (press Windows key, type cmd, press Enter).
3. Install requirements (only once):
   py -m pip install -r "%USERPROFILE%\Desktop\Tumosan_SmartFuel_App\requirements.txt"
   (If your folder is elsewhere, replace that path.)
4. Go to the extracted folder:
   cd /d "%USERPROFILE%\Desktop\Tumosan_SmartFuel_App"
5. Start the app:
   py -m streamlit run app.py
6. The application opens in your browser, usually at http://localhost:8501
7. Stop the app by pressing Ctrl+C in Command Prompt.

The trained model, metadata, predictions, and original dataset are included.
IMPORTANT: Fuel rate unit L/h and vehicle speed unit km/h should be verified against original dataset documentation.
The trained model requires compatible versions of scikit-learn; if you see a model version warning, recreate the model using the supplied modelling notebook in your environment.
