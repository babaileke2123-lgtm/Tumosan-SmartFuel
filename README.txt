TUMOSAN SMARTFUEL — LIVE PLAYBACK EDITION

Four sections: Manual Fuel Prediction, Interactive Dashboard, Live Playback Monitor, Model & Methodology.

Live Playback Monitor uses RECORDED historical test data (not a live tractor connection).
Press Start/Resume, Pause, or Restart. Select playback speed. The app replays the held-out test period and shows measured and model-predicted fuel rates.

Run locally:
python -m pip install -r requirements.txt
python -m streamlit run app.py

To update the existing online app: replace app.py, requirements.txt and model_metadata.json in your existing GitHub repository. The existing raw .tab dataset and model file must remain in the same repository folder.

Units based on supplied tractor CAN-bus signal table: fuel rate L/h, engine speed rpm, torque %, wheel speed km/h.

Dataset redistribution: the researcher permitted academic use and requested citation; confirm whether that covers public redistribution of the raw file and provide appropriate attribution.
