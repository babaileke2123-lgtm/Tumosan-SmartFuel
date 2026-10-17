from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title='FuelPulse AI | Tractor Fuel Analytics', page_icon='💚', layout='wide')
BASE = Path(__file__).resolve().parent
FEATURES = ['EngineSpeed', 'ActualEngine_PercTorque', 'WheelBasedVehicleSpeed']
TARGET = 'EngFuelRate'

@st.cache_resource
def load_model():
    return joblib.load(BASE / 'tractor_fuel_model.joblib')

@st.cache_data
def load_data():
    p = BASE / 'FIELD_1_TUMOSAN_81110P_TILLAGE_OP_RAW.tab'
    return pd.read_csv(p, sep='\t').sort_values('time_ms').reset_index(drop=True)

model = load_model()
metadata = json.loads((BASE / 'model_metadata.json').read_text(encoding='utf-8'))
# FuelPulse AI visual system. The model and recorded data are unchanged.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');
:root { --fp-forest:#153D32; --fp-emerald:#218B63; --fp-cream:#F5F8F5; --fp-text:#25362F; }
html, body, [class*="css"], [data-testid="stApp"], [data-testid="stSidebar"],
button, input, textarea, select, label, p, span, div { font-family:'Montserrat',sans-serif; }
[data-testid="stAppViewContainer"] { background:linear-gradient(180deg,#F5F8F5 0%,#FFFFFF 45%); }
[data-testid="stSidebar"] { background:#EEF5F0; border-right:1px solid #DDE8E0; }
[data-testid="stSidebar"] * { font-family:'Montserrat',sans-serif; }
.block-container { padding-top:1.5rem; padding-bottom:3rem; max-width:1320px; }
h1,h2,h3,h4 { color:#153D32 !important; font-family:'Montserrat',sans-serif !important; letter-spacing:-.025em; }
h1 { font-weight:800 !important; } h2,h3 { font-weight:700 !important; }
[data-testid="stMetric"] { background:#FFFFFF; border:1px solid #DDE8E0; border-radius:16px; padding:18px 20px; box-shadow:0 5px 20px rgba(21,61,50,.04); }
[data-testid="stMetricLabel"] { color:#546C5F; font-weight:600; }
[data-testid="stMetricValue"] { color:#153D32; font-weight:800; }
.stButton > button[kind="primary"] { background:#218B63; border-color:#218B63; color:white; font-weight:700; border-radius:11px; }
.stButton > button { border-radius:11px; font-weight:600; }
[data-testid="stAlert"] { border-radius:12px; }
[data-testid="stPlotlyChart"] { background:white; border:1px solid #E4ECE6; border-radius:16px; padding:6px; }
.fp-hero { background:linear-gradient(115deg,#153D32,#22694E); padding:29px 32px; border-radius:20px; margin-bottom:18px; color:#FFFFFF; }
.fp-kicker { color:#B5E5C7; text-transform:uppercase; letter-spacing:.14em; font-size:.70rem; font-weight:700; margin:0 0 8px; }
.fp-title { font-size:clamp(2rem,4vw,3.15rem); font-weight:800; letter-spacing:-.04em; line-height:1.1; margin:0; color:white; }
.fp-subtitle { font-size:.97rem; color:#D8EDE0; margin:13px 0 0; line-height:1.65; }
.fp-pill { display:inline-block; border:1px solid #79BA97; color:#D7F5E4; padding:6px 11px; border-radius:99px; margin-top:17px; font-size:.72rem; font-weight:600; }
.fp-note { color:#62786B; font-size:.78rem; margin:0 0 20px; line-height:1.7; }
@media(max-width:700px){ .fp-hero{padding:22px 20px;} .block-container{padding-left:1rem;padding-right:1rem;} }
</style>
""", unsafe_allow_html=True)

# Consistent Plotly chart palette (no changes to chart data).
px.defaults.template = 'plotly_white'
px.defaults.color_discrete_sequence = ['#218B63', '#E4A647', '#3576A8', '#7D6AAF']

st.markdown("""
<div class="fp-hero">
  <div class="fp-kicker">Agricultural machinery intelligence</div>
  <div class="fp-title">◉ FuelPulse AI</div>
  <p class="fp-subtitle">Intelligent tractor fuel prediction, analytics &amp; recorded-data playback.</p>
  <span class="fp-pill">Machine learning powered · Research demonstration</span>
</div>
<div class="fp-note">Model developed using Tümosan 81.110 soil-tillage CAN-bus data. Documented units: fuel rate L/h (SPN 183), vehicle speed km/h (SPN 84), engine speed rpm (SPN 190), and torque %.</div>
""", unsafe_allow_html=True)
st.sidebar.markdown('### ◉ FuelPulse AI')
st.sidebar.caption('Tractor fuel analytics')
page = st.sidebar.radio('Navigate', ['Manual Fuel Prediction', 'Interactive Dashboard', 'Live Playback Monitor', 'Model & Methodology'])

if page == 'Manual Fuel Prediction':
    st.subheader('Predict engine fuel consumption rate')
    st.write('Adjust the operating conditions below. The prediction is an estimate from recorded data, not a controlled experiment.')
    a,b,c = st.columns(3)
    with a: rpm = st.number_input('Engine speed (rpm)', min_value=0.0, max_value=3000.0, value=1550.0, step=50.0)
    with b: torque = st.number_input('Engine torque (% of reference)', min_value=0.0, max_value=110.0, value=35.0, step=1.0)
    with c: speed = st.number_input('Vehicle speed (km/h)', min_value=0.0, max_value=20.0, value=5.0, step=0.5)
    hours = st.number_input('Operating duration (hours)', min_value=0.1, max_value=100.0, value=2.0, step=0.5)
    input_df = pd.DataFrame([[rpm, torque, speed]], columns=FEATURES)
    if st.button('Predict fuel rate', type='primary'):
        limits = metadata['trainval_feature_ranges']
        outside = [f'{f}: {input_df.iloc[0][f]:g} (trained range {v["min"]:g}–{v["max"]:g})' for f,v in limits.items() if not v['min'] <= input_df.iloc[0][f] <= v['max']]
        if outside:
            st.warning('Input outside observed training/validation range. Prediction may be unreliable: ' + '; '.join(outside))
        pred = max(0.0, float(model.predict(input_df)[0]))
        x,y = st.columns(2)
        x.metric('Estimated fuel rate', f'{pred:.2f} L/h')
        y.metric('Estimated fuel for duration', f'{pred*hours:.2f} L')
        st.caption('Total fuel estimate assumes a constant predicted fuel rate throughout the chosen duration.')
    st.divider()
    st.subheader('Explore engine-speed scenarios')
    rpm_values = np.linspace(600, 2400, 30)
    scenarios = pd.DataFrame({'EngineSpeed':rpm_values,'ActualEngine_PercTorque':torque,'WheelBasedVehicleSpeed':speed})
    scenarios['Predicted fuel rate'] = np.maximum(0,model.predict(scenarios))
    fig = px.line(scenarios, x='EngineSpeed', y='Predicted fuel rate', labels={'EngineSpeed':'Engine speed (rpm)','Predicted fuel rate':'Predicted fuel rate (L/h)'})
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Scenario curve holds torque and travel speed fixed; changing one parameter does not prove a causal fuel saving.')

elif page == 'Interactive Dashboard':
    try:
        df = load_data()
    except FileNotFoundError:
        st.error('Raw .tab dataset missing. Place FIELD_1_TUMOSAN_81110P_TILLAGE_OP_RAW.tab in the same folder as app.py.')
        st.stop()
    st.subheader('Tillage telemetry overview')
    m1,m2,m3,m4 = st.columns(4)
    m1.metric('Observations', f'{len(df):,}')
    m2.metric('Recorded duration', f'{(df.time_ms.max()-df.time_ms.min())/60000:.1f} min')
    m3.metric('Mean fuel rate', f'{df[TARGET].mean():.2f} L/h')
    m4.metric('Maximum fuel rate', f'{df[TARGET].max():.2f} L/h')
    st.caption('Recorded engine fuel rate is in L/h (dataset signal documentation, SPN 183).')
    units = {'EngFuelRate':'L/h','EngineSpeed':'rpm','ActualEngine_PercTorque':'%','WheelBasedVehicleSpeed':'km/h','EngOilTemp1':'°C','EngCoolantTemp':'°C','EngOilPress':'kPa'}
    columns = [TARGET]+FEATURES+['EngOilTemp1','EngCoolantTemp','EngOilPress']
    chosen = st.selectbox('Variable to explore', columns)
    df['Elapsed time (min)'] = (df.time_ms-df.time_ms.iloc[0])/60000
    time_max = float(df['Elapsed time (min)'].max())
    start,end = st.slider('Operating time window (minutes)',0.0,time_max,(0.0,time_max),step=0.5)
    sample = df.loc[df['Elapsed time (min)'].between(start,end)]
    if sample.empty:
        st.warning('No observations in this interval.')
    else:
        st.plotly_chart(px.line(sample.iloc[::max(1,len(sample)//2500)], x='Elapsed time (min)', y=chosen, title=f'{chosen} over time', labels={chosen:f'{chosen} ({units[chosen]})'}), use_container_width=True)
        left,right = st.columns(2)
        with left:
            st.plotly_chart(px.histogram(sample, x=chosen, nbins=50, title='Distribution', labels={chosen:f'{chosen} ({units[chosen]})'}),use_container_width=True)
        with right:
            st.plotly_chart(px.scatter(sample.iloc[::max(1,len(sample)//2000)],x=chosen if chosen!=TARGET else 'EngineSpeed',y=TARGET,opacity=0.4,title='Relationship with fuel rate', labels={TARGET:'Engine fuel rate (L/h)',chosen:f'{chosen} ({units[chosen]})'}),use_container_width=True)
        st.subheader('Descriptive statistics')
        st.dataframe(sample[columns].describe().T.round(3),use_container_width=True)
        st.subheader('Correlation matrix')
        st.dataframe(sample[columns].corr().round(3),use_container_width=True)
    st.download_button('Download filtered observations (CSV)', data=sample.to_csv(index=False).encode('utf-8'),file_name='tractor_filtered_data.csv',mime='text/csv')

elif page == 'Model & Methodology':
    st.subheader('Model evaluation and methodology')
    st.write('Chronological split: 70% training, 15% validation, 15% testing. Candidate algorithms included mean baseline, Random Forest, ANN and Histogram Gradient Boosting. Model selection used validation RMSE; final scores below are from the held-out test period.')
    st.write('**Selected model:**',metadata['selected_model'])
    st.write('**Inputs:**',', '.join(metadata['features']))
    st.write('**Target:**',metadata['target'])
    st.dataframe(pd.DataFrame([metadata['final_test_metrics']]).round(4),hide_index=True,use_container_width=True)
    pred_path = BASE/'test_predictions.csv'
    if pred_path.exists():
        preds = pd.read_csv(pred_path)
        st.subheader('Test predictions')
        st.dataframe(preds.head(25),use_container_width=True)
        st.download_button('Download test predictions',preds.to_csv(index=False),file_name='test_predictions.csv',mime='text/csv')
    st.warning('Results describe a later portion of the same recording. They do not validate performance on new tractors, fields or seasons. The model is not a causal simulator.')
    st.info('Dataset credit: Tümosan 81.110 tractor soil-tillage CAN-bus dataset, shared by Ali Can Tellioğlu and research collaborators. Add the full publication citation once verified. FuelPulse AI is an independent student-developed research application, not an official Tümosan product.')


else:
    st.subheader('Live Playback Monitor — recorded tractor data')
    st.info('REPLAY MODE: Historical CAN-bus measurements, not a live connection to a tractor. The saved model generates predictions for the held-out test period.')
    st.caption('The test period was not used for model training or selection. Replay starts at its first observation (approximately 68.35 minutes into the original recording).')

    @st.cache_data(show_spinner='Preparing held-out test predictions…')
    def playback_data():
        raw = load_data()
        # The held-out test segment begins after the original 70% + 15% chronological split.
        test_start = int(metadata['train_rows']) + int(metadata['validation_rows'])
        test = raw.iloc[test_start:].copy().reset_index(drop=True)
        test['Predicted fuel rate (L/h)'] = np.maximum(0.0, model.predict(test[FEATURES]))
        test['Actual fuel rate (L/h)'] = test[TARGET]
        test['Elapsed time (min)'] = (test['time_ms'] - raw['time_ms'].iloc[0]) / 60000.0
        dt_hours = test['time_ms'].diff().fillna(0).clip(lower=0) / 3600000.0
        test['Estimated fuel used (L)'] = (test['Predicted fuel rate (L/h)'].shift(1).fillna(test['Predicted fuel rate (L/h)']) * dt_hours).cumsum()
        test['Recorded-rate fuel estimate (L)'] = (test['Actual fuel rate (L/h)'].shift(1).fillna(test['Actual fuel rate (L/h)']) * dt_hours).cumsum()
        return test

    test = playback_data()
    if 'playback_running' not in st.session_state:
        st.session_state.playback_running = False
    if 'playback_index' not in st.session_state:
        st.session_state.playback_index = 0
    st.session_state.playback_index = min(st.session_state.playback_index, len(test)-1)
    speed = st.select_slider('Recorded time advanced per update (every ~1 second)', options=[1, 5, 10], value=5, format_func=lambda x: f'{x} recorded seconds')
    c1, c2, c3 = st.columns(3)
    if c1.button('▶ Start / Resume', use_container_width=True):
        st.session_state.playback_running = True
    if c2.button('⏸ Pause', use_container_width=True):
        st.session_state.playback_running = False
    if c3.button('↺ Restart', use_container_width=True):
        st.session_state.playback_running = False
        st.session_state.playback_index = 0
    st.caption('Playback is accelerated: each second of real time advances 1, 5, or 10 seconds of historical tractor data. Pause to inspect any point.')

    @st.fragment(run_every='1s')
    def playback_panel():
        if st.session_state.playback_running:
            # Original measurements are sampled at 0.1-second intervals.
            st.session_state.playback_index = min(len(test)-1, st.session_state.playback_index + speed*10)
            if st.session_state.playback_index >= len(test)-1:
                st.session_state.playback_running = False
        i = st.session_state.playback_index
        current = test.iloc[i]
        st.progress((i+1)/len(test), text=f'Record {i+1:,} / {len(test):,}  •  Elapsed tractor time {current["Elapsed time (min)"]:.2f} min  •  {"Playing" if st.session_state.playback_running else "Paused"}')
        a,b,c,d = st.columns(4)
        a.metric('Engine speed', f'{current["EngineSpeed"]:,.0f} rpm')
        b.metric('Engine torque', f'{current["ActualEngine_PercTorque"]:.1f} %')
        c.metric('Vehicle speed', f'{current["WheelBasedVehicleSpeed"]:.2f} km/h')
        d.metric('Predicted fuel rate', f'{current["Predicted fuel rate (L/h)"]:.2f} L/h')
        a,b,c = st.columns(3)
        a.metric('Recorded fuel rate', f'{current["Actual fuel rate (L/h)"]:.2f} L/h')
        b.metric('Fuel-rate error (actual − predicted)', f'{current["Actual fuel rate (L/h)"]-current["Predicted fuel rate (L/h)"]:+.2f} L/h')
        c.metric('Estimated fuel since replay start', f'{current["Estimated fuel used (L)"]:.3f} L')
        # Plot recent history; restrict point count to keep the web UI responsive.
        visible = test.iloc[max(0,i-1199):i+1: max(1,(i-max(0,i-1199))//600)]
        chart = px.line(visible, x='Elapsed time (min)', y=['Actual fuel rate (L/h)','Predicted fuel rate (L/h)'], labels={'value':'Fuel rate (L/h)','variable':'Series'}, title='Recorded versus predicted engine fuel rate')
        chart.update_layout(legend_title_text='Series')
        st.plotly_chart(chart, use_container_width=True)
        pred_chart = px.line(visible, x='Elapsed time (min)', y='Predicted fuel rate (L/h)', title='Predicted fuel rate over time', labels={'Predicted fuel rate (L/h)':'Predicted fuel rate (L/h)'})
        st.plotly_chart(pred_chart, use_container_width=True)
        with st.expander('Current observation and source data'):
            st.dataframe(test.iloc[[i]][['time_ms']+FEATURES+[TARGET,'Predicted fuel rate (L/h)']],hide_index=True,use_container_width=True)
        st.caption('Fuel-volume estimate integrates predicted fuel rate over the recorded timestamps. It is not a measured fuel-volume reading.')
    playback_panel()
