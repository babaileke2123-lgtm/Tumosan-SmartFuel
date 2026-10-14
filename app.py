from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title='Tümosan SmartFuel', page_icon='🚜', layout='wide')
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
st.title('🚜 Tümosan SmartFuel')
st.caption('Fuel consumption prediction and monitoring • Tümosan 81.110 soil tillage CAN-bus data')
st.info('Fuel-rate units are provisionally L/h, based on the typical J1939 engine fuel-rate parameter. Confirm the dataset documentation before presenting the units as verified.')
page = st.sidebar.radio('Navigate', ['Manual Fuel Prediction', 'Interactive Dashboard', 'Model & Methodology'])

if page == 'Manual Fuel Prediction':
    st.subheader('Predict engine fuel consumption rate')
    st.write('Adjust the operating conditions below. The prediction is an estimate from recorded data, not a controlled experiment.')
    a,b,c = st.columns(3)
    with a: rpm = st.number_input('Engine speed (rpm)', min_value=0.0, max_value=3000.0, value=1550.0, step=50.0)
    with b: torque = st.number_input('Engine torque (% of reference)', min_value=0.0, max_value=110.0, value=35.0, step=1.0)
    with c: speed = st.number_input('Vehicle speed (dataset units; verify km/h)', min_value=0.0, max_value=20.0, value=5.0, step=0.5)
    hours = st.number_input('Operating duration (hours)', min_value=0.1, max_value=100.0, value=2.0, step=0.5)
    input_df = pd.DataFrame([[rpm, torque, speed]], columns=FEATURES)
    if st.button('Predict fuel rate', type='primary'):
        limits = metadata['trainval_feature_ranges']
        outside = [f'{f}: {input_df.iloc[0][f]:g} (trained range {v["min"]:g}–{v["max"]:g})' for f,v in limits.items() if not v['min'] <= input_df.iloc[0][f] <= v['max']]
        if outside:
            st.warning('Input outside observed training/validation range. Prediction may be unreliable: ' + '; '.join(outside))
        pred = max(0.0, float(model.predict(input_df)[0]))
        x,y = st.columns(2)
        x.metric('Estimated fuel rate', f'{pred:.2f} L/h*')
        y.metric('Estimated fuel for duration', f'{pred*hours:.2f} L*')
        st.caption('*Unit provisional. Total fuel assumes a constant predicted rate throughout the chosen duration.')
    st.divider()
    st.subheader('Explore engine-speed scenarios')
    rpm_values = np.linspace(600, 2400, 30)
    scenarios = pd.DataFrame({'EngineSpeed':rpm_values,'ActualEngine_PercTorque':torque,'WheelBasedVehicleSpeed':speed})
    scenarios['Predicted fuel rate'] = np.maximum(0,model.predict(scenarios))
    fig = px.line(scenarios, x='EngineSpeed', y='Predicted fuel rate', labels={'EngineSpeed':'Engine speed (rpm)','Predicted fuel rate':'Predicted fuel rate (L/h, provisional)'})
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
    m3.metric('Mean fuel rate', f'{df[TARGET].mean():.2f} L/h*')
    m4.metric('Maximum fuel rate', f'{df[TARGET].max():.2f} L/h*')
    st.caption('*Fuel-rate unit provisional.')
    columns = [TARGET]+FEATURES+['EngOilTemp1','EngCoolantTemp','EngOilPress']
    chosen = st.selectbox('Variable to explore', columns)
    df['Elapsed time (min)'] = (df.time_ms-df.time_ms.iloc[0])/60000
    time_max = float(df['Elapsed time (min)'].max())
    start,end = st.slider('Operating time window (minutes)',0.0,time_max,(0.0,time_max),step=0.5)
    sample = df.loc[df['Elapsed time (min)'].between(start,end)]
    if sample.empty:
        st.warning('No observations in this interval.')
    else:
        st.plotly_chart(px.line(sample.iloc[::max(1,len(sample)//2500)], x='Elapsed time (min)', y=chosen, title=f'{chosen} over time'), use_container_width=True)
        left,right = st.columns(2)
        with left:
            st.plotly_chart(px.histogram(sample, x=chosen, nbins=50, title='Distribution'),use_container_width=True)
        with right:
            st.plotly_chart(px.scatter(sample.iloc[::max(1,len(sample)//2000)],x=chosen if chosen!=TARGET else 'EngineSpeed',y=TARGET,opacity=0.4,title='Relationship with fuel rate'),use_container_width=True)
        st.subheader('Descriptive statistics')
        st.dataframe(sample[columns].describe().T.round(3),use_container_width=True)
        st.subheader('Correlation matrix')
        st.dataframe(sample[columns].corr().round(3),use_container_width=True)
    st.download_button('Download filtered observations (CSV)', data=sample.to_csv(index=False).encode('utf-8'),file_name='tractor_filtered_data.csv',mime='text/csv')

else:
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
