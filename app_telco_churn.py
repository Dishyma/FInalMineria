import streamlit as st
import pandas as pd
import os
import pickle

st.set_page_config(page_title='Telco Churn', page_icon='📞')
st.title('📞 Telco Customer Churn')

@st.cache_resource
def load_pipeline():
    """Carga el pipeline desde rutas comunes del repo o permite subirlo."""
    candidate_paths = [
        'pipeline_telco_churn_final.pkl',
        'Final/pipeline_telco_churn_final.pkl',
        'Final/FInalMineria/pipeline_telco_churn_final.pkl'
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            with open(p, 'rb') as f:
                bundle = pickle.load(f)
            return bundle['pipeline']
    return None

pipeline = load_pipeline()
if pipeline is None:
    st.warning('No se encontró el archivo pipeline_telco_churn_final.pkl en el repo. Sube el .pkl:')
    up = st.file_uploader('Sube pipeline_telco_churn_final.pkl', type=['pkl'])
    if up is not None:
        bundle = pickle.load(up)
        pipeline = bundle['pipeline']

if pipeline is None:
    st.stop()

st.subheader('Entrada de datos')
modo = st.radio('Selecciona el modo de entrada', ['Formulario', 'CSV'])


def input_form():
    with st.form('form_telco'):
        col1, col2 = st.columns(2)
        with col1:
            senior = st.selectbox('SeniorCitizen', [0, 1], index=0)
            tenure = st.number_input('tenure', min_value=0, max_value=100, value=12)
            internet = st.selectbox('InternetService', ['DSL', 'Fiber optic', 'No'])
            contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
            paperless = st.selectbox('PaperlessBilling', ['Yes', 'No'])
        with col2:
            onsec = st.selectbox('OnlineSecurity', ['Yes', 'No', 'No internet service'])
            onback = st.selectbox('OnlineBackup', ['Yes', 'No', 'No internet service'])
            device = st.selectbox('DeviceProtection', ['Yes', 'No', 'No internet service'])
            tech = st.selectbox('TechSupport', ['Yes', 'No', 'No internet service'])
            pay = st.selectbox('PaymentMethod', [
                'Electronic check', 'Mailed check',
                'Bank transfer (automatic)', 'Credit card (automatic)'
            ])
        submitted = st.form_submit_button('Predecir')
    if submitted:
        df = pd.DataFrame([{
            'customerID': 'TEMP-0000',
            'gender': 'Female',
            'SeniorCitizen': senior,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': tenure,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': internet,
            'OnlineSecurity': onsec,
            'OnlineBackup': onback,
            'DeviceProtection': device,
            'TechSupport': tech,
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': pay,
            'MonthlyCharges': 0.0,
            'TotalCharges': '0.0'
        }])
        return df
    return None


def input_csv():
    file = st.file_uploader('Sube un CSV con el formato del dataset original', type=['csv'])
    if file is not None:
        try:
            df = pd.read_csv(file)
            st.write('Vista previa:', df.head())
            return df
        except Exception as e:
            st.error(f'Error leyendo CSV: {e}')
    return None


df_input = input_form() if modo == 'Formulario' else input_csv()
if df_input is not None:
    try:
        pred = pipeline.predict(df_input)
        st.subheader('Resultado')
        st.write('Predicciones:', pred)
        try:
            prob = pipeline.predict_proba(df_input)
            st.write('Probabilidades (No, Sí):')
            st.write(prob)
        except Exception:
            st.info('El modelo no expone predict_proba')

        # Descarga de resultados
        out = df_input.copy()
        out['Prediccion'] = pred
        st.download_button(
            'Descargar predicciones CSV', out.to_csv(index=False),
            file_name='predicciones_telco.csv', mime='text/csv'
        )
    except Exception as e:
        st.error(f'Error realizando la predicción: {e}')
