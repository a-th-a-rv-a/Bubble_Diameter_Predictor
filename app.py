import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool

# Load your trained model
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model('catboost_model.cbm')
    return model

model = load_model()

# Function for prediction
def predict_bubble_diameter(input_data):
    # Reorder the input features to match the model's expected order
    feature_order = [
        'System', 'Density of gas', 'Viscosity of gas', 'Particle density',
        'Particle diameter', 'Geldart Particle Type', 'Column Diameter', 'Column Height', 'Packing Height', 'Absolute Pressure', 'Superficial Gas Velocity',
        'Minimum fluidization velocity', 'Axial Location'
    ]
    

# 'System', 'Density of gas', 'Viscosity of gas', 'Particle density', 'Particle diameter', 
# 'Geldart Particle Type', 'Column Diameter', 'Column Height', 'Packing Height', 'Absolute Pressure', 'Superficial Gas Velocity', 'Minimum fluidization velocity', 'Axial Location'

    # Reorder the input data to match the model's expected order
    input_df = pd.DataFrame([input_data])[feature_order]
    
    # Assuming the model takes the same features as in training (except target)
    cat_features = ['System', 'Geldart Particle Type']
    input_pool = Pool(input_df, cat_features=cat_features)
    # X_input = input_df
    
    # Predict using the model
    prediction = model.predict(input_pool)
    return prediction[0]

# Streamlit app UI
st.title('Bubble Diameter Prediction')

st.write("""
This app predicts the bubble diameter in fluidization systems based on various features.
""")

# Input form for user to enter values for prediction
st.sidebar.header('Input Features')

system = st.sidebar.selectbox('System', options=['Air-Aluminium oxide', 'Air-Coal', 'Air-FCC catalyst', 'Air-Glass',
    'Air-Ilmenite', 'Air-Magnetite', 'Air-Polyethylene', 'Air-Quartz',
    'Air-Sand', 'Air-Silica sand', 'Air-Solid'])  # replace with actual categories
geldart_particle_type = st.sidebar.selectbox('Geldart Particle Type', options=['A', 'B', 'A/B', 'D'])  # replace with actual categories
density_of_gas = st.sidebar.number_input('Density of Gas (kg/m^3)', min_value=0.0)
viscosity_of_gas = st.sidebar.number_input('Viscosity of Gas (Pa.s)', min_value=0.0, format="%.2e")
particle_density = st.sidebar.number_input('Particle Density (kg/m^3)', min_value=0.0)
particle_diameter = st.sidebar.number_input('Particle Diameter (m)', min_value=0.0, format="%.2e")
column_diameter = st.sidebar.number_input('Column Diameter (m)', min_value=0.0)
column_height = st.sidebar.number_input('Column Height (m)', min_value=0.0)
packing_height = st.sidebar.number_input('Packing Height (m)', min_value=0.0)
absolute_pressure = st.sidebar.number_input('Absolute Pressure (Pa)', min_value=0.0)
temperature = st.sidebar.number_input('Temperature (K)', min_value=0.0)
superficial_gas_velocity = st.sidebar.number_input('Superficial Gas Velocity (m/s)', min_value=0.0)
minimum_fluidization_velocity = st.sidebar.number_input('Minimum Fluidization Velocity (m/s)', min_value=0.0)
axial_location = st.sidebar.number_input('Axial Location', min_value=0.0)

# Gather inputs into a dictionary
input_data = {
    'System': system,
    'Geldart Particle Type': geldart_particle_type,
    'Density of gas': density_of_gas,
    'Viscosity of gas': viscosity_of_gas,
    'Particle density': particle_density,
    'Particle diameter': particle_diameter,
    'Column Diameter': column_diameter,
    'Column Height': column_height,
    'Packing Height': packing_height,
    'Absolute Pressure': absolute_pressure,
    'Temperature': temperature,
    'Superficial Gas Velocity': superficial_gas_velocity,
    'Minimum fluidization velocity': minimum_fluidization_velocity,
    'Axial Location': axial_location,
}

# Prediction button
if st.sidebar.button('Predict Bubble Diameter'):
    prediction = predict_bubble_diameter(input_data)
    st.write(f"Predicted Bubble Diameter: {prediction:.8f} m")

# Display feature ranges
st.markdown("### Feature Value Ranges:")
st.write("""
- **Density of gas:** 1.1690760147434667 - 37.410432471790934
- **Viscosity of gas:** 1.81e-05 - 1.87e-05
- **Particle density:** 800.0 - 8600.0
- **Particle diameter:** 8.3e-05 - 0.00138
- **Column Diameter:** 0.083 - 1.0
- **Column Height:** 0.6 - 3.5
- **Packing Height:** 0.15 - 0.61
- **Absolute Pressure:** 100.0 - 3200.0
- **Superficial Gas Velocity:** 0.0187 - 11.55
- **Minimum fluidization velocity:** 0.0051 - 0.77
- **Axial Location:** 0.004 - 0.881249897
""")