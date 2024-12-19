from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pickle

app = Flask(__name__)

# --- Load the Trained Model and Preprocessor ---
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
except FileNotFoundError:
    # If model files don't exist create a basic model and preprocessor, and save it

    # Define Data Ranges and other data generation logic (copied from previous code)
    np.random.seed(42)
    num_samples = 500
    materials = ['Cotton', 'Polyester', 'Linen', 'Viscose', 'Recycled Cotton', 'Recycled Polyester']
    sources = ['Conventional', 'Organic', 'Recycled']
    spinning_methods = ['Ring-Spun', 'Open-End', 'Air-Jet']
    weaving_methods = ['Woven', 'Knitted']
    dyeing_methods = ['Conventional', 'Natural', 'Solution Dyeing']
    finishing_methods = ['Chemical', 'Mechanical', 'None']
    energy_sources = ['Fossil Fuels', 'Renewable', 'Mixed']
    transport_modes = ['Road', 'Sea', 'Air']

    # Create Data
    data = {
        'Material': np.random.choice(materials, num_samples),
        'Source': np.random.choice(sources, num_samples),
        'Spinning': np.random.choice(spinning_methods, num_samples),
        'Weaving': np.random.choice(weaving_methods, num_samples),
        'Dyeing': np.random.choice(dyeing_methods, num_samples),
        'Finishing': np.random.choice(finishing_methods, num_samples),
        'Energy': np.random.choice(energy_sources, num_samples),
        'Transport_Distance': np.random.uniform(100, 5000, num_samples),  # km
        'Transport_Mode': np.random.choice(transport_modes, num_samples),
        'Production_Volume': np.random.uniform(1000, 100000, num_samples),  # in kg
        'Recycled_Content': np.random.uniform(0, 1, num_samples),  # Percentage (0-1)
        'Water_Consumption': np.random.uniform(1, 20, num_samples),  # L/kg
        'Chemical_Use': np.random.uniform(0.1, 5, num_samples)  # kg/kg
    }

    df = pd.DataFrame(data)

    # --- Create a Synthetic CO2 emission (Target) ---

    # Define some base emission values
    material_emission = {'Cotton': 10, 'Polyester': 25, 'Linen': 12, 'Viscose': 18,
                        'Recycled Cotton': 5, 'Recycled Polyester': 15}
    source_emission = {'Conventional': 1, 'Organic': 0.5, 'Recycled': 0.2}
    spinning_emission = {'Ring-Spun': 2, 'Open-End': 1.5, 'Air-Jet': 1.8}
    weaving_emission = {'Woven': 1, 'Knitted': 0.8}
    dyeing_emission = {'Conventional': 3, 'Natural': 0.8, 'Solution Dyeing': 1.2}
    finishing_emission = {'Chemical': 2, 'Mechanical': 1, 'None': 0.2}
    energy_emission = {'Fossil Fuels': 5, 'Renewable': 1, 'Mixed': 3}
    transport_emission_factor = {'Road': 0.01, 'Sea': 0.005, 'Air': 0.05}  # gCO2e/km
    water_emission_factor = 0.01  # gCO2e/l
    chemical_emission_factor = 0.5  # gCO2e/kg

    df['CO2_emissions'] = 0

    for index, row in df.iterrows():
        emissions = (
            material_emission[row['Material']]
            + source_emission[row['Source']]
            + spinning_emission[row['Spinning']]
            + weaving_emission[row['Weaving']]
            + dyeing_emission[row['Dyeing']]
            + finishing_emission[row['Finishing']]
            + energy_emission[row['Energy']]
            + row['Transport_Distance'] * transport_emission_factor[row['Transport_Mode']]
            + row['Water_Consumption'] * water_emission_factor
            + row['Chemical_Use'] * chemical_emission_factor
        )

        emissions = emissions * (1 - row['Recycled_Content'])

        emissions = emissions * row['Production_Volume'] / 1000  # scaling emissions by production volume and bringing it to Kg instead of ton

        df.loc[index, 'CO2_emissions'] = emissions

    # Separate features and target variable
    X = df.drop('CO2_emissions', axis=1)
    y = df['CO2_emissions']

    # Define categorical and numerical features
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    # Define the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ],
        remainder='passthrough'
    )

    # Train the model and preprocessor with all data
    X_processed = preprocessor.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_processed, y)


    # save model and preprocessor
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
      try:
        # Retrieve input data from the form
        input_data = {
            'Material': request.form['material'],
            'Source': request.form['source'],
            'Spinning': request.form['spinning'],
            'Weaving': request.form['weaving'],
            'Dyeing': request.form['dyeing'],
            'Finishing': request.form['finishing'],
            'Energy': request.form['energy'],
            'Transport_Distance': float(request.form['transport_distance']),
            'Transport_Mode': request.form['transport_mode'],
            'Production_Volume': float(request.form['production_volume']),
            'Recycled_Content': float(request.form['recycled_content']),
             'Water_Consumption': float(request.form['water_consumption']),
            'Chemical_Use': float(request.form['chemical_use'])
        }

        input_df = pd.DataFrame([input_data])

        # Preprocess input data
        input_processed = preprocessor.transform(input_df)

        # Predict using the model
        prediction = model.predict(input_processed)[0]

        return render_template('index.html', prediction=prediction, input_data=input_data)
      except Exception as e:
          return render_template('index.html', error=f"An error occurred: {e}")


    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)