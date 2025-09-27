from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("car_price_model.pkl")

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        # Get form data
        year = int(request.form['year'])
        mileage = float(request.form['mileage'])
        car_model = int(request.form['model'])
        fuel_type = int(request.form['fueltype'])
        doors = int(request.form['doors'])
        color = int(request.form['color'])
        reference_price = float(request.form['reference_price'])  # NEW FEATURE

        # Prepare features array in the same order as model training
        features = np.array([year, mileage, car_model, fuel_type, doors, color, reference_price]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
