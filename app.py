from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
with open("model/wine_cultivar_model.pkl", "rb") as f:
    model, scaler, features = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            input_data = [
                float(request.form[feature]) for feature in features
            ]

            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = scaler.transform(input_array)

            pred_class = model.predict(input_scaled)[0]
            prediction = f"Cultivar {pred_class + 1}"

        except:
            prediction = "Invalid input. Please enter numeric values."

    return render_template("index.html",
                           features=features,
                           prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
