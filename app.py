from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
loaded_model = joblib.load("linear_regression_housing_vastral.pkl")

@app.route("/")
def home():
    return render_template("housing.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    data = {}

    if request.args:
        data = request.args
    elif request.is_json:
        data = request.get_json()
    else:
        data = request.form

    area = data.get("area")
    bedrooms = data.get("bedrooms")
    bathrooms = data.get("bathrooms")
    age = data.get("age_of_house")   # ✅ matches HTML

    if not all([area, bedrooms, bathrooms, age]):
        return render_template("housing.html", prediction_text="Error: Missing input fields")

    try:
        features = [[
            float(area),
            int(bedrooms),
            int(bathrooms),
            int(age)
        ]]
        prediction = loaded_model.predict(features)[0]

        if request.args or request.is_json:
            return jsonify({
                "area": float(area),
                "bedrooms": int(bedrooms),
                "bathrooms": int(bathrooms),
                "age_of_house": int(age),
                "predicted_price": round(float(prediction), 2)
            })

        return render_template(
            "housing.html",
            prediction_text=f"🏠 Predicted Price: ₹{round(float(prediction), 2):,.2f}"
        )

    except Exception as e:
        return render_template("housing.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)