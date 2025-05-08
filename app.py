from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load pre-trained models and encoder
yield_model = joblib.load(r"C:\Users\narsi\Downloads\crop Yield fertilizer\crop Yield fertilizer\rf_yield_model.pkl")
fertilizer_model = joblib.load(r"C:\Users\narsi\Downloads\crop Yield fertilizer\crop Yield fertilizer\rf_fert_model.pkl")
label_encoder = joblib.load(r"C:\Users\narsi\Downloads\crop Yield fertilizer\crop Yield fertilizer\fert_label_encoder.pkl")  # Only used for fertilizer encoding/decoding

# Crop mapping for reference
CROP_MAPPING = {
    0: "Corn",
    1: "Rice",
    2: "Soybeans",
    3: "Wheat"
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON data from request
        data = request.get_json()

        # Validate required fields
        required_fields = ['crop', 'soil_ph', 'organic_matter', 'n_content',
                           'p_content', 'k_content', 'rainfall', 'temperature']

        for field in required_fields:
            if field not in data or data[field] == '':
                return jsonify({"error": f"Field '{field}' is required"}), 400

        # Get crop value (already numeric)
        crop_value = int(data['crop'])
        if crop_value not in CROP_MAPPING:
            return jsonify({"error": "Invalid crop selection"}), 400

        # Prepare features for yield prediction
        yield_features = np.array([[
            crop_value,
            float(data['soil_ph']),
            float(data['organic_matter']),
            float(data['n_content']),
            float(data['p_content']),
            float(data['k_content']),
            float(data['rainfall']),
            float(data['temperature'])
        ]])

        # Prepare features for fertilizer recommendation
        fert_features = np.array([[
            crop_value,
            float(data['n_content']),
            float(data['p_content']),
            float(data['k_content'])
        ]])

        # Make predictions
        predicted_yield = yield_model.predict(yield_features)[0]
        fertilizer_encoded = fertilizer_model.predict(fert_features)[0]

        # Decode fertilizer recommendation
        recommended_fertilizer = label_encoder.inverse_transform([fertilizer_encoded])[0]

        # Return results as JSON
        return jsonify({
            "predicted_yield": float(predicted_yield),
            "recommended_fertilizer": recommended_fertilizer,
            "crop_name": CROP_MAPPING[crop_value]
        })

    except ValueError as e:
        return jsonify({"error": "Invalid input values. Please check all fields contain valid numbers."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)