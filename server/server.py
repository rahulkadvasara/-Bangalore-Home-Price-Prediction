# from flask import Flask, request, jsonify
# import util

# app = Flask(__name__)

# @app.route('/get_location_names', methods=['GET'])
# def get_location_names():
#     response = jsonify({
#         'locations': util.get_location_names()
#     })
#     response.headers.add('Access-Control-Allow-Origin', '*')

#     return response

# @app.route('/predict_home_price', methods=['GET', 'POST'])
# def predict_home_price():
#     total_sqft = float(request.form['total_sqft'])
#     location = request.form['location']
#     bhk = int(request.form['bhk'])
#     bath = int(request.form['bath'])

#     response = jsonify({
#         'estimated_price': util.get_estimated_price(location,total_sqft,bhk,bath)
#     })
#     response.headers.add('Access-Control-Allow-Origin', '*')

#     return response

# if __name__ == "__main__":
#     print("Starting Python Flask Server For Home Price Prediction...")
#     util.load_saved_artifacts()
#     app.run()



from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import util

app = FastAPI()

# CORS middleware (equivalent to Access-Control-Allow-Origin in Flask)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/get_location_names")
def get_location_names():
    locations = util.get_location_names()
    return JSONResponse(content={"locations": locations})

@app.post("/predict_home_price")
def predict_home_price(
    total_sqft: float = Form(...),
    location: str = Form(...),
    bhk: int = Form(...),
    bath: int = Form(...)
):
    estimated_price = util.get_estimated_price(location, total_sqft, bhk, bath)
    return JSONResponse(content={"estimated_price": estimated_price})

# Start-up event to load saved artifacts
@app.on_event("startup")
def startup_event():
    print("Starting FastAPI Server For Home Price Prediction...")
    util.load_saved_artifacts()
