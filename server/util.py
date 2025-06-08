# import pickle
# import json
# import numpy as np

# __locations = None
# __data_columns = None
# __model = None

# def get_estimated_price(location,sqft,bhk,bath):
#     try:
#         loc_index = __data_columns.index(location.lower())
#     except:
#         loc_index = -1

#     x = np.zeros(len(__data_columns))
#     x[0] = sqft
#     x[1] = bath
#     x[2] = bhk
#     if loc_index>=0:
#         x[loc_index] = 1

#     return round(__model.predict([x])[0],2)


# def load_saved_artifacts():
#     print("loading saved artifacts...start")
#     global  __data_columns
#     global __locations

#     with open("./artifacts/columns.json", "r") as f:
#         __data_columns = json.load(f)['data_columns']
#         __locations = __data_columns[3:]  # first 3 columns are sqft, bath, bhk

#     global __model
#     if __model is None:
#         with open('./artifacts/banglore_home_prices_model.pickle', 'rb') as f:
#             __model = pickle.load(f)
#     print("loading saved artifacts...done")

# def get_location_names():
#     return __locations

# def get_data_columns():
#     return __data_columns

# if __name__ == '__main__':
#     load_saved_artifacts()
#     print(get_location_names())
#     print(get_estimated_price('1st Phase JP Nagar',1000, 3, 3))
#     print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
#     print(get_estimated_price('Kalhalli', 1000, 2, 2))
#     print(get_estimated_price('Ejipura', 1000, 2, 2))  





import pickle
import json
import numpy as np
from threading import Lock
from typing import List, Optional

# Globals for model and metadata
__locations: Optional[List[str]] = None
__data_columns: Optional[List[str]] = None
__model = None
_model_lock = Lock()  # Thread safety

def get_estimated_price(location: str, sqft: float, bhk: int, bath: int) -> float:
    """Estimate price based on location and features."""
    try:
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        print(f"[WARN] Location not found in data columns: {location}")
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    with _model_lock:
        prediction = __model.predict([x])[0]

    return round(prediction, 2)

def load_saved_artifacts() -> None:
    """Load model and metadata from disk."""
    print("[INFO] Loading saved artifacts...")

    global __data_columns, __locations, __model

    # Load data columns
    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # Assume first 3 are sqft, bath, bhk

    # Load model only once
    if __model is None:
        with open("./artifacts/banglore_home_prices_model.pickle", "rb") as f:
            __model = pickle.load(f)

    print("[INFO] Artifacts loaded successfully.")

def get_location_names() -> List[str]:
    """Return list of location names."""
    return __locations

def get_data_columns() -> List[str]:
    """Return list of all data columns (used internally)."""
    return __data_columns
