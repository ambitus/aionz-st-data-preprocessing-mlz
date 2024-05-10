import requests
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import os

load_dotenv()


class InvalidWMLzPasswordException(Exception):
    "Raised when the WMLz Password has been expired"
    pass

def get_auth_token():
    url = 'https://' + os.environ["MLZ_IP_W_PORT"] + '/auth/generateToken'
    username = os.environ["MLZ_USER"]
    password = os.environ["MLZ_PASS"]

    payload = json.dumps({
        "username": username,
        "password": password
    })

    headers = {
        "Content-Type": "application/json",
        "Control": "no-cache"
    }

    response = requests.request(
        "POST", url, headers=headers, data=payload, verify=False)
    auth_response_json = response.json()
    try:
        auth_token = auth_response_json["token"]
    except:
        raise InvalidWMLzPasswordException

    return auth_token



# Get WMLz details from env vars
WMLZ_TOKEN = "CHANGE_ME"
try:
    WMLZ_TOKEN = get_auth_token()
except InvalidWMLzPasswordException:
    raise InvalidWMLzPasswordException

MLZ_IP_W_PORT = os.environ["MLZ_IP_W_PORT"]
MLZ_USER = os.environ["MLZ_USER"]
MLZ_PASS = os.environ["MLZ_PASS"]
MODEL_URL = os.environ["MODEL_URL"]
PREPROCESSING_URL = os.environ["PREPROCESSING_URL"]

data_to_clean =  [
    {
        "User": "0",
        "Card": "0",
        "Year": "2002",
        "Day": "1",
        "Month": "9",
        "Time": "06:21",
        "Amount": "$134.09",
        "Use Chip": "Swipe Transaction",
        "Merchant Name": "3527213246127876953",
        "Zip": "91750.0"
    }
]

print('Original Data:')
print(json.dumps(data_to_clean[0], indent=2))

data_to_preproces =  [
    {
        "User": "0",
        "Card": "0",
        "Year": "2002",
        "Day": "1",
        "Month": "9",
        "Time": "0621",
        "Amount": "134.09",
        "Use Chip": "Swipe Transaction",
        "Merchant Name": "3527213246127876953",
        "Zip": "91750.0"
    }
]

print('\nCleaned Data:')
print(json.dumps(data_to_preproces[0], indent=2))


headers = {'content-type': 'application/json', 'Authorization': 'Bearer ' + WMLZ_TOKEN}
response = requests.post(PREPROCESSING_URL, json=data_to_preproces, headers=headers, verify=False)
preprocessed_data = response.json()[0]
print('\nPreprocessed Data:')
print(json.dumps(preprocessed_data, indent=2))


data_to_inference = [
    {
        "x1": preprocessed_data['standardScaler(User)'],
        "x10": preprocessed_data['standardScaler(Card)'],
        "x2": preprocessed_data['standardScaler(Year)'],
        "x3": preprocessed_data['standardScaler(Day)'],
        "x4": preprocessed_data['standardScaler(Month)'],
        "x5": preprocessed_data['standardScaler(Time)'],
        "x6": preprocessed_data['standardScaler(Amount)'],
        "x7": preprocessed_data['encoder(Use Chip)'],
        "x8": preprocessed_data['standardScaler(Merchant Name)'],
        "x9": preprocessed_data['standardScaler(Zip)']
    }
]

response = requests.post(MODEL_URL, json=data_to_inference, headers=headers, verify=False)
inference_data = response.json()[0]
print('\nInference Data:')
print(json.dumps(inference_data, indent=2))
print('\n')


if inference_data['probability(1)'] > .8:
    print('Prediction:\nFraud')
else:
    print('Prediction:\nNot Fraud')