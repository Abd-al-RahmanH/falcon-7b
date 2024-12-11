import requests
import streamlit as st

# IBM Cloud API Key
api_key = "lxSvrj38wGGmzduRfNhTS1jMH2yHCA835Hilf64P_9go"

# Function to fetch a new access token
def get_access_token(api_key):
    token_url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key,
    }
    response = requests.post(token_url, headers=headers, data=data)
    if response.status_code != 200:
        st.error(f"Failed to get access token: {response.text}")
        return None
    return response.json()["access_token"]

# Streamlit App
st.title("Falcon 7B Instruct Model Text Generation")

# User input
prompt = st.text_input("Enter your prompt:", "Explain about transformers in ML")
max_new_tokens = st.slider("Max New Tokens:", min_value=50, max_value=300, value=150)
decoding_method = st.selectbox("Decoding Method:", ["greedy", "beam", "sampling"])
repetition_penalty = st.slider("Repetition Penalty:", min_value=1.0, max_value=2.0, value=1.1)

if st.button("Generate Text"):
    # Fetch access token
    access_token = get_access_token(api_key)
    if access_token is None:
        st.stop()

    # API URL
    url = "https://us-south.ml.cloud.ibm.com/ml/v1/deployments/02ba55f0-6620-412f-8eac-0f4ce2face3b/text/generation?version=2023-05-29"

    # Request body
    body = {
        "input": prompt,
        "parameters": {
            "decoding_method": decoding_method,
            "max_new_tokens": max_new_tokens,
            "stop_sequences": [],  # You can add stop sequences here if needed
            "repetition_penalty": repetition_penalty,
        },
    }

    # Headers
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    # Make request
    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 200:
        st.error(f"Request failed: {response.text}")
    else:
        data = response.json()
        generated_text = data.get("results", [{}])[0].get("generated_text", "No output.")
        st.text_area("Generated Text:", generated_text, height=200)
