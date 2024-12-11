import requests
import streamlit as st

# API Keys and Deployment URLs (hardcoded for development use)
models = {
    "Falcon 7B": {
        "api_key": "r6zSAPJm7t8GbkqJENPzmXPpOKokltDGcMREKRr5fWdh",
        "url": "https://us-south.ml.cloud.ibm.com/ml/v1/deployments/ff2ad53a-6aa0-440b-8311-447844c24376/text/generation?version=2023-05-29",
    },
    "Falcon 7B Instruct": {
        "api_key": "lxSvrj38wGGmzduRfNhTS1jMH2yHCA835Hilf64P_9go",
        "url": "https://us-south.ml.cloud.ibm.com/ml/v1/deployments/02ba55f0-6620-412f-8eac-0f4ce2face3b/text/generation?version=2023-05-29",
    },
}

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

# Streamlit Sidebar for Model Selection
st.sidebar.title("Falcon Model Selector")
model_choice = st.sidebar.radio("Choose a model:", list(models.keys()))

# Fetch selected model details
selected_model = models[model_choice]
api_key = selected_model["api_key"]
url = selected_model["url"]

# Streamlit App Title
st.title("Chat with Falcon Models")

# User Inputs
prompt = st.text_input("Enter your prompt:", "Explain about transformers in ML")
max_new_tokens = st.slider("Max New Tokens:", min_value=50, max_value=300, value=150)
decoding_method = st.selectbox("Decoding Method:", ["greedy", "beam", "sampling"])
repetition_penalty = st.slider("Repetition Penalty:", min_value=1.0, max_value=2.0, value=1.1)
temperature = st.slider("Temperature:", min_value=0.1, max_value=1.0, value=0.7)
top_k = st.slider("Top-k:", min_value=0, max_value=50, value=10)
top_p = st.slider("Top-p:", min_value=0.0, max_value=1.0, value=0.9)

if st.button("Generate Text"):
    # Fetch access token
    access_token = get_access_token(api_key)
    if access_token is None:
        st.stop()

    # Request body
    body = {
        "input": prompt,
        "parameters": {
            "decoding_method": decoding_method,
            "max_new_tokens": max_new_tokens,
            "stop_sequences": [],
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
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
