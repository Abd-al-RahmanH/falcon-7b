import requests
import streamlit as st

# Hardcoded API Key and Deployment Details
API_KEY = "r6zSAPJm7t8GbkqJENPzmXPpOKokltDGcMREKRr5fWdh"
DEPLOYMENT_URL = "https://us-south.ml.cloud.ibm.com/ml/v1/deployments/02ba55f0-6620-412f-8eac-0f4ce2face3b/text/generation?version=2023-05-29"

# Streamlit App
st.title("Falcon 7B Model Text Generation")

# User input
prompt = st.text_area("Enter your prompt:", "Explain about transformers in ML")
max_new_tokens = st.slider("Max New Tokens:", min_value=50, max_value=300, value=150)
decoding_method = st.selectbox("Decoding Method:", ["greedy", "beam", "sampling"])
repetition_penalty = st.slider("Repetition Penalty:", min_value=1.0, max_value=2.0, value=1.1)

if st.button("Generate Text"):
    # Request body
    body = {
        "input": prompt,
        "parameters": {
            "decoding_method": decoding_method,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
        },
    }

    # Headers
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    # Make the API request
    response = requests.post(DEPLOYMENT_URL, headers=headers, json=body)

    if response.status_code != 200:
        st.error(f"Request failed: {response.text}")
    else:
        data = response.json()
        generated_text = data.get("results", [{}])[0].get("generated_text", "No output.")
        st.text_area("Generated Text:", generated_text, height=200)
