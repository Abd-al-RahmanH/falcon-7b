import streamlit as st
import requests

# Streamlit App Title and Layout
st.set_page_config(page_title="Falcon 7B Chat App", layout="centered")
st.title("Falcon 7B Chat App")

# IBM Cloud Credentials
API_KEY = "r6zSAPJm7t8GbkqJENPzmXPpOKokltDGcMREKRr5fWdh"  # Replace with your actual API key
FALCON_URL = "https://us-south.ml.cloud.ibm.com/ml/v1/deployments/ff2ad53a-6aa0-440b-8311-447844c24376/text/generation?version=2023-05-29"

# Fetch Access Token
def get_access_token(api_key):
    """Retrieve access token from IBM Cloud."""
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key,
    }
    try:
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        return response.json().get("access_token")
    except Exception as e:
        st.error(f"Failed to fetch access token: {e}")
        return None

# Make a request to Falcon Model
def fetch_falcon_response(access_token, user_input):
    """Send a user query to the Falcon 7B model and retrieve a response."""
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    body = {
        "input": user_input,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 150,
            "repetition_penalty": 1.1,
            "stop_sequences": [],
        },
    }
    try:
        st.write("Request Body:", body)  # Debugging log
        response = requests.post(FALCON_URL, headers=headers, json=body)
        st.write("Response Status Code:", response.status_code)  # Debugging log
        st.write("Response Body:", response.text)  # Debugging log

        if response.status_code == 200:
            return response.json().get("results", [{}])[0].get("generated_text", "No response from the model.")
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        st.error(f"Failed to fetch response from Falcon model: {e}")
        return None

# Streamlit App Logic
st.subheader("Chat History")
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_input = st.text_input("Your Question", placeholder="Type your question here...")
if user_input:
    access_token = get_access_token(API_KEY)
    if access_token:
        model_response = fetch_falcon_response(access_token, user_input)
        if model_response:
            st.session_state["chat_history"].append(f"User: {user_input}")
            st.session_state["chat_history"].append(f"Assistant: {model_response}")
    else:
        st.error("Could not generate access token. Please check your API key.")

# Display Chat History
for message in st.session_state["chat_history"]:
    if "User:" in message:
        st.markdown(f"**{message}**")
    else:
        st.markdown(f"{message}")
