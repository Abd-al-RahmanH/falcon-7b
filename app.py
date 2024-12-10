import requests
import streamlit as st

# Falcon 7B model endpoint
FALCON_URL = "https://api-inference.huggingface.co/models/falcon-7b"

# Streamlit app setup
st.set_page_config(page_title="Falcon 7B Chat App", page_icon="🤖", layout="centered")
st.title("Falcon 7B Chat App")
st.markdown("### Chat History")

# Chat history initialization
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Function to fetch Falcon response
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
            "decoding_method": "top_k_sampling",
            "top_k": 50,
            "top_p": 0.9,
            "temperature": 0.7,
            "max_new_tokens": 50,
            "repetition_penalty": 1.2,
            "stop_sequences": ["\n"]
        }
    }
    try:
        response = requests.post(FALCON_URL, headers=headers, json=body)
        if response.status_code == 200:
            result = response.json().get("results", [{}])[0].get("generated_text", "")
            clean_response = result.split('.')[0] + '.'
            return clean_response
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        st.error(f"Failed to fetch response from Falcon model: {e}")
        return None

# Input access token
access_token = st.text_input("Enter your API Access Token", type="password")
user_input = st.text_input("Your Question", key="user_input")

# Display chat history
for message in st.session_state["chat_history"]:
    st.markdown(message)

# Fetch and display response on button click
if st.button("Send"):
    if access_token and user_input:
        st.session_state["chat_history"].append(f"User: {user_input}")
        response = fetch_falcon_response(access_token, user_input)
        if response:
            st.session_state["chat_history"].append(f"Assistant: {response}")
        else:
            st.session_state["chat_history"].append("Assistant: No response from the model.")
        st.experimental_rerun()
    else:
        st.warning("Please enter both an access token and your question.")
