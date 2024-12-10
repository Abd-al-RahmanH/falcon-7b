import streamlit as st
import requests

# URL to fetch the access token
TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"
# URL for Falcon 7B deployment
FALCON_URL = "https://us-south.ml.cloud.ibm.com/ml/v1/deployments/ff2ad53a-6aa0-440b-8311-447844c24376/text/generation?version=2023-05-29"

# Function to fetch a new access token
def get_access_token(api_key):
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key,
    }
    response = requests.post(TOKEN_URL, headers=headers, data=data)
    if response.status_code != 200:
        st.error(f"Failed to get access token: {response.text}")
        return None
    return response.json().get("access_token")

# Function to send the request to Falcon 7B
def fetch_falcon_response(access_token, user_input):
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
    response = requests.post(FALCON_URL, headers=headers, json=body)
    if response.status_code != 200:
        st.error(f"Request failed: {response.text}")
        return None
    return response.json()

# Streamlit app layout and logic
def main():
    st.title("Falcon 7B Chat App")
    st.sidebar.write("Developed by **Abdul Rahman H**")
    
    # Session state to manage history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # IBM Cloud API key input
    api_key = st.sidebar.text_input("IBM Cloud API Key", type="password", help="Enter your IBM Cloud API Key.")
    if not api_key:
        st.warning("Please enter your IBM Cloud API key to proceed.")
        st.stop()
    
    # Display chat history
    st.subheader("Chat History")
    for message in st.session_state.messages:
        role = "User" if message["role"] == "user" else "Assistant"
        st.markdown(f"**{role}:** {message['content']}")
    
    # User input
    user_input = st.text_input("Your Question", placeholder="Type your question here...", disabled=not api_key)
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get Access Token
        access_token = get_access_token(api_key)
        if not access_token:
            st.stop()
        
        # Fetch response
        response = fetch_falcon_response(access_token, user_input)
        if response:
            assistant_reply = response.get("generated_text", "No response from the model.")
            st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
            st.success("Response received!")
        else:
            st.error("Failed to fetch response. Check your inputs or API deployment.")
    
    # Display the latest response
    if st.session_state.messages:
        latest_message = st.session_state.messages[-1]
        if latest_message["role"] == "assistant":
            st.markdown(f"**Assistant:** {latest_message['content']}")

if __name__ == "__main__":
    main()
