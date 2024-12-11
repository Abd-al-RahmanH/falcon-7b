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

# Function to generate text using Falcon 7B
def generate_falcon(api_key, url, prompt, parameters):
    access_token = get_access_token(api_key)
    if access_token is None:
        st.stop()

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    response = requests.post(url, headers=headers, json={"input": prompt, "parameters": parameters})
    if response.status_code != 200:
        st.error(f"Request failed: {response.text}")
        return ""

    data = response.json()
    return data.get("results", [{}])[0].get("generated_text", "No output.")

# Streamlit Sidebar for Model Selection
st.sidebar.title("Model Parameters")
model_choice = st.sidebar.radio("Choose a model:", list(models.keys()))

# Fetch selected model details
selected_model = models[model_choice]
api_key = selected_model["api_key"]
url = selected_model["url"]

# Sidebar Parameters
st.sidebar.subheader("Generation Parameters")
max_new_tokens = st.sidebar.slider("Max New Tokens:", min_value=50, max_value=2000, value=600)
decoding_method = st.sidebar.selectbox("Decoding Method:", ["greedy", "beam", "sampling"])
repetition_penalty = st.sidebar.slider("Repetition Penalty:", min_value=1.0, max_value=2.0, value=1.1)
if decoding_method != "greedy":
    temperature = st.sidebar.slider("Temperature:", min_value=0.1, max_value=1.0, value=0.7)
    top_k = st.sidebar.slider("Top-k:", min_value=0, max_value=50, value=10)
    top_p = st.sidebar.slider("Top-p:", min_value=0.0, max_value=1.0, value=0.9)

# Streamlit App Layout
st.title("Multi-Document Retrieval with Watsonx")
st.markdown("Developed by Abdul Rahman H")

# Chat History
def display_chat(chat_history):
    for entry in chat_history:
        if entry["role"] == "user":
            st.markdown(f"**You:** {entry['content']}")
        elif entry["role"] == "assistant":
            st.markdown(f"**Model:** {entry['content']}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat Box
st.subheader("Chat with the Model")
prompt = st.text_input("Enter your question:", "", key="user_input")
if st.button("Ask") and prompt:
    # Add user input to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Prepare parameters for the request
    parameters = {
        "decoding_method": decoding_method,
        "max_new_tokens": max_new_tokens,
        "stop_sequences": [],
        "repetition_penalty": repetition_penalty,
    }

    if decoding_method != "greedy":
        parameters.update({
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        })

    # Generate text using the selected model
    response = generate_falcon(api_key, url, prompt, parameters)

    # Add model response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Display chat history
display_chat(st.session_state.chat_history)
