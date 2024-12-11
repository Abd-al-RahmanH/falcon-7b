import requests
import streamlit as st

# Hardcoded API details (update keys as needed)
API_DETAILS = {
    "Falcon 7B": {
        "url": "https://us-south.ml.cloud.ibm.com/ml/v1/deployments/02ba55f0-6620-412f-8eac-0f4ce2face3b/text/generation?version=2023-05-29",
    },
    "Falcon 7B Instruct": {
        "url": "https://us-south.ml.cloud.ibm.com/ml/v1/deployments/OTHER_DEPLOY_ID/text/generation?version=2023-05-29",
    },
}

# Hardcoded API key
API_KEY = "r6zSAPJm7t8GbkqJENPzmXPpOKokltDGcMREKRr5fWdh"

# Function to generate an access token from the API key
def get_access_token(api_key):
    auth_url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key,
    }
    response = requests.post(auth_url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        st.error("Failed to fetch access token. Please check your API key.")
        return None

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Falcon 7B"

if "access_token" not in st.session_state:
    # Generate the token automatically using the hardcoded API key
    st.session_state.access_token = get_access_token(API_KEY)

# Sidebar
st.sidebar.title("Model Controls")
st.sidebar.selectbox(
    "Select Model:",
    options=list(API_DETAILS.keys()),
    index=list(API_DETAILS.keys()).index(st.session_state.selected_model),
    key="selected_model",
)

st.sidebar.write("### Parameters")
temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
max_new_tokens = st.sidebar.slider("Max New Tokens:", min_value=50, max_value=2048, value=150)
decoding_method = st.sidebar.selectbox("Decoding Method:", ["greedy", "beam", "sampling"])

# Main chat interface
st.title("Chat Interface")

if not st.session_state.access_token:
    st.error("Access token generation failed. Please check the hardcoded API key.")
else:
    with st.form("chat_form"):
        user_input = st.text_area("Enter your message:", key="user_input")
        submit = st.form_submit_button("Send")

    if submit and user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "message": user_input})

        # API call
        model_details = API_DETAILS[st.session_state.selected_model]
        body = {
            "input": user_input,
            "parameters": {
                "decoding_method": decoding_method,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            },
        }

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {st.session_state.access_token}",
        }

        response = requests.post(model_details["url"], headers=headers, json=body)

        if response.status_code != 200:
            bot_message = f"Error: {response.text}"
        else:
            data = response.json()
            bot_message = data.get("results", [{}])[0].get("generated_text", "No output generated.")

        # Add bot message to history
        st.session_state.chat_history.append({"role": "bot", "message": bot_message})

    # Display chat history
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f"**You:** {chat['message']}")
        else:
            st.markdown(f"**Bot:** {chat['message']}")
