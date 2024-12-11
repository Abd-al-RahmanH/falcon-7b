import requests
import streamlit as st

# Hardcoded API details
API_DETAILS = {
    "Falcon 7B": {
        "url": "https://us-south.ml.cloud.ibm.com/ml/v1/deployments/02ba55f0-6620-412f-8eac-0f4ce2face3b/text/generation?version=2023-05-29",
        "key": "r6zSAPJm7t8GbkqJENPzmXPpOKokltDGcMREKRr5fWdh",  # Replace with actual API key
    },
    "Falcon 7B Instruct": {
        "url": "https://us-south.ml.cloud.ibm.com/ml/v1/deployments/OTHER_DEPLOY_ID/text/generation?version=2023-05-29",
        "key": "r6zSAPJm7t8GbkqJENPzmXPpOKokltDGcMREKRr5fWdh",  # Replace with actual API key
    },
}

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Falcon 7B"

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
        "Authorization": f"Bearer {model_details['key']}",
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
