# app.py

import os
import json
import streamlit as st
from dotenv import load_dotenv
from llm_agent import create_agent

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Create or retrieve the agent from session state
if "agent" not in st.session_state:
    st.session_state.agent = create_agent(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4o",
        temperature=0.0
    )
agent = st.session_state.agent

st.title("Physician Finder Chat Agent Feb 18 2025")
st.write("""
**Welcome!** Ask for a physician by specifying your needs.
For example: "I need a cardiologist in LA, California"
""")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -- 1) On first load, invoke agent with hidden "Hello" so we get an initial greeting --
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    first_response = agent.invoke("Hello")
    # If it's a dict with an "output" key, extract that text. Otherwise use the raw string.
    if isinstance(first_response, dict) and "output" in first_response:
        greeting_text = first_response["output"]
    else:
        greeting_text = str(first_response)
    st.session_state.chat_history.append({"user": "", "agent": greeting_text})


# -- 2) Display conversation so far, skipping any empty user text --
for chat in st.session_state.chat_history:
    if chat["user"]:  # Only show if user text is non-empty
        with st.chat_message("user"):
            st.markdown(chat["user"])
    with st.chat_message("assistant"):
        st.markdown(chat["agent"])

# -- 3) Chat input widget for new questions --
user_input = st.chat_input("Type your message...")

if user_input:
    # Display the user's new message
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Processing..."):
        response = agent.invoke(user_input)
        try:
            parsed = json.loads(response)
            display_response = parsed.get("output", response)
        except Exception:
            display_response = str(response)
            
        # Same logic: if it's a dict with 'output', use that; else raw string
        if isinstance(response, dict) and "output" in response:
            display_response = response["output"]
        else:
            display_response = str(response)

    # Show the agent's response
    with st.chat_message("assistant"):
        st.markdown(display_response)

    # Save this turn to the chat history
    st.session_state.chat_history.append({"user": user_input, "agent": display_response})
