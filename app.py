import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
import os

# Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCFbnID7J4KnD-hoveRc37CEx_MV9eXUEk"

# Initialize the memory for storing conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the Google Gemini model
chat_model = GoogleGenerativeAI(model="gemini-1.5-pro")

# Function to handle user input and generate responses
def generate_response(user_input):
    chat_history = memory.load_memory_variables({}).get("chat_history", [])
    
    # Append the new user message
    chat_history.append(HumanMessage(content=user_input))
    
    # Generate AI response
    response = chat_model.invoke(chat_history[-5:])  # Limit history to 5 messages
    
    # Ensure response is handled correctly
    response_text = response if isinstance(response, str) else getattr(response, "content", str(response))
    response_message = AIMessage(content=response_text)
    
    # Append AI response to history
    chat_history.append(response_message)
    
    # Save updated history
    memory.save_context({"input": user_input}, {"output": response_message.content})
    
    return response_message.content

# Streamlit UI
st.set_page_config(page_title="AI Mechanical Engineering Tutor", page_icon="🤖")
st.title("🤖 Banu AI Conversational Mechanical Engineering Tutor")
st.write("Hi Banu Ask any mechanical engineering-related questions!")

# Initialize chat history in session state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

# User input
user_input = st.chat_input("Type your mechanical engineering question...")
if user_input:
    # Append user input to session state
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    # Get AI response
    response = generate_response(user_input)
    
    # Append AI response to session state
    st.session_state.messages.append(AIMessage(content=response))
    
    # Display response
    st.chat_message("assistant").write(response)
