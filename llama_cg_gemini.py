from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

# Assign the environmental variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# Langsmith tracking
os.environ['LANGCHAIN_TRACING_V2'] = "true"

# Initialize Streamlit session state to store conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Prompt function to include conversation history
def get_prompt(conversation_history, current_question):
    messages = [
        ("system", "you are a helpful assistant who responds to the user queries. Please be honest with the answers and mention if you are not sure about the answer")
    ]
    for user_message, assistant_message in conversation_history:
        messages.append(("user", user_message))
        messages.append(("assistant", assistant_message))
    messages.append(("user", f"Question: {current_question}"))
    return ChatPromptTemplate.from_messages(messages)

# Streamlit UI
st.title('Langchain with Ollama')

# Display the conversation history at the top (outside the if block)
st.write("## Conversation History")
chat_container = st.container()
with chat_container:
    for i, (user_message, assistant_message) in enumerate(st.session_state.conversation_history, 1):
        st.write(f"**User:** {user_message}")
        st.write(f"**Assistant:** {assistant_message}")

# User input
input_text = st.text_input("Ask a Question", key="user_input")

# Checkbox for clearing input (independent)
clear_on_submit = st.checkbox("Clear Input After Submit")

# Ollama LLM
llm = Ollama(model="llama2")
outputparser = StrOutputParser()

# Process and display response whenever input text changes
if input_text:
    # Generate prompt with conversation history
    prompt = get_prompt(st.session_state.conversation_history, input_text)
    chain = prompt | llm | outputparser

    # Get the response
    response = chain.invoke({'question': input_text})

    # Update conversation history
    st.session_state.conversation_history.append((input_text, response))

    # Display the response regardless of checkbox state
    st.write(f"**Assistant:** {response}")

    # Clear the input box based on checkbox state (optional)
    if clear_on_submit:
        st.session_state['user_input'] = ""

# Add custom CSS to style the chat history like a messenger
st.markdown("""
<style>
.css-1kiw93k {
  flex-direction: column-reverse;
}
.stTextInput > div > div > input {
  margin-top: 20px;
}
.stContainer {
  display: flex;
  flex-direction: column;
  max-height: 70vh;
  overflow-y: auto;
  border: 1px solid #ccc;
  padding: 10px;
  border-radius: 10px;
}
.stContainer::-webkit-scrollbar {
  width: 8px;
}
.stContainer::-webkit-scrollbar-thumb {
  background-color: #888;
  border-radius: 10px;
}
.stContainer::-webkit-scrollbar-thumb:hover {
  background-color: #555;
}
</style>
""", unsafe_allow_html=True)
