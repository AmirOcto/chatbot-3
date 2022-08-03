import streamlit as st
from Chatbot_Project import *


st.write("NLP Chatbot")
user_message = st.text_input("Lets have a chat, enter your question")

if user_message:
    st.write("Response:")
    with st.spinner("I am thinking"):
        bot_message= get_response(user_message)
        st.write(bot_message)
    st.write("")
