import streamlit as st
from langchain.llms import Bedrock

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.memory import RedisChatMessageHistory

template = "{chat_history}\n\nHuman:{human_input}\n\nAssistant:"

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)

redis_chat_memory = RedisChatMessageHistory(session_id="42")
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=redis_chat_memory, ai_prefix="\n\nAssistant", human_prefix="\n\nHuman")

st.title("Echo Bot")

llm = Bedrock(
        credentials_profile_name="default",
        model_id="anthropic.claude-v2",
        streaming=True,
    )
    
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = llm_chain.predict(human_input=prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})