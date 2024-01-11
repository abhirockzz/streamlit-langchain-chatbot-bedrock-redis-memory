import streamlit as st
import os
import uuid

from langchain.llms import Bedrock
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.memory import RedisChatMessageHistory

template = "{chat_history}\n\nHuman:{human_input}\n\nAssistant:"

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)

# Initialize chat ID
if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())

redis_conn_string = os.environ.get("REDIS_CONN_STRING")

redis_chat_memory = RedisChatMessageHistory(url=redis_conn_string,session_id=st.session_state.chat_id)
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=redis_chat_memory, ai_prefix="\n\nAssistant", human_prefix="\n\nHuman")

st.title("Streamlit chatbot")
st.caption("Powered by Amazon Bedrock, Langchain and Redis")

llm = Bedrock(
        model_id="anthropic.claude-v2",
        streaming=True,
    )
    
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your message"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = llm_chain.predict(human_input=prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})