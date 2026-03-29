import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

## Set up the Streamlit app
st.set_page_config(
    page_title="TheoremAI | Intelligent Math & Research Assistant",
    page_icon="∞",
    layout="centered"
)

st.title("✨ TheoremAI")
st.markdown("##### Your Intelligent Mathematical & Logical Reasoning Agent")

st.sidebar.header("Configuration ⚙️")
groq_api_key=st.sidebar.text_input(label="Groq API Key",type="password", help="Enter your Groq API key to power the logic engine.")


if not groq_api_key:
    st.info("👋 Please add your Groq API key in the sidebar to awaken the agent.")
    st.stop()

llm=ChatGroq(model="llama-3.3-70b-versatile",groq_api_key=groq_api_key)


## Initializing the tools
wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the Internet to find various scientific and historical information on mentioned topics."

)

## Initialize the Math tool
import numexpr

def evaluate_math(expression: str) -> str:
    try:
        # Clean up the expression in case the LLM wrapped it in backticks or words
        clean_expr = expression.strip().replace('`', '')
        if clean_expr.startswith('text'):
            clean_expr = clean_expr[4:]
        return str(numexpr.evaluate(clean_expr.strip()))
    except Exception as e:
        return f"Math Error: {e}"

calculator=Tool(
    name="Calculator",
    func=evaluate_math,
    description="A tool for answering math-related questions. Only a raw mathematical expression should be provided."
)

prompt="""
You are an advanced reasoning agent tasked with solving the user's mathematical and text-based logic questions. 
Logically arrive at the solution, provide a detailed step-by-step reasoning process, and display it clearly point-wise.
Question: {question}
Answer:
"""

prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

## Combine all the tools into chain
chain=LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool=Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

## initialize the agents

assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi! I'm **TheoremAI**. Ask me a complex math problem or logic puzzle, and I'll break it down for you."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

## Let's start the interaction
question=st.text_area(
    "💬 What logic puzzle can I solve for you today?",
    "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?",
    height=120
)

if st.button("Calculate Answer 🚀"):
    if question:
        with st.spinner("Reasoning and executing tools..."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb]
                                         )
            st.session_state.messages.append({'role':'assistant',"content":response})
            st.write('### Response:')
            st.success(response)

    else:
        st.warning("Please enter the question")