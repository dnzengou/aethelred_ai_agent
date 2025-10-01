from langchain import PromptTemplate, LLMChain, ConversationChain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import gradio as gr

# 1. Load your open source LLM locally or from HuggingFace Hub
pipe = pipeline(
    "text-generation", 
    model="tiiuae/falcon-7b-instruct",  # example open LLM model
    max_length=512,
    do_sample=True,
    top_p=0.95,
    temperature=0.7
)

llm = HuggingFacePipeline(pipeline=pipe)

# 2. Define your System/Agent Prompt with embedded strategic knowledge
system_template = """
You are Aethelred, a Senior Geopolitical Strategist and Historical Military Philosopher.
Base your answers on the principles from Sun Tzu's The Art of War, Unrestricted Warfare, and Machiavelli's The Prince.
Answer questions related to global geopolitical conflicts, challenges, and policy implications with rigor and clarity.

User's Question: {input}
"""

prompt = PromptTemplate(
    input_variables=["input"],
    template=system_template
)

# 3. Create a conversational chain
conversation = ConversationChain(llm=llm, prompt=prompt)

# 4. Define Gradio UI function
def respond_to_query(user_input, chat_history):
    response = conversation.predict(input=user_input)
    chat_history = chat_history or []
    chat_history.append((user_input, response))
    return chat_history, chat_history

# 5. Build Gradio Interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask Aethelred a geopolitical question...")
    state = gr.State()
    msg.submit(respond_to_query, inputs=[msg, state], outputs=[chatbot, state])

demo.launch()
