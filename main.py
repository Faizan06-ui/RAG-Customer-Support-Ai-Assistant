import gradio as gr
from pipeline import Pipeline

# Load the RAG pipeline once
rag_pipeline = Pipeline()

# Define a function to be used in Gradio
def generate_response(user_query):
    return rag_pipeline.run(user_query)

# Create Gradio interface
iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=2, placeholder="Enter your query here..."),
    outputs=gr.Textbox(label="Response"),
    title="Customer Support AI Assistant",
    description="Ask your question and get a response from the RAG-powered assistant."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
