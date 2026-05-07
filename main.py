import gradio as gr
from pipeline import Pipeline

# Load RAG pipeline
rag_pipeline = Pipeline()


# Function to generate response
def generate_response(user_query):

    try:
        result = rag_pipeline.run(user_query)

        # If result is dictionary
        if isinstance(result, dict):

            answer = result.get("response", "No response generated.")
            sources = result.get("sources", [])
            confidence = result.get("confidence", 0.0)

            response = f"""
Answer:
{answer}


Sources:
{sources}

Confidence Score:
{confidence}
"""

            return response

        return result

    except Exception as e:
        return f"Error: {str(e)}"


# Custom CSS
custom_css = """
textarea {
    font-size: 16px !important;
}

.output-text {
    font-size: 16px !important;
}

footer {
    visibility: hidden;
}
"""


# Create UI
with gr.Blocks(
    css=custom_css,
    theme=gr.themes.Soft(),
    title="Customer Support AI Assistant"
) as app:

    gr.Markdown("""
# 🤖 Customer Support AI Assistant

Ask your questions and get intelligent responses powered by RAG.
""")

    user_input = gr.Textbox(
        label="Enter Your Query",
        placeholder="Type your question here...",
        lines=2
    )

    submit_btn = gr.Button("Submit Query")

    output = gr.Textbox(
        label="Assistant Response",
        lines=15
    )

    # Button click event
    submit_btn.click(
        fn=generate_response,
        inputs=user_input,
        outputs=output
    )


# Launch app
if __name__ == "__main__":
    app.launch()