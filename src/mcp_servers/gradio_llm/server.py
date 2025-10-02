
import argparse
import gradio as gr
import uvicorn
from mcp.server.fastmcp import FastMCP
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load lightweight LLM model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

mcp = FastMCP("gradio-llm-mcp")

@mcp.tool()
def llm_inference(prompt: str, max_length: int = 100) -> str:
    """Generate text using lightweight LLM based on the input prompt."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text[len(prompt):].strip()  # Return only the generated part

# Gradio interface for demo
def gradio_chat(message, history):
    response = llm_inference(message)
    history.append((message, response))
    return history, ""

def create_gradio_demo():
    demo = gr.ChatInterface(
        gradio_chat,
        title="MCP LLM Inference Demo",
        description="Chat with a lightweight LLM for medical queries (using DistilGPT2)."
    )
    return demo

def main():
    parser = argparse.ArgumentParser(description="Gradio LLM MCP Server")
    parser.add_argument("--host", type=str, required=True, help="Host to bind the server")
    parser.add_argument("--port", type=int, required=True, help="Port to bind the server")
    args = parser.parse_args()

    demo = create_gradio_demo()

    from gradio import mount_gradio_app

    # Mount Gradio on the MCP FastAPI app
    mcp_app = mcp.streamable_http_app()
    mount_gradio_app(mcp_app, demo, path="/ui")

    uvicorn.run(mcp_app, host=args.host, port=args.port, workers=1)

if __name__ == "__main__":
    main()
