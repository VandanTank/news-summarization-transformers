# app.py - Streamlit Web Application for News Headline Generation

import streamlit as st
from transformers import pipeline
import torch # Import torch to check for GPU availability

# --- Configuration ---
MODEL_NAME = "PEGASUS"
# This path points to the folder you downloaded and unzipped from Google Drive
MODEL_PATH = "./PEGASUS-best-finetuned" 

# --- Model Loading ---
# Use Streamlit's caching to load the model only once
@st.cache_resource
def get_summarizer(use_gpu_flag):
    """Loads the fine-tuned PEGASUS model and pipeline."""
    st.write("Loading fine-tuned PEGASUS model...")
    
    # Determine device: Use GPU (device 0) if available and selected, otherwise CPU (-1)
    device_option = 0 if use_gpu_flag and torch.cuda.is_available() else -1
    if device_option == 0:
        st.write("✅ GPU detected and selected for faster inference.")
    else:
        st.write("⏳ Using CPU for inference (may be slower).")

    # Initialize the summarization pipeline
    try:
        summarizer = pipeline(
            "summarization",
            model=MODEL_PATH,
            tokenizer=MODEL_PATH, # Load tokenizer from the same path
            device=device_option 
        )
        st.success("Model loaded successfully!")
        return summarizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please ensure the model files are correctly placed in the folder specified by MODEL_PATH.")
        return None

# --- Streamlit Interface ---
st.set_page_config(layout="wide") # Use wider layout
st.title("📰 Abstractive News Headline Generator")
st.markdown("_(Fine-tuned PEGASUS model on XSum dataset)_")

# Sidebar for controls
st.sidebar.title("⚙️ Settings")

# Allow user to attempt GPU usage if available
allow_gpu = torch.cuda.is_available()
use_gpu = st.sidebar.checkbox("Try using GPU (if available)", value=False, disabled=not allow_gpu, 
                              help="Requires correct CUDA setup locally. Defaults to CPU if GPU not found or causes errors.")
if not allow_gpu:
    st.sidebar.caption("No local CUDA GPU detected by PyTorch.")

max_len = st.sidebar.slider("Max Headline Length (tokens)", 10, 60, 40)
min_len = st.sidebar.slider("Min Headline Length (tokens)", 5, 25, 10)

# Input text area
article_text = st.text_area(
    "Paste the full News Article below:",
    height=350,
    placeholder="Enter the news article text here..."
)

# Generate button
if st.button("Generate Headline", type="primary", use_container_width=True):
    if article_text:
        # Load the summarizer (will be cached after first run)
        summarizer = get_summarizer(use_gpu)
        
        if summarizer:
            with st.spinner("Generating summary... Please wait."):
                try:
                    # Generate the summary using the pipeline
                    summary_output = summarizer(
                        article_text,
                        max_length=max_len,
                        min_length=min_len,
                        do_sample=False, # Use deterministic beam search for better quality
                        truncation=True  # Ensure long articles are handled
                    )
                    
                    st.markdown("---")
                    st.markdown("### ✨ Generated Headline:")
                    # Display the generated text
                    st.info(summary_output[0]['summary_text'])
                    
                except Exception as e:
                    st.error(f"Error during summarization: {e}")
                    st.error("This might happen if the input text is too long or malformed.")
        else:
            # Error message if model loading failed
            st.error("Model could not be loaded. Please check the terminal for errors.")
            
    else:
        # Warning if the text area is empty
        st.warning("⚠️ Please paste an article text into the box above.")

st.sidebar.markdown("---")
st.sidebar.caption("Project by: Vandan Tank") # Optional: Add your name