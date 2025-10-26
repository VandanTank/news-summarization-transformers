# Comparative Analysis of Transformer Models for Abstractive News Summarization

# Comparative Analysis of Transformer Models for Abstractive News Summarization

## 🎯 Goal
This project undertakes a comprehensive comparative analysis of three prominent Transformer architectures – BART, T5, and PEGASUS – for the task of abstractive news headline generation. The primary objective was to fine-tune these models on the challenging XSum dataset, evaluate their performance using both quantitative (ROUGE scores) and qualitative methods, identify the best-performing model, and deploy it in an interactive application.

## 📊 Dataset
* **Dataset:** XSum (Extreme Summarization)
* **Source:** BBC Articles (2010-2017), known for requiring highly abstractive summaries.
* **Task:** Generate a concise, single-sentence summary (headline) capturing the core information of a news article.
* **Link:** [XSum Dataset on Hugging Face](https://huggingface.co/datasets/EdinburghNLP/xsum)
* **Strategy:** For efficient iteration and training feasibility, a representative sample was used: 5,000 training, 1,000 validation, and 1,000 test examples.

## ⚙️ Methodology & Workflow
1.  **Environment:** Project developed using Python 3.10, VS Code, Jupyter Notebooks, and virtual environments. Key libraries include `transformers`, `datasets`, `evaluate`, `torch`, and `streamlit`.
2.  **Data Preprocessing:** Articles and summaries were loaded and tokenized using model-specific tokenizers from Hugging Face. Special handling was implemented for the `t5-small` model's required "summarize: " prefix. Input sequences were truncated to 512 tokens and target sequences (headlines) to 128 tokens.
3.  **Models Compared:**
    * `facebook/bart-base` (General-purpose seq2seq model)
    * `t5-small` (Text-to-text framework)
    * `google/pegasus-xsum` (Model pre-trained specifically on XSum)
4.  **Fine-Tuning:**
    * Models were fine-tuned for 3 epochs using the Hugging Face `Seq2SeqTrainer`.
    * To leverage GPU acceleration for efficient training, **Google Colab (T4 GPU)** was utilized.
    * The best checkpoint for each model was saved based on the validation ROUGE-2 score, stored directly to Google Drive during training.
5.  **Evaluation:**
    * **Quantitative:** Performance was rigorously measured using standard ROUGE F1 scores (ROUGE-1, ROUGE-2, ROUGE-Lsum) calculated on the validation set.
    * **Qualitative:** Generated headlines from the best-performing model (PEGASUS) were manually inspected against reference headlines and source articles to assess fluency, relevance, and factual consistency.

## 📈 Results & Analysis

### Quantitative Results (Validation Set @ Epoch 3)

| Model   |   Validation Loss |   Rouge1 |   Rouge2 |   RougeLsum |
| :------ | ------------------: | -------: | -------: | ----------: |
| BART    |              0.72 |    34.37 |    13.57 |       27.75 |
| T5      |              0.67 |    21.11 |     5.44 |       17.31 |
| PEGASUS |              0.55 |    40.35 |    17.94 |       36.06 |

**Key Findings (Quantitative):**
* **PEGASUS** (`google/pegasus-xsum`) significantly outperformed BART and T5 across all ROUGE metrics, achieving the highest ROUGE-2 score (17.94). This aligns with expectations due to its specialized pre-training.
* **BART** showed reasonable performance, demonstrating successful transfer learning.
* **T5-small** struggled considerably on this highly abstractive task according to ROUGE scores.

### Qualitative Analysis (Focus on PEGASUS - The Best Model Quantitatively)

While PEGASUS excelled in metrics, manual inspection revealed important limitations inherent to abstractive models:

* **Good Summaries:** For some articles, PEGASUS generated accurate and fluent headlines, effectively capturing the main point.
    * **Example (Sports):**
        * *Original Article Snippet:* > The champions paid for a lacklustre first half... Leicester improved but ran into a stubborn home defence... goalkeeper Artur Boruc produced a key save...
        * *Reference Headline:* > Bournemouth moved to the highest league position in their history as Leicester's miserable run away from home continued with defeat at Vitality Stadium.
        * *Generated Headline (PEGASUS):* > Leicester's miserable start to the Premier League season continued as they were beaten at Bournemouth. *(Captures key info well)*
* **Challenges Observed:**
    * **Factual Hallucination:** The model sometimes introduced plausible but incorrect details not present in the source text (e.g., specific summit names/locations, sports venues).
    * **Overly Generic Summaries:** For longer, complex articles, the generated headline occasionally lacked crucial specifics, summarizing the broad topic rather than the key information.

**Overall Conclusion:** The quantitative results clearly favor PEGASUS based on ROUGE scores. However, the qualitative analysis reveals limitations regarding factual consistency and specificity, demonstrating that ROUGE scores alone don't fully capture summary quality. This project highlights the importance of mixed-method evaluation for abstractive summarization and the practical challenges of deploying these models reliably.

## 🚀 Demo (Streamlit App)

An interactive web application was built using Streamlit to demonstrate the fine-tuned PEGASUS model.

**To run locally:**
1. Clone this repository.
2. **Download the fine-tuned model weights (`model.safetensors` - Approx. 2.1 GiB) from [this Google Drive link](https://drive.google.com/file/d/1gui4hSjzA6LnQefdVR5r3_T8ObmexZvG/view?usp=sharing)**.
3. Place the downloaded `model.safetensors` file inside the `PEGASUS-best-finetuned` folder within the cloned repository.
4. Navigate to the project directory.
5. Set up and activate a Python virtual environment.
6. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
7. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

`![Streamlit Demo](./streamlit_screenshot.png)`

[*(Optional but Recommended) Insert Link to Deployed Streamlit Cloud App Here]*

## 📁 File Structure
├── PEGASUS-best-finetuned/ # Fine-tuned PEGASUS model files ├── venv/ # Virtual environment (add to .gitignore) ├── 01_Data_Exploration.ipynb # Data loading and initial analysis ├── 02_Model_Preprocessing.ipynb # Tokenization and data preparation ├── 03_Model_Training.ipynb # Model fine-tuning (run in Colab) ├── app.py # Streamlit application script ├── requirements.txt # Python dependencies └── README.md # This documentation file


---
*Project by: Vandan Tank*