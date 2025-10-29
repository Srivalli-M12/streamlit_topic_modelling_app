# streamlit_topic_modelling_app
Parallel Text Processor: Interactive Topic Modeling

# Project Overview

This project, developed as part of the Infosys Springboard internship, is an interactive web application built with Streamlit. It provides an end-to-end workflow for topic modeling on user-uploaded text data. The application allows users to load datasets, preprocess text, apply and compare two popular topic modeling techniques (LDA and BERTopic), analyze processing times (including sequential vs. parallel comparisons for specific tasks), and export the results.

# Features

1) Multi-Page Interface: Guided workflow through separate pages for each step.

2) Data Loading: Upload CSV or Excel files.

3) Configuration: Select the text column and the number of records to process.

4) Text Preprocessing: Automated cleaning pipeline including:-

   Lowercasing, URL/HTML tag removal

   Punctuation, number, non-ASCII character removal

   Lemmatization (using spaCy)

   Stopword removal (using NLTK)

5) Word Count Analysis: Calculate and display top N words with timing comparison between sequential and parallel processing (using multiprocessing).

6) LDA Modeling: Train a Latent Dirichlet Allocation model (using Gensim) with a user-specified number of topics. Includes tokenization timing comparison.

7) BERTopic Modeling: Train a BERTopic model, leveraging sentence transformers, UMAP, and HDBSCAN. Includes tokenization for coherence calculation.

8) Model Comparison: Side-by-side comparison of LDA and BERTopic based on:

   Coherence Score (C_v)

   Perplexity (LDA only)

   Processing Time

9) Results Visualization: Bar charts for timing comparisons and coherence/time trade-offs. Tables for topic words and data previews.

10) Export: Download processed data with topic assignments as a CSV file.

11) Email: Send model summaries or results CSV via email (requires Gmail App Password configuration).

12) Persistence: Session state management keeps results visible across pages during a user session.

13) Basic Login: Simple email/password authentication (using Streamlit secrets).

# Milestones Summary

Milestone 1: Data loading (Tweets dataset via Kagglehub), basic exploration (Pandas), initial storage (SQLite).

Milestone 2: Text preprocessing implementation (regex, NLTK, spaCy), sequential vs. parallel processing comparison for tokenization/word count, LDA model training (Gensim), evaluation (Coherence, Perplexity), and basic email summary.

Milestone 3: BERTopic model implementation, performance comparison with LDA, saving BERTopic results, emailing results with attachments.

Milestone 4: Development of the multi-page Streamlit application to integrate all previous steps into an interactive UI, adding login, persistent results display, separate comparison/download/email pages, and refined visualizations.

# Project Structure

streamlit_topic_app/
├── .venv/                     # Virtual environment
├── pages/                     # Streamlit pages (steps 1-7)
│   ├── 1_Load_Data.py
│   ├── 2_Preprocess_WordCount.py
│   ├── 3_LDA_Model.py
│   ├── 4_BERTopic_Model.py
│   ├── 5_Compare_Models.py
│   ├── 6_Download_Results.py
│   └── 7_Email_Results.py
├── app.py                     # Main app file (Login/Welcome page)
├── utils.py                   # Helper functions (preprocessing, email, plots)
├── requirements.txt           # Python dependencies
└── README.md                  # This file


# Technologies Used

1) Python 3.11 (Recommended)

2) Streamlit (Web App Framework)

3) Pandas (Data Manipulation)

4) NLTK, spaCy (Text Preprocessing)

5) Gensim (LDA Implementation)

6) BERTopic, Sentence-Transformers, UMAP-learn, HDBSCAN (BERTopic Implementation)

7) Scikit-learn (Used by BERTopic)

8) Plotly (Data Visualization)

9) NumPy, SciPy (Numerical Computing)

# Installation

1) Prerequisites:

Python (Version 3.11 recommended for compatibility). Ensure Python is added to your system PATH.

pip (Python package installer).

Git (for cloning the repository).

Microsoft C++ Build Tools: Required for compiling dependencies like hdbscan on Windows. Download from Microsoft and install the "Desktop development with C++" workload.

2) Clone the Repository:

git clone [https://github.com/Srivalli-M12/streamlit_topic_modelling_app.git](https://github.com/Srivalli-M12/streamlit_topic_modelling_app.git)
cd streamlit_topic_modelling_app


3) Create and Activate Virtual Environment:

# Use python -m venv or py -3.11 -m venv depending on your setup
python -m venv .venv
# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# Activate (Git Bash / Linux / macOS)
# source .venv/bin/activate
Note: On Windows PowerShell, you might need to run Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser once.

4) Install Dependencies:
pip install -r requirements.txt
 
 Download Language Models/Data:
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords


5) Configure Secrets:

Create a folder named .streamlit inside your project directory.

Inside .streamlit, create a file named secrets.toml.

Add the following content, replacing placeholders with your actual credentials:

# .streamlit/secrets.toml
app_user_email = "user@example.com" # Login email
app_password = "your_chosen_password" # Login password

gmail_sender_email = "your_sending_gmail@gmail.com"
gmail_app_password = "your_google_app_password" # Use a Google App Password

IMPORTANT: Add .streamlit/secrets.toml to your .gitignore file to avoid committing credentials to GitHub.

# Usage

Ensure your virtual environment is activated.

Run the Streamlit app from the project root directory:

streamlit run app.py

The application will open in your web browser.

Log in using the credentials defined in secrets.toml.

Follow the steps in the sidebar: Load data, preprocess, run models, compare, and export.

# Potential Improvements

1) Implement more robust user authentication.

2) Add more sophisticated preprocessing options (e.g., n-grams, custom stopword lists).

3) Incorporate visualization options for topics (e.g., word clouds, intertopic distance maps).

4) Allow users to tune hyperparameters for LDA and BERTopic.

5) Add support for different embedding models in BERTopic.

6) Implement more evaluation metrics (e.g., Topic Diversity).

7) Optimize parallel processing further, potentially using libraries like Dask for larger datasets.


