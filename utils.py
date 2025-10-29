import streamlit as st
import pandas as pd
import re
import spacy
import nltk
from nltk.corpus import stopwords
import time
from multiprocessing import Pool, cpu_count, Manager
from collections import Counter
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
import plotly.graph_objects as go
import numpy as np # Needed for array_split

# --- Preprocessing ---

# Load spacy model once
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("Spacy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
        st.stop()

nlp = load_spacy_model()

# Ensure stopwords are downloaded
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    st.info("Downloading NLTK stopwords...")
    try:
        nltk.download('stopwords')
        stop_words = set(stopwords.words("english"))
    except Exception as e:
        st.error(f"Failed to download NLTK stopwords: {e}")
        stop_words = set() # Fallback to empty set


def clean_and_normalize(text):
    """Cleans text: removes links, tags, non-ASCII, punctuation, numbers; lowercases; lemmatizes; removes stopwords."""
    if not isinstance(text, str):
        return "" # Return empty string for non-string inputs

    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # remove links first
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation (keeps words and spaces)
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = re.sub(r'[^\x00-\x7F]+', '', text) # Remove emojis/non-ASCII
    text = text.lower() # Lowercase
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with single space

    # Use spaCy for tokenization and lemmatization
    doc = nlp(text)
    # Lemmatize and remove stopwords and single characters
    lemmatized = [token.lemma_ for token in doc if token.text not in stop_words and len(token.lemma_) > 1 and not token.is_space]

    return " ".join(lemmatized)

# --- Tokenization ---
def tokenize(text):
    if not isinstance(text, str):
        return []
    return text.split()

# --- Word Count ---
# Worker function for multiprocessing - counts words in a single string
def count_words_simple(text_string):
    if not isinstance(text_string, str):
        return Counter()
    return Counter(text_string.split())

def word_count_sequential(series, top_n):
    """Counts words sequentially."""
    start_time = time.perf_counter()
    # Combine all cleaned text into one large string
    all_text = " ".join(series.astype(str).dropna())
    total_counts = count_words_simple(all_text)
    end_time = time.perf_counter()
    return total_counts.most_common(top_n), end_time - start_time

# Helper function to process a chunk of data for multiprocessing word count
def process_chunk_word_count(chunk):
    """Processes a list (chunk) of text strings for word counting."""
    local_counter = Counter()
    for text in chunk:
        local_counter.update(text.split())
    return local_counter

def word_count_parallel(series, top_n):
    """Counts words using multiprocessing. Optimized for chunking."""
    start_time = time.perf_counter()
    word_lists = series.astype(str).dropna().tolist()
    if not word_lists:
        return [], 0.0

    num_processes = max(1, cpu_count() - 1) # Use most cores, leave one free
    # Split data into chunks for better parallel performance
    chunks = np.array_split(word_lists, num_processes)
    # Convert chunks back to lists as required by pool.map
    chunks = [chunk.tolist() for chunk in chunks]

    total_counts = Counter()
    try:
        # Use Manager().list() if more complex state sharing needed, not required here
        # with Manager() as manager: # If complex objects needed
        with Pool(processes=num_processes) as pool:
            # Map the processing function to the chunks
            partial_counts_list = pool.map(process_chunk_word_count, chunks)

        # Sum up the counters from all processes
        for partial_counts in partial_counts_list:
            total_counts.update(partial_counts)

    except Exception as e:
        st.error(f"Multiprocessing error in word count: {e}. Overhead might be high for small data.")
        # Fallback for safety, though less likely needed with chunking
        all_text = " ".join(word_lists)
        total_counts = count_words_simple(all_text)


    end_time = time.perf_counter()
    return total_counts.most_common(top_n), end_time - start_time


# --- Timing Comparison Plot ---
def plot_timing_comparison(seq_time, mp_time, task_name="Task"):
    """Generates a Plotly bar chart comparing sequential and multiprocessing times."""
    # Handle cases where multiprocessing might not have run or returned a time
    mp_time_val = mp_time if mp_time is not None else 0
    labels = ['Sequential', 'Multiprocessing']
    times = [seq_time, mp_time_val]

    fig = go.Figure(data=[go.Bar(x=labels, y=times, text=times, textposition='auto')])
    fig.update_traces(texttemplate='%{text:.3f}s', textfont_size=12) # Show time on bars
    fig.update_layout(
        title=f'{task_name} Time Comparison',
        yaxis_title="Time (seconds)",
        xaxis_title="Processing Method",
        bargap=0.4 # Adjust gap between bars
    )
    return fig

# --- Email Functions ---
# (Keep send_email_alert and send_email_with_attachment as they were,
# ensuring they accept sender_email and sender_password as arguments)
def send_email_alert(to_email, subject, message, sender_email, sender_password):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    if not sender_email or not sender_password:
        st.error("Sender email credentials are not configured in secrets.")
        return False # Indicate failure

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(message, "plain"))

    server = None # Initialize server variable
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        # st.success(f"Email sent successfully to {to_email}") # Moved success msg to page
        return True # Indicate success
    except Exception as e:
        st.error(f"❌ Email Error: {e}")
        return False # Indicate failure
    finally:
        if server:
            try:
                server.quit()
            except Exception:
                 pass # Ignore errors during quit


def send_email_with_attachment(to_email, subject, body, file_path, sender_email, sender_password):
    if not sender_email or not sender_password:
        st.error("Sender email credentials are not configured in secrets.")
        return False # Indicate failure

    if not os.path.exists(file_path):
        st.error(f"Attachment file not found: {file_path}")
        return False # Indicate failure

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # Attach CSV
    try:
        with open(file_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(file_path)}")
        msg.attach(part)
    except Exception as e:
        st.error(f"Error attaching file: {e}")
        return False # Indicate failure

    # Send Email
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    server = None # Initialize server variable
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        # st.success(f"Email with attachment sent successfully to {to_email}") # Moved success msg to page
        return True # Indicate success
    except Exception as e:
        st.error(f"❌ Email Error: {e}")
        return False # Indicate failure
    finally:
         if server:
            try:
                server.quit()
            except Exception:
                 pass # Ignore errors during quit

# --- CSV Conversion for Download ---
@st.cache_data # Use cache_data for data conversions
def convert_df_to_csv(df):
   # IMPORTANT: Cache the conversion to prevent computation on every rerun
   if df is None:
       return b"" # Return empty bytes if df is None
   return df.to_csv(index=False).encode('utf-8')
