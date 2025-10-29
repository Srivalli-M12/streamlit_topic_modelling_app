import streamlit as st
import pandas as pd
from utils import clean_and_normalize, word_count_sequential, word_count_parallel, plot_timing_comparison
import time
import os # Make sure os is imported

# --- Check Login Status ---
# (Keep this block as it is)
if not st.session_state.get("logged_in", False):
    st.warning("Please log in first on the main page.")
    st.stop()

# --- Sidebar ---
# (Keep this block as it is)
def handle_logout_sidebar():
    st.session_state['logged_in'] = False
    st.session_state['current_user_email'] = None
    st.success("You have been logged out.")
    st.switch_page("app.py")

if st.session_state.get('logged_in', False):
    # Determine current page for unique key (moved outside the button check)
    st.session_state['current_page'] = os.path.basename(__file__)
    logout_key = f"logout_btn_{st.session_state['current_page']}"

    st.sidebar.success(f"Logged in as: {st.session_state.get('current_user_email', 'N/A')}")
    st.sidebar.markdown("---")
    st.sidebar.header("Navigation")
    st.sidebar.info("Select a page above.")
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", key=logout_key):
         handle_logout_sidebar()

# --- Main Page Header ---
st.header("2. Preprocess Text and Count Words")

# --- Check Prerequisites ---
# (Keep this block as it is)
if 'df' not in st.session_state or st.session_state['df'] is None:
    st.warning("⬅️ Please load data on the 'Load Data' page first.")
    st.stop()
if 'text_column' not in st.session_state or st.session_state['text_column'] is None:
    st.warning("⬅️ Please select the text column on the 'Load Data' page.")
    st.stop()
if 'num_records' not in st.session_state or st.session_state['num_records'] is None:
    st.warning("⬅️ Please select the number of records on the 'Load Data' page.")
    st.stop()

# --- Get Data from State ---
df = st.session_state['df']
text_column = st.session_state['text_column']
num_records = st.session_state['num_records']

# --- Preprocessing Section ---
st.subheader("Text Preprocessing")

# --- Display existing preprocessing results FIRST ---
# This block runs every time if data is available
if st.session_state.get('cleaned_data_available', False) and 'df_processed' in st.session_state and st.session_state['df_processed'] is not None:
     st.success(f"Preprocessing previously completed in {st.session_state.get('preprocess_time', 0):.3f} seconds.")
     st.subheader("Preview: Original vs. Cleaned Text (First 5)")
     original_text_col_name = st.session_state.get('text_column', None) # Get original column name
     # Check if both original and cleaned columns exist before displaying
     if original_text_col_name and original_text_col_name in st.session_state['df_processed'].columns and 'cleaned_text' in st.session_state['df_processed'].columns:
          st.dataframe(st.session_state['df_processed'][[original_text_col_name, 'cleaned_text']].head())
     elif 'cleaned_text' in st.session_state['df_processed'].columns:
          st.dataframe(st.session_state['df_processed'][['cleaned_text']].head())
          st.warning("Original text column not found for preview.")
     st.markdown("---") # Separator

# --- Preprocessing Button and Logic ---
# This button calculates and stores the results, then reruns
if st.button("🚀 Start/Rerun Preprocessing", key="start_preprocess_btn"):
    # Ensure df_raw exists from loading stage before slicing
    if 'df_raw' in st.session_state and st.session_state['df_raw'] is not None:
        df_process = st.session_state['df_raw'].head(num_records).copy() # Work on a copy of the RAW slice
    else:
        st.error("Raw dataframe not found. Please reload data on Page 1.")
        st.stop()

    if text_column not in df_process.columns:
        st.error(f"Selected text column '{text_column}' not found in the dataframe.")
        st.stop()

    st.info(f"Preprocessing {num_records} records from column '{text_column}'...")
    st.markdown("""
    **Steps performed:**
    1. Remove URLs/Links
    2. Remove HTML tags
    3. Remove punctuation
    4. Remove numbers
    5. Remove non-ASCII 
    6. Convert to lowercase 
    7. Remove extra whitespace,
    8. Lemmatize words
    9. Remove stopwords
    10. Remove single-character tokens
    """)
    progress_bar = st.progress(0)
    status_text = st.empty()
    cleaned_texts = []
    total_records = len(df_process)

    start_time_preprocess = time.perf_counter()
    for i, text in enumerate(df_process[text_column]):
        cleaned_texts.append(clean_and_normalize(text))
        progress = (i + 1) / total_records
        if (i + 1) % 50 == 0 or (i + 1) == total_records:
             progress_bar.progress(progress)
             status_text.text(f"Processing record {i+1}/{total_records} ({(progress*100):.1f}%)")

    # Add original text column back for preview consistency before storing
    df_process['cleaned_text'] = cleaned_texts
    if text_column in df_process.columns: # Keep original column if it exists
         df_process = df_process[[text_column, 'cleaned_text']] # Select relevant columns

    end_time_preprocess = time.perf_counter()
    processing_time = end_time_preprocess - start_time_preprocess
    # Don't show success message here, it's shown above after rerun

    # Store results in session state
    st.session_state['df_processed'] = df_process
    st.session_state['cleaned_data_available'] = True
    st.session_state['preprocess_time'] = processing_time

    # Clear downstream model results
    st.session_state['lda_results_available'] = False
    st.session_state['bt_results_available'] = False
    # Also clear word count results as they depend on preprocessing
    st.session_state['word_count_results'] = None
    st.session_state['word_count_fig'] = None
    st.session_state['word_count_results_seq'] = None
    st.session_state['word_count_results_mp'] = None
    st.session_state['word_count_seq_time'] = None
    st.session_state['word_count_mp_time'] = None


    # Rerun to display the results using the block above
    st.rerun()

# --- Word Count Section ---
st.markdown("---")
st.subheader("Word Count Analysis")

# Only show if preprocessing has been done in this session
if st.session_state.get('cleaned_data_available', False):
    # Retrieve potentially existing processed df
    df_processed_wc = st.session_state.get('df_processed')
    if df_processed_wc is None or 'cleaned_text' not in df_processed_wc.columns:
         st.warning("Processed data with 'cleaned_text' not found. Please rerun preprocessing.")
         st.stop()

    # Slider for top N words
    top_n_words = st.slider("Select number of top words to display:", min_value=5, max_value=50, value=st.session_state.get('top_n_words', 10), key='top_n_slider_wc')
    # Store slider value immediately in case button isn't clicked
    st.session_state['top_n_words'] = top_n_words

    # --- Display existing word count results FIRST ---
    # This block runs every time if results are available
    if st.session_state.get('word_count_results') is not None:
        seq_time = st.session_state.get('word_count_seq_time')
        mp_time = st.session_state.get('word_count_mp_time')
        top_words_seq_disp = st.session_state.get('word_count_results_seq', [])
        top_words_mp_disp = st.session_state.get('word_count_results_mp', [])
        top_n_disp = st.session_state.get('top_n_words', 10) # Get the count used for display

        st.write(f"**Top {top_n_disp} Words (Sequential Time: {seq_time:.3f}s):**")
        st.table(pd.DataFrame(top_words_seq_disp, columns=['Word', 'Count']))

        st.write(f"**Top {top_n_disp} Words (Multiprocessing Time: {mp_time:.3f}s):**")
        st.table(pd.DataFrame(top_words_mp_disp, columns=['Word', 'Count']))

        st.markdown("""
        *Note on Timing:* Multiprocessing introduces overhead. For fast tasks like word counting on smaller datasets, this overhead can make it slower than sequential processing. Benefits appear with larger datasets or more complex tasks.
        """)

        st.subheader("Word Count Timing Comparison")
        fig_wc = st.session_state.get('word_count_fig')
        if fig_wc:
            st.plotly_chart(fig_wc)
        st.markdown("---") # Separator

    # --- Word Count Button and Logic ---
    # This button calculates/recalculates, stores results, and reruns
    if st.button("📊 Calculate/Recalculate Word Counts", key="calc_word_counts_btn"):
        with st.spinner("Calculating word counts (Sequential)..."):
            # Ensure using the latest processed df from session state
            df_to_count = st.session_state['df_processed']
            if 'cleaned_text' in df_to_count.columns:
                top_words_seq, seq_time = word_count_sequential(df_to_count['cleaned_text'], top_n_words)
                # Store results
                st.session_state['word_count_seq_time'] = seq_time
                st.session_state['word_count_results_seq'] = top_words_seq
            else:
                st.error("Cleaned text column not found for sequential count.")
                st.stop()

        with st.spinner("Calculating word counts (Multiprocessing)..."):
            df_to_count = st.session_state['df_processed'] # Re-fetch just in case
            if 'cleaned_text' in df_to_count.columns:
                 top_words_mp, mp_time = word_count_parallel(df_to_count['cleaned_text'], top_n_words)
                 # Store results
                 st.session_state['word_count_mp_time'] = mp_time
                 st.session_state['word_count_results_mp'] = top_words_mp
            else:
                 st.error("Cleaned text column not found for parallel processing.")
                 st.stop()

        # Store combined results dict and figure
        st.session_state['word_count_results'] = {'seq_time': seq_time, 'mp_time': mp_time, 'top_words': top_words_mp}
        fig = plot_timing_comparison(seq_time, mp_time if mp_time is not None else 0, "Word Count")
        st.session_state['word_count_fig'] = fig

        # Rerun to display results using the blocks above
        st.rerun()

    # Success message after word count section (if applicable)
    st.success("Ready for Topic Modeling.")

else:
    # Message if preprocessing hasn't been done yet in this session
    st.info("Click 'Start/Rerun Preprocessing' first to generate cleaned text.")

