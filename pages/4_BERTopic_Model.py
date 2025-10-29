import streamlit as st
import pandas as pd
from utils import tokenize, plot_timing_comparison
import time
from multiprocessing import Pool, cpu_count
import numpy as np
from bertopic import BERTopic
# Need gensim dictionary and coherence model for BERTopic coherence
from gensim import corpora
from gensim.models import CoherenceModel

# --- Check Login Status ---
if not st.session_state.get("logged_in", False):
    st.warning("Please log in first on the main page.")
    st.stop()
# --- Sidebar ---
# Placed in each page script to ensure persistence

# Logout Button Logic (Requires session state check)
def handle_logout_sidebar():
    st.session_state['logged_in'] = False
    st.session_state['current_user_email'] = None
    # Optionally clear other results, or rely on Load Data page's clear button
    st.success("You have been logged out.") # Use success/info for feedback
    # No st.rerun() needed here usually, Streamlit handles page switch

# Display sidebar content only if logged in
if st.session_state.get('logged_in', False):
    st.sidebar.success(f"Logged in as: {st.session_state.get('current_user_email', 'N/A')}")
    st.sidebar.markdown("---")
    st.sidebar.header("Navigation")
    st.sidebar.info("Select a page above.")
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", key=f"logout_btn_{st.session_state.get('current_page', 'app')}"): # Unique key per page
         handle_logout_sidebar()
         st.switch_page("app.py") # Redirect to login page after logout

# --- Login Check ---
# Also add this check at the top of every page script (except app.py itself)
# To prevent access if not logged in
# (The version you already have in the pages is fine)
# if not st.session_state.get("logged_in", False):
#     st.warning("Please log in first on the main page.")
#     st.stop() # Stop execution if not logged in

# Store current page name for unique keys if needed (optional but good practice)
# This assumes your page filenames are descriptive
import os
st.session_state['current_page'] = os.path.basename(__file__)

st.header("4. BERTopic Modeling")

# --- Check Prerequisites ---
if not st.session_state.get('cleaned_data_available', False) or 'df_processed' not in st.session_state:
    st.warning("⬅️ Please preprocess data on the 'Preprocess & WordCount' page first.")
    st.stop()

df_processed = st.session_state['df_processed']


# --- Tokenization for Coherence ---
st.subheader("Tokenization (Required for BERTopic Coherence Calculation)")
st.markdown("""
Although BERTopic doesn't require pre-tokenized text for *training*, we need the tokenized version to calculate the C_v coherence score, which allows comparison with LDA.
We compare sequential vs. parallel processing times here.
""")

# Button to trigger tokenization timing
if st.button("⏱️ Prepare Tokens & Compare Times", key="bt_tokenize_btn"):
    texts_to_tokenize = df_processed["cleaned_text"].astype(str).dropna().tolist()
    if not texts_to_tokenize:
        st.warning("No cleaned text available for tokenization.")
        st.stop()

    with st.spinner("Running Sequential Tokenization..."):
        start_seq = time.perf_counter()
        tokens_seq = [tokenize(text) for text in texts_to_tokenize]
        end_seq = time.perf_counter()
        st.session_state['bt_tokenization_seq_time'] = end_seq - start_seq

    with st.spinner("Running Parallel Tokenization..."):
        start_mp = time.perf_counter()
        try:
            num_processes = max(1, cpu_count() - 1)
            with Pool(processes=num_processes) as pool:
                tokens_mp = pool.map(tokenize, texts_to_tokenize)
            end_mp = time.perf_counter()
            st.session_state['bt_tokenization_mp_time'] = end_mp - start_mp
            # Store tokens & dictionary for Coherence calculation
            st.session_state['bt_tokens'] = tokens_mp
            st.session_state['bt_dictionary'] = corpora.Dictionary(tokens_mp)

        except Exception as e:
            st.error(f"Multiprocessing error during tokenization: {e}. Storing sequential results.")
            st.session_state['bt_tokenization_mp_time'] = None
            st.session_state['bt_tokens'] = tokens_seq # Fallback
            st.session_state['bt_dictionary'] = corpora.Dictionary(tokens_seq)


    # Plot comparison and store figure
    seq_time = st.session_state.get('bt_tokenization_seq_time', 0)
    mp_time = st.session_state.get('bt_tokenization_mp_time', seq_time) # Use seq_time if mp failed
    fig = plot_timing_comparison(seq_time, mp_time if mp_time is not None else 0, "BERTopic Tokenization")
    st.session_state['bt_tokenization_fig'] = fig
    st.session_state['bt_tokenization_times'] = {'seq': seq_time, 'mp': mp_time}
    st.rerun() # Rerun to display results


# --- Display Tokenization Results ---
if 'bt_tokens' in st.session_state and st.session_state['bt_tokens'] is not None:
    st.write("**Tokenization Timing Results:**")
    seq_time_bt_tok = st.session_state.get('bt_tokenization_seq_time')
    mp_time_bt_tok = st.session_state.get('bt_tokenization_mp_time')
    if seq_time_bt_tok is not None:
         st.write(f"- Sequential Time: {seq_time_bt_tok:.3f} seconds")
    if mp_time_bt_tok is not None:
         st.write(f"- Multiprocessing Time: {mp_time_bt_tok:.3f} seconds")
    else:
         st.write("- Multiprocessing failed or was not run.")

    fig_bt_tok = st.session_state.get('bt_tokenization_fig')
    if fig_bt_tok:
        st.plotly_chart(fig_bt_tok)
        st.markdown("""
        *Note on Timing:* Similar to word counting, the overhead for multiprocessing might make it slower than sequential processing for tokenization, especially if the dataset size isn't very large.
        """)

    # Show Token Preview (Requirement 5)
    st.write("**Preview of Tokenized Text (First 3 Documents):**")
    for i, doc_tokens in enumerate(st.session_state['bt_tokens'][:3]):
        st.text(f"Doc {i+1}: {' | '.join(doc_tokens[:20])} {'...' if len(doc_tokens) > 20 else ''}") # Show first 20 tokens


    # --- Run BERTopic Model ---
    st.markdown("---")
    st.subheader("Train BERTopic Model")
    st.markdown("BERTopic uses sentence transformers for embeddings, UMAP for dimensionality reduction, and HDBSCAN for clustering.")
    # Add any BERTopic specific parameters here if needed (e.g., min_topic_size)
    # min_topic_size = st.number_input("Minimum Topic Size:", min_value=5, value=10, step=1)

    if st.button("🚀 Train BERTopic Model", key="train_bt_btn"):
        docs = df_processed["cleaned_text"].astype(str).dropna().tolist()
        # Check again in case state was cleared
        if not docs:
            st.error("Cleaned text list is empty. Cannot train BERTopic.")
            st.stop()

        try:
            with st.spinner("Training BERTopic Model (this can take a significant amount of time)..."):
                start_train = time.perf_counter()
                # Initialize BERTopic model
                # Consider adding parameters like min_topic_size if desired
                topic_model = BERTopic(language="english", calculate_probabilities=False, verbose=True) # Probabilities off saves memory/time
                topics, _ = topic_model.fit_transform(docs)
                end_train = time.perf_counter()
                st.session_state['bt_model'] = topic_model
                st.session_state['bt_training_time'] = end_train - start_train

                # Assign Topics
                # Ensure column exists
                if 'dominant_topic_bt' not in df_processed.columns:
                     df_processed['dominant_topic_bt'] = None
                df_processed['dominant_topic_bt'] = topics
                # Update the main processed df in session state
                st.session_state['df_processed'] = df_processed

            # Calculate Coherence (if tokens/dictionary available)
            with st.spinner("Calculating Coherence Score..."):
                coherence_bt = None # Default
                if 'bt_tokens' in st.session_state and 'bt_dictionary' in st.session_state:
                    try:
                        tokens = st.session_state['bt_tokens']
                        dictionary = st.session_state['bt_dictionary']
                        # Prepare topics in the format CoherenceModel expects
                        bertopic_topics_raw = topic_model.get_topics()
                        # Filter out the outlier topic (-1) and get only words
                        bertopic_topics_for_coherence = [
                            [word for word, _ in topic_model.get_topic(topic_id)]
                            for topic_id in bertopic_topics_raw if topic_id != -1
                        ]

                        if bertopic_topics_for_coherence and tokens and dictionary:
                            coherence_model_bt = CoherenceModel(topics=bertopic_topics_for_coherence, texts=tokens, dictionary=dictionary, coherence='c_v')
                            coherence_bt = coherence_model_bt.get_coherence()
                        else:
                            st.warning("Could not calculate coherence. Topics, tokens, or dictionary might be missing/empty.")

                    except Exception as ce:
                        st.error(f"Error calculating BERTopic coherence: {ce}")
                else:
                    st.warning("Tokens/Dictionary not generated. Cannot calculate BERTopic coherence. Run 'Prepare Tokens' first.")

                st.session_state['bt_metrics'] = {'coherence_cv': coherence_bt}
                # Store topic info DataFrame
                st.session_state['bt_topic_info_df'] = topic_model.get_topic_info()


            st.session_state['bt_results_available'] = True
            st.success("BERTopic Model Training Complete!")
            st.rerun() # Rerun to display results

        except Exception as e:
            st.error(f"Error during BERTopic processing: {e}")
            st.session_state['bt_results_available'] = False
            # Clear potentially incomplete results
            st.session_state['bt_model'] = None
            st.session_state['bt_metrics'] = None
            st.session_state['bt_topic_info_df'] = None


# --- Display BERTopic Results ---
if st.session_state.get('bt_results_available', False):
    st.markdown("---")
    st.subheader("BERTopic Model Results")

    # Display Timings
    train_time_bt = st.session_state.get('bt_training_time')
    if train_time_bt is not None:
        st.write(f"**BERTopic Modeling Time:** {train_time_bt:.3f} seconds")

    # Display Metrics
    metrics_bt = st.session_state.get('bt_metrics', {})
    if metrics_bt:
        coh_cv_bt = metrics_bt.get('coherence_cv', None)
        st.write(f"**Coherence Score (c_v):** {coh_cv_bt:.4f}" if isinstance(coh_cv_bt, (int, float)) else "**Coherence Score (c_v):** N/A (Calculation failed or not performed)")

    # Display Topics Table
    topic_info_df = st.session_state.get('bt_topic_info_df')
    if topic_info_df is not None:
        st.write("**Discovered Topics Info**") # Removed bracket part (Requirement 6)
        st.dataframe(topic_info_df)
        # Explain Columns (Requirement 6)
        st.markdown("""
        **Table Column Explanations:**
        * **Topic:** The unique ID assigned to the topic. Topic -1 contains outlier documents that don't fit well into any specific cluster.
        * **Count:** The number of documents assigned to this topic.
        * **Name:** An automatically generated name for the topic (e.g., `-1_outlier_topic_documents`). You can often customize these.
        * **Representation:** The list of top words that represent the topic, based on the c-TF-IDF scoring used by BERTopic.
        * **Representative_Docs:** A sample of the actual documents from your dataset that are most representative of this topic, based on their proximity to the topic's embedding centroid.
        """)


    # Display DataFrame preview with topics
    st.write("**Preview with BERTopic Topic Assignment:**")
    df_with_topics_bt = st.session_state.get('df_processed')
    if df_with_topics_bt is not None and 'dominant_topic_bt' in df_with_topics_bt.columns:
        st.dataframe(df_with_topics_bt[['cleaned_text', 'dominant_topic_bt']].head())
    else:
        st.warning("Could not display preview. Processed data or BERTopic topics missing.")

    st.success("BERTopic results generated. Proceed to compare or export.")


elif 'bt_tokens' in st.session_state and st.session_state['bt_tokens'] is not None:
     # If tokens are ready but model not trained, show info
    st.info("Tokens for coherence are ready. Click the 'Train BERTopic' button above.")
