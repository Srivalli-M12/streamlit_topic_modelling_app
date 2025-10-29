import streamlit as st
import pandas as pd
from utils import tokenize, plot_timing_comparison
import time
from multiprocessing import Pool, cpu_count
import numpy as np
from gensim import corpora, models
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

st.header("3. LDA Topic Modeling (Gensim)")

# --- Check Prerequisites ---
if not st.session_state.get('cleaned_data_available', False) or 'df_processed' not in st.session_state:
    st.warning("⬅️ Please preprocess data on the 'Preprocess & WordCount' page first.")
    st.stop()

df_processed = st.session_state['df_processed']

# --- LDA Configuration ---
num_topics_lda = st.number_input("Enter the number of topics for LDA:", min_value=2, max_value=50,
                                 value=st.session_state.get('lda_num_topics', 5), step=1, key='lda_num_topics_input')
# Store selection in session state
st.session_state['lda_num_topics'] = num_topics_lda


# --- Tokenization for LDA ---
st.subheader("Tokenization (Required for LDA Dictionary/Corpus)")
st.markdown("""
This step splits the cleaned text into individual words (tokens), which are needed to build the vocabulary (dictionary) and document representations (corpus) for LDA.
We compare sequential vs. parallel processing times here.
""")

# Button to trigger tokenization timing
if st.button("⏱️ Prepare Tokens & Compare Times", key="lda_tokenize_btn"):
    texts_to_tokenize = df_processed["cleaned_text"].astype(str).dropna().tolist()
    if not texts_to_tokenize:
        st.warning("No cleaned text available for tokenization.")
        st.stop()

    with st.spinner("Running Sequential Tokenization..."):
        start_seq = time.perf_counter()
        tokens_seq = [tokenize(text) for text in texts_to_tokenize]
        end_seq = time.perf_counter()
        st.session_state['lda_tokenization_seq_time'] = end_seq - start_seq

    with st.spinner("Running Parallel Tokenization..."):
        start_mp = time.perf_counter()
        try:
            num_processes = max(1, cpu_count() - 1)
            # Simple map should be okay for tokenization as it's less complex than counting
            with Pool(processes=num_processes) as pool:
                tokens_mp = pool.map(tokenize, texts_to_tokenize)
            end_mp = time.perf_counter()
            st.session_state['lda_tokenization_mp_time'] = end_mp - start_mp
            # Store tokens for LDA use (use parallel result if successful)
            st.session_state['lda_tokens'] = tokens_mp
        except Exception as e:
            st.error(f"Multiprocessing error during tokenization: {e}. Storing sequential results.")
            st.session_state['lda_tokenization_mp_time'] = None
            st.session_state['lda_tokens'] = tokens_seq # Fallback

    # Plot comparison and store figure
    seq_time = st.session_state.get('lda_tokenization_seq_time', 0)
    mp_time = st.session_state.get('lda_tokenization_mp_time', seq_time) # Use seq_time if mp failed
    fig = plot_timing_comparison(seq_time, mp_time if mp_time is not None else 0, "LDA Tokenization")
    st.session_state['lda_tokenization_fig'] = fig
    st.session_state['lda_tokenization_times'] = {'seq': seq_time, 'mp': mp_time}
    st.rerun() # Rerun to display results

# --- Display Tokenization Results ---
if 'lda_tokens' in st.session_state and st.session_state['lda_tokens'] is not None:
    st.write("**Tokenization Timing Results:**")
    seq_time_lda_tok = st.session_state.get('lda_tokenization_seq_time')
    mp_time_lda_tok = st.session_state.get('lda_tokenization_mp_time')
    if seq_time_lda_tok is not None:
         st.write(f"- Sequential Time: {seq_time_lda_tok:.3f} seconds")
    if mp_time_lda_tok is not None:
         st.write(f"- Multiprocessing Time: {mp_time_lda_tok:.3f} seconds")
    else:
         st.write("- Multiprocessing failed or was not run.")

    fig_lda_tok = st.session_state.get('lda_tokenization_fig')
    if fig_lda_tok:
        st.plotly_chart(fig_lda_tok)
        st.markdown("""
        *Note on Timing:* Similar to word counting, the overhead for multiprocessing might make it slower than sequential processing for tokenization, especially if the dataset size isn't very large.
        """)

    # Show Token Preview (Requirement 5)
    st.write("**Preview of Tokenized Text (First 3 Documents):**")
    for i, doc_tokens in enumerate(st.session_state['lda_tokens'][:3]):
        st.text(f"Doc {i+1}: {' | '.join(doc_tokens[:20])} {'...' if len(doc_tokens) > 20 else ''}") # Show first 20 tokens


    # --- Run LDA Model ---
    st.markdown("---")
    st.subheader("Train LDA Model")
    if st.button(f"🔥 Train LDA with {num_topics_lda} Topics", key="train_lda_btn"):
        tokens = st.session_state['lda_tokens']
        # Check again in case state was cleared between button clicks
        if not tokens:
            st.error("Token list is empty. Please re-run tokenization.")
            st.stop()

        try:
            with st.spinner(f"Creating Dictionary and Corpus..."):
                dictionary = corpora.Dictionary(tokens)
                corpus = [dictionary.doc2bow(text) for text in tokens]
                # Store dictionary and corpus in session state
                st.session_state['lda_dictionary'] = dictionary
                st.session_state['lda_corpus'] = corpus

            with st.spinner(f"Training LDA Model ({num_topics_lda} topics)..."):
                start_train = time.perf_counter()
                lda_model = models.LdaModel(
                    corpus=corpus,
                    num_topics=num_topics_lda,
                    id2word=dictionary,
                    passes=10, # Good default
                    random_state=42 # for reproducibility
                )
                end_train = time.perf_counter()
                st.session_state['lda_model'] = lda_model
                st.session_state['lda_training_time'] = end_train - start_train

            # Assign Dominant Topic
            with st.spinner("Assigning dominant topics..."):
                start_assign = time.perf_counter()
                def get_dominant_topic(bow):
                    if not bow: return -1 # Assign -1 or None for empty docs
                    try:
                        topics = lda_model.get_document_topics(bow, minimum_probability=0.1) # Set a threshold
                        if topics:
                             return max(topics, key=lambda x: x[1])[0]
                        else:
                             return -1 # Assign -1 if no topic meets threshold
                    except Exception:
                        return -1 # Handle errors

                dominant_topics = [get_dominant_topic(doc) for doc in corpus]
                # Ensure the column exists before assigning
                if 'dominant_topic_lda' not in df_processed.columns:
                     df_processed['dominant_topic_lda'] = None # Initialize column if needed
                df_processed['dominant_topic_lda'] = dominant_topics

                # Update the *main* processed df in session state
                st.session_state['df_processed'] = df_processed
                end_assign = time.perf_counter()
                st.session_state['lda_assignment_time'] = end_assign - start_assign

            # Calculate Metrics
            with st.spinner("Calculating Coherence and Perplexity..."):
                coherence_model_lda = CoherenceModel(model=lda_model, texts=tokens, dictionary=dictionary, coherence='c_v')
                coherence_lda = coherence_model_lda.get_coherence()
                perplexity_lda = lda_model.log_perplexity(corpus) # Lower is generally better
                st.session_state['lda_metrics'] = {'coherence_cv': coherence_lda, 'perplexity': perplexity_lda}

                # Store topics as a DataFrame for easier access/display
                topics_data = []
                num_words_to_show = 10
                for idx, topic_str in lda_model.print_topics(num_words=num_words_to_show):
                     words = [word.split('*')[1].replace('"', '').strip() for word in topic_str.split(' + ')]
                     topics_data.append({'Topic': idx, 'Top Words': ', '.join(words)})
                st.session_state['lda_topics_df'] = pd.DataFrame(topics_data)


            st.session_state['lda_results_available'] = True
            st.success("LDA Model Training Complete!")
            st.rerun() # Rerun to display results

        except Exception as e:
            st.error(f"Error during LDA processing: {e}")
            st.session_state['lda_results_available'] = False
            # Clear potentially incomplete results
            st.session_state['lda_model'] = None
            st.session_state['lda_metrics'] = None
            st.session_state['lda_topics_df'] = None


# --- Display LDA Results ---
if st.session_state.get('lda_results_available', False):
    st.markdown("---")
    st.subheader("LDA Model Results")

    # Display Timings
    train_time = st.session_state.get('lda_training_time', None)
    assign_time = st.session_state.get('lda_assignment_time', None)
    if train_time is not None:
        st.write(f"**LDA Training Time:** {train_time:.3f} seconds")
    if assign_time is not None:
        st.write(f"**Topic Assignment Time:** {assign_time:.3f} seconds")

    # Display Metrics
    metrics = st.session_state.get('lda_metrics', {})
    if metrics:
        coh_cv = metrics.get('coherence_cv', 'N/A')
        perp = metrics.get('perplexity', 'N/A')
        st.write(f"**Coherence Score (c_v):** {coh_cv:.4f}" if isinstance(coh_cv, (int, float)) else "**Coherence Score (c_v):** N/A")
        st.write(f"**Perplexity:** {perp:.4f}" if isinstance(perp, (int, float)) else "**Perplexity:** N/A")


    # Display Topics Table
    topics_df = st.session_state.get('lda_topics_df')
    if topics_df is not None:
        st.write("**Discovered Topics & Top Words:**")
        st.dataframe(topics_df)

    # Display DataFrame preview with topics
    st.write("**Preview with Dominant LDA Topic:**")
    df_with_topics = st.session_state.get('df_processed')
    if df_with_topics is not None and 'dominant_topic_lda' in df_with_topics.columns:
        st.dataframe(df_with_topics[['cleaned_text', 'dominant_topic_lda']].head())
    else:
        st.warning("Could not display preview. Processed data or LDA topics missing.")

    st.success("LDA results generated. Proceed to compare or export.")

elif 'lda_tokens' in st.session_state and st.session_state['lda_tokens'] is not None:
    # If tokens are ready but model not trained, show info
    st.info("Tokens are ready. Click the 'Train LDA' button above.")
