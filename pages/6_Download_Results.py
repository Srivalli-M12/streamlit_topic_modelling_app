import streamlit as st
import pandas as pd
from utils import convert_df_to_csv

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
st.header("6. Download Results")

# --- Check Prerequisites ---
if 'df_processed' not in st.session_state or st.session_state['df_processed'] is None:
    st.warning("⬅️ Processed data not found. Please run preprocessing first.")
    st.stop()

lda_available = st.session_state.get('lda_results_available', False)
bt_available = st.session_state.get('bt_results_available', False)

df_processed = st.session_state['df_processed']

# --- Select Columns for Download ---
st.subheader("Select Data to Include in Download")

# Base columns
cols_options = ['original_text', 'cleaned_text'] # Add original text option
cols_selected_base = st.multiselect(
    "Select base text columns:",
    cols_options,
    default=['cleaned_text'] # Default to cleaned
)

# Optional original text column retrieval
if 'original_text' in cols_selected_base:
     if 'df_raw' in st.session_state and st.session_state.get('text_column') in st.session_state['df_raw'].columns:
         # Merge original text back based on index (assuming order is preserved)
         original_text_col = st.session_state['text_column']
         # Ensure indices align before merging if lengths differ (use df_processed index)
         df_processed['original_text'] = st.session_state['df_raw'].loc[df_processed.index, original_text_col]
     else:
         st.warning("Original raw data or text column selection not found. Cannot include original text.")
         cols_selected_base.remove('original_text')


# Model columns
model_cols_options = []
if lda_available and 'dominant_topic_lda' in df_processed.columns:
    model_cols_options.append('dominant_topic_lda')
if bt_available and 'dominant_topic_bt' in df_processed.columns:
    model_cols_options.append('dominant_topic_bt')

cols_selected_model = []
if model_cols_options:
     cols_selected_model = st.multiselect(
        "Select model topic assignment columns:",
        model_cols_options,
        default=model_cols_options # Default to all available
    )
else:
    st.info("No topic model results available to include.")

# Combine selected columns
final_cols_to_download = cols_selected_base + cols_selected_model

# --- Prepare Download ---
if final_cols_to_download:
    st.subheader("Download Combined CSV")
    df_download = df_processed[final_cols_to_download].copy() # Create df with selected columns

    # Provide a preview
    st.write("Preview of data to be downloaded:")
    st.dataframe(df_download.head())

    # Generate CSV data
    csv_data_download = convert_df_to_csv(df_download)
    file_name_download = "topic_modeling_combined_results.csv"

    st.download_button(
       label="⬇️ Download Selected Data as CSV",
       data=csv_data_download,
       file_name=file_name_download,
       mime='text/csv',
       key='download_combined_csv_btn'
    )
else:
    st.warning("Please select at least one column to include in the download.")

# --- Optional: Download Raw Model Info ---
st.markdown("---")
st.subheader("Download Raw Model Topic Information (Optional)")

if lda_available and 'lda_topics_df' in st.session_state and st.session_state['lda_topics_df'] is not None:
     df_lda_topics = st.session_state['lda_topics_df']
     csv_lda_topics = convert_df_to_csv(df_lda_topics)
     st.download_button(
        label="⬇️ Download LDA Topics CSV",
        data=csv_lda_topics,
        file_name="lda_topic_words.csv",
        mime='text/csv',
        key='download_lda_topics_btn'
     )

if bt_available and 'bt_topic_info_df' in st.session_state and st.session_state['bt_topic_info_df'] is not None:
     df_bt_topics = st.session_state['bt_topic_info_df']
     csv_bt_topics = convert_df_to_csv(df_bt_topics)
     st.download_button(
        label="⬇️ Download BERTopic Info CSV",
        data=csv_bt_topics,
        file_name="bertopic_info.csv",
        mime='text/csv',
        key='download_bt_info_btn'
     )
