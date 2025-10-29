import streamlit as st
import pandas as pd
import io
import os # Make sure os is imported for basename

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
    st.switch_page("app.py") # Redirect to login page after logout

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
st.header("1. Load and Prepare Dataset")

# --- Clear Session State Function ---
# (Keep this function as it is)
def clear_session_state_on_button():
    logged_in = st.session_state.get('logged_in', False)
    user_email = st.session_state.get('current_user_email', None)
    keys_to_clear = [k for k in st.session_state.keys() if k not in ['logged_in', 'current_user_email']]
    for key in keys_to_clear:
        del st.session_state[key]
    st.session_state['logged_in'] = logged_in
    st.session_state['current_user_email'] = user_email
    st.session_state['cleaned_data_available'] = False
    st.session_state['lda_results_available'] = False
    st.session_state['bt_results_available'] = False
    st.success("Previous data and model results cleared.")
    st.rerun()
    
# --- File Uploader ---
# (Keep this widget as it is)
uploaded_file = st.file_uploader("Upload a new CSV or Excel file to replace current data:", type=['csv', 'xlsx', 'xls'])

# --- Clear Button ---
# (Keep this section as it is)
st.warning("Clearing results will remove all loaded data and model outputs.")
if st.button("🚨 Clear All Data and Models"):
    clear_session_state_on_button()

# --- Display Existing Data (Moved Up) ---
# This block now runs every time the page loads, if data exists in state
if 'df' in st.session_state and st.session_state['df'] is not None:
    st.markdown("---") # Add separator
    st.subheader(f"Current Dataset: `{st.session_state.get('current_file_name', 'Unknown')}`")
    st.subheader("Dataset Preview (First 5 Rows)")
    st.dataframe(st.session_state['df'].head())

    st.subheader("Dataset Info")
    buffer = io.StringIO()
    # Use a temporary variable to avoid potential issues if df gets modified elsewhere
    temp_df_info = st.session_state['df']
    temp_df_info.info(buf=buffer)
    st.text(buffer.getvalue())
    st.markdown("---") # Add separator


# --- File Loading Logic ---
# (Keep this block largely as it is - it handles loading *new* files)
if uploaded_file is not None:
    # Check if it's a new file or the same one
    is_new_file = ('current_file_name' not in st.session_state or
                   st.session_state.current_file_name != uploaded_file.name or
                   st.session_state.get('df_raw') is None)

    if is_new_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                try:
                    df_loaded = pd.read_csv(uploaded_file)
                except UnicodeDecodeError:
                    try:
                       uploaded_file.seek(0)
                       df_loaded = pd.read_csv(uploaded_file, encoding='latin1')
                    except Exception as e:
                       st.error(f"Error reading CSV: {e}. Try saving the file as UTF-8.")
                       st.stop()
            else: # Excel file
                 df_loaded = pd.read_excel(uploaded_file)

            # Store the raw dataframe and working copy
            st.session_state['df_raw'] = df_loaded.copy()
            st.session_state['df'] = df_loaded.copy() # Use a copy to avoid modifying raw
            st.session_state['current_file_name'] = uploaded_file.name

            # **Important: Clear results from previous file when a new file is loaded**
            keys_to_reset = ['df_processed', 'cleaned_data_available', 'lda_results_available',
                             'bt_results_available', 'lda_model', 'bt_model', 'word_count_results',
                             'lda_metrics', 'bt_metrics', 'lda_tokenization_times', 'bt_tokenization_times',
                             'lda_topics_df', 'bt_topic_info_df', 'lda_tokenization_fig', 'bt_tokenization_fig',
                             'word_count_fig', 'lda_tokens', 'bt_tokens', 'lda_dictionary', 'bt_dictionary',
                             'lda_corpus', 'text_column', 'num_records'] # Also reset config selections
            for key in keys_to_reset:
                 if key in st.session_state:
                     st.session_state[key] = None # Reset to None or initial state
            st.session_state['cleaned_data_available'] = False # Ensure flags are reset
            st.session_state['lda_results_available'] = False
            st.session_state['bt_results_available'] = False

            st.success(f"Successfully loaded new file: `{uploaded_file.name}`. Previous results cleared.")
            # Rerun AFTER loading a new file to display its preview immediately
            st.rerun()

        except Exception as e:
            st.error(f"An error occurred loading the file: {e}")
            # Reset state variables related to file loading on error
            st.session_state['current_file_name'] = None
            st.session_state['df_raw'] = None
            st.session_state['df'] = None
            st.stop()


# --- Configuration Section (Moved Down, Conditional) ---
# This block now runs every time the page loads, if data exists in state
if 'df' in st.session_state and st.session_state['df'] is not None:
    st.subheader("Select Configuration")
    df_config = st.session_state['df'] # Get the current dataframe for config
    columns = df_config.columns.tolist()

    # Text column selection
    text_col_index = 0
    # If a text_column is already stored and exists in the current df, use it
    if st.session_state.get('text_column') in columns:
        text_col_index = columns.index(st.session_state['text_column'])
    # Otherwise, the default index 0 will be used

    text_column_choice = st.selectbox(
        "Select the column containing text data:",
        columns,
        index=text_col_index,
        key='text_column_selector'
    )
    # Update session state if selection changed (triggers rerun implicitly via selectbox)
    if st.session_state.get('text_column') != text_column_choice:
         st.session_state['text_column'] = text_column_choice
         # Invalidate downstream results if column changes
         st.session_state['cleaned_data_available'] = False
         st.session_state['lda_results_available'] = False
         st.session_state['bt_results_available'] = False
         st.warning("Text column changed. Preprocessing and model results need regeneration.")
         st.rerun() # Rerun to ensure warnings/state changes are reflected


    # Number of records selection
    max_records = len(df_config)
    min_slider_val = min(100, max_records) # Ensure min isn't > max

    num_records_value = st.session_state.get('num_records')
    if num_records_value is None: # Default if not set
        num_records_value = max_records
    # Clamp value within current valid range
    num_records_value = min(max(min_slider_val, num_records_value), max_records)

    num_records_choice = st.slider(
        "Select number of records to process:",
        min_value=min_slider_val,
        max_value=max_records,
        value=num_records_value,
        step=100 if max_records > 200 else 10,
        key='num_records_slider'
    )
     # Update session state if selection changed (triggers rerun implicitly via slider)
    if st.session_state.get('num_records') != num_records_choice:
        st.session_state['num_records'] = num_records_choice
        # Invalidate downstream results if count changes
        st.session_state['cleaned_data_available'] = False
        st.session_state['lda_results_available'] = False
        st.session_state['bt_results_available'] = False
        st.warning("Number of records changed. Preprocessing and model results need regeneration.")
        st.rerun() # Rerun to ensure warnings/state changes are reflected


    # Display selected config info (always displayed if df exists)
    st.info(f"Selected Text Column: `{st.session_state.get('text_column', 'Not selected')}`")
    st.info(f"Number of Records to Process: `{st.session_state.get('num_records', 'Not selected')}`")

    # Confirmation message
    if st.session_state.get('text_column') and st.session_state.get('num_records'):
        st.success("Configuration set. Proceed to the next step.")
    else:
        st.warning("Please select a text column and number of records.")

# Message if no file has ever been uploaded in this session
elif uploaded_file is None and ('df' not in st.session_state or st.session_state['df'] is None):
    st.info("Upload a file to get started or clear previous results using the button above.")

