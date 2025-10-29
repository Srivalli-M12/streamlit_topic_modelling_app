import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="Topic Modeling UI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Initialize Session State ---
# Use functions to avoid re-initializing unnecessarily
def initialize_state():
    # Flags
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'current_user_email' not in st.session_state:
        st.session_state['current_user_email'] = None
    if 'cleaned_data_available' not in st.session_state:
        st.session_state['cleaned_data_available'] = False
    if 'lda_results_available' not in st.session_state:
        st.session_state['lda_results_available'] = False
    if 'bt_results_available' not in st.session_state:
        st.session_state['bt_results_available'] = False

    # Data & Models (Initialize to None, pages will populate)
    data_keys = ['df_raw', 'df', 'df_processed', 'text_column', 'num_records', 'current_file_name']
    for key in data_keys:
        if key not in st.session_state:
            st.session_state[key] = None

    word_count_keys = ['word_count_seq_time', 'word_count_mp_time', 'word_count_results', 'word_count_fig']
    for key in word_count_keys:
        if key not in st.session_state:
            st.session_state[key] = None

    lda_keys = ['lda_tokens', 'lda_tokenization_times', 'lda_tokenization_fig', 'lda_dictionary', 'lda_corpus',
                'lda_model', 'lda_training_time', 'lda_assignment_time', 'lda_metrics', 'lda_topics_df']
    for key in lda_keys:
        if key not in st.session_state:
            st.session_state[key] = None

    bt_keys = ['bt_tokens', 'bt_dictionary', 'bt_tokenization_times', 'bt_tokenization_fig',
               'bt_model', 'bt_training_time', 'bt_metrics', 'bt_topic_info_df']
    for key in bt_keys:
        if key not in st.session_state:
            st.session_state[key] = None

initialize_state()

# --- Authentication Logic ---
def check_login(email, password):
    """Checks credentials against streamlit secrets."""
    correct_email = st.secrets.get("app_user_email")
    correct_password = st.secrets.get("app_password")

    if not correct_email or not correct_password:
        st.error("Login configuration missing in secrets.toml. Please add 'app_user_email' and 'app_password'.")
        return False

    if email == correct_email and password == correct_password:
        return True
    return False

def display_login_form():
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.svg", width=200)
    st.markdown("Please log in to continue.")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if check_login(email, password):
                st.session_state['logged_in'] = True
                st.session_state['current_user_email'] = email
                st.rerun() # Rerun the script to reflect the logged-in state
            else:
                st.error("😕 Email or Password incorrect")

def handle_logout():
    # Clear sensitive/user-specific state on logout
    st.session_state['logged_in'] = False
    st.session_state['current_user_email'] = None
    # Optionally clear other results on logout, or keep them
    # for key in st.session_state.keys():
    #     if key not in ['logged_in', 'current_user_email']: # Keep login state vars
    #         del st.session_state[key]
    # initialize_state() # Re-initialize state
    st.info("You have been logged out.")
    st.rerun()

# --- Main App Interface ---
if not st.session_state['logged_in']:
    display_login_form()
    st.stop() # Stop execution if not logged in
else:
    # --- Sidebar ---
    st.sidebar.success(f"Logged in as: {st.session_state['current_user_email']}")
    st.sidebar.markdown("---")
    st.sidebar.header("Navigation")
    st.sidebar.info("Select a page above to proceed through the workflow.")
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        handle_logout()

    # --- Main Page Content (after login) ---
    st.title("🎉 Welcome to the Topic Modeling Application!")
    st.markdown(f"""
    You are logged in as **{st.session_state['current_user_email']}**.

    Use the sidebar to navigate through the steps:
    1.  **Load Data:** Upload your dataset (CSV/Excel).
    2.  **Preprocess & WordCount:** Clean the text and analyze word frequencies.
    3.  **LDA Model:** Train a Latent Dirichlet Allocation model.
    4.  **BERTopic Model:** Train a BERTopic model.
    5.  **Compare Models:** View performance metrics side-by-side.
    6.  **Download Results:** Download the processed data with topic assignments.
    7.  **Email Results:** Send results via email.

    **Persistence:** Your uploaded data and results will persist during this session as you navigate between pages. Use the 'Clear All Data and Models' button on the 'Load Data' page to reset the application state.
    """)
    st.balloons()
