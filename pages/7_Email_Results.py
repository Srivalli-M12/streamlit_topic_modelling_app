import streamlit as st
import pandas as pd
from utils import send_email_alert, send_email_with_attachment, convert_df_to_csv
import os
import tempfile

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
st.header("7. Email Results")

# --- Check Prerequisites ---
df_processed = st.session_state.get('df_processed', None)
lda_available = st.session_state.get('lda_results_available', False)
bt_available = st.session_state.get('bt_results_available', False)

if df_processed is None:
     st.warning("⬅️ Processed data not found. Please load and preprocess data first.")
     st.stop()
if not lda_available and not bt_available:
     st.warning("⬅️ No model results found. Please run LDA or BERTopic first.")
     # Allow sending just processed data if needed? For now, require model results.
     st.stop()


# --- Email Configuration ---
st.subheader("Configure Email")

# Get email credentials from secrets
sender_email = st.secrets.get("gmail_sender_email", "")
sender_password = st.secrets.get("gmail_app_password", "") # Use App Password

if not sender_email or not sender_password:
    st.error("Email credentials not found in Streamlit secrets (.streamlit/secrets.toml). Please configure `gmail_sender_email` and `gmail_app_password` to enable emailing.")
    email_configured = False
    st.stop() # Stop if email isn't configured
else:
    email_configured = True
    #st.caption(f"Emails will be sent from: {sender_email}")

# Recipient Email - Default to logged-in user (Requirement 1)
default_recipient = st.session_state.get('current_user_email', "")
recipient_email = st.text_input("Recipient email address:", value=default_recipient, key='email_recipient_input')

email_subject_default = "Topic Modeling Results Summary"
email_subject = st.text_input("Email Subject:", value=email_subject_default, key='email_subject_input')

email_body_default = "Attached are the results from the topic modeling process."
email_body = st.text_area("Email Body:", value=email_body_default, height=100, key='email_body_input')

# --- Select Content to Email ---
st.subheader("Select Content to Send")

email_options = []
# Option to send combined CSV
cols_to_email = ['cleaned_text'] # Start with cleaned text
if lda_available and 'dominant_topic_lda' in df_processed.columns:
    cols_to_email.append('dominant_topic_lda')
if bt_available and 'dominant_topic_bt' in df_processed.columns:
    cols_to_email.append('dominant_topic_bt')
if len(cols_to_email) > 1: # Only offer CSV if topics exist
    email_options.append('Send Combined Results CSV')

# Option to send LDA Summary
if lda_available and 'lda_model' in st.session_state:
    email_options.append('Send LDA Topic Summary Only')

# Option to send BERTopic Info
if bt_available and 'bt_topic_info_df' in st.session_state and st.session_state['bt_topic_info_df'] is not None:
     email_options.append('Send BERTopic Info CSV')


if not email_options:
    st.warning("No results available to email.")
    st.stop()

email_content_choice = st.radio(
    "Choose what to email:",
    email_options,
    key='email_content_radio'
    )

# --- Send Button ---
if st.button("📧 Send Email", key="send_email_final_btn"):
    if not recipient_email:
        st.error("Please enter a recipient email address.")
    elif '@' not in recipient_email or '.' not in recipient_email: # Basic validation
         st.error("Please enter a valid email address.")
    else:
        success = False # Flag for final message

        if email_content_choice == 'Send Combined Results CSV':
            # Create a temporary file for the attachment
            df_email = df_processed[cols_to_email].copy()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmpfile:
                df_email.to_csv(tmpfile.name, index=False)
                temp_file_path = tmpfile.name

            with st.spinner(f"Sending CSV to {recipient_email}..."):
                 success = send_email_with_attachment(
                    to_email=recipient_email,
                    subject=email_subject,
                    body=email_body,
                    file_path=temp_file_path,
                    sender_email=sender_email,
                    sender_password=sender_password
                )
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        elif email_content_choice == 'Send LDA Topic Summary Only':
            lda_model = st.session_state.get('lda_model')
            if lda_model and 'dominant_topic_lda' in df_processed.columns:
                topic_counts = df_processed['dominant_topic_lda'].value_counts().sort_index().to_dict()
                summary_lines = [f"--- Topic Distribution ({len(df_processed)} Records) ---"]
                summary_lines.extend([f"Topic {k}: {v} documents ({v/len(df_processed)*100:.1f}%)" for k, v in topic_counts.items() if k != -1 and k is not None]) # Exclude -1 if present
                summary_lines.append("\n--- Top Words per Topic (LDA) ---")
                topics = lda_model.print_topics(num_words=10)
                for idx, topic in topics:
                    summary_lines.append(f"Topic {idx}: {topic}")

                full_summary = "\n".join(summary_lines)

                with st.spinner(f"Sending LDA Summary to {recipient_email}..."):
                     success = send_email_alert(
                        to_email=recipient_email,
                        subject=f"{email_subject} - LDA Summary",
                        message=full_summary,
                        sender_email=sender_email,
                        sender_password=sender_password
                    )
            else:
                st.error("LDA results are not available to generate a summary.")

        elif email_content_choice == 'Send BERTopic Info CSV':
             df_bt_topics = st.session_state['bt_topic_info_df']
             with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmpfile:
                df_bt_topics.to_csv(tmpfile.name, index=False)
                temp_file_path = tmpfile.name

             with st.spinner(f"Sending BERTopic Info CSV to {recipient_email}..."):
                 success = send_email_with_attachment(
                    to_email=recipient_email,
                    subject=f"{email_subject} - BERTopic Info",
                    body="Attached is the BERTopic model information.",
                    file_path=temp_file_path,
                    sender_email=sender_email,
                    sender_password=sender_password
                 )
             # Clean up
             if os.path.exists(temp_file_path):
                 os.remove(temp_file_path)

        # Display final status
        if success:
             st.success(f"Email sent successfully to {recipient_email}!")
        # else: # Error message is displayed within the send functions
        #      st.error("Email sending failed. Check console/logs if necessary.")
