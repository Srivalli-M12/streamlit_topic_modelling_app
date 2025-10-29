import streamlit as st
import pandas as pd
import plotly.graph_objects as go

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

st.header("5. Compare Model Performance")

# --- Check Prerequisites ---
lda_available = st.session_state.get('lda_results_available', False)
bt_available = st.session_state.get('bt_results_available', False)

if not lda_available and not bt_available:
    st.warning("⬅️ Please run at least one topic model (LDA or BERTopic) first.")
    st.stop()
if 'df_processed' not in st.session_state or st.session_state['df_processed'] is None:
     st.warning("⬅️ Processed data not found. Please ensure preprocessing was completed.")
     st.stop()

# --- Build Comparison Table ---
comparison_data = []

# LDA Data
if lda_available:
    lda_metrics = st.session_state.get('lda_metrics', {})
    lda_train_time = st.session_state.get('lda_training_time', None)
    lda_assign_time = st.session_state.get('lda_assignment_time', None)
    total_lda_time = (lda_train_time or 0) + (lda_assign_time or 0)
    lda_coh = lda_metrics.get('coherence_cv')
    lda_perp = lda_metrics.get('perplexity')
    comparison_data.append({
        'Model': 'LDA (Gensim)',
        'Coherence (c_v)': f"{lda_coh:.4f}" if isinstance(lda_coh, float) else 'N/A',
        'Perplexity': f"{lda_perp:.4f}" if isinstance(lda_perp, float) else 'N/A',
        'Total Time (s)': f"{total_lda_time:.3f}" if total_lda_time > 0 else 'N/A'
    })

# BERTopic Data
if bt_available:
    bt_metrics = st.session_state.get('bt_metrics', {})
    bt_train_time = st.session_state.get('bt_training_time', None)
    # BERTopic assigns topics during fit_transform, so training time is the main component
    total_bt_time = bt_train_time
    bt_coh = bt_metrics.get('coherence_cv')
    comparison_data.append({
        'Model': 'BERTopic',
        'Coherence (c_v)': f"{bt_coh:.4f}" if isinstance(bt_coh, float) else 'N/A',
        'Perplexity': 'N/A', # Not comparable
        'Total Time (s)': f"{total_bt_time:.3f}" if total_bt_time is not None else 'N/A'
    })

# --- Display Comparison Table ---
st.subheader("📊 Metrics Comparison")
if comparison_data:
    df_compare = pd.DataFrame(comparison_data).set_index('Model')
    st.table(df_compare)
    st.markdown("""
    * **Coherence (c_v):** Higher is generally better (topic words co-occur meaningfully).
    * **Perplexity (LDA only):** Lower is generally better (model fits data well). Not directly comparable to BERTopic.
    * **Total Time:** Approximate time for model training and topic assignment.
    """)
else:
    st.info("No models have been run yet.")

# --- Display Comparison Graph (Coherence & Time) ---
st.subheader("📈 Visual Comparison")
if len(comparison_data) > 0:
    models = [d['Model'] for d in comparison_data]

    # --- Coherence Score Chart ---
    try:
        # Attempt to convert coherence scores to float, handle errors
        coherence_scores = []
        for d in comparison_data:
            coh_str = d.get('Coherence (c_v)', 'N/A')
            if coh_str != 'N/A':
                try:
                    coherence_scores.append(float(coh_str))
                except (ValueError, TypeError):
                    coherence_scores.append(0) # Use 0 if conversion fails
                    st.warning(f"Could not convert coherence '{coh_str}' to float for plotting.")
            else:
                 coherence_scores.append(0) # Use 0 if N/A

        if any(c > 0 for c in coherence_scores): # Check if there's valid data to plot
            fig_coh = go.Figure(data=[go.Bar(
                x=models,
                y=coherence_scores,
                name='Coherence (c_v)',
                marker_color='indianred',
                text=[f"{c:.4f}" for c in coherence_scores], # Format text label
                textposition='auto'
            )])
            fig_coh.update_layout(
                title='Model Coherence Comparison (c_v)',
                xaxis_title='Model',
                yaxis_title='Coherence Score (Higher is better)',
                bargap=0.4
            )
            st.plotly_chart(fig_coh)
        else:
            st.info("No valid Coherence (c_v) scores available to plot.")

    except Exception as e:
         st.error(f"An error occurred while plotting coherence scores: {e}")


    # --- Time Comparison Chart ---
    try:
        # Attempt to convert times to float, handle errors
        times = []
        for d in comparison_data:
            time_str = d.get('Total Time (s)', 'N/A')
            if time_str != 'N/A':
                try:
                    times.append(float(time_str))
                except (ValueError, TypeError):
                    times.append(0) # Use 0 if conversion fails
                    st.warning(f"Could not convert time '{time_str}' to float for plotting.")
            else:
                times.append(0) # Use 0 if N/A

        if any(t > 0 for t in times): # Check if there's valid data to plot
            fig_time = go.Figure(data=[go.Bar(
                x=models,
                y=times,
                name='Total Time (s)',
                marker_color='lightsalmon',
                text=[f"{t:.2f}s" for t in times], # Format text label
                textposition='auto'
            )])
            fig_time.update_layout(
                title='Model Total Time Comparison',
                xaxis_title='Model',
                yaxis_title='Time (seconds)',
                bargap=0.4
            )
            st.plotly_chart(fig_time)
        else:
            st.info("No valid Time scores available to plot.")

    except Exception as e:
         st.error(f"An error occurred while plotting time scores: {e}")


else:
    st.info("Run at least one model to see visual comparisons.")


# --- Display Combined Data Preview ---
# (Keep the rest of the file from here downwards the same)
st.markdown("---")
st.subheader("📄 Data Preview with Both Topic Assignments")
# ... (rest of the code remains unchanged) ...
df_processed = st.session_state.get('df_processed')

if df_processed is not None:
    cols_to_show = ['cleaned_text']
    if lda_available and 'dominant_topic_lda' in df_processed.columns:
        cols_to_show.append('dominant_topic_lda')
    if bt_available and 'dominant_topic_bt' in df_processed.columns:
        cols_to_show.append('dominant_topic_bt')

    if len(cols_to_show) > 1: # Check if at least one topic column exists
        st.dataframe(df_processed[cols_to_show])
    else:
        st.warning("No topic assignments found in the processed data.")
else:
    st.warning("Processed data not available.")
