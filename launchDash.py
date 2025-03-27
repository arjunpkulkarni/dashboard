import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import requests
 
# ----------------------------------
# Utility Functions
# ----------------------------------
 
def extract_disease_state(text):
    """
    Extracts disease state from a string by looking for content in parentheses.
    Special-cases 'DLBCL' if present.
    """
    if isinstance(text, str):
        # Look for all parentheses
        matches = re.findall(r'\((.*?)\)', text) 
        if "DLBCL" in text:
            return text
        elif matches:
            # Join the extracted parentheses contents
            return ', '.join(matches)
        else:
            # If no parentheses, return the original text
            return text
    return None
 
def classify_overlap(x):
    """
    Classify overlap percentage into one of four categories:
    - "0%" when x is 0.
    - ">0-50%" when x is greater than 0 and up to 50.
    - "50-70%" when x is greater than 50 and up to 70.
    - ">70%" when x is greater than 70.
    """
    if x == 0:
        return "0%"
    elif x > 0 and x <= 50:
        return ">0-50%"
    elif x > 50 and x <= 70:
        return "50-70%"
    elif x > 70:
        return ">70%"
 
def detect_outliers(df):
    """
    Applies classify_overlap to a numeric column (reuse_p_global) to label each row.
    Returns a small subset of columns with the classification.
    """
    df['outlier_class'] = df['reuse_p_global'].apply(classify_overlap)
    return df[['document_id', 'disease_state', 'reuse_p_global', 'outlier_class']]
 
def plot_stacked_bar(df, column='outlier_class'):
    """
    Plots a stacked bar chart of overlap categories by disease_state.
    The `column` argument should be a categorical column with overlap classes.
    """
    categories = [
        "0%",
        ">0-50%",
        "50-70%",
        ">70%"
    ]
    colors = ['#A9A9A9', '#87CEEB', '#4682B4', '#1E3A5F']
 
    # Get unique disease states
    disease_states = df['disease_state'].dropna().unique()
    # Abbreviate disease state to first 3 characters for brevity
    abbreviations = {state: state[:3].upper() for state in disease_states}
    df['disease_state'] = df['disease_state'].map(abbreviations)
 
    data = []
    for state in df['disease_state'].unique():
        subset = df[df['disease_state'] == state][column].value_counts(normalize=True) * 100
        row = [subset.get(cat, 0) for cat in categories]
        data.append(row)
 
    df_plot = pd.DataFrame(data, index=df['disease_state'].unique(), columns=categories)
   
    if df_plot.empty:
        st.warning("No data available to plot.")
        return
 
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot.plot(kind='bar', stacked=True, color=colors, ax=ax)
    ax.set_xlabel("Disease State")
    ax.set_ylabel("Percentage of Documents")
    ax.set_title("Global Document Overlap Categories by Disease State")
    ax.legend(title="Overlap Category")
    st.pyplot(fig)
 
def plot_local_document_overlap(df, column='local_outlier_class'):
    """
    Plots a stacked bar chart of overlap categories by disease_state based on local reuse.
    It creates a new classification based on 'reuse_p_local'.
    """
    categories = ["0%", ">0-50%", "50-70%", ">70%"]
    colors = ['#A9A9A9', '#87CEEB', '#4682B4', '#1E3A5F']
 
    # Compute local overlap class
    df['local_outlier_class'] = df['reuse_p_local'].apply(classify_overlap)
   
    # Get unique disease states and abbreviate them
    disease_states = df['disease_state'].dropna().unique()
    abbreviations = {state: state[:3].upper() for state in disease_states}
    df['disease_state'] = df['disease_state'].map(abbreviations)
 
    data = []
    for state in df['disease_state'].unique():
        subset = df[df['disease_state'] == state][column].value_counts(normalize=True) * 100
        row = [subset.get(cat, 0) for cat in categories]
        data.append(row)
   
    df_plot = pd.DataFrame(data, index=df['disease_state'].unique(), columns=categories)
   
    if df_plot.empty:
        st.warning("No data available to plot.")
        return
 
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot.plot(kind='bar', stacked=True, color=colors, ax=ax)
    ax.set_xlabel("Disease State")
    ax.set_ylabel("Percentage of Documents")
    ax.set_title("Local Document Overlap Categories by Disease State")
    ax.legend(title="Overlap Category")
    st.pyplot(fig)
 
def plot_slides_disease_state():
    # Read the tab-delimited files
    global_slides = pd.read_csv('output/global_documents.csv', sep='\t')
    local_slides = pd.read_csv('output/local_documents.csv', sep='\t')

    # Rename columns for consistency
    global_slides.rename(columns={'global_document': 'slide_id'}, inplace=True)
    local_slides.rename(columns={'local_document': 'slide_id'}, inplace=True)

    # Function to extract abbreviation from disease state
    def extract_abbreviation(text):
        if pd.isna(text):
            return None
        match = re.search(r'\((.*?)\)', str(text))
        if match:
            return match.group(1)
        return text

    # Extract abbreviations from disease states
    global_slides['disease_state'] = global_slides['disease_state'].apply(extract_abbreviation)
    local_slides['disease_state'] = local_slides['disease_state'].apply(extract_abbreviation)

    # Define the 4 disease states to show
    disease_states = ['AML', 'CLL', 'DLBCL', 'FL']

    # Filter for only the specified disease states
    global_slides = global_slides[global_slides['disease_state'].isin(disease_states)]
    local_slides = local_slides[local_slides['disease_state'].isin(disease_states)]

    # Check if we have any data after filtering
    if global_slides.empty and local_slides.empty:
        st.warning("No data available for the specified disease states (AML, CLL, DLBCL, FL).")
        return

    # Count slides by disease state
    global_counts = global_slides.groupby('disease_state').size().reset_index(name='count')
    local_counts = local_slides.groupby('disease_state').size().reset_index(name='count')

    # Plot Global Slides chart
    fig_global, ax_global = plt.subplots(figsize=(12, 6))
    global_counts.plot(x='disease_state', y='count', kind='bar', ax=ax_global, color='#4682B4')
    ax_global.set_xlabel('Disease State')
    ax_global.set_ylabel('Count of Global Slides')
    ax_global.set_title('Global Slides per Disease State')
    ax_global.set_xticklabels(global_counts['disease_state'], rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_global)

    # Plot Local Slides chart
    fig_local, ax_local = plt.subplots(figsize=(12, 6))
    local_counts.plot(x='disease_state', y='count', kind='bar', ax=ax_local, color='#4682B4')
    ax_local.set_xlabel('Disease State')
    ax_local.set_ylabel('Count of Local Slides')
    ax_local.set_title('Local Slides per Disease State')
    ax_local.set_xticklabels(local_counts['disease_state'], rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_local)
 
def plot_overlap_pie_charts(df):
    """
    For each disease state in the dataframe, this function creates a pair of pie charts—
    one for global overlap and one for local overlap—showing the percentage distribution
    of overlap categories.
 
    It assumes that:
    - 'reuse_p_global' and 'reuse_p_local' columns exist in df.
    - The global overlap categories are computed in the 'outlier_class' column.
    - The local overlap categories are computed in the 'local_outlier_class' column.
   
    If these classification columns are missing, they are computed using the
    classify_overlap function.
    """
 
    # Compute global and local overlap classification if not already present
    if 'outlier_class' not in df.columns:
        df['outlier_class'] = df['reuse_p_global'].apply(classify_overlap)
    if 'local_outlier_class' not in df.columns:
        df['local_outlier_class'] = df['reuse_p_local'].apply(classify_overlap)
 
    # Get unique disease states
    disease_states = df['disease_state'].unique()
 
    # Iterate over each disease state and plot pie charts for both global and local overlaps
    for state in disease_states:
        subset = df[df['disease_state'] == state]
       
        # Calculate percentage counts for global and local overlap categories
        global_counts = subset['outlier_class'].value_counts(normalize=True) * 100
        local_counts = subset['local_outlier_class'].value_counts(normalize=True) * 100
 
        # Create a figure with two subplots: one for global and one for local
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
       
        # Global Overlap Pie Chart
        axes[0].pie(
            global_counts,
            labels=global_counts.index,
            autopct='%1.1f%%',
            startangle=90
        )
        axes[0].set_title(f"{state} - Global Overlap")
 
        # Local Overlap Pie Chart
        axes[1].pie(
            local_counts,
            labels=local_counts.index,
            autopct='%1.1f%%',
            startangle=90
        )
        axes[1].set_title(f"{state} - Local Overlap")
       
        # Display the figure using Streamlit
        st.pyplot(fig)
 
def plot_slide_reuse_comparison():
    """
    Reads global and local slide data, computes slide reuse (Overlap vs No Overlap) for each disease state,
    and creates a grouped stacked bar chart. For each disease state, two bars are plotted side by side:
    one for Global slides (left) and one for Local slides (right).
 
    Global slides:
      - Overlap: slides present in both datasets (merge indicator 'both')
      - No Overlap: slides present only in the global set (merge indicator 'left_only')
 
    Local slides:
      - Overlap: slides present in both datasets (merge indicator 'both')
      - No Overlap: slides present only in the local set (merge indicator 'right_only')
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import streamlit as st
    from matplotlib.patches import Patch
 
    # Read the tab-delimited CSV files
    global_slides = pd.read_csv('output/global_slides.csv', sep='\t')
    local_slides = pd.read_csv('output/local_slides.csv', sep='\t')
 
    # Rename columns for consistency
    global_slides.rename(columns={'global_document': 'slide_id'}, inplace=True)
    local_slides.rename(columns={'local_document': 'slide_id'}, inplace=True)
 
    # Merge datasets on common columns with an indicator for merge status
    merged_slides = pd.merge(
        global_slides,
        local_slides,
        on=['slide_id', 'document_name', 'document_number', 'disease_state'],
        how='outer',
        indicator=True
    )
 
    # Abbreviate disease_state to the first 3 characters for brevity
    merged_slides['disease_state'] = merged_slides['disease_state'].str[:3]
 
    # Group by disease_state and merge indicator
    grouped = merged_slides.groupby(['disease_state', '_merge']).size().unstack(fill_value=0)
 
    # For Global Slides:
    # Overlap: slides present in both datasets (indicator == 'both')
    # No Overlap: slides present only in the global set (indicator == 'left_only')
    global_data = pd.DataFrame({
        "Overlap": grouped.get("both", 0),
        "No Overlap": grouped.get("left_only", 0)
    })
 
    # For Local Slides:
    # Overlap: slides present in both datasets (indicator == 'both')
    # No Overlap: slides present only in the local set (indicator == 'right_only')
    local_data = pd.DataFrame({
        "Overlap": grouped.get("both", 0),
        "No Overlap": grouped.get("right_only", 0)
    })
 
    # Ensure both DataFrames have the same disease states
    disease_states = sorted(set(global_data.index).union(set(local_data.index)))
    global_data = global_data.reindex(disease_states, fill_value=0)
    local_data = local_data.reindex(disease_states, fill_value=0)
 
    # Set up the bar positions
    n = len(disease_states)
    x = np.arange(n)
    width = 0.35
 
    fig, ax = plt.subplots(figsize=(12, 6))
 
    # Global stacked bars (positioned at x - width/2)
    global_no = global_data["No Overlap"].values
    global_overlap = global_data["Overlap"].values
    ax.bar(x - width/2, global_no, width, color='#A9A9A9')
    ax.bar(x - width/2, global_overlap, width, bottom=global_no, color='#4682B4')
 
    # Local stacked bars (positioned at x + width/2)
    local_no = local_data["No Overlap"].values
    local_overlap = local_data["Overlap"].values
    ax.bar(x + width/2, local_no, width, color='#A9A9A9')
    ax.bar(x + width/2, local_overlap, width, bottom=local_no, color='#4682B4')
 
    # Customize the axes and title
    ax.set_xlabel("Disease State (Abbrev.)")
    ax.set_ylabel("Count of Slides")
    ax.set_title("Slide Reuse Comparison by Disease State (Left: Global, Right: Local)")
    ax.set_xticks(x)
    ax.set_xticklabels(disease_states)
 
    # Create a custom legend for the overlap status
    legend_elements = [
        Patch(facecolor='#4682B4', label='Overlap'),
        Patch(facecolor='#A9A9A9', label='No Overlap')
    ]
    ax.legend(handles=legend_elements, title="Slide Status")
 
    plt.tight_layout()
    st.pyplot(fig)
 
@st.cache_data
def load_and_process_data(global_csv_path, local_csv_path):
    """
    Loads two CSV files (tab-delimited), merges them, and processes the columns.
    Adjust the paths/column names to match your actual data.
    """
    try:
        # Load tab-delimited CSVs
        global_docs = pd.read_csv(global_csv_path, sep='\t')
        local_docs = pd.read_csv(local_csv_path, sep='\t')
    except Exception as e:
        st.error(f"Error loading CSV files: {e}")
        return None, None
   
    # Rename for consistency if needed
    if 'global_document' in global_docs.columns:
        global_docs.rename(columns={'global_document': 'document_id'}, inplace=True)
    if 'local_document' in local_docs.columns:
        local_docs.rename(columns={'local_document': 'document_id'}, inplace=True)
   
    # Merge datasets
    merged_docs = pd.merge(
        global_docs,
        local_docs,
        on='document_id',
        suffixes=('_global', '_local'),
        how='outer'
    )
   
    # Create a single disease_state column from whichever is non-null
    if 'disease_state_global' in merged_docs.columns and 'disease_state_local' in merged_docs.columns:
        merged_docs['disease_state'] = merged_docs['disease_state_global'].fillna(
            merged_docs['disease_state_local']
        )
    else:
        merged_docs['disease_state'] = merged_docs.get('disease_state', 'Unknown')
   
    # If these columns exist, fill NaN with 0, then sum
    merged_docs['reuse_p_global'] = merged_docs.get('reuse_p_global', 0).fillna(0)
    merged_docs['reuse_p_local']  = merged_docs.get('reuse_p_local', 0).fillna(0)
   
    merged_docs['total_reuse'] = merged_docs['reuse_p_global'] + merged_docs['reuse_p_local']
    merged_docs['rounded_reuse'] = merged_docs['total_reuse'].round()
   
    # Process disease state to extract from parentheses
    merged_docs['disease_state'] = merged_docs['disease_state'].astype(str).apply(extract_disease_state)
   
    # Detect outliers / classify
    outliers_detected = detect_outliers(merged_docs)
   
    return merged_docs, outliers_detected
 
# ----------------------------------
# Main Streamlit App
# ----------------------------------
 
st.title("Document Overlap Analysis Dashboard")
 
st.sidebar.title("About")
st.sidebar.info("This dashboard visualizes document overlap analysis and similarity scores for pharmaceutical research documents.")
 
st.sidebar.subheader("Methodology")
st.sidebar.write("""
The similarity scores are computed based on document reuse percentages.
Overlap categories have been updated to:
- **0%**
- **>0-50%**
- **50-70%**
- **>70%**
""")
 
# ------------------------------
# Load data & display
# ------------------------------
 
# Replace these with your actual file paths:
global_csv_path = "output/global_documents.csv"
local_csv_path = "output/local_documents.csv"
 
merged_docs, outliers_detected = load_and_process_data(global_csv_path, local_csv_path)
 
if merged_docs is None:
    st.error("Failed to load and process data. Please check file paths or data format.")
else:
    # Create a tab bar with three tabs
    tabs = st.tabs(["Oncology", "Derm", "Rheum", "Gastro"])
 
    with tabs[0]:
        st.subheader("Document Overlap")
        # Filter for Oncology disease states
        oncology_states = ['AML', 'CLL', 'DLBCL', 'FL']
        oncology_docs = merged_docs[merged_docs['disease_state'].isin(oncology_states)]
        oncology_outliers = outliers_detected[outliers_detected['disease_state'].isin(oncology_states)]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Global Document Overlap")
            plot_stacked_bar(oncology_outliers, column='outlier_class')
        with col2:
            st.markdown("Local Document Overlap")
            plot_local_document_overlap(oncology_docs, column='local_outlier_class')

        plot_slides_disease_state()
        plot_overlap_pie_charts(oncology_docs)
        plot_slide_reuse_comparison()

    with tabs[1]:
        st.subheader("Derm")
        # Filter for Derm disease states
        derm_states = ['PSO', 'AD', 'HS']  # Add actual Derm disease states here
        derm_docs = merged_docs[merged_docs['disease_state'].isin(derm_states)]
        derm_outliers = outliers_detected[outliers_detected['disease_state'].isin(derm_states)]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Global Document Overlap")
            plot_stacked_bar(derm_outliers, column='outlier_class')
        with col2:
            st.markdown("Local Document Overlap")
            plot_local_document_overlap(derm_docs, column='local_outlier_class')

        plot_slides_disease_state()
        plot_overlap_pie_charts(derm_docs)
        plot_slide_reuse_comparison()

    with tabs[2]:
        st.subheader("Rheum")
        # Filter for Rheum disease states
        rheum_states = ['RA', 'PsA', 'AS']  # Add actual Rheum disease states here
        rheum_docs = merged_docs[merged_docs['disease_state'].isin(rheum_states)]
        rheum_outliers = outliers_detected[outliers_detected['disease_state'].isin(rheum_states)]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Global Document Overlap")
            plot_stacked_bar(rheum_outliers, column='outlier_class')
        with col2:
            st.markdown("Local Document Overlap")
            plot_local_document_overlap(rheum_docs, column='local_outlier_class')

        plot_slides_disease_state()
        plot_overlap_pie_charts(rheum_docs)
        plot_slide_reuse_comparison()

    with tabs[3]:
        st.subheader("Gastro")
        # Filter for Gastro disease states
        gastro_states = ['UC', 'CD', 'IBD']  # Add actual Gastro disease states here
        gastro_docs = merged_docs[merged_docs['disease_state'].isin(gastro_states)]
        gastro_outliers = outliers_detected[outliers_detected['disease_state'].isin(gastro_states)]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Global Document Overlap")
            plot_stacked_bar(gastro_outliers, column='outlier_class')
        with col2:
            st.markdown("Local Document Overlap")
            plot_local_document_overlap(gastro_docs, column='local_outlier_class')

        plot_slides_disease_state()
        plot_overlap_pie_charts(gastro_docs)
        plot_slide_reuse_comparison()
        