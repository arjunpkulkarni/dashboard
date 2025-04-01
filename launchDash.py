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
        subset = df[df['disease_state'] == state][column].value_counts()
        row = [subset.get(cat, 0) for cat in categories]
        data.append(row)
 
    df_plot = pd.DataFrame(data, index=df['disease_state'].unique(), columns=categories)
   
    if df_plot.empty:
        st.warning("No data available to plot.")
        return
 
    # Plot with increased figure size
    fig, ax = plt.subplots(figsize=(24, 14))  # Increased from (20, 12)
    df_plot.plot(kind='bar', stacked=True, color=colors, ax=ax)
    ax.set_xlabel("Disease State", fontsize=24)  # Increased from 20
    ax.set_ylabel("Number of Documents", fontsize=24)  # Increased from 20
    ax.set_title("Global Document Overlap Categories by Disease State", pad=30, fontsize=28, fontweight='bold')  # Increased from 24
    ax.legend(title="Overlap Category", fontsize=22, title_fontsize=24)  # Increased from 18/20
    
    # Rotate x-axis labels to horizontal
    plt.xticks(rotation=0, ha='center', fontsize=22)  # Increased from 18
    
    # Add value labels on the bars with increased font size
    for c in ax.containers:
        ax.bar_label(c, label_type='center', fontsize=20)  # Increased from 16
    
    # Adjust layout with more padding
    plt.tight_layout(pad=3.0)  # Increased padding
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
        subset = df[df['disease_state'] == state][column].value_counts()
        row = [subset.get(cat, 0) for cat in categories]
        data.append(row)
   
    df_plot = pd.DataFrame(data, index=df['disease_state'].unique(), columns=categories)
   
    if df_plot.empty:
        st.warning("No data available to plot.")
        return
 
    # Plot with increased figure size
    fig, ax = plt.subplots(figsize=(24, 14))  # Increased from (20, 12)
    df_plot.plot(kind='bar', stacked=True, color=colors, ax=ax)
    ax.set_xlabel("Disease State", fontsize=24)  # Increased from 20
    ax.set_ylabel("Number of Documents", fontsize=24)  # Increased from 20
    ax.set_title("Local Document Overlap Categories by Disease State", pad=30, fontsize=28, fontweight='bold')  # Increased from 24
    ax.legend(title="Overlap Category", fontsize=22, title_fontsize=24)  # Increased from 18/20
    
    # Rotate x-axis labels to horizontal
    plt.xticks(rotation=0, ha='center', fontsize=22)  # Increased from 18
    
    # Add value labels on the bars with increased font size
    for c in ax.containers:
        ax.bar_label(c, label_type='center', fontsize=20)  # Increased from 16
    
    # Adjust layout with more padding
    plt.tight_layout(pad=3.0)  # Increased padding
    st.pyplot(fig)

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
 
def plot_slide_reuse_comparison(df):
    """
    Creates slide reuse comparisons for each disease state (AML, CLL, DLBCL)
    showing global and local content on the same plot with document counts and percentages
    """
    # Calculate metrics for each disease state
    metrics = df.groupby('disease_state').agg({
        'num_pages_global': 'sum',
        'unique_global_page_count': 'sum',
        'num_pages_local': 'sum',
        'unique_local_page_count': 'sum'
    }).reset_index()
    
    # Calculate overlaps and percentages with 3 significant figures
    metrics['global_overlap'] = metrics['num_pages_global'] - metrics['unique_global_page_count']
    metrics['local_overlap'] = metrics['num_pages_local'] - metrics['unique_local_page_count']
    
    metrics['global_overlap_pct'] = (metrics['global_overlap'] / metrics['num_pages_global'] * 100).round(3)
    metrics['local_overlap_pct'] = (metrics['local_overlap'] / metrics['num_pages_local'] * 100).round(3)

    # Colors
    overlap_color = '#4CAF50'  # Green
    no_overlap_color = '#9E9E9E'  # Grey

    for _, row in metrics.iterrows():
        disease = row['disease_state']
        
        # Create single figure for each disease state with increased size
        fig, ax = plt.subplots(figsize=(36, 14))  # Increased from (32, 12)
        
        # Get overlap counts
        global_overlap = row['global_overlap']
        global_no_overlap = row['unique_global_page_count']
        
        local_overlap = row['local_overlap']
        local_no_overlap = row['unique_local_page_count']

        # Plot Global Content (top bar) - always show overlap (green) on the left
        ax.barh(1, global_overlap, color=overlap_color)
        ax.barh(1, global_no_overlap, left=global_overlap, color=no_overlap_color)
        # Add count and percentage labels with increased font size
        ax.text(global_overlap/2, 1, f"{int(global_overlap)}\n({row['global_overlap_pct']:.3f}%)", 
                va='center', ha='center', color='white', fontweight='bold', fontsize=24)  # Increased from 20
        ax.text(global_overlap + global_no_overlap/2, 1, f"{int(global_no_overlap)}\n({100-row['global_overlap_pct']:.3f}%)", 
                va='center', ha='center', color='white', fontweight='bold', fontsize=24)  # Increased from 20

        # Plot US Content (bottom bar) - always show overlap (green) on the left
        ax.barh(0, local_overlap, color=overlap_color)
        ax.barh(0, local_no_overlap, left=local_overlap, color=no_overlap_color)
        # Add count and percentage labels with increased font size
        ax.text(local_overlap/2, 0, f"{int(local_overlap)}\n({row['local_overlap_pct']:.3f}%)", 
                va='center', ha='center', color='white', fontweight='bold', fontsize=24)  # Increased from 20
        ax.text(local_overlap + local_no_overlap/2, 0, f"{int(local_no_overlap)}\n({100-row['local_overlap_pct']:.3f}%)", 
                va='center', ha='center', color='white', fontweight='bold', fontsize=24)  # Increased from 20

        # Add n values with increased font size
        ax.text(-0.02, 1, f"Global (n = {int(row['num_pages_global'])})", 
                va='center', ha='right', transform=ax.get_yaxis_transform(), fontsize=24)  # Increased from 20
        ax.text(-0.02, 0, f"US (n = {int(row['num_pages_local'])})", 
                va='center', ha='right', transform=ax.get_yaxis_transform(), fontsize=24)  # Increased from 20

        # Set up axes with more spacing
        ax.set_yticks([0, 1])
        ax.set_yticklabels([])  # We're using custom labels
        ax.set_xlim(0, max(row['num_pages_global'], row['num_pages_local']))  # Set x-axis limit to max count
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add more padding to the plot
        ax.margins(x=0.1)  # Add horizontal margins
        ax.margins(y=0.2)  # Add vertical margins

        # Set title and labels with increased padding and font size
        ax.set_title(f'{disease} Content Overlap', pad=40, fontsize=30, fontweight='bold')  # Increased from 26
        ax.set_xlabel('Number of Documents', labelpad=20, fontsize=26)  # Increased from 22

        # Add legend with more padding and increased font size, positioned in the right corner
        ax.legend(['Overlap', 'No overlap'], 
                 loc='center left', 
                 bbox_to_anchor=(1, 0.5),  # Position in the right center
                 ncol=1,  # Single column for better fit
                 fontsize=24)  # Increased from 20

        # Adjust layout with more padding
        plt.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.85)  # Adjusted right margin to make room for legend
        plt.tight_layout(pad=5.0)
        
        # Display the plot
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
    
def plot_page_overlap(df):
    """
    Plots overlap analysis using page counts for both local and global documents
    Creates horizontal stacked bar charts showing overlap vs no overlap percentages
    """
    # Calculate metrics for each disease state
    metrics = df.groupby('disease_state').agg({
        'num_pages_global': 'sum',
        'unique_global_page_count': 'sum',
        'num_pages_local': 'sum',
        'unique_local_page_count': 'sum'
    }).reset_index()
    
    # Calculate overlap and no overlap for global with 3 significant figures
    metrics['global_overlap'] = (metrics['num_pages_global'] - metrics['unique_global_page_count']) / metrics['num_pages_global'] * 100
    metrics['global_no_overlap'] = 100 - metrics['global_overlap']
    
    # Calculate overlap and no overlap for local with 3 significant figures
    metrics['local_overlap'] = (metrics['num_pages_local'] - metrics['unique_local_page_count']) / metrics['num_pages_local'] * 100
    metrics['local_no_overlap'] = 100 - metrics['local_overlap']

    # Colors
    overlap_color = '#4CAF50'  # Green
    no_overlap_color = '#9E9E9E'  # Grey

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 16))  # Increased from (28, 14)

    # Global Content Plot
    y_pos = range(len(metrics))
    
    # Plot in reverse order (AML, CLL, DLBCL)
    for i, (no_overlap, overlap, total, state) in enumerate(zip(
        metrics['global_no_overlap'].iloc[::-1], 
        metrics['global_overlap'].iloc[::-1],
        metrics['num_pages_global'].iloc[::-1],
        metrics['disease_state'].iloc[::-1]
    )):
        # Always plot overlap (green) on the left
        ax1.barh(i, overlap, color=overlap_color)
        ax1.barh(i, no_overlap, left=overlap, color=no_overlap_color)
        # Add percentage labels with increased font size
        ax1.text(overlap/2, i, f"{overlap:.3f}%", va='center', ha='center', color='white', fontweight='bold', fontsize=20)  # Increased from 16
        ax1.text(overlap + no_overlap/2, i, f"{no_overlap:.3f}%", va='center', ha='center', color='white', fontweight='bold', fontsize=20)  # Increased from 16
        
        # Add disease state and n value with increased font size
        ax1.text(-0.02, i, state, va='center', ha='right', transform=ax1.get_yaxis_transform(), fontsize=20)  # Increased from 16
        ax1.text(-0.02, i-0.25, f"n = {int(total)}", va='top', ha='right', transform=ax1.get_yaxis_transform(), fontsize=18)  # Increased from 14

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([])  # Remove default y-labels since we're adding them manually
    ax1.set_title('Global Content', pad=30, fontsize=26, fontweight='bold')  # Increased from 22
    ax1.set_xlabel('% Global slides', fontsize=22)  # Increased from 18

    # US Content Plot
    for i, (no_overlap, overlap, total, state) in enumerate(zip(
        metrics['local_no_overlap'].iloc[::-1], 
        metrics['local_overlap'].iloc[::-1],
        metrics['num_pages_local'].iloc[::-1],
        metrics['disease_state'].iloc[::-1]
    )):
        # Always plot overlap (green) on the left
        ax2.barh(i, overlap, color=overlap_color)
        ax2.barh(i, no_overlap, left=overlap, color=no_overlap_color)
        # Add percentage labels with increased font size
        ax2.text(overlap/2, i, f"{overlap:.3f}%", va='center', ha='center', color='white', fontweight='bold', fontsize=20)  # Increased from 16
        ax2.text(overlap + no_overlap/2, i, f"{no_overlap:.3f}%", va='center', ha='center', color='white', fontweight='bold', fontsize=20)  # Increased from 16
        
        # Add disease state and n value with increased font size
        ax2.text(-0.02, i, state, va='center', ha='right', transform=ax2.get_yaxis_transform(), fontsize=20)  # Increased from 16
        ax2.text(-0.02, i-0.25, f"n = {int(total)}", va='top', ha='right', transform=ax2.get_yaxis_transform(), fontsize=18)  # Increased from 14

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])  # Remove default y-labels since we're adding them manually
    ax2.set_title('US Content', pad=30, fontsize=26, fontweight='bold')  # Increased from 22
    ax2.set_xlabel('% US slides', fontsize=22)  # Increased from 18

    # Add legend with increased font size
    fig.legend(['Overlap', 'No overlap'], loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=20)  # Increased from 16

    # Set x-axis limits to 0-100 for both plots
    ax1.set_xlim(0, 100)
    ax2.set_xlim(0, 100)

    # Adjust layout with more padding
    plt.tight_layout(pad=3.0)
    
    # Display the plot
    st.pyplot(fig)

# ----------------------------------
# Main Streamlit App
# ----------------------------------
 
st.title("GLASS Dashboard")
 
st.sidebar.title("About")
st.sidebar.info("This dashboard compares local and global documents on process adherence to the 'make a copy' process.")

# Project Overview
st.sidebar.subheader("Project Overview")
st.sidebar.write("""
This dashboard analyzes document overlap and similarity across different disease states in pharmaceutical research, focusing on:
- Document reuse patterns
- Similarity scores between documents
- Disease state-specific analysis
""")

# Time Frame and Data Source
st.sidebar.subheader("Data Information")
st.sidebar.write("""
**Time Frame:**
- Global Documents: 2023-07-09 to 2025-03-01
- US Affiliate Documents: 2023-07-09 to 2025-03-01

**Data Source:**
- Global Documents: Global Medical Affairs Team
- US Affiliate Documents: US Medical Affairs Team
""")

# Disease States
st.sidebar.subheader("Disease States")
st.sidebar.write("""
**Oncology:**
- Chronic Lymphocytic Leukemia (CLL) with Venetoclax
- Acute Myeloid Leukemia (AML) with Venetoclax
- Diffuse Large B-cell Lymphoma (DLBCL) and Follicular Lymphoma (FL) with Epcoritamab
""")

# Impact and Usage
st.sidebar.subheader("Impact & Usage")
st.sidebar.write("""
This dashboard serves as a valuable tool for:
- Analyzing document reuse patterns across different disease states
- Identifying potential areas of research overlap
- Supporting strategic decision-making in pharmaceutical research
- Facilitating efficient resource allocation
- Ensuring compliance and proper documentation practices
""")
 
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

        st.subheader("Slide Overlap")
        plot_page_overlap(oncology_docs)

        st.subheader("Disease State Overlap")
        plot_overlap_pie_charts(oncology_docs)

        st.subheader("Slide Reuse Comparison")
        plot_slide_reuse_comparison(oncology_docs)

    