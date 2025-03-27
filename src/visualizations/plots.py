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
    
    # Calculate overlap and no overlap for global
    metrics['global_overlap'] = (metrics['num_pages_global'] - metrics['unique_global_page_count']) / metrics['num_pages_global'] * 100
    metrics['global_no_overlap'] = 100 - metrics['global_overlap']
    
    # Calculate overlap and no overlap for local
    metrics['local_overlap'] = (metrics['num_pages_local'] - metrics['unique_local_page_count']) / metrics['num_pages_local'] * 100
    metrics['local_no_overlap'] = 100 - metrics['local_overlap']

    # Colors
    overlap_color = '#4CAF50'  # Green
    no_overlap_color = '#9E9E9E'  # Grey

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Global Content Plot
    y_pos = range(len(metrics))
    
    # Determine order based on larger percentage
    for i, (no_overlap, overlap, total, state) in enumerate(zip(
        metrics['global_no_overlap'], 
        metrics['global_overlap'],
        metrics['num_pages_global'],
        metrics['disease_state']
    )):
        # Plot larger percentage first
        if no_overlap > overlap:
            ax1.barh(i, no_overlap, color=no_overlap_color)
            ax1.barh(i, overlap, left=no_overlap, color=overlap_color)
            # Add percentage labels
            ax1.text(no_overlap/2, i, f"{no_overlap:.0f}%", va='center', ha='center', color='white', fontweight='bold')
            ax1.text(no_overlap + overlap/2, i, f"{overlap:.0f}%", va='center', ha='center', color='white', fontweight='bold')
        else:
            ax1.barh(i, overlap, color=overlap_color)
            ax1.barh(i, no_overlap, left=overlap, color=no_overlap_color)
            # Add percentage labels
            ax1.text(overlap/2, i, f"{overlap:.0f}%", va='center', ha='center', color='white', fontweight='bold')
            ax1.text(overlap + no_overlap/2, i, f"{no_overlap:.0f}%", va='center', ha='center', color='white', fontweight='bold')
        
        # Add disease state and n value
        ax1.text(-0.02, i, state, va='bottom', ha='right', transform=ax1.get_yaxis_transform())
        ax1.text(-0.02, i-0.25, f"n = {int(total)}", va='top', ha='right', transform=ax1.get_yaxis_transform())

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([])  # Remove default y-labels since we're adding them manually
    ax1.set_title('Global Content', pad=20, fontsize=14, fontweight='bold')
    ax1.set_xlabel('% Global slides')

    # US Content Plot
    for i, (no_overlap, overlap, total, state) in enumerate(zip(
        metrics['local_no_overlap'], 
        metrics['local_overlap'],
        metrics['num_pages_local'],
        metrics['disease_state']
    )):
        # Plot larger percentage first
        if no_overlap > overlap:
            ax2.barh(i, no_overlap, color=no_overlap_color)
            ax2.barh(i, overlap, left=no_overlap, color=overlap_color)
            # Add percentage labels
            ax2.text(no_overlap/2, i, f"{no_overlap:.0f}%", va='center', ha='center', color='white', fontweight='bold')
            ax2.text(no_overlap + overlap/2, i, f"{overlap:.0f}%", va='center', ha='center', color='white', fontweight='bold')
        else:
            ax2.barh(i, overlap, color=overlap_color)
            ax2.barh(i, no_overlap, left=overlap, color=no_overlap_color)
            # Add percentage labels
            ax2.text(overlap/2, i, f"{overlap:.0f}%", va='center', ha='center', color='white', fontweight='bold')
            ax2.text(overlap + no_overlap/2, i, f"{no_overlap:.0f}%", va='center', ha='center', color='white', fontweight='bold')
        
        # Add disease state and n value
        ax2.text(-0.02, i, state, va='bottom', ha='right', transform=ax2.get_yaxis_transform())
        ax2.text(-0.02, i-0.25, f"n = {int(total)}", va='top', ha='right', transform=ax2.get_yaxis_transform())

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])  # Remove default y-labels since we're adding them manually
    ax2.set_title('US Content', pad=20, fontsize=14, fontweight='bold')
    ax2.set_xlabel('% US slides')

    # Add legend
    fig.legend(['No overlap', 'Overlap'], loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)

    # Set x-axis limits to 0-100 for both plots
    ax1.set_xlim(0, 100)
    ax2.set_xlim(0, 100)

    # Adjust layout
    plt.tight_layout()
    
    # Display the plot
    st.pyplot(fig)

    # Display summary statistics in a table
    st.subheader("Page Counts Summary")
    summary_df = metrics[['disease_state', 'num_pages_global', 'unique_global_page_count', 
                         'num_pages_local', 'unique_local_page_count']]
    summary_df.columns = ['Disease State', 'Global Pages', 'Global Unique Pages', 
                         'US Pages', 'US Unique Pages']
    st.dataframe(summary_df) 