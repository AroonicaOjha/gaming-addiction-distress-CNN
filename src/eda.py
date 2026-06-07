import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set premium visualization styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid", palette="muted")

# Custom harmonious color palettes
METALLIC_PALETTE = ["#2b2d42", "#8d99ae", "#ef233c", "#d90429"]
PASTEL_PALETTE = ["#4EA8DE", "#56CFE1", "#72EFDD", "#80FFDB"]
PURPLE_PINK_PALETTE = ["#7209B7", "#3F37C9", "#4895EF", "#F72585"]
MODALITIES_COLORS = {
    'Healthy': '#4EA8DE',
    'Distressed Only': '#7209B7',
    'Addicted Only': '#F72585',
    'Both Addicted & Distressed': '#EF233C'
}


def generate_eda(df, output_dir):
    """
    Step 2: Exploratory Data Analysis.
    Generates and saves visual plots under output_dir.
    """
    print("\n--- Step 2: Exploratory Data Analysis ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Class Distribution Plot
    plot_class_distribution(df, output_dir)
    
    # 2. Correlation Heatmap
    plot_correlation_heatmap(df, output_dir)
    
    # 3. Feature Histograms (Select core features representing each modality)
    plot_feature_histograms(df, output_dir)
    
    # 4. Addiction vs Distress comparisons
    plot_addiction_vs_distress(df, output_dir)
    
    print(f"EDA plots successfully generated and saved in: {output_dir}")

def plot_class_distribution(df, output_dir):
    """
    Generates a premium bar chart showing multiclass labels and individual labels.
    """
    plt.figure(figsize=(10, 6))
    
    # Map multiclass_label to descriptive names
    label_map = {
        0: 'Healthy (0)',
        1: 'Distressed Only (1)',
        2: 'Addicted Only (2)',
        3: 'Both Addicted & Distressed (3)'
    }
    df_temp = df.copy()
    df_temp['class_name'] = df_temp['multiclass_label'].map(label_map)
    
    class_counts = df_temp['class_name'].value_counts().sort_index()
    class_percentages = df_temp['class_name'].value_counts(normalize=True).sort_index() * 100
    
    # Plot using our custom palette
    colors = [MODALITIES_COLORS[name.split(' (')[0]] for name in class_counts.index]
    
    ax = sns.barplot(
        x=class_counts.index, 
        y=class_counts.values, 
        palette=colors,
        hue=class_counts.index,
        legend=False
    )
    
    # Annotate bars with counts and percentages
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.annotate(
            f'{int(height)}\n({class_percentages.iloc[i]:.1f}%)',
            (p.get_x() + p.get_width() / 2., height),
            ha='center', 
            va='center', 
            xytext=(0, 10), 
            textcoords='offset points',
            fontsize=10, 
            weight='bold'
        )
        
    plt.title('Player Modality Class Distribution', fontsize=14, weight='bold', pad=20)
    plt.xlabel('Modality Class', fontsize=12, labelpad=10)
    plt.ylabel('Number of Players', fontsize=12, labelpad=10)
    plt.ylim(0, max(class_counts.values) * 1.15)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved class distribution plot to: {output_path}")

def plot_correlation_heatmap(df, output_dir):
    """
    Generates a correlation heatmap of all behavioral, gameplay, and social columns.
    """
    # Exclude non-numeric fields like player_id and labels we don't want to correlate directly
    exclude_cols = ['player_id', 'addicted_label', 'distressed_label', 'multiclass_label']
    corr_cols = [col for col in df.columns if col not in exclude_cols]
    
    plt.figure(figsize=(12, 10))
    corr_matrix = df[corr_cols].corr()
    
    # Draw heatmap with a premium diverging colormap
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        vmin=-1.0, 
        vmax=1.0,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 8}
    )
    
    plt.title('Correlation Heatmap of Behavioral Features', fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved correlation heatmap to: {output_path}")

def plot_feature_histograms(df, output_dir):
    """
    Generates histograms with KDE for key representative features.
    """
    features_to_plot = [
        ('avg_daily_hours', 'Session Behaviour'),
        ('performance_volatility', 'Gameplay Metrics'),
        ('toxic_chat_score', 'Social Signals'),
        ('phq9_score', 'Clinical Indicators')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for i, (col, title) in enumerate(features_to_plot):
        if col in df.columns:
            sns.histplot(
                data=df, 
                x=col, 
                kde=True, 
                ax=axes[i], 
                color=METALLIC_PALETTE[i % len(METALLIC_PALETTE)],
                alpha=0.6
            )
            axes[i].set_title(f'{col} ({title})', fontsize=12, weight='bold')
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Count')
            
    plt.suptitle('Distribution of Core Features Across Modalities', fontsize=15, weight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'feature_histograms.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved feature histograms to: {output_path}")

def plot_addiction_vs_distress(df, output_dir):
    """
    Generates a scatter plot comparing IGD score (addiction indicator) 
    vs. PHQ-9 score (distress indicator), colored by multiclass labels.
    """
    plt.figure(figsize=(10, 8))
    
    # Map multiclass_label to descriptive names for the legend
    label_map = {
        0: 'Healthy',
        1: 'Distressed Only',
        2: 'Addicted Only',
        3: 'Both Addicted & Distressed'
    }
    df_temp = df.copy()
    df_temp['Modality'] = df_temp['multiclass_label'].map(label_map)
    
    # Custom colors mapping
    sns.scatterplot(
        data=df_temp,
        x='igd_score',
        y='phq9_score',
        hue='Modality',
        palette=MODALITIES_COLORS,
        alpha=0.8,
        s=80,
        edgecolor='w',
        linewidth=0.5
    )
    
    # Draw threshold indicator lines (e.g. IGD >= 35 indicating addiction and PHQ-9 >= 10 indicating distress)
    plt.axvline(x=35, color='#F72585', linestyle='--', alpha=0.5, label='IGD Score Threshold (35)')
    plt.axhline(y=10, color='#7209B7', linestyle='--', alpha=0.5, label='PHQ-9 Distress Threshold (10)')
    
    plt.title('Clinical Severity Space: Addiction (IGD) vs. Distress (PHQ-9)', fontsize=14, weight='bold', pad=15)
    plt.xlabel('IGD-20 Addiction Score (Higher = Addicted)', fontsize=12, labelpad=10)
    plt.ylabel('PHQ-9 Distress Score (Higher = Distressed)', fontsize=12, labelpad=10)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'addiction_vs_distress.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved addiction vs distress scatter plot to: {output_path}")
