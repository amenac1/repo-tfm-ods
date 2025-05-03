import seaborn as sns
from itertools import cycle

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
})

def df_histogram(df, column, plot_title, x_label, y_label, save_file=None):
    if column in df.columns:
        sns.set_theme(style='white', palette='colorblind')
        plt.figure(figsize=(8, 6))
        unique_vals = sorted(df[column].dropna().unique())
        counts = df[column].value_counts().loc[unique_vals]
        positions = range(len(unique_vals))
        plt.bar(positions, counts, color='mediumseagreen', edgecolor='black', 
                linewidth=2, width=0.8)  
        plt.title(plot_title, fontsize=16, fontweight='bold')
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.xticks(positions, unique_vals)
        
        if save_file:
            plt.savefig(save_file, dpi=300)
        
        plt.show()
    else:
        print(f"Column doesn't exist: {column}.")

def df_histogram_ods(df, column, plot_title, x_label, y_label, colors, save_file=None):
    if column in df.columns:
        sns.set_theme(style='white', palette='colorblind', rc={
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
        })
        plt.figure(figsize=(8, 6))
        unique_vals = sorted(df[column].dropna().unique())
        counts = df[column].value_counts().loc[unique_vals]
        positions = range(len(unique_vals))
        bars = plt.bar(positions, counts, edgecolor='black', linewidth=1.5, width=0.8)
        plt.title(plot_title, fontsize=16, fontweight='bold')
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.xticks(positions, unique_vals)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        color_cycle = cycle(colors)
        for bar in bars:
            bar.set_facecolor(next(color_cycle))
        
        if save_file:
            plt.savefig(save_file, dpi=300)
        
        plt.show()
    else:
        print(f"Column doesn't exist: {column}.")
