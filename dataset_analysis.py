import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import data_loader
import json
import os

# Create results directory
os.makedirs('results', exist_ok=True)

print("Analyzing dataset...")

# Load data
X_train, X_val, X_test, y_train, y_val, y_test = data_loader.prepare_data()

# Combine all data for analysis
X_all = pd.concat([X_train, X_val, X_test])
y_all = pd.concat([y_train, y_val, y_test])

# Class distribution
class_counts = y_all.value_counts()

# Document length statistics
doc_lengths = X_all.str.split().str.len()

# By class
sport_docs = X_all[y_all == 'Sport']
politics_docs = X_all[y_all == 'Politics']
sport_lengths = sport_docs.str.split().str.len()
politics_lengths = politics_docs.str.split().str.len()

# Character length statistics
char_lengths = X_all.str.len()

# Vocabulary analysis
all_words = ' '.join(X_all).split()
unique_words = set(all_words)

# Save statistics to file
stats_dict = {
    'total_documents': len(X_all),
    'sport_documents': int(class_counts['Sport']),
    'politics_documents': int(class_counts['Politics']),
    'avg_words_per_doc': float(doc_lengths.mean()),
    'median_words_per_doc': float(doc_lengths.median()),
    'min_words': int(doc_lengths.min()),
    'max_words': int(doc_lengths.max()),
    'avg_chars_per_doc': float(char_lengths.mean()),
    'total_words': len(all_words),
    'vocabulary_size': len(unique_words),
    'sport_avg_words': float(sport_lengths.mean()),
    'politics_avg_words': float(politics_lengths.mean())
}

with open('results/dataset_statistics.json', 'w') as f:
    json.dump(stats_dict, f, indent=2)

# Generate basic visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Class distribution
ax1 = axes[0, 0]
colors = ['#FF9999', '#66B2FF']
class_counts.plot(kind='bar', ax=ax1, color=colors)
ax1.set_title('Class Distribution', fontsize=11, fontweight='bold')
ax1.set_ylabel('Documents')
ax1.set_xlabel('Class')
ax1.grid(axis='y', alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)

# Document length distribution
ax2 = axes[0, 1]
ax2.hist(sport_lengths, bins=40, alpha=0.6, label='Sport', color='#FF9999')
ax2.hist(politics_lengths, bins=40, alpha=0.6,
         label='Politics', color='#66B2FF')
ax2.set_xlabel('Words per Document')
ax2.set_ylabel('Frequency')
ax2.set_title('Document Length Distribution', fontsize=11, fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Box plot comparison
ax3 = axes[1, 0]
data_to_plot = [sport_lengths, politics_lengths]
ax3.boxplot(data_to_plot, tick_labels=['Sport', 'Politics'], patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.7),
            medianprops=dict(color='red', linewidth=2))
ax3.set_ylabel('Words per Document')
ax3.set_title('Length Comparison', fontsize=11, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Data splits
ax4 = axes[1, 1]
splits = ['Train', 'Val', 'Test']
split_sizes = [len(X_train), len(X_val), len(X_test)]
colors_split = ['#90EE90', '#FFD700', '#FFA07A']
ax4.bar(splits, split_sizes, color=colors_split)
ax4.set_ylabel('Documents')
ax4.set_title('Data Split', fontsize=11, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
for i, v in enumerate(split_sizes):
    ax4.text(i, v + 30, str(v), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('results/dataset_analysis.png', dpi=200, bbox_inches='tight')

print(f"Total: {len(X_all)} documents")
print(
    f"Sport: {class_counts['Sport']} ({class_counts['Sport']/len(y_all)*100:.1f}%)")
print(
    f"Politics: {class_counts['Politics']} ({class_counts['Politics']/len(y_all)*100:.1f}%)")
print(f"Avg length: {doc_lengths.mean():.0f} words")
print(f"Vocabulary: {len(unique_words):,} words")
print("Saved: results/dataset_statistics.json, results/dataset_analysis.png")
