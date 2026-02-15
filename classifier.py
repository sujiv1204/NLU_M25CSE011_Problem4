import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import data_loader

# Set random seed for reproducibility
np.random.seed(42)

# Create results directory
os.makedirs('results', exist_ok=True)

print("Loading data \n")
X_train, X_val, X_test, y_train, y_val, y_test = data_loader.prepare_data()

# Define feature extraction methods
feature_methods = {
    'Bag of Words': CountVectorizer(max_features=5000, stop_words='english'),
    'TF-IDF Unigram': TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 1)),
    'TF-IDF Bigram': TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
}

# Define models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Linear SVM': LinearSVC(random_state=42, max_iter=2000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

# Train and evaluate all combinations
print("Training models \n")

results = []

for feature_name, vectorizer in feature_methods.items():
    # Transform data
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    for model_name, model in models.items():
        print(f"  {model_name} + {feature_name}")

        # Train model
        model.fit(X_train_vec, y_train)

        # Predict on validation set
        y_val_pred = model.predict(X_val_vec)
        val_accuracy = accuracy_score(y_val, y_val_pred)

        # Predict on test set
        y_test_pred = model.predict(X_test_vec)

        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(
            y_test, y_test_pred, pos_label='Sport')
        test_recall = recall_score(y_test, y_test_pred, pos_label='Sport')
        test_f1 = f1_score(y_test, y_test_pred, pos_label='Sport')

        # Store results
        results.append({
            'Model': model_name,
            'Features': feature_name,
            'Val_Accuracy': val_accuracy,
            'Test_Accuracy': test_accuracy,
            'Precision': test_precision,
            'Recall': test_recall,
            'F1_Score': test_f1
        })

# Save results
print("\nSaving results \n")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Test_Accuracy', ascending=False)
results_df.to_csv('results/classification_results.csv', index=False)

# Find best model
best_row = results_df.iloc[0]
print(
    f"Best: {best_row['Model']} + {best_row['Features']} = {best_row['Test_Accuracy']:.4f}")

# Save best model and vectorizer
best_feature_name = best_row['Features']
best_model_name = best_row['Model']

# Re-train best model on train+val data
best_vectorizer = feature_methods[best_feature_name]
best_model = models[best_model_name]

X_combined = pd.concat([X_train, X_val])
y_combined = pd.concat([y_train, y_val])

X_combined_vec = best_vectorizer.fit_transform(X_combined)
best_model.fit(X_combined_vec, y_combined)

X_test_vec = best_vectorizer.transform(X_test)
final_test_accuracy = best_model.score(X_test_vec, y_test)

# Save to files
with open('results/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('results/best_vectorizer.pkl', 'wb') as f:
    pickle.dump(best_vectorizer, f)
with open('results/best_model_info.txt', 'w') as f:
    f.write(f"Model: {best_model_name}\n")
    f.write(f"Features: {best_feature_name}\n")
    f.write(f"Test Accuracy: {final_test_accuracy:.4f}\n")

# Generate visualizations
print("Generating visualizations")

# Plot 1: Accuracy comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Accuracy by model
ax1 = axes[0, 0]
model_groups = results_df.groupby(
    'Model')['Test_Accuracy'].mean().sort_values(ascending=False)
ax1.barh(model_groups.index, model_groups.values, color='skyblue')
ax1.set_xlabel('Average Test Accuracy')
ax1.set_title('Model Performance Comparison')
ax1.grid(axis='x', alpha=0.3)

# Accuracy by feature type
ax2 = axes[0, 1]
feature_groups = results_df.groupby(
    'Features')['Test_Accuracy'].mean().sort_values(ascending=False)
ax2.barh(feature_groups.index, feature_groups.values, color='lightcoral')
ax2.set_xlabel('Average Test Accuracy')
ax2.set_title('Feature Type Comparison')
ax2.grid(axis='x', alpha=0.3)

# Precision vs Recall
ax3 = axes[1, 0]
for model in results_df['Model'].unique():
    model_data = results_df[results_df['Model'] == model]
    ax3.scatter(model_data['Precision'], model_data['Recall'],
                label=model, s=100, alpha=0.7)
ax3.set_xlabel('Precision')
ax3.set_ylabel('Recall')
ax3.set_title('Precision vs Recall Trade-off')
ax3.legend()
ax3.grid(alpha=0.3)

# F1 Score comparison
ax4 = axes[1, 1]
pivot_data = results_df.pivot(
    index='Model', columns='Features', values='F1_Score')
pivot_data.plot(kind='bar', ax=ax4)
ax4.set_xlabel('Model')
ax4.set_ylabel('F1 Score')
ax4.set_title('F1 Score by Model and Feature Type')
ax4.legend(title='Features', bbox_to_anchor=(1.05, 1), loc='upper left')
ax4.grid(axis='y', alpha=0.3)
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')

# Plot 2: Confusion matrices for top 4 models
top_4 = results_df.head(4)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (_, row) in enumerate(top_4.iterrows()):
    feature_name = row['Features']
    model_name = row['Model']

    # Get the vectorizer and model
    vectorizer = feature_methods[feature_name]
    model = models[model_name]

    # Re-fit on training data (we need to do this to get predictions)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=['Politics', 'Sport'])

    # Plot
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Politics', 'Sport'],
                yticklabels=['Politics', 'Sport'])
    axes[idx].set_title(f'{model_name}\n({feature_name})')
    axes[idx].set_ylabel('Actual')
    axes[idx].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')

# Plot 3: Metrics heatmap
plt.figure(figsize=(10, 8))

metrics_data = results_df.pivot_table(
    index='Model',
    columns='Features',
    values='Test_Accuracy'
)

sns.heatmap(metrics_data, annot=True, fmt='.3f', cmap='YlGnBu',
            cbar_kws={'label': 'Test Accuracy'})
plt.title('Test Accuracy Heatmap: Models vs Features')
plt.tight_layout()
plt.savefig('results/accuracy_heatmap.png', dpi=300, bbox_inches='tight')

print("\nComplete! Results saved to results/classification_results.csv")
print(f"Best model saved: {best_model_name} + {best_feature_name}")
