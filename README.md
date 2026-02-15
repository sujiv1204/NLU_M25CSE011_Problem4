# Sport vs Politics Text Classification

**M25CSE011**

## Description

This classifier uses machine learning to categorize text documents as either Sport or Politics. Compares 4 different ML algorithms with 3 feature methods.

## Files

- `data_loader.py` - Loads and preprocesses mixed dataset (20 Newsgroups + BBC)
- `dataset_analysis.py` - Basic dataset statistics and visualization
- `classifier.py` - Main training and evaluation script
- `test_predictions.py` - Interactive prediction tool (uses saved model)
- `M25CSE011_prob4.pdf` - The detailed report for this problem


## Generated Files

All generated files are saved in the `results/` directory:

- `classification_results.csv` - All experiment results
- `dataset_statistics.json` - Dataset statistics
- `dataset_analysis.png` - Dataset visualizations (4 plots)
- `best_model.pkl` - Saved best model
- `best_vectorizer.pkl` - Saved vectorizer
- `best_model_info.txt` - Model information
- `model_comparison.png` - Performance comparison plots
- `confusion_matrices.png` - Top 4 model confusion matrices
- `accuracy_heatmap.png` - Model vs feature accuracy heatmap

## Dataset

- **20 Newsgroups**: ~4600 documents (baseball, hockey, politics topics)
- **BBC News**: ~900 documents (sport and politics categories)
- **Combined**: ~5250 documents after preprocessing
- **Split**: 70% train, 15% validation, 15% test

## Models

1. Naive Bayes
2. Logistic Regression
3. Linear SVM
4. Random Forest

## Features

1. Bag of Words
2. TF-IDF Unigrams
3. TF-IDF Bigrams

**TOTAL**: 12 experiments (4 models × 3 features)

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn requests
```

## How to Run

### 1. Analyze dataset (optional but recommended)

```bash
python3 dataset_analysis.py
```

- Generates basic statistics and visualizations
- Takes 1-2 minutes

### 2. Train classifiers

```bash
python3 classifier.py
```

- Trains all 12 combinations, saves best model, creates plots
- Takes 5-10 minutes

### 3. Test predictions (optional)

```bash
python3 test_predictions.py
```

- Interactive tool - enter text to get predictions
- Uses the saved best model (no retraining)


## Metrics

- **Accuracy**: Overall correctness
- **Precision**: Correct positive predictions / total positive predictions
- **Recall**: Correct positive predictions / actual positives
- **F1-Score**: Harmonic mean of precision and recall

All metrics calculated on test set (15% of data held aside from training)

## Outputs

All outputs are saved in `results/` directory:

- `classification_results.csv` - Complete results for all 12 combinations
- 3 visualization PNG files showing model comparisons

