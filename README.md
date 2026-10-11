# SimobiPlus Review Sentiment Analysis with Pseudo-Labeling

An Indonesian-language NLP project that classifies Google Play reviews of the SimobiPlus banking app as **negative, neutral, or positive**. I built the workflow from review collection and manual annotation through semi-supervised labeling, model training, and evaluation.

## At a glance

| | |
|---|---|
| **Problem** | Thousands of user reviews are difficult to analyze manually, while high-quality sentiment labels are expensive to produce. |
| **Approach** | Train an IndoBERT teacher on manually labeled reviews, then use its high-confidence predictions to expand the training set. Compare SVM, Bi-LSTM, and IndoBERT on the resulting dataset. |
| **Data** | 10,000 collected reviews; 976 manually labeled reviews plus 7,789 accepted pseudo-labels produced a final dataset of 8,765 reviews. |
| **Recorded result** | Fine-tuned IndoBERT reached 99.00% test accuracy and 0.93 macro F1 in the reported 70:30 split. |
| **My work** | Data collection, Indonesian text preprocessing, annotation workflow, pseudo-labeling, model experiments, and evaluation. |

## Why pseudo-labeling instead of a sentiment lexicon?

A sentiment lexicon assigns polarity from a predefined word list. That can miss context in informal app reviews, such as negation, slang, or a word whose meaning changes with the surrounding sentence. In this project, I used **manually labeled examples to train an IndoBERT teacher model**, then used the teacher's predictions to label additional reviews. I kept only predictions with at least **90% confidence**. The pseudo-labels therefore come from a contextual model trained on the project's annotations, **not from lexicon-based sentiment scores**.

The preprocessing notebook does use a slang-normalization dictionary. Its role is to standardize text; it is not used to assign sentiment labels.

## Workflow

1. [Collect reviews](notebook/01_scrapping_data.ipynb) from Google Play and save them in [the raw dataset](data/raw/simobi_reviews.csv).
2. [Clean and split the data](notebook/02_preprocessing_and_data_splitting.ipynb). This includes case folding, removing unwanted text elements, and normalizing Indonesian slang. A rating-stratified sample of 1,000 reviews was prepared for manual annotation; 976 manually labeled reviews were used in the teacher-model stage.
3. [Train the teacher and generate pseudo-labels](notebook/03_pseudo_labeling.ipynb). The IndoBERT teacher predicted labels for 9,000 unlabeled reviews; 7,789 predictions met the 90% confidence threshold. Together with the manual labels, they formed [the final dataset](data/processed/final/dataset_final_10000.csv). Despite the filename, the reported final dataset has **8,765 rows**.
4. [Train and evaluate three classifiers](notebook/04_model_training_and_evaluation.ipynb): SVM with TF-IDF, Bi-LSTM, and fine-tuned IndoBERT.

The final class distribution is **4,812 negative**, **192 neutral**, and **3,761 positive**. Because the neutral class is small, macro F1 and per-class results matter alongside accuracy.

## Reported results

| Model | Test split | Accuracy | Macro F1 | Neutral F1 |
|---|---:|---:|---:|---:|
| SVM + TF-IDF | 20% (1,753 reviews) | 97.15% | 0.69 | 0.10 |
| Bi-LSTM | 20% (1,753 reviews) | 97.03% | 0.78 | 0.38 |
| IndoBERT | 30% (2,630 reviews) | **99.00%** | **0.93** | **0.80** |

These figures are reported in the project experiment. IndoBERT recorded the strongest result, including a substantially higher F1 score for neutral reviews. Its test split differs from the other models, so the table is a summary of recorded experiments rather than a comparison on one identical test set.

## Evaluation limits

- The final dataset contains both human labels and model-generated pseudo-labels. A test split from this dataset is **not equivalent to an independent, fully human-labeled test set**.
- Neutral reviews account for only 192 of 8,765 examples; overall accuracy can hide weak performance on this minority class.
- Model weights referenced in the original README are **not present in this repository**. The notebooks and data can be reviewed, but an inference example that loads those weights cannot run directly from a fresh clone.

## Explore or run the project

Start with the four numbered [notebooks](notebook/) or inspect the supplied [data](data/) and results above. To run the notebooks, install the listed dependencies with `pip install -r requirements.txt`, then adjust any local or Google Colab paths for your environment. Training IndoBERT and Bi-LSTM requires suitable compute resources. This repository documents an experiment, not a production inference service.

**Tools:** Python, pandas, scikit-learn, TensorFlow/Keras, PyTorch, Hugging Face Transformers, IndoBERT, and `google-play-scraper`. See [requirements.txt](requirements.txt).

**Author:** [Muhammad Attan](https://github.com/attmhd)
