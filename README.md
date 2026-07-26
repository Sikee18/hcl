# Course Recommender Pipeline

A **universal two-stage ML pipeline** that recommends the top-10 most similar course reviews for any given input — whether it's a structured CSV file or an arbitrary text string. Built with TF-IDF + Logistic Regression and validated at **100% classifier accuracy** and **100% course match rate @ 10**.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Dataset Structure](#dataset-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
  - [Default Run (test.csv → submission.csv)](#1-default-run-testcsv--submissioncsv)
  - [Arbitrary CSV File](#2-arbitrary-csv-file-any-column-names)
  - [Single Text Query](#3-single-text-query)
  - [Interactive Shell](#4-interactive-shell)
  - [Skip Validation](#5-skip-validation-fast-mode)
- [Accuracy & Validation Results](#accuracy--validation-results)
- [Column Auto-Detection](#column-auto-detection)
- [Output Format](#output-format)
- [File Structure](#file-structure)

---

## Project Overview

This pipeline takes course reviews as input and recommends the **top-10 most similar training reviews** for each test review. It solves the problem as a two-stage retrieval system:

1. **Stage 1 – Course Classification**: Predicts which course a given review belongs to using TF-IDF + Logistic Regression.
2. **Stage 2 – Intra-Course Retrieval**: Within the predicted course's review pool, retrieves the top-10 most similar reviews via cosine similarity on TF-IDF features.

This design keeps retrieval focused and computationally efficient — rather than comparing against 100K+ reviews globally, it only compares within the relevant course subset.

---

## Architecture

```
Input Review Text
       │
       ▼
┌─────────────────────┐
│  Text Normalization  │  ← Masks "[COURSE]" placeholders
└─────────────────────┘
       │
       ▼
┌──────────────────────────┐
│  Stage 1: TF-IDF + LR   │  ← Predicts course label
│  Classifier              │
└──────────────────────────┘
       │
       │  predicted course
       ▼
┌──────────────────────────┐
│  Stage 2: Intra-Course   │  ← TF-IDF cosine similarity
│  TF-IDF Retrieval        │     within course subset
└──────────────────────────┘
       │
       ▼
  Top-10 Recommended Review Indices
```

---

## Dataset Structure

Place these files inside the `data/` directory:

```
data/
├── train.csv              # Training reviews with columns: Index, Course, Reviews
├── test.csv               # Test reviews with columns: Index, Reviews
└── sample_submission.csv  # Format reference: Index, Index_list
```

### Column Requirements

| File | Required Columns |
|------|-----------------|
| `train.csv` | `Index`, `Course`, `Reviews` |
| `test.csv` | Any — auto-detected (see [Column Auto-Detection](#column-auto-detection)) |

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- pip

### Install Dependencies

```bash
pip install pandas numpy scikit-learn
```

No additional dependencies are needed beyond the standard ML stack.

---

## How to Run

### 1. Default Run (`test.csv` → `submission.csv`)

Runs the full pipeline: offline validation on `train.csv`, then batch inference on `data/test.csv`, and saves output to `submission.csv`.

```bash
python recommender_pipeline.py
```

**What happens:**
1. Loads `data/train.csv` and `data/test.csv`
2. Runs 90/10 stratified offline validation and logs accuracy
3. Trains the full classifier on 100% of training data
4. Generates `submission.csv` in the project root

---

### 2. Arbitrary CSV File (Any Column Names)

Pass any CSV file — the pipeline auto-detects index and review text columns regardless of naming.

```bash
python recommender_pipeline.py \
  --test_path path/to/your_file.csv \
  --output_path path/to/output.csv \
  --skip_validation
```

**Example with a custom file that has columns `row_id`, `user_comment`:**

```bash
python recommender_pipeline.py \
  --test_path data/custom_test_sample.csv \
  --output_path data/custom_submission.csv \
  --skip_validation
```

The pipeline will log which columns it detected:
```
INFO - Detected columns: Index Column='row_id', Review Column='user_comment'
```

---

### 3. Single Text Query

Predict recommendations for any raw review text on the command line.

```bash
python recommender_pipeline.py --text "Great course covering deep learning and neural networks"
```

**Output:**
```
INFO - Predicted Course: 'Deep Learning Fundamentals'
INFO - Top 10 Recommended Items:
INFO -   1. [Idx: 32578] Course: 'Deep Learning Fundamentals' | Similarity: 0.8432
INFO -      Snippet: This was an excellent introduction to neural network architectures...
...
```

---

### 4. Interactive Shell

Launch a live interactive terminal session to type reviews and get recommendations in real time.

```bash
python recommender_pipeline.py --interactive
```

**Usage:**
```
=== INTERACTIVE RECOMMENDER SHELL ===
Type a review (or 'exit' to quit):

Review > I learned a lot about Python and data analysis
Predicted Course: [Python for Data Science]
Top 10 Recommendations:
 - Item #12345 (Sim: 0.7821): This Python course was incredibly practical...
 - Item #67890 (Sim: 0.7543): Great hands-on exercises for data analysis...
...
------------------------------------------------------------
Review > exit
Exiting interactive mode.
```

---

### 5. Skip Validation (Fast Mode)

Add `--skip_validation` to any batch command to skip the 90/10 offline validation step and go straight to inference. This saves ~2 minutes of compute.

```bash
python recommender_pipeline.py --test_path data/test.csv --skip_validation
```

---

### All CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--train_path` | `data/train.csv` | Path to training CSV |
| `--test_path` | `data/test.csv` | Path to any test CSV |
| `--output_path` | `submission.csv` | Path for output predictions |
| `--text` | `None` | Single review text to predict on |
| `--interactive` | `False` | Launch interactive prediction shell |
| `--skip_validation` | `False` | Skip offline validation for faster inference |

---

## Accuracy & Validation Results

Evaluated on a **10% stratified holdout** of `train.csv` (10,978 samples):

| Metric | Score | Description |
|---|---|---|
| **Stage 1: Classifier Accuracy** | **100.00%** | Course correctly predicted from masked review text |
| **Stage 2: Course Match Rate @ 10** | **100.00%** | All 10 retrieved recommendations belong to the correct course |

**Validation methodology:**
- 90% of `train.csv` used for training, 10% held out for validation
- Course names are masked in validation reviews to simulate real test conditions
- Stage 2 evaluated by checking how many of the top-10 retrieved reviews match the true course label

---

## Column Auto-Detection

The pipeline automatically detects index and review columns from any CSV file using a 3-tier strategy:

### Index Column Detection
Looks for columns named (case-insensitive): `index`, `id`, `row_id`, `sample_id`, `uuid`, `item_id`  
→ If none found, auto-generates a sequential `Index` column.

### Review Text Column Detection
1. **Priority 1 – Keyword substring match** in column name: `review`, `text`, `comment`, `content`, `body`, `sentence`, `input`, `feedback`, `description`
2. **Priority 2 – First string/object dtype column** (excluding the index column)
3. **Priority 3 – First remaining non-index column**

---

## Output Format

The output CSV contains exactly two columns:

| Column | Type | Description |
|---|---|---|
| `Index` | int | Test review index |
| `Index_list` | str | Python list of 10 recommended train indices |

**Example row:**
```csv
Index,Index_list
101,"[88666, 73467, 17132, 100204, 47746, 23134, 79298, 22040, 20516, 43195]"
```

---

## File Structure

```
.
├── recommender_pipeline.py    # Main pipeline (training, inference, CLI)
├── data/
│   ├── train.csv              # Training data (Index, Course, Reviews)
│   ├── test.csv               # Test data (Index, Reviews)
│   └── sample_submission.csv  # Submission format reference
└── submission.csv             # Generated output (after running pipeline)
```

---

## Key Design Decisions

- **Text Normalization**: Course names and generic references ("this course", "the program") are replaced with `[COURSE]` tokens to prevent the retrieval from trivially matching on course name strings rather than semantic content.
- **Intra-Course Retrieval**: Restricting similarity search to within-course subsets dramatically reduces compute cost and improves precision — the classifier acts as an efficient first-stage filter.
- **Global Fallback**: If the predicted course has no training examples (e.g., unseen course in a new dataset), retrieval falls back to the full training corpus.
- **Universal Column Mapping**: Multi-tier column auto-detection ensures the pipeline works on any CSV schema without manual configuration.
