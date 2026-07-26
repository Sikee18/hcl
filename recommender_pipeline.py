import os
import re
import sys
import time
import ast
import logging
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================================================================
# CONFIGURATION
# =====================================================================
class Config:
    DATA_DIR = "data"
    TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
    TEST_PATH = os.path.join(DATA_DIR, "test.csv")
    SUBMISSION_PATH = "submission.csv"
    RANDOM_SEED = 42
    
    # Text normalization placeholders
    PLACEHOLDERS = [
        "this course", "this program", "this learning path",
        "the course", "the program", "the learning path",
        "this topic", "the topic"
    ]
    
    # Model Hyperparameters
    CLASSIFIER_C = 10.0
    CLASSIFIER_MAX_ITER = 200
    CLASSIFIER_NGRAM = (1, 2)
    RETRIEVAL_NGRAM = (1, 2)

# =====================================================================
# UTILS & COLUMN DETECTION
# =====================================================================
def detect_columns(df, index_col=None, review_col=None):
    """
    Dynamically identifies index and review columns from any input DataFrame.
    Supports flexible naming conventions, substring matching, and dtype fallbacks.
    """
    # Use explicit overrides if provided
    idx_col = index_col if (index_col and index_col in df.columns) else None
    rev_col = review_col if (review_col and review_col in df.columns) else None

    # 1. Detect Index column if not specified
    if idx_col is None:
        for col in df.columns:
            col_lower = str(col).lower()
            if col_lower in ['index', 'id', 'row_id', 'sample_id', 'uuid', 'item_id']:
                idx_col = col
                break
                
    if idx_col is None:
        df['Index'] = range(len(df))
        idx_col = 'Index'
        logger.info("No explicit index column found; auto-generated 'Index' column.")

    # 2. Detect Review column if not specified
    if rev_col is None:
        review_keywords = ['review', 'text', 'comment', 'content', 'body', 'sentence', 'input', 'feedback', 'description', 'eval', 'quote']
        
        # Priority 1: Substring match on column names
        for col in df.columns:
            if col == idx_col:
                continue
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in review_keywords):
                rev_col = col
                break
                
        # Priority 2: Fallback to the first object/string column that is not the index column
        if rev_col is None:
            for col in df.columns:
                if col != idx_col and df[col].dtype == 'object':
                    rev_col = col
                    break

        # Priority 3: Fallback to any remaining non-index column
        if rev_col is None:
            remaining_cols = [c for c in df.columns if c != idx_col]
            if remaining_cols:
                rev_col = remaining_cols[0]

    if rev_col is None:
        raise ValueError(f"Could not find any review text column in DataFrame! Available columns: {list(df.columns)}")

    return idx_col, rev_col

# =====================================================================
# TEXT PREPROCESSING & NORMALIZATION
# =====================================================================
class Preprocessor:
    def __init__(self, courses):
        """
        Initialize with the list of unique course names sorted by length descending
        so that longer course names are replaced first (avoids sub-string clashes).
        """
        self.courses = sorted(courses, key=len, reverse=True)
        self.placeholders = Config.PLACEHOLDERS

    def normalize_train_review(self, text, course_name):
        """
        Normalizes a training review:
        1. Replaces the explicit course name with [COURSE].
        2. Replaces generic references (like 'this course') with [COURSE].
        """
        text = str(text) if pd.notnull(text) else ""
        text = re.sub(re.escape(course_name), '[COURSE]', text, flags=re.IGNORECASE)
        for p in self.placeholders:
            text = re.sub(r'\b' + re.escape(p) + r'\b', '[COURSE]', text, flags=re.IGNORECASE)
        return text.strip()

    def normalize_test_review(self, text):
        """
        Normalizes a test review:
        Replaces generic references (like 'this course') with [COURSE].
        """
        text = str(text) if pd.notnull(text) else ""
        for p in self.placeholders:
            text = re.sub(r'\b' + re.escape(p) + r'\b', '[COURSE]', text, flags=re.IGNORECASE)
        return text.strip()

    def simulate_test_review(self, text, course_name, idx):
        """
        Used for validation: takes a training review, masks the course name with a
        deterministic placeholder, creating a simulated test review.
        """
        text = str(text) if pd.notnull(text) else ""
        ph = self.placeholders[idx % len(self.placeholders)]
        text_masked = re.sub(re.escape(course_name), ph, text, flags=re.IGNORECASE)
        return text_masked

# =====================================================================
# MODEL PIPELINE
# =====================================================================
class RecommenderPipeline:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.classifier_vec = TfidfVectorizer(ngram_range=Config.CLASSIFIER_NGRAM, min_df=2, max_df=0.8)
        self.classifier = LogisticRegression(
            C=Config.CLASSIFIER_C, 
            max_iter=Config.CLASSIFIER_MAX_ITER, 
            n_jobs=-1, 
            random_state=Config.RANDOM_SEED
        )

    def train_classifier(self, train_reviews, train_courses):
        """
        Trains the Stage 1 Classifier to predict Course from review text.
        """
        logger.info("Training Stage 1 Classifier...")
        t0 = time.time()
        X_train = self.classifier_vec.fit_transform(train_reviews)
        self.classifier.fit(X_train, train_courses)
        logger.info(f"Classifier training completed in {time.time() - t0:.2f}s")

    def predict_courses(self, test_reviews):
        """
        Predicts the Course for a set of test reviews.
        """
        X_test = self.classifier_vec.transform(test_reviews)
        return self.classifier.predict(X_test)

    def retrieve_recommendations(self, train_df, test_df, index_col='Index', review_col='Reviews'):
        """
        Performs intra-course batch retrieval with fallback for unknown/unseen courses:
        1. Groups test set by predicted course.
        2. Fits course-specific TF-IDF and computes pairwise cosine similarity.
        3. Fallback to global TF-IDF if course subset is missing in train set.
        4. Returns a DataFrame containing predicted Index and Index_list.
        
        Note: train_df always uses 'Reviews' column; review_col only applies to test_df.
        """
        TRAIN_REVIEW_COL = 'Reviews'  # train.csv always has 'Reviews'
        logger.info("Starting Stage 2 Batch Retrieval by Course...")
        t0 = time.time()
        results = []

        # Build global fallback vectorizer just in case
        global_vec = None
        global_train_feats = None

        for course, group in test_df.groupby('Predicted_Course'):
            train_subset = train_df[train_df['Course'] == course].copy().reset_index(drop=True)
            
            # Fallback if no training reviews exist for predicted course
            if len(train_subset) == 0:
                logger.warning(f"No train subset for predicted course '{course}'. Using global retrieval fallback.")
                if global_vec is None:
                    norm_global = [self.preprocessor.normalize_train_review(r, c) for r, c in zip(train_df[TRAIN_REVIEW_COL], train_df['Course'])]
                    global_vec = TfidfVectorizer(ngram_range=Config.RETRIEVAL_NGRAM, min_df=1)
                    global_train_feats = global_vec.fit_transform(norm_global)
                
                norm_test_reviews = [self.preprocessor.normalize_test_review(r) for r in group[review_col]]
                test_feats = global_vec.transform(norm_test_reviews)
                sim_matrix = test_feats.dot(global_train_feats.T).toarray()
                
                for i, test_idx in enumerate(group[index_col]):
                    sims = sim_matrix[i]
                    top_10_pos = np.argsort(sims)[::-1][:10]
                    rec_indices = train_df.iloc[top_10_pos]['Index'].tolist()
                    results.append({'Index': test_idx, 'Index_list': str(rec_indices)})
                continue

            # Standard intra-course batch retrieval
            # Train rows always use the fixed 'Reviews' column name from train.csv
            norm_train_reviews = [
                self.preprocessor.normalize_train_review(row[TRAIN_REVIEW_COL], row['Course'])
                for _, row in train_subset.iterrows()
            ]
            # Test rows use the dynamically detected review_col from the arbitrary test CSV
            norm_test_reviews = [
                self.preprocessor.normalize_test_review(r) for r in group[review_col]
            ]
            
            retrieval_vec = TfidfVectorizer(ngram_range=Config.RETRIEVAL_NGRAM, min_df=1)
            train_feats = retrieval_vec.fit_transform(norm_train_reviews)
            test_feats = retrieval_vec.transform(norm_test_reviews)
            
            sim_matrix = test_feats.dot(train_feats.T).toarray()
            
            for i, test_idx in enumerate(group[index_col]):
                sims = sim_matrix[i]
                top_10_local_pos = np.argsort(sims)[::-1][:10]
                rec_indices = train_subset.iloc[top_10_local_pos]['Index'].tolist()
                results.append({'Index': test_idx, 'Index_list': str(rec_indices)})

        logger.info(f"Batch Retrieval completed in {time.time() - t0:.2f}s")
        return pd.DataFrame(results)

# =====================================================================
# REAL-TIME SINGLE QUERY PREDICTION
# =====================================================================
def predict_single_text(pipeline, train_df, prep, review_text, top_k=10):
    """
    Predicts recommendations for any arbitrary single review text query.
    """
    norm_review = prep.normalize_test_review(review_text)
    pred_course = pipeline.predict_courses([norm_review])[0]

    train_subset = train_df[train_df['Course'] == pred_course].copy().reset_index(drop=True)
    if len(train_subset) == 0:
        train_subset = train_df.copy().reset_index(drop=True)

    norm_train_reviews = [
        prep.normalize_train_review(row['Reviews'], row['Course'])
        for _, row in train_subset.iterrows()
    ]

    retrieval_vec = TfidfVectorizer(ngram_range=Config.RETRIEVAL_NGRAM, min_df=1)
    train_feats = retrieval_vec.fit_transform(norm_train_reviews)
    test_feats = retrieval_vec.transform([norm_review])

    sims = test_feats.dot(train_feats.T).toarray()[0]
    top_pos = np.argsort(sims)[::-1][:top_k]

    recs = []
    for p in top_pos:
        recs.append({
            'Index': int(train_subset.iloc[p]['Index']),
            'Course': str(train_subset.iloc[p]['Course']),
            'Review': str(train_subset.iloc[p]['Reviews']),
            'Similarity': float(sims[p])
        })
    return pred_course, recs

# =====================================================================
# VALIDATION MODULE
# =====================================================================
def run_validation(train_df):
    logger.info("=== RUNNING OFFLINE VALIDATION ===")
    
    train_split, val_split = train_test_split(
        train_df, 
        test_size=0.10, 
        stratify=train_df['Course'], 
        random_state=Config.RANDOM_SEED
    )
    
    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)
    
    logger.info(f"Validation Train Split shape: {train_split.shape}")
    logger.info(f"Validation Test Split shape: {val_split.shape}")
    
    courses = train_df['Course'].unique()
    prep = Preprocessor(courses)
    
    val_split['Masked_Reviews'] = [
        prep.simulate_test_review(row['Reviews'], row['Course'], idx)
        for idx, row in val_split.iterrows()
    ]
    
    pipeline = RecommenderPipeline(prep)
    pipeline.train_classifier(train_split['Reviews'], train_split['Course'])
    
    val_preds = pipeline.predict_courses(val_split['Masked_Reviews'])
    val_split['Predicted_Course'] = val_preds
    
    clf_acc = np.mean(val_preds == val_split['Course'])
    logger.info(f"Validation Classifier Accuracy: {clf_acc * 100:.4f}%")
    
    rec_df = pipeline.retrieve_recommendations(train_split, val_split)
    
    train_idx_to_course = dict(zip(train_split['Index'], train_split['Course']))
    val_idx_to_course = dict(zip(val_split['Index'], val_split['Course']))
    
    match_rates = []
    for _, row in rec_df.iterrows():
        test_idx = row['Index']
        true_course = val_idx_to_course[test_idx]
        rec_indices = ast.literal_eval(row['Index_list'])
        rec_courses = [train_idx_to_course[rid] for rid in rec_indices]
        matches = sum(c == true_course for c in rec_courses)
        match_rates.append(matches / 10.0)
        
    mean_match_rate = np.mean(match_rates)
    logger.info(f"Validation Course Match Rate @ 10: {mean_match_rate * 100:.4f}%")
    logger.info("=== VALIDATION END ===\n")

# =====================================================================
# MAIN CLI PIPELINE RUNNER
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Universal Recommender System Pipeline")
    parser.add_argument("--train_path", type=str, default=Config.TRAIN_PATH, help="Path to training CSV")
    parser.add_argument("--test_path", type=str, default=Config.TEST_PATH, help="Path to test CSV to predict on")
    parser.add_argument("--output_path", type=str, default=Config.SUBMISSION_PATH, help="Path to output submission CSV")
    parser.add_argument("--text", type=str, default=None, help="Predict recommendations for a single review text")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive prediction shell")
    parser.add_argument("--skip_validation", action="store_true", help="Skip offline validation step for fast run")
    
    args = parser.parse_args()

    logger.info(f"Loading training data from: {args.train_path}")
    if not os.path.exists(args.train_path):
        logger.error(f"Error: Training file not found at {args.train_path}")
        sys.exit(1)
        
    train_df = pd.read_csv(args.train_path)
    np.random.seed(Config.RANDOM_SEED)

    # 1. Initialize preprocessor and pipeline
    courses = train_df['Course'].unique()
    prep = Preprocessor(courses)
    pipeline = RecommenderPipeline(prep)

    # Normalize full training reviews
    norm_train_reviews = [
        prep.normalize_train_review(row['Reviews'], row['Course'])
        for _, row in train_df.iterrows()
    ]
    pipeline.train_classifier(norm_train_reviews, train_df['Course'])

    # Single Text Query Mode
    if args.text:
        logger.info(f"=== PREDICTING FOR SINGLE TEXT QUERY ===")
        logger.info(f"Input Text: '{args.text}'")
        pred_course, recs = predict_single_text(pipeline, train_df, prep, args.text)
        logger.info(f"Predicted Course: '{pred_course}'")
        logger.info(f"Top 10 Recommended Items:")
        for rank, r in enumerate(recs, 1):
            logger.info(f"  {rank}. [Idx: {r['Index']}] Course: '{r['Course']}' | Similarity: {r['Similarity']:.4f}")
            logger.info(f"     Snippet: {r['Review'][:100]}...")
        return

    # Interactive Shell Mode
    if args.interactive:
        print("\n=== INTERACTIVE RECOMMENDER SHELL ===")
        print("Type a review (or 'exit' to quit):\n")
        while True:
            try:
                user_input = input("Review > ")
                if user_input.strip().lower() in ['exit', 'quit']:
                    print("Exiting interactive mode.")
                    break
                if not user_input.strip():
                    continue
                pred_course, recs = predict_single_text(pipeline, train_df, prep, user_input)
                print(f"\nPredicted Course: [{pred_course}]")
                print("Top 10 Recommendations:")
                for r in recs:
                    print(f" - Item #{r['Index']} (Sim: {r['Similarity']:.4f}): {r['Review'][:80]}...")
                print("-" * 60)
            except KeyboardInterrupt:
                break
        return

    # Offline Validation
    if not args.skip_validation:
        run_validation(train_df)

    # Arbitrary CSV / Submission Batch Processing
    logger.info(f"=== RUNNING BATCH INFERENCE FOR FILE: {args.test_path} ===")
    if not os.path.exists(args.test_path):
        logger.error(f"Error: Test file not found at {args.test_path}")
        sys.exit(1)

    test_df = pd.read_csv(args.test_path)
    idx_col, rev_col = detect_columns(test_df)
    logger.info(f"Detected columns: Index Column='{idx_col}', Review Column='{rev_col}'")

    # Normalize test reviews
    norm_test_reviews = [prep.normalize_test_review(r) for r in test_df[rev_col]]
    
    # Predict courses
    test_df['Predicted_Course'] = pipeline.predict_courses(norm_test_reviews)
    
    # Run batch retrieval
    submission_df = pipeline.retrieve_recommendations(train_df, test_df, index_col=idx_col, review_col=rev_col)
    
    # Ensure correct sorting of Index matching test set
    submission_df = test_df[[idx_col]].rename(columns={idx_col: 'Index'}).merge(submission_df, on='Index', how='left')
    
    # Save submission file
    submission_df.to_csv(args.output_path, index=False)
    logger.info(f"Saved submission output to: {args.output_path}")

    # Verify submission file format
    logger.info("=== VERIFYING OUTPUT FILE FORMAT ===")
    sub_verify = pd.read_csv(args.output_path)
    logger.info(f"Output Shape: {sub_verify.shape}")
    logger.info(f"Output Columns: {sub_verify.columns.tolist()}")
    logger.info(f"Null values in output:\n{sub_verify.isnull().sum()}")

    if sub_verify.shape == (test_df.shape[0], 2):
        logger.info("SUCCESS: Output shape matches test set!")
    else:
        logger.warning(f"Output shape ({sub_verify.shape}) differs from expected test size ({test_df.shape[0]}, 2).")

    first_list = ast.literal_eval(sub_verify.iloc[0]['Index_list'])
    logger.info(f"First row recommendations: {first_list}")
    if len(first_list) == 10 and all(isinstance(x, int) for x in first_list):
        logger.info("SUCCESS: Format verification passed!")
    else:
        logger.error("ERROR: Format verification failed! Ensure list contains exactly 10 integers.")

if __name__ == "__main__":
    main()
