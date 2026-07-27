import os
import re
import sys
import time
import json
import logging
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

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

    # ---------------------------------------------------------------
    # KEY HYPERPARAMETERS (tuned from competition sweep data)
    # Sweep result: 6-grams + max_df=0.2 gave 76.59% (best in sweep)
    # Higher n-grams keep helping: bigrams=64%, trigrams=71%, 4-grams=73%, 5-grams=75%, 6-grams=76.59%
    # ---------------------------------------------------------------
    NGRAM_MAX = 6           # Upper n-gram bound (sweet spot per sweep; try 7 or 8 if time allows)
    MAX_DF = 0.2            # Confirmed sweet spot: removes very common/uninformative terms
    MIN_DF = 2              # Ignore extremely rare terms (noise)
    SUBLINEAR_TF = True     # log(1+tf) — prevents high-freq terms from dominating
    CLASSIFIER_C = 10.0     # Regularization strength for LogisticRegression
    CLASSIFIER_MAX_ITER = 1000  # Enough iterations for convergence on large vocab

# =====================================================================
# COLUMN DETECTION UTILS
# =====================================================================
def detect_columns(df, index_col=None, review_col=None):
    """
    Dynamically identifies index and review columns from any input DataFrame.
    Supports flexible naming conventions, substring matching, and dtype fallbacks.
    """
    idx_col = index_col if (index_col and index_col in df.columns) else None
    rev_col = review_col if (review_col and review_col in df.columns) else None

    # --- Index column detection ---
    if idx_col is None:
        for col in df.columns:
            if str(col).lower() in ['index', 'id', 'row_id', 'sample_id', 'uuid', 'item_id']:
                idx_col = col
                break
    if idx_col is None:
        df['Index'] = range(len(df))
        idx_col = 'Index'
        logger.info("No explicit index column found; auto-generated 'Index' column.")

    # --- Review column detection ---
    if rev_col is None:
        review_keywords = ['review', 'text', 'comment', 'content', 'body',
                           'sentence', 'input', 'feedback', 'description', 'eval', 'quote']
        for col in df.columns:
            if col == idx_col:
                continue
            if any(kw in str(col).lower() for kw in review_keywords):
                rev_col = col
                break
        if rev_col is None:
            for col in df.columns:
                if col != idx_col and df[col].dtype == 'object':
                    rev_col = col
                    break
        if rev_col is None:
            remaining = [c for c in df.columns if c != idx_col]
            if remaining:
                rev_col = remaining[0]

    if rev_col is None:
        raise ValueError(f"Could not detect review column. Available columns: {list(df.columns)}")

    return idx_col, rev_col

# =====================================================================
# CORE PIPELINE
# =====================================================================
class RecommenderPipeline:
    """
    Two-stage recommender:
      Stage 1 - TF-IDF + Logistic Regression: predicts which Course a review belongs to.
      Stage 2 - Global TF-IDF Cosine Similarity: retrieves Top-K most similar training
                reviews within the predicted course subset.

    Key design decisions vs. previous version:
      - Single SHARED global TF-IDF vectorizer (fit on all train reviews) for BOTH stages.
        This gives retrieval access to the full vocabulary rather than a tiny per-course vocab.
      - Raw review text used (no [COURSE] masking). Normalization was hurting retrieval
        by removing discriminative course-name context from the feature space.
      - ngram_range=(1, NGRAM_MAX=6): high n-grams capture meaningful phrases
        (competition sweep confirmed monotonic improvement: bigrams 64% → 6-grams 76.59%).
      - max_df=0.2: aggressively removes very common cross-course terms.
      - sublinear_tf=True: log-normalizes term frequencies.
      - Output uses json.dumps for spec-compliant Index_list formatting.
    """

    def __init__(self):
        ngram = (1, Config.NGRAM_MAX)
        self.vec = TfidfVectorizer(
            ngram_range=ngram,
            min_df=Config.MIN_DF,
            max_df=Config.MAX_DF,
            sublinear_tf=Config.SUBLINEAR_TF,
        )
        self.clf = LogisticRegression(
            C=Config.CLASSIFIER_C,
            max_iter=Config.CLASSIFIER_MAX_ITER,
            n_jobs=-1,
            random_state=Config.RANDOM_SEED,
            solver='saga',       # saga scales better for large sparse matrices
        )
        self.X_train = None      # Cached global train TF-IDF matrix
        self.train_df = None     # Reference to train_df for course grouping

    def fit(self, train_df):
        """
        Fits the global TF-IDF vectorizer and the Stage 1 classifier on the full
        training set. The same vectorizer is shared for Stage 2 retrieval.
        """
        self.train_df = train_df.reset_index(drop=True)

        logger.info(f"Fitting global TF-IDF (ngram=(1,{Config.NGRAM_MAX}), max_df={Config.MAX_DF}, sublinear_tf={Config.SUBLINEAR_TF}) ...")
        t0 = time.time()
        self.X_train = self.vec.fit_transform(self.train_df['Reviews'])
        logger.info(f"Vocabulary size: {len(self.vec.get_feature_names_out()):,} | Done in {time.time()-t0:.1f}s")

        logger.info("Training Stage 1 Classifier (LogisticRegression)...")
        t0 = time.time()
        self.clf.fit(self.X_train, self.train_df['Course'])
        logger.info(f"Classifier training completed in {time.time()-t0:.1f}s")

        # Pre-build course→row-positions lookup for fast retrieval
        self._course_positions = self.train_df.groupby('Course').indices

    def predict_courses(self, reviews):
        """Predict course labels for a list of raw review strings."""
        X = self.vec.transform(reviews)
        return self.clf.predict(X)

    def retrieve_recommendations(self, test_df, index_col='Index', review_col='Reviews', top_k=10):
        """
        Stage 2: For each test review, find the top-k most similar training reviews
        WITHIN the predicted course subset using the shared global TF-IDF matrix.

        Uses the precomputed self.X_train matrix — no per-course re-fitting needed.
        Fallback to full corpus if predicted course is absent from training data.
        """
        logger.info("Starting Stage 2 Batch Retrieval by Course...")
        t0 = time.time()

        reviews_list = test_df[review_col].tolist()
        X_test = self.vec.transform(reviews_list)
        pred_courses = test_df['Predicted_Course'].tolist()
        indices = test_df[index_col].tolist()

        results = []
        for i, (test_idx, course) in enumerate(zip(indices, pred_courses)):
            if i % 2000 == 0 and i > 0:
                logger.info(f"  Retrieval progress: {i}/{len(indices)} rows...")

            pos = self._course_positions.get(course, None)

            # Fallback: if predicted course missing from train, use all train rows
            if pos is None or len(pos) == 0:
                logger.warning(f"Course '{course}' not in train set. Using global fallback for idx={test_idx}.")
                pos = np.arange(len(self.train_df))

            X_train_subset = self.X_train[pos]
            sims = cosine_similarity(X_test[i], X_train_subset).ravel()
            top_local = np.argsort(-sims)[:top_k]
            rec_indices = self.train_df.iloc[pos[top_local]]['Index'].tolist()
            results.append({
                'Index': test_idx,
                'Index_list': json.dumps([int(x) for x in rec_indices])
            })

        logger.info(f"Stage 2 Retrieval completed in {time.time()-t0:.1f}s")
        return pd.DataFrame(results)

# =====================================================================
# VALIDATION
# =====================================================================
def run_validation(train_df):
    """
    90/10 stratified holdout validation.
    Uses RAW reviews (no masking) to faithfully reflect real test conditions.
    Reports Stage 1 classifier accuracy and Stage 2 course match rate @ 10.
    """
    logger.info("=== RUNNING OFFLINE VALIDATION ===")
    train_split, val_split = train_test_split(
        train_df, test_size=0.10, stratify=train_df['Course'],
        random_state=Config.RANDOM_SEED
    )
    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)
    logger.info(f"Train split: {train_split.shape} | Val split: {val_split.shape}")

    pipe = RecommenderPipeline()
    pipe.fit(train_split)

    val_preds = pipe.predict_courses(val_split['Reviews'].tolist())
    clf_acc = np.mean(val_preds == val_split['Course'].values)
    logger.info(f"Stage 1 Classifier Accuracy: {clf_acc * 100:.4f}%")

    val_split['Predicted_Course'] = val_preds
    rec_df = pipe.retrieve_recommendations(val_split, index_col='Index', review_col='Reviews')

    train_idx_to_course = dict(zip(train_split['Index'], train_split['Course']))
    val_idx_to_course   = dict(zip(val_split['Index'],   val_split['Course']))

    match_rates = []
    for _, row in rec_df.iterrows():
        true_course = val_idx_to_course[row['Index']]
        rec_indices = json.loads(row['Index_list'])
        rec_courses = [train_idx_to_course.get(rid) for rid in rec_indices]
        matches = sum(c == true_course for c in rec_courses if c is not None)
        match_rates.append(matches / float(len(rec_indices)))

    mean_match = np.mean(match_rates)
    logger.info(f"Stage 2 Course Match Rate @ 10: {mean_match * 100:.4f}%")
    logger.info("=== VALIDATION END ===\n")

# =====================================================================
# SINGLE TEXT QUERY (interactive / --text mode)
# =====================================================================
def predict_single_text(pipeline, review_text, top_k=10):
    """Returns (predicted_course, list of recommendation dicts) for a single query."""
    pred_course = pipeline.predict_courses([review_text])[0]

    pos = pipeline._course_positions.get(pred_course, None)
    if pos is None or len(pos) == 0:
        pos = np.arange(len(pipeline.train_df))

    X_test_single = pipeline.vec.transform([review_text])
    X_train_subset = pipeline.X_train[pos]
    sims = cosine_similarity(X_test_single, X_train_subset).ravel()
    top_local = np.argsort(-sims)[:top_k]

    recs = []
    for p in top_local:
        row = pipeline.train_df.iloc[pos[p]]
        recs.append({
            'Index': int(row['Index']),
            'Course': str(row['Course']),
            'Review': str(row['Reviews']),
            'Similarity': float(sims[p])
        })
    return pred_course, recs

# =====================================================================
# MAIN CLI
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Personalized Learning Path Recommender Pipeline")
    parser.add_argument("--train_path",    type=str, default=Config.TRAIN_PATH)
    parser.add_argument("--test_path",     type=str, default=Config.TEST_PATH)
    parser.add_argument("--output_path",   type=str, default=Config.SUBMISSION_PATH)
    parser.add_argument("--text",          type=str, default=None, help="Single review text query")
    parser.add_argument("--interactive",   action="store_true",    help="Launch interactive shell")
    parser.add_argument("--skip_validation", action="store_true",  help="Skip offline validation")
    args = parser.parse_args()

    # --- Load training data ---
    if not os.path.exists(args.train_path):
        logger.error(f"Training file not found: {args.train_path}")
        sys.exit(1)
    logger.info(f"Loading training data from: {args.train_path}")
    train_df = pd.read_csv(args.train_path)
    np.random.seed(Config.RANDOM_SEED)

    # --- Offline validation ---
    if not args.skip_validation and not args.text and not args.interactive:
        run_validation(train_df)

    # --- Train full pipeline on 100% of data ---
    pipeline = RecommenderPipeline()
    pipeline.fit(train_df)

    # --- Single Text Query Mode ---
    if args.text:
        logger.info(f"=== SINGLE TEXT QUERY ===")
        logger.info(f"Input: '{args.text}'")
        pred_course, recs = predict_single_text(pipeline, args.text)
        logger.info(f"Predicted Course: '{pred_course}'")
        logger.info("Top 10 Recommended Items:")
        for rank, r in enumerate(recs, 1):
            logger.info(f"  {rank}. [Idx: {r['Index']}] Course: '{r['Course']}' | Sim: {r['Similarity']:.4f}")
            logger.info(f"     Snippet: {r['Review'][:100]}...")
        return

    # --- Interactive Shell Mode ---
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
                pred_course, recs = predict_single_text(pipeline, user_input)
                print(f"\nPredicted Course: [{pred_course}]")
                print("Top 10 Recommendations:")
                for r in recs:
                    print(f" - Item #{r['Index']} (Sim: {r['Similarity']:.4f}): {r['Review'][:80]}...")
                print("-" * 60)
            except KeyboardInterrupt:
                break
        return

    # --- Batch Inference Mode ---
    if not os.path.exists(args.test_path):
        logger.error(f"Test file not found: {args.test_path}")
        sys.exit(1)

    logger.info(f"=== BATCH INFERENCE: {args.test_path} ===")
    test_df = pd.read_csv(args.test_path)
    idx_col, rev_col = detect_columns(test_df)
    logger.info(f"Detected columns → Index='{idx_col}', Review='{rev_col}'")

    # Stage 1: Predict courses
    test_df['Predicted_Course'] = pipeline.predict_courses(test_df[rev_col].tolist())

    # Stage 2: Retrieve recommendations
    submission_df = pipeline.retrieve_recommendations(test_df, index_col=idx_col, review_col=rev_col)

    # Preserve original test order
    submission_df = test_df[[idx_col]].rename(columns={idx_col: 'Index'}).merge(
        submission_df, on='Index', how='left'
    )

    # Save
    submission_df.to_csv(args.output_path, index=False)
    logger.info(f"Saved submission to: {args.output_path} | Shape: {submission_df.shape}")

    # Verify format
    logger.info("=== FORMAT VERIFICATION ===")
    sub_check = pd.read_csv(args.output_path)
    logger.info(f"Columns: {sub_check.columns.tolist()} | Nulls: {sub_check.isnull().sum().to_dict()}")
    first_list = json.loads(sub_check.iloc[0]['Index_list'])
    if len(first_list) == 10 and all(isinstance(x, int) for x in first_list):
        logger.info(f"SUCCESS: Format verified. Sample row: {first_list}")
    else:
        logger.error("ERROR: Format verification failed. Check Index_list values.")

if __name__ == "__main__":
    main()
