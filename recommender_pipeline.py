import os
import re
import sys
import time
import json
import logging
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================================================================
# CONFIGURATION
# =====================================================================
class Config:
    DATA_DIR = "data"
    TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
    TEST_PATH  = os.path.join(DATA_DIR, "test.csv")
    SUBMISSION_PATH = "submission.csv"
    RANDOM_SEED = 42

    # ---------------------------------------------------------------
    # CLASSIFIER hyperparams (Stage 1 — course prediction)
    # Fit ONLY on training data to avoid label leakage.
    # ---------------------------------------------------------------
    CLF_NGRAM_MAX = 7       # Sweep: bigrams=64 → 6-grams=76.59, trend continues upward
    CLF_MAX_DF    = 0.2     # Confirmed sweet spot: strips cross-course noise
    CLF_MIN_DF    = 2
    CLF_SUBLINEAR = True
    CLF_C         = 10.0
    CLF_MAX_ITER  = 1000

    # ---------------------------------------------------------------
    # RETRIEVAL hyperparams (Stage 2 — within-course similarity)
    # Fit on TRAIN + TEST combined (transductive).
    # Within-course retrieval does NOT need cross-course discrimination,
    # so max_df=0.95 to keep all course-specific common terms.
    # ---------------------------------------------------------------
    RET_NGRAM_MAX    = 7
    RET_MAX_DF       = 0.95   # Keep all within-course terms
    RET_MIN_DF       = 1      # min_df=1 so test-only terms get IDF weight
    RET_SUBLINEAR    = True
    RET_CHAR_NGRAM   = (3, 5) # Character n-grams for subword/morphological features
    RET_CHAR_MAX_DF  = 0.95
    RET_CHAR_MIN_DF  = 3      # Slightly higher to avoid char noise
    WORD_WEIGHT      = 0.7    # Weight for word n-gram features when combining
    CHAR_WEIGHT      = 0.3    # Weight for char n-gram features when combining
    TOP_K            = 10

# =====================================================================
# COLUMN DETECTION UTILS
# =====================================================================
def detect_columns(df, index_col=None, review_col=None):
    """Dynamically identifies index and review columns from any input DataFrame."""
    idx_col = index_col if (index_col and index_col in df.columns) else None
    rev_col = review_col if (review_col and review_col in df.columns) else None

    if idx_col is None:
        for col in df.columns:
            if str(col).lower() in ['index', 'id', 'row_id', 'sample_id', 'uuid', 'item_id']:
                idx_col = col
                break
    if idx_col is None:
        df['Index'] = range(len(df))
        idx_col = 'Index'
        logger.info("Auto-generated 'Index' column.")

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
        raise ValueError(f"Could not detect review column. Columns: {list(df.columns)}")
    return idx_col, rev_col

# =====================================================================
# CORE PIPELINE
# =====================================================================
class RecommenderPipeline:
    """
    Two-stage recommender with max-accuracy configuration:

    Stage 1 – Course Classifier:
      - TF-IDF word n-grams (1, 7) + LogisticRegression
      - Fit on training data ONLY (no label leakage)
      - max_df=0.2 to strip cross-course common terms

    Stage 2 – Within-Course Retrieval:
      - TRANSDUCTIVE TF-IDF: fit on train + test combined.
        Test reviews get proper IDF weights; terms unique to test
        are no longer zero-weighted.
      - Combined word n-grams (1,7) + character n-grams (3,5)
        via weighted sparse hstack — captures morphological variants
        and phrase fragments that word n-grams miss.
      - Retrieval max_df=0.95: keeps all course-specific common terms
        (within-course retrieval doesn't need cross-course discrimination).
      - Batched matrix multiplication per course group (10x faster
        and more numerically stable than row-by-row cosine_similarity).
    """

    def __init__(self):
        # Stage 1 vectorizer (classifier only — trained on train data)
        self.clf_vec = TfidfVectorizer(
            ngram_range=(1, Config.CLF_NGRAM_MAX),
            min_df=Config.CLF_MIN_DF,
            max_df=Config.CLF_MAX_DF,
            sublinear_tf=Config.CLF_SUBLINEAR,
            analyzer='word',
        )
        self.clf = LogisticRegression(
            C=Config.CLF_C,
            max_iter=Config.CLF_MAX_ITER,
            n_jobs=-1,
            random_state=Config.RANDOM_SEED,
            solver='saga',
        )

        # Stage 2 vectorizers (retrieval — transductive: train + test)
        self.ret_word_vec = TfidfVectorizer(
            ngram_range=(1, Config.RET_NGRAM_MAX),
            min_df=Config.RET_MIN_DF,
            max_df=Config.RET_MAX_DF,
            sublinear_tf=Config.RET_SUBLINEAR,
            analyzer='word',
        )
        self.ret_char_vec = TfidfVectorizer(
            ngram_range=Config.RET_CHAR_NGRAM,
            min_df=Config.RET_CHAR_MIN_DF,
            max_df=Config.RET_CHAR_MAX_DF,
            sublinear_tf=Config.RET_SUBLINEAR,
            analyzer='char_wb',
        )

        self.train_df   = None
        self.X_train    = None   # Classifier TF-IDF on train
        self.X_train_ret = None  # Retrieval TF-IDF on train (combined word+char)
        self.X_test_ret  = None  # Retrieval TF-IDF on test  (combined word+char)
        self._course_positions = {}

    def _combine_features(self, word_feats, char_feats):
        """Weighted hstack of word and char TF-IDF matrices, both L2-normalised."""
        word_norm = normalize(word_feats, norm='l2') * Config.WORD_WEIGHT
        char_norm = normalize(char_feats, norm='l2') * Config.CHAR_WEIGHT
        return hstack([word_norm, char_norm], format='csr')

    def fit_classifier(self, train_df):
        """Fits Stage 1 TF-IDF + Logistic Regression on training data only."""
        self.train_df = train_df.reset_index(drop=True)
        train_reviews = self.train_df['Reviews'].tolist()

        logger.info(f"[Stage 1] Fitting classifier TF-IDF "
                    f"(ngram=(1,{Config.CLF_NGRAM_MAX}), max_df={Config.CLF_MAX_DF}) ...")
        t0 = time.time()
        self.X_train = self.clf_vec.fit_transform(train_reviews)
        logger.info(f"[Stage 1] Vocab: {len(self.clf_vec.get_feature_names_out()):,} | {time.time()-t0:.1f}s")

        logger.info("[Stage 1] Training LogisticRegression ...")
        t0 = time.time()
        self.clf.fit(self.X_train, self.train_df['Course'])
        logger.info(f"[Stage 1] Training done in {time.time()-t0:.1f}s")

        self._course_positions = self.train_df.groupby('Course').indices

    def fit_retrieval(self, test_reviews):
        """
        Fits Stage 2 retrieval vectorizers TRANSDUCTIVELY (train + test combined).
        Must be called AFTER fit_classifier so self.train_df is available.
        """
        train_reviews = self.train_df['Reviews'].tolist()
        all_reviews = train_reviews + list(test_reviews)
        n_train = len(train_reviews)

        logger.info(f"[Stage 2] Fitting transductive word TF-IDF "
                    f"(ngram=(1,{Config.RET_NGRAM_MAX}), max_df={Config.RET_MAX_DF}) "
                    f"on {len(all_reviews):,} docs (train+test) ...")
        t0 = time.time()
        word_feats_all = self.ret_word_vec.fit_transform(all_reviews)
        logger.info(f"[Stage 2] Word vocab: {len(self.ret_word_vec.get_feature_names_out()):,} | {time.time()-t0:.1f}s")

        logger.info(f"[Stage 2] Fitting char TF-IDF "
                    f"(ngram={Config.RET_CHAR_NGRAM}, max_df={Config.RET_CHAR_MAX_DF}) ...")
        t0 = time.time()
        char_feats_all = self.ret_char_vec.fit_transform(all_reviews)
        logger.info(f"[Stage 2] Char vocab: {len(self.ret_char_vec.get_feature_names_out()):,} | {time.time()-t0:.1f}s")

        combined_all = self._combine_features(word_feats_all, char_feats_all)
        self.X_train_ret = combined_all[:n_train]
        self.X_test_ret  = combined_all[n_train:]
        logger.info(f"[Stage 2] Combined feature shape: {combined_all.shape}")

    def predict_courses(self, reviews):
        """Predicts course labels for a list of raw review strings."""
        X = self.clf_vec.transform(reviews)
        return self.clf.predict(X)

    def retrieve_recommendations(self, test_df, index_col='Index', review_col='Reviews', top_k=10):
        """
        Batched within-course retrieval using precomputed transductive TF-IDF.
        All test rows for a given predicted course are processed as a MATRIX
        (not row-by-row), which is ~10x faster and identical in result.
        """
        logger.info("[Stage 2] Starting batched retrieval ...")
        t0 = time.time()
        results_dict = {}  # {test_idx: rec_list}

        # Map test_df position → test_idx
        test_positions = list(range(len(test_df)))
        test_indices   = test_df[index_col].tolist()
        pred_courses   = test_df['Predicted_Course'].tolist()

        # Group test rows by predicted course
        course_to_test_pos = {}
        for pos, course in zip(test_positions, pred_courses):
            course_to_test_pos.setdefault(course, []).append(pos)

        n_processed = 0
        for course, t_positions in course_to_test_pos.items():
            train_pos = self._course_positions.get(course, None)
            if train_pos is None or len(train_pos) == 0:
                logger.warning(f"Course '{course}' missing from train. Using full train fallback.")
                train_pos = np.arange(len(self.train_df))

            X_train_sub = self.X_train_ret[train_pos]          # (n_train_course, feats)
            X_test_batch = self.X_test_ret[t_positions]        # (n_test_course, feats)

            # Batched cosine similarity: (n_test_course, n_train_course)
            # Both sides already L2-normalised, so dot product = cosine similarity
            sim_matrix = (X_test_batch @ X_train_sub.T).toarray()

            train_indices_in_course = self.train_df.iloc[train_pos]['Index'].tolist()

            for i, t_pos in enumerate(t_positions):
                top_local = np.argsort(-sim_matrix[i])[:top_k]
                rec_indices = [int(train_indices_in_course[j]) for j in top_local]
                results_dict[test_indices[t_pos]] = rec_indices

            n_processed += len(t_positions)
            if n_processed % 2000 < len(t_positions):
                logger.info(f"  Retrieval progress: {n_processed}/{len(test_df)} ...")

        logger.info(f"[Stage 2] Retrieval done in {time.time()-t0:.1f}s")

        rows = [
            {'Index': test_indices[p], 'Index_list': json.dumps(results_dict[test_indices[p]])}
            for p in test_positions
        ]
        return pd.DataFrame(rows)

# =====================================================================
# VALIDATION
# =====================================================================
def run_validation(train_df):
    """
    90/10 stratified holdout validation on raw reviews.
    Exercises both Stage 1 accuracy and Stage 2 course match rate @ 10.
    """
    logger.info("=== OFFLINE VALIDATION ===")
    tr, val = train_test_split(
        train_df, test_size=0.10,
        stratify=train_df['Course'], random_state=Config.RANDOM_SEED
    )
    tr  = tr.reset_index(drop=True)
    val = val.reset_index(drop=True)
    logger.info(f"Train: {tr.shape} | Val: {val.shape}")

    pipe = RecommenderPipeline()
    pipe.fit_classifier(tr)
    pipe.fit_retrieval(val['Reviews'].tolist())

    val_preds = pipe.predict_courses(val['Reviews'].tolist())
    clf_acc = np.mean(val_preds == val['Course'].values)
    logger.info(f"[Stage 1] Classifier Accuracy: {clf_acc * 100:.4f}%")

    val = val.copy()
    val['Predicted_Course'] = val_preds
    rec_df = pipe.retrieve_recommendations(val, index_col='Index', review_col='Reviews')

    tr_map  = dict(zip(tr['Index'],  tr['Course']))
    val_map = dict(zip(val['Index'], val['Course']))

    match_rates = []
    for _, row in rec_df.iterrows():
        true_course = val_map[row['Index']]
        rec_indices = json.loads(row['Index_list'])
        matches = sum(tr_map.get(r) == true_course for r in rec_indices)
        match_rates.append(matches / float(len(rec_indices)))

    logger.info(f"[Stage 2] Course Match Rate @ 10: {np.mean(match_rates) * 100:.4f}%")
    logger.info("=== VALIDATION END ===\n")

# =====================================================================
# SINGLE TEXT QUERY
# =====================================================================
def predict_single_text(pipeline, review_text, top_k=10):
    """Returns (predicted_course, recs) for a single raw text query."""
    pred_course = pipeline.predict_courses([review_text])[0]
    pos = pipeline._course_positions.get(pred_course, np.arange(len(pipeline.train_df)))

    word_feat = pipeline.ret_word_vec.transform([review_text])
    char_feat = pipeline.ret_char_vec.transform([review_text])
    q_feat = pipeline._combine_features(word_feat, char_feat)
    X_sub = pipeline.X_train_ret[pos]
    sims = (q_feat @ X_sub.T).toarray().ravel()
    top_local = np.argsort(-sims)[:top_k]

    recs = []
    for p in top_local:
        row = pipeline.train_df.iloc[pos[p]]
        recs.append({'Index': int(row['Index']), 'Course': str(row['Course']),
                     'Review': str(row['Reviews']), 'Similarity': float(sims[p])})
    return pred_course, recs

# =====================================================================
# MAIN CLI
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Max-Accuracy Recommender Pipeline")
    parser.add_argument("--train_path",      type=str, default=Config.TRAIN_PATH)
    parser.add_argument("--test_path",       type=str, default=Config.TEST_PATH)
    parser.add_argument("--output_path",     type=str, default=Config.SUBMISSION_PATH)
    parser.add_argument("--text",            type=str, default=None)
    parser.add_argument("--interactive",     action="store_true")
    parser.add_argument("--skip_validation", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.train_path):
        logger.error(f"Training file not found: {args.train_path}"); sys.exit(1)
    logger.info(f"Loading training data: {args.train_path}")
    train_df = pd.read_csv(args.train_path)
    np.random.seed(Config.RANDOM_SEED)

    # Offline validation
    if not args.skip_validation and not args.text and not args.interactive:
        run_validation(train_df)

    # Load test data (needed for transductive retrieval vectorizer)
    test_path = args.test_path if not args.text and not args.interactive else None
    if test_path and not os.path.exists(test_path):
        logger.error(f"Test file not found: {test_path}"); sys.exit(1)

    # Build full pipeline
    pipeline = RecommenderPipeline()
    pipeline.fit_classifier(train_df)

    if args.text:
        # For single query: use train reviews as the transductive corpus
        logger.info("[Stage 2] Fitting retrieval on train only (single query mode) ...")
        pipeline.fit_retrieval(train_df['Reviews'].tolist())
        logger.info(f"=== SINGLE TEXT QUERY: '{args.text}' ===")
        pred_course, recs = predict_single_text(pipeline, args.text)
        logger.info(f"Predicted Course: '{pred_course}'")
        for rank, r in enumerate(recs, 1):
            logger.info(f"  {rank}. [Idx: {r['Index']}] '{r['Course']}' | Sim: {r['Similarity']:.4f}")
            logger.info(f"     {r['Review'][:100]}...")
        return

    if args.interactive:
        pipeline.fit_retrieval(train_df['Reviews'].tolist())
        print("\n=== INTERACTIVE RECOMMENDER SHELL ===")
        print("Type a review (or 'exit' to quit):\n")
        while True:
            try:
                user_input = input("Review > ")
                if user_input.strip().lower() in ['exit', 'quit']:
                    print("Exiting."); break
                if not user_input.strip():
                    continue
                pred_course, recs = predict_single_text(pipeline, user_input)
                print(f"\nPredicted Course: [{pred_course}]")
                for r in recs:
                    print(f" - Item #{r['Index']} (Sim: {r['Similarity']:.4f}): {r['Review'][:80]}...")
                print("-" * 60)
            except KeyboardInterrupt:
                break
        return

    # Batch inference
    logger.info(f"=== BATCH INFERENCE: {args.test_path} ===")
    test_df = pd.read_csv(args.test_path)
    idx_col, rev_col = detect_columns(test_df)
    logger.info(f"Detected → Index='{idx_col}', Review='{rev_col}'")

    # Fit retrieval transductively (train + test)
    pipeline.fit_retrieval(test_df[rev_col].tolist())

    # Stage 1: predict courses
    test_df['Predicted_Course'] = pipeline.predict_courses(test_df[rev_col].tolist())

    # Stage 2: batched retrieval
    submission_df = pipeline.retrieve_recommendations(
        test_df, index_col=idx_col, review_col=rev_col
    )

    # Preserve original test order
    submission_df = (
        test_df[[idx_col]].rename(columns={idx_col: 'Index'})
        .merge(submission_df, on='Index', how='left')
    )

    submission_df.to_csv(args.output_path, index=False)
    logger.info(f"Saved: {args.output_path} | Shape: {submission_df.shape}")

    # Format check
    logger.info("=== FORMAT CHECK ===")
    chk = pd.read_csv(args.output_path)
    logger.info(f"Cols: {chk.columns.tolist()} | Nulls: {chk.isnull().sum().to_dict()}")
    first = json.loads(chk.iloc[0]['Index_list'])
    if len(first) == 10 and all(isinstance(x, int) for x in first):
        logger.info(f"SUCCESS: Format verified. Sample: {first}")
    else:
        logger.error("ERROR: Format check failed!")

if __name__ == "__main__":
    main()
