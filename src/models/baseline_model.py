import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import dagshub
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer


# ---------------- Logging setup ----------------

LOGGER_NAME = "sentiment_model_pipeline"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging(logging.INFO)


# ---------------- MLflow configuration ----------------

DAGSHUB_REPO_OWNER = "jubito27"
DAGSHUB_REPO_NAME = "commentlens"
MLFLOW_TRACKING_URI = "https://dagshub.com/jubito27/commentlens.mlflow"
MLFLOW_EXPERIMENT_NAME = "baseline-models"


def configure_mlflow() -> None:
    """Initialize DagsHub and configure MLflow tracking."""
    try:
        dagshub.init(
            repo_owner=DAGSHUB_REPO_OWNER,
            repo_name=DAGSHUB_REPO_NAME,
            mlflow=True,
        )

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        logger.info("MLflow tracking configured successfully")
        logger.debug("MLflow tracking URI: %s", mlflow.get_tracking_uri())

    except Exception as error:
        logger.exception("Failed to configure MLflow tracking")
        raise RuntimeError(f"Failed to configure MLflow tracking: {error}") from error


# ---------------- Data functions ----------------

LABEL_MAPPING = {
    "positive": 1,
    "negative": -1,
    "neutral": 0,
}


def validate_dataframe(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
) -> None:
    """Validate that the sentiment dataset is usable."""
    if df.empty:
        raise ValueError("Input dataframe is empty")

    missing_columns = [
        column
        for column in [text_column, label_column]
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    if df[text_column].isna().any():
        raise ValueError(f"Text column '{text_column}' contains missing values")

    if df[label_column].isna().any():
        raise ValueError(f"Label column '{label_column}' contains missing values")

    invalid_labels = set(df[label_column].unique()) - set(LABEL_MAPPING.keys())

    if invalid_labels:
        raise ValueError(
            f"Invalid labels found: {sorted(invalid_labels)}. "
            f"Allowed labels: {sorted(LABEL_MAPPING.keys())}"
        )

    logger.debug(
        "Data validation successful. Rows: %s, Label distribution: %s",
        df.shape[0],
        df[label_column].value_counts().to_dict(),
    )


def prepare_data(
    df: pd.DataFrame,
    text_column: str = "comments",
    label_column: str = "category",
) -> tuple[pd.Series, pd.Series]:
    """Prepare text features and encoded sentiment labels."""
    validate_dataframe(
        df=df,
        text_column=text_column,
        label_column=label_column,
    )

    x = df[text_column].astype(str)
    y = df[label_column].map(LABEL_MAPPING)

    logger.debug("Text and labels prepared successfully")
    return x, y


# ---------------- Feature encoding ----------------

def encode_text(
    encoding: str,
    x_train: pd.Series,
    x_test: pd.Series,
    max_features: int,
    embedding_model_name: str = "all-MiniLM-L6-v2",
):
    """Encode text using embeddings, TF-IDF, or CountVectorizer."""
    encoding = encoding.lower().strip()

    if encoding == "embedding":
        encoder = SentenceTransformer(embedding_model_name)

        x_train_encoded = encoder.encode(
            x_train.tolist(),
            show_progress_bar=True,
        )
        x_test_encoded = encoder.encode(
            x_test.tolist(),
            show_progress_bar=True,
        )

        vectorizer = None

    elif encoding == "tfidf":
        vectorizer = TfidfVectorizer(max_features=max_features)

        x_train_encoded = vectorizer.fit_transform(x_train)
        x_test_encoded = vectorizer.transform(x_test)

    elif encoding == "cv":
        vectorizer = CountVectorizer(max_features=max_features)

        x_train_encoded = vectorizer.fit_transform(x_train)
        x_test_encoded = vectorizer.transform(x_test)

    else:
        raise ValueError(
            f"Unsupported encoding method: '{encoding}'. "
            "Allowed values are: 'embedding', 'tfidf', 'cv'."
        )

    logger.debug(
        "Text encoding completed. Train shape: %s, Test shape: %s",
        x_train_encoded.shape,
        x_test_encoded.shape,
    )

    return x_train_encoded, x_test_encoded, vectorizer


# ---------------- Evaluation ----------------

def calculate_classification_metrics(
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Calculate multiclass classification metrics."""
    labels = sorted(set(y_test.unique()) | set(y_pred))

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(
            y_test,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_test,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_test,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "precision_weighted": precision_score(
            y_test,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0,
        ),
        "recall_weighted": recall_score(
            y_test,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0,
        ),
        "f1_weighted": f1_score(
            y_test,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0,
        ),
    }

    logger.debug("Calculated classification metrics: %s", metrics)
    return metrics


# ---------------- Model training ----------------

def train_and_evaluate_models(
    models: Dict[str, Any],
    x_train,
    x_test,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """Train and evaluate every provided model."""
    model_results: Dict[str, Dict[str, float]] = {}

    for model_name, model in models.items():
        logger.info("Training model: %s", model_name)

        try:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            metrics = calculate_classification_metrics(
                y_test=y_test,
                y_pred=y_pred,
            )

            model_results[model_name] = metrics

            logger.info(
                "Model '%s' results: %s",
                model_name,
                metrics,
            )

            with mlflow.start_run(
                run_name=f"{model_name}_sentiment_classification",
                nested=True,
            ) as model_run:
                mlflow.set_tags(
                    {
                        "project": "sentiment_classification",
                        "model_name": model_name,
                        "model_class": model.__class__.__name__,
                    }
                )

                mlflow.log_params(
                    {
                        "model_name": model_name,
                        "model_class": model.__class__.__name__,
                        "n_training_samples": x_train.shape[0],
                        "n_test_samples": x_test.shape[0],
                        "n_features": x_train.shape[1],
                        "n_classes": len(np.unique(y_train)),
                    }
                )

                mlflow.log_metrics(metrics)

                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="sentiment_classifier",
                    input_example=x_train[:5]
                    if hasattr(x_train, "shape")
                    else None,
                )

                logger.debug(
                    "MLflow run completed for model: %s, Run ID: %s",
                    model_name,
                    model_run.info.run_id,
                )

        except Exception:
            logger.exception("Failed to train or evaluate model: %s", model_name)
            raise

    return model_results


# ---------------- Main pipeline ----------------

def pipeline(
    df: pd.DataFrame,
    models: Dict[str, Any],
    test_size: float = 0.2,
    encoding: str = "embedding",
    max_features: int = 1500,
    text_column: str = "comments",
    label_column: str = "category",
    embedding_model_name: str = "all-MiniLM-L6-v2",
    random_state: int = 42,
    experiment_name: str = "sentiment_classification",
) -> Dict[str, Dict[str, float]]:
    """
    Train and evaluate multiple sentiment-classification models.

    Args:
        df: Input dataframe containing text and sentiment labels.
        models: Dictionary mapping model names to unfitted sklearn classifiers.
        test_size: Proportion of data reserved for testing.
        encoding: Text representation method: 'embedding', 'tfidf', or 'cv'.
        max_features: Maximum vocabulary size for TF-IDF or CountVectorizer.
        text_column: Name of the text input column.
        label_column: Name of the sentiment label column.
        embedding_model_name: SentenceTransformer model name.
        random_state: Random seed for reproducibility.
        experiment_name: MLflow experiment name.

    Returns:
        Dictionary containing evaluation metrics for every model.
    """
    if not 0 < test_size < 1:
        raise ValueError(
            f"test_size must be between 0 and 1. Received: {test_size}"
        )

    if not models:
        raise ValueError("At least one model must be provided")

    if max_features < 1:
        raise ValueError(f"max_features must be positive. Received: {max_features}")

    encoding = encoding.lower().strip()

    x, y = prepare_data(
        df=df,
        text_column=text_column,
        label_column=label_column,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        "Data split completed. Train rows: %s, Test rows: %s",
        x_train.shape[0],
        x_test.shape[0],
    )

    x_train_encoded, x_test_encoded, _ = encode_text(
        encoding=encoding,
        x_train=x_train,
        x_test=x_test,
        max_features=max_features,
        embedding_model_name=embedding_model_name,
    )

    configure_mlflow()
    mlflow.set_experiment(experiment_name)

    labels = sorted(set(y.unique()))

    all_model_results: Dict[str, Dict[str, float]] = {}

    with mlflow.start_run(
        run_name=f"sentiment_pipeline_{encoding}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags={
            "project": "sentiment_classification",
            "encoding": encoding,
            "pipeline": "text_encoding+classification",
        },
    ):
        mlflow.log_params(
            {
                "encoding": encoding,
                "max_features": max_features,
                "test_size": test_size,
                "random_state": random_state,
                "text_column": text_column,
                "label_column": label_column,
                "embedding_model_name": embedding_model_name,
                "n_total_rows": df.shape[0],
                "n_train_rows": x_train.shape[0],
                "n_test_rows": x_test.shape[0],
                "n_models": len(models),
                "labels": ", ".join(str(label) for label in labels),
            }
        )

        mlflow.log_artifact(
            __file__,
            artifact_path="source_code",
        )

        # Save a sample of the original dataset for lineage.
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_path = Path(temp_dir) / "dataset_sample.csv"
            df.head(50).to_csv(sample_path, index=False)

            mlflow.log_artifact(
                str(sample_path),
                artifact_path="data_samples",
            )

        all_model_results = train_and_evaluate_models(
            models=models,
            x_train=x_train_encoded,
            x_test=x_test_encoded,
            y_train=y_train,
            y_test=y_test,
        )

        # Log a comparison metric for the parent run.
        best_model_name = max(
            all_model_results,
            key=lambda model_name: all_model_results[model_name]["f1_macro"],
        )

        mlflow.set_tags(
            {
                "best_model": best_model_name,
                "best_f1_macro": str(
                    all_model_results[best_model_name]["f1_macro"]
                ),
            }
        )

        logger.info(
            "Best model based on macro F1-score: %s",
            best_model_name,
        )

    return all_model_results