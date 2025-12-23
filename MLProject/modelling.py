import argparse
import os
import json
import numpy as np

import mlflow
import mlflow.sklearn

from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def load_csr_from_npz(npz_path: str) -> csr_matrix:
    z = np.load(npz_path, allow_pickle=True)
    required = {"data", "indices", "indptr", "shape"}
    if not required.issubset(set(z.files)):
        raise ValueError(f"NPZ CSR format not recognized. Keys found: {z.files}")
    return csr_matrix((z["data"], z["indices"], z["indptr"]), shape=z["shape"])


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="adult_income_preprocessing")
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--solver", type=str, default="liblinear")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--run_name", type=str, default="ci_retrain")
    parser.add_argument("--out_dir", type=str, default="artifacts_out")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, args.data_dir)
    out_dir = os.path.join(here, args.out_dir)
    ensure_dir(out_dir)

    X_train = load_csr_from_npz(os.path.join(data_dir, "X_train.npz"))
    X_test  = load_csr_from_npz(os.path.join(data_dir, "X_test.npz"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"), allow_pickle=True)
    y_test  = np.load(os.path.join(data_dir, "y_test.npy"), allow_pickle=True)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"file://{os.path.join(here, 'mlruns')}")
    exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "ci-workflow")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)

    model = LogisticRegression(
        max_iter=args.max_iter,
        solver=args.solver,
        C=args.C
    )

    with mlflow.start_run(run_name=args.run_name) as run:
        # Train
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # -Metrics
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, pos_label=">50K")

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("solver", args.solver)
        mlflow.log_param("C", args.C)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "model")

        local_model_dir = os.path.join(out_dir, "model")
        mlflow.sklearn.save_model(model, local_model_dir)

        report = classification_report(y_test, preds)
        report_path = os.path.join(out_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        ConfusionMatrixDisplay.from_predictions(y_test, preds, display_labels=["<=50K", ">50K"])
        plt.tight_layout()
        cm_path = os.path.join(out_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()
        mlflow.log_artifact(cm_path)

        meta = {
            "run_id": run.info.run_id,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": int(X_train.shape[1]),
            "pos_label": ">50K",
            "accuracy": float(acc),
            "f1": float(f1),
        }
        meta_path = os.path.join(out_dir, "run_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        mlflow.log_artifact(meta_path)

        print("Test accuracy:", acc)
        print("Test f1:", f1)
        print("Saved model for Docker at:", local_model_dir)
        print("Artifacts out:", out_dir)


if __name__ == "__main__":
    main()

