from pathlib import Path
from datasets import Dataset, load_dataset
from sklearn.datasets import fetch_openml, load_breast_cancer
import kagglehub

def openml_to_huggingface_dataset(data, target):
    return Dataset.from_pandas(
        data.assign(label=target)
    )

def kaggle_to_huggingface_dataset(dataset_name):
    path = kagglehub.dataset_download(dataset_name)
    csv_files = [str(p) for p in Path(path).iterdir() if p.suffix.lower() == ".csv"]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {path}")

    return Dataset.from_csv(csv_files)

def load_uci_heart_disease_dataset():
    dataset = fetch_openml(
        name="heart-disease",
        version=1,
        as_frame=True,
    )
    return (
        openml_to_huggingface_dataset(dataset.data, dataset.target)
        .remove_columns(["label"])
    )

def load_breast_cancer_wisconsin_dataset():
    dataset = load_breast_cancer(
        as_frame=True,
    )
    return openml_to_huggingface_dataset(dataset.data, dataset.target)

def load_ncctg_lung_cancer_dataset():
    return (
        kaggle_to_huggingface_dataset("ukveteran/ncctg-lung-cancer-data")
        .remove_columns(["Unnamed: 0"])
    )

heart_disease = load_uci_heart_disease_dataset()
breast_cancer = load_breast_cancer_wisconsin_dataset()
lung_cancer = load_ncctg_lung_cancer_dataset()

heart_disease.to_csv("heart_disease.csv")
breast_cancer.to_csv("breast_cancer.csv")
lung_cancer.to_csv("lung_cancer.csv")