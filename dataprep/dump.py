from datasets import Dataset
from sklearn.datasets import fetch_openml, load_breast_cancer

def to_huggingface_dataset(data, target):
    return Dataset.from_pandas(
        data.assign(label=target)
    )

def load_uci_heart_disease_dataset():
    dataset = fetch_openml(
        name="heart-disease",
        version=1,
        as_frame=True,
    )
    return to_huggingface_dataset(dataset.data, dataset.target)

def load_breast_cancer_wisconsin_dataset():
    dataset = load_breast_cancer(
        as_frame=True,
    )
    return to_huggingface_dataset(dataset.data, dataset.target)

def load_ncctg_lung_cancer_dataset():
    dataset = fetch_openml(
        name="lung-cancer",
        version=1,
        as_frame=True,
    )
    return to_huggingface_dataset(dataset.data, dataset.target)

heart_disease = load_uci_heart_disease_dataset()
breast_cancer = load_breast_cancer_wisconsin_dataset()
lung_cancer = load_ncctg_lung_cancer_dataset()

heart_disease.to_csv("heart_disease.csv")
breast_cancer.to_csv("breast_cancer.csv")
lung_cancer.to_csv("lung_cancer.csv")