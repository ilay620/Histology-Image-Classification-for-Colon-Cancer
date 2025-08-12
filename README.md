# Hierarchical Attention CNN for Fine-Grained Histology Classification

A comprehensive project for the classification of 8 tissue types in colorectal cancer histology images. This project details a systematic journey from baseline model evaluation to the development of an advanced, hierarchical system featuring an attention mechanism and targeted data augmentation to solve fine-grained classification challenges.

**Final Macro F1-Score Achieved: [הכנס כאן את הציון הסופי שלך, למשל 0.952]**

---

### About The Project

This project, developed for the Machine Learning course at Holon Institute of Technology (HIT), tackles the challenge of classifying histopathological images from the **Kather_texture_2016 dataset**. The primary goal was to build a robust model capable of distinguishing between 8 tissue types, with a special focus on overcoming the significant visual similarity between the `STROMA`, `COMPLEX`, and `LYMPHO` classes.

The project is contained within a single, end-to-end Jupyter Notebook: `ML_Project.ipynb`.

### The Methodological Journey

The project followed a systematic, multi-stage approach within the notebook:

1.  **Baseline Evaluation:** We began by comparing 9 baseline models, combining three feature extraction methods (Flattened Pixels, PCA, and a pre-trained VGG16) with three classifiers (SVM, Softmax Regression, and a simple Neural Network). The results clearly demonstrated the superiority of the features extracted by the VGG16 model.

2.  **Hyperparameter Optimization:** Using the **Optuna** framework, we conducted an extensive search to find the optimal hyperparameters for a VGG16 + Neural Network model with Fine-Tuning. This process identified a "champion" single model that achieved an F1-score of **~0.95**.

3.  **Error Analysis & Problem Identification:** A deep dive into the champion model's confusion matrix revealed its primary weakness: a persistent confusion between the `STROMA` and `COMPLEX` classes.

4.  **The Hierarchical Solution:** To address this specific issue, we developed a novel hierarchical system:
    * A **Generalist Model** (our champion) performs the initial 8-class classification.
    * A **Specialist Model** is triggered if the Generalist predicts one of the problematic classes. This expert model was specifically trained on a curated dataset of the most confusing classes (`STROMA`, `COMPLEX`, `LYMPHO`) and "hard negatives" (`TUMOR`, `DEBRIS`).

5.  **Optimizing the Specialist:** The specialist model itself is an advanced architecture featuring:
    * A deep classifier head with 2-3 hidden layers.
    * An **Attention Mechanism (SE-Block)** to help the model focus on the most discriminative features.
    * **Targeted Augmentations:** Different, aggressive augmentation pipelines from Albumentations were applied to each problematic class to maximize robustness.

### How to Run

1.  Clone this repository.
2.  Open and run the `ML_Project.ipynb` notebook in a Jupyter environment (like Jupyter Lab or VS Code). Ensure you have a GPU available for the deep learning sections.

### Key Technologies
* **Python**
* **TensorFlow & Keras**
* **Scikit-learn**
* **Optuna** (for Hyperparameter Optimization)
* **Albumentations** (for Advanced Data Augmentation)
* **Weights & Biases** (for Experiment Tracking)