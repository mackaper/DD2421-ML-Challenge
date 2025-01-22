# Machine Learning Challenge 2024

This repository showcases my participation in the DD2421 Machine Learning Challenge. The task involved building a high-performing classifier to predict labels for an unlabeled evaluation dataset. The project demonstrates my ability to preprocess data, train machine learning models, and optimize performance metrics.

## Challenge Overview

- **Objective:** Train a classifier using `TrainOnMe.csv` and predict labels for `EvaluateOnMe.csv`.
- **Deliverables:** Submit the predicted labels in `predictions.txt` and the code in a zip file.
- **Evaluation:** Model performance was graded using a predefined scoring curve based on prediction accuracy.

## Repository Contents

- `main.py`: Core implementation of the challenge, including data preprocessing, model training, and prediction generation.
- `TrainOnMe.csv`: Training dataset with labeled examples.
- `EvaluateOnMe.csv`: Evaluation dataset without labels.
- `EvaluateOnMe_predictions.csv`: Generated predictions for the evaluation dataset.
- `predictions.txt`: Final label predictions in submission format.
- `requirements.txt`: Dependencies required to run the project.
- `submission.zip`: Packaged submission file.

## Key Highlights

- **Preprocessing:** Applied feature encoding, alignment, and handling of categorical and boolean data.
- **Model Training:** Trained a Random Forest Classifier with cross-validation for robust performance.
- **Dimensionality Reduction:** Leveraged PCA to improve computational efficiency and model accuracy.
- **Predictions:** Generated and validated predictions that align with the challenge requirements.

## Instructions to Run

1. Install dependencies: `pip install -r requirements.txt`
2. Run the main script: `python main.py`
3. Predictions will be saved in `predictions.txt`.

For more details on the challenge and implementation, see `main.py`.

---

Feel free to explore this project to understand my approach to solving data-driven machine learning challenges!
