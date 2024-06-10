# Fake News Detection Using Machine Learning

This project involves developing and deploying a machine learning model to detect fake news using Logistic Regression and Random Forest algorithms. The goal is to classify news articles as real or fake with high accuracy.

## Overview

Fake news detection is a critical task in today's digital age where misinformation can spread rapidly. This project utilizes two popular machine learning algorithms, Logistic Regression and Random Forest, to build an effective fake news detection model. The project includes data preprocessing, feature engineering, model training, evaluation, and optimization.

## Project Structure

- `data/`: Contains the dataset used for training and testing the models.
- `notebooks/`: Jupyter notebooks with detailed steps of data preprocessing, model training, and evaluation.
- `src/`: Source code for data preprocessing, feature extraction, model training, and evaluation.
- `models/`: Saved models for Logistic Regression and Random Forest.
- `results/`: Evaluation metrics and comparison between models.

## Key Features

- **High Accuracy**: Achieved 92% accuracy with Logistic Regression and 98% accuracy with Random Forest.
- **Comprehensive Data Processing**: Includes tokenization, stop word removal, and TF-IDF vectorization.
- **Robust Model Evaluation**: Uses accuracy, precision, recall, and F1-score for evaluating model performance.
- **Optimization**: Hyperparameter tuning implemented for both models to enhance performance.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Libraries: pandas, numpy, sklearn, matplotlib, nltk

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git
   cd fake-news-detection
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Data Preprocessing**:
   - Load and clean the dataset.
   - Perform tokenization, stop word removal, and TF-IDF vectorization.

2. **Model Training**:
   - Train the Logistic Regression model:
     ```python
     python src/train_logistic_regression.py
     ```
   - Train the Random Forest model:
     ```python
     python src/train_random_forest.py
     ```

3. **Model Evaluation**:
   - Evaluate the models using the provided Jupyter notebooks in the `notebooks/` directory.
   - Compare the results and metrics.

## Results

- **Logistic Regression**: Achieved an accuracy of 92%.
- **Random Forest**: Achieved an accuracy of 98%.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or additions.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

- Special thanks to the authors of the dataset used in this project.
- Inspiration from various open-source projects and tutorials.

---

Feel free to modify the README file to better suit your specific project details and requirements.
