# Titanic-Survival-Prediction
# Titanic Survival Prediction 🚢

## Overview
This project is a Machine Learning model that predicts whether a passenger survived the Titanic disaster. The model uses classification techniques based on passenger details such as age, gender, ticket class, fare, and cabin information.

## Dataset
The dataset includes:
- **Numerical Features**: Age, Fare, SibSp (siblings/spouses aboard), Parch (parents/children aboard)
- **Categorical Features**: Sex, Pclass (ticket class), Embarked (boarding port)

## Features & Preprocessing
- Handled missing values for `Age` and `Fare`.
- Encoded categorical variables (`Sex`, `Embarked`).
- Standardized numerical features.
- Dropped unnecessary columns (`PassengerId`, `Name`, `Ticket`, `Cabin`).

## Model & Evaluation
- **Model Used**: Logistic Regression
- **Performance Metrics**: Accuracy, Precision, Recall, F1-score

## Installation & Usage
### Requirements
Install dependencies with:
```bash
pip install pandas scikit-learn numpy
```

### Run the Model
```bash
python titanic_survival.py
```

## Results
The model predicts Titanic survival with strong accuracy. Further improvements can be made using hyperparameter tuning and ensemble models.

## Contributing
Feel free to fork and improve the project! PRs are welcome. 🚀

## License
This project is open-source under the MIT License.

