import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
    return df

def preprocess_data(df):
    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    df["Age"] = imputer.fit_transform(df[["Age"]])
    df["Fare"] = imputer.fit_transform(df[["Fare"]])
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    df["Sex"] = label_encoder.fit_transform(df["Sex"])
    df["Embarked"] = label_encoder.fit_transform(df["Embarked"])
    
    return df

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def main():
    file_path = "tested.csv"  # Update with actual file path
    df = load_data(file_path)
    df = preprocess_data(df)
    
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = train_model(X_train, y_train)
    accuracy, report = evaluate_model(model, X_test, y_test)
    
    print(f"Model Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)

if __name__ == "__main__":
    main()
