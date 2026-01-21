# model_development.py

import pickle
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def main():
    wine = load_wine(as_frame=True)
    df = wine.frame

    features = [
        'alcohol',
        'malic_acid',
        'alcalinity_of_ash',
        'magnesium',
        'flavanoids',
        'color_intensity'
    ]

    X = df[features]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    with open("wine_cultivar_model.pkl", "wb") as f:
        pickle.dump((model, scaler, features), f)

    print("\nModel saved successfully.")


if __name__ == "__main__":
    main()
