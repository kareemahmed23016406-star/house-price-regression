import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor


def convert(floors):
    if floors == 1:
        return 0
    else:
        return 1


def main(data_path: str):
    data = pd.read_csv(data_path)

    # Your original selection
    x = data.iloc[:, 3:].copy()
    y = data.iloc[:, 1:2].copy()

    # Convert floors to binary (same logic as your notebook)
    if "floors" in x.columns:
        x["floors"] = x["floors"].apply(convert)

    # Train/test split (same params)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=42
    )

    # Keep numeric only (same idea)
    x_train = x_train.select_dtypes(include=["int64", "float64"])
    x_test = x_test.select_dtypes(include=["int64", "float64"])

    # Impute missing values (mean)
    imp = SimpleImputer(strategy="mean")
    x_train = imp.fit_transform(x_train)
    x_test = imp.transform(x_test)

    # Linear Regression
    model1 = LinearRegression()
    model1.fit(x_train, y_train)
    print("LinearRegression test score:", model1.score(x_test, y_test))

    # Polynomial Regression
    poly = PolynomialFeatures(degree=2)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.transform(x_test)

    model2 = LinearRegression()
    model2.fit(x_train_poly, y_train)
    print("Polynomial(2) LinearRegression test score:", model2.score(x_test_poly, y_test))

    # KNN Regressor (your notebook prints train score)
    model3 = KNeighborsRegressor()
    model3.fit(x_train, y_train)
    print("KNNRegressor train score:", model3.score(x_train, y_train))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/data.csv")
    args = parser.parse_args()
    main(args.data_path)
