"""Feature preprocessing: impute, scale numbers, one-hot encode categories."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(params):
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([("num", num, params["numeric_features"]),
                              ("cat", cat, params["categorical_features"])])
