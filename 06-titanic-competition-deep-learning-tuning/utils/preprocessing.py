from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def preprocess_data(train_data, test_data, label_value, cols_to_drop):
    X = train_data.copy()
    X_test = test_data.copy()

    # Drop rows with empty label value
    X.dropna(axis=0, subset=[label_value], inplace=True)
    y = X.Survived
    X.drop([label_value], axis=1, inplace=True)

    # Dropping columns
    X.drop(cols_to_drop, axis=1, inplace=True)
    X_test.drop(cols_to_drop, axis=1, inplace=True)

    # Select categorical columns
    categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
    # Select numerical columns
    numerical_cols = [
        cname for cname in X.columns if X[cname].dtype in ["int64", "float64"]
    ]
    # Keep selected columns only
    my_cols = categorical_cols + numerical_cols

    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy="mean")

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Transform both the training and the test data
    X = preprocessor.fit_transform(X)
    X_test = preprocessor.transform(X_test)

    return X, y.to_numpy(), X_test


def get_variable_name(variable):
    for name in globals():
        if id(globals()[name]) == id(variable):
            return name
    for name in locals():
        if id(locals()[name]) == id(variable):
            return name
    return None


def get_num_missing_values(df):
    missing_val_count_by_column = df.isnull().sum()
    df_name = get_variable_name(df)
    print(f"Missing values in {df_name} columns:")
    print(missing_val_count_by_column[missing_val_count_by_column > 0])


def get_categorical_columns(df):
    object_cols = [col for col in df.columns if df[col].dtype == "object"]
    object_nunique = list(map(lambda col: df[col].nunique(), object_cols))
    d = dict(zip(object_cols, object_nunique))
    df_name = get_variable_name(df)
    print(f"{df_name} columns with categorical data:")
    print(sorted(d.items(), key=lambda x: x[1]))
