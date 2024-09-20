from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def preprocess_data(train_data, test_data, label_value):
    X = train_data.copy()
    X_test = test_data.copy()

    # Drop rows with empty label value
    X.dropna(axis=0, subset=[label_value], inplace=True)
    y = X.Survived
    X.drop([label_value], axis=1, inplace=True)

    # Editing columns X
    X["Ticket"] = X["Ticket"].apply(
        lambda x: x.split(" ")[0] if len(x.split(" ")) > 1 else "None"
    )
    X["Cabin"] = X["Cabin"].apply(lambda x: x[0] if type(x) == str else "None")
    X["Name"] = X["Name"].apply(lambda x: x.split(",")[0])

    # Editing columns X_test
    X_test["Ticket"] = X_test["Ticket"].apply(
        lambda x: x.split(" ")[0] if len(x.split(" ")) > 1 else "None"
    )
    X_test["Cabin"] = X_test["Cabin"].apply(
        lambda x: x[0] if type(x) == str else "None"
    )
    X_test["Name"] = X_test["Name"].apply(lambda x: x.split(",")[0])

    # Select categorical columns
    categorical_cols_one_hot = ["Sex", "Embarked"]
    categorical_cols_ordinal = ["Name", "Ticket", "Cabin"]

    # Select numerical columns
    numerical_cols = [
        cname for cname in X.columns if X[cname].dtype in ["int64", "float64"]
    ]

    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy="mean")

    # Preprocessing for categorical data
    categorical_transformer_one_hot = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    categorical_transformer_ordinal = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat_one_hot", categorical_transformer_one_hot, categorical_cols_one_hot),
            ("cat_ordinal", categorical_transformer_ordinal, categorical_cols_ordinal),
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
