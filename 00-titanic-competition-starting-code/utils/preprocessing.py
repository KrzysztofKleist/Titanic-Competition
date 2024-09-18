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
