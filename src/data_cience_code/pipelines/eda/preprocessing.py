from pandas import DataFrame


def split_blood_pressure(df: DataFrame) -> DataFrame:
    """Split Blood Pressure into systolic and diastolic columns."""
    df = df.copy()
    bp_split = df["Blood Pressure"].str.split("/", expand=True)
    df["bp_systolic"] = bp_split[0].astype(int)
    df["bp_diastolic"] = bp_split[1].astype(int)
    df = df.drop(columns=["Blood Pressure"])
    return df


def fill_sleep_disorder_none(df: DataFrame) -> DataFrame:
    """Fill missing Sleep Disorder with 'No disorder' (no disorder)."""
    df = df.copy()
    df["Sleep Disorder"] = df["Sleep Disorder"].fillna("No disorder")
    return df


def standardize_bmi_category(df: DataFrame) -> DataFrame:
    """Standardize BMI Category: 'Normal Weight' -> 'Normal'."""
    df = df.copy()
    df["BMI Category"] = df["BMI Category"].replace("Normal Weight", "Normal")
    return df


def drop_person_id_feature(df: DataFrame) -> DataFrame:
    df = df.copy()
    df = df.drop(columns=["Person ID"])
    return df


def preprocess_dataset(df: DataFrame) -> DataFrame:
    """Run full preprocessing pipeline for EDA."""
    df = split_blood_pressure(df)
    df = fill_sleep_disorder_none(df)
    df = standardize_bmi_category(df)
    df = drop_person_id_feature(df)
    return df


def preprocess(df: DataFrame) -> DataFrame:
    """Backward compatible alias."""
    return preprocess_dataset(df)


def clean_dataset(df: DataFrame) -> DataFrame:
    """Alias for preprocessing."""
    return preprocess_dataset(df)
