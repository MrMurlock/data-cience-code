import pandas as pd

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    return df.drop_duplicates()


def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Remove outliers using IQR method."""
    df = df.copy()
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]


def split_blood_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """Split Blood Pressure into systolic and diastolic columns."""
    df = df.copy()
    bp_split = df["Blood Pressure"].str.split("/", expand=True)
    df["bp_systolic"] = bp_split[0].astype(int)
    df["bp_diastolic"] = bp_split[1].astype(int)
    return df


def fill_sleep_disorder_none(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing Sleep Disorder with 'None' (no disorder)."""
    df = df.copy()
    df["Sleep Disorder"] = df["Sleep Disorder"].fillna("N/A")
    return df


def standardize_bmi_category(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize BMI Category: 'Normal Weight' -> 'Normal'."""
    df = df.copy()
    df["BMI Category"] = df["BMI Category"].replace("Normal Weight", "Normal")
    return df


def create_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Create age groups: 27-35 (Young), 36-45 (Middle), 46-59 (Senior)."""
    df = df.copy()
    bins = [26, 35, 45, 60]
    labels = ["Young", "Middle", "Senior"]
    df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
    return df


def create_quality_group(df: pd.DataFrame) -> pd.DataFrame:
    """Create sleep quality groups: 4-5 (Low), 6-7 (Medium), 8-9 (High)."""
    df = df.copy()
    bins = [3, 5, 7, 10]
    labels = ["Low", "Medium", "High"]
    df["quality_group"] = pd.cut(df["Quality of Sleep"], bins=bins, labels=labels, right=False)
    return df


def convert_to_category(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert columns to category type."""
    df = df.copy()
    for col in columns:
        df[col] = df[col].astype("category")
    return df


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run full cleaning pipeline."""
    print("=== Cleaning Pipeline ===")
    print(f"Initial shape: {df.shape}")

    df = split_blood_pressure(df)
    print("✓ Split Blood Pressure")

    df = fill_sleep_disorder_none(df)
    print("✓ Fill Sleep Disorder NaN -> N/A")

    df = standardize_bmi_category(df)
    print("✓ Standardize BMI Category")

    df = create_age_group(df)
    print("✓ Create age_group")

    df = create_quality_group(df)
    print("✓ Create quality_group")

    cat_cols = ["Gender", "Occupation", "BMI Category", "Sleep Disorder", "age_group", "quality_group"]
    df = convert_to_category(df, cat_cols)
    print("✓ Convert to category")

    print(f"Final shape: {df.shape}")
    return df