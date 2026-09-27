
from pathlib import Path

import pandas as pd


# Project paths
ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = ROOT / "data" / "raw" / "sales_data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = PROCESSED_DIR / "sales_cleaned.csv"


def load_data():
    """Load the raw sales dataset."""
    df = pd.read_csv(RAW_PATH)
    df["OrderDate"] = pd.to_datetime(
        df["OrderDate"],
        errors="coerce"
    )
    return df


def clean_data(df):
    """Clean and validate the sales dataset."""
    df = df.copy()

    # Remove duplicate transactions
    df = df.drop_duplicates(subset=["OrderID"])

    # Remove records with missing essential values
    essential_columns = [
        "OrderID",
        "OrderDate",
        "Product",
        "Quantity",
        "UnitPrice",
        "Revenue",
        "Cost",
        "Profit"
    ]

    df = df.dropna(subset=essential_columns)

    # Remove invalid records
    df = df[
        (df["Quantity"] > 0)
        & (df["UnitPrice"] >= 0)
        & (df["Revenue"] >= 0)
        & (df["Cost"] >= 0)
    ].copy()

    # Recalculate financial fields consistently
    df["GrossSales"] = df["Quantity"] * df["UnitPrice"]

    df["Revenue"] = (
        df["GrossSales"] * (1 - df["Discount"] / 100)
    ).round(2)

    df["Profit"] = (df["Revenue"] - df["Cost"]).round(2)

    # Add date-based features for analysis
    df["Year"] = df["OrderDate"].dt.year
    df["Month"] = df["OrderDate"].dt.month
    df["MonthName"] = df["OrderDate"].dt.strftime("%b")
    df["Quarter"] = df["OrderDate"].dt.to_period("Q").astype(str)

    # Save the cleaned dataset
    df = df.sort_values("OrderDate").reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)

    return df


if __name__ == "__main__":
    raw_df = load_data()
    cleaned_df = clean_data(raw_df)

    print("DATA CLEANING SUMMARY")
    print("-" * 40)
    print("Raw rows:", len(raw_df))
    print("Cleaned rows:", len(cleaned_df))
    print("Rows removed:", len(raw_df) - len(cleaned_df))
    print("Missing values after cleaning:", cleaned_df.isnull().sum().sum())
    print("Duplicate OrderIDs:", cleaned_df["OrderID"].duplicated().sum())
    print("Cleaned file:", OUTPUT_PATH)

    print("\nDataset information:")
    print(cleaned_df.info())

    print("\nNumerical summary:")
    print(cleaned_df.describe().round(2))