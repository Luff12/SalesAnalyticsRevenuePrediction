
from pathlib import Path

import numpy as np
import pandas as pd


# Reproducibility: generates the same dataset each time
np.random.seed(42)

# Project paths
ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Dataset settings
N_TRANSACTIONS = 5000
N_CUSTOMERS = 1000

# Product catalogue: category, selling price, cost per unit
products = {
    "Laptop": {
        "category": "Electronics",
        "price": 65000,
        "cost": 52000,
    },
    "Smartphone": {
        "category": "Electronics",
        "price": 25000,
        "cost": 19000,
    },
    "Headphones": {
        "category": "Accessories",
        "price": 2500,
        "cost": 1400,
    },
    "Keyboard": {
        "category": "Accessories",
        "price": 1800,
        "cost": 1000,
    },
    "Office Chair": {
        "category": "Furniture",
        "price": 8500,
        "cost": 6000,
    },
    "Desk": {
        "category": "Furniture",
        "price": 12000,
        "cost": 8500,
    },
    "Monitor": {
        "category": "Electronics",
        "price": 15000,
        "cost": 11000,
    },
    "Printer": {
        "category": "Electronics",
        "price": 10000,
        "cost": 7500,
    },
}

product_names = list(products.keys())

# Product probabilities: more affordable products sell more often
product_probabilities = [
    0.08, 0.20, 0.19, 0.16,
    0.10, 0.07, 0.13, 0.07
]

# Generate dates from 2022 through 2025
date_range = pd.date_range(
    start="2022-01-01",
    end="2025-12-31",
    freq="D"
)

# Give some months higher sales activity
# December and festival-season months have higher weights
month_weights = {
    1: 0.90, 2: 0.90, 3: 1.00, 4: 1.00,
    5: 1.05, 6: 0.95, 7: 1.00, 8: 1.10,
    9: 1.15, 10: 1.25, 11: 1.20, 12: 1.50
}

date_weights = np.array([
    month_weights[date.month]
    for date in date_range
], dtype=float)

date_weights = date_weights / date_weights.sum()

order_dates = np.random.choice(
    date_range,
    size=N_TRANSACTIONS,
    p=date_weights
)

# Generate transaction attributes
selected_products = np.random.choice(
    product_names,
    size=N_TRANSACTIONS,
    p=product_probabilities
)

regions = np.random.choice(
    ["North", "South", "East", "West", "Central"],
    size=N_TRANSACTIONS,
    p=[0.22, 0.25, 0.15, 0.23, 0.15]
)

channels = np.random.choice(
    ["Online", "Retail", "Distributor"],
    size=N_TRANSACTIONS,
    p=[0.45, 0.35, 0.20]
)

quantities = np.random.choice(
    [1, 2, 3, 4, 5, 6, 8, 10],
    size=N_TRANSACTIONS,
    p=[0.28, 0.22, 0.16, 0.12, 0.09, 0.06, 0.04, 0.03]
)

# Discounts are percentages, e.g. 10 means 10%
discounts = np.random.choice(
    [0, 5, 10, 15, 20],
    size=N_TRANSACTIONS,
    p=[0.25, 0.25, 0.25, 0.15, 0.10]
)

# Create transaction-level records
records = []

for i in range(N_TRANSACTIONS):
    product = selected_products[i]
    product_info = products[product]

    # Small price variation between transactions
    unit_price = round(
        product_info["price"] * np.random.uniform(0.95, 1.05),
        2
    )

    unit_cost = product_info["cost"]

    quantity = int(quantities[i])
    discount = int(discounts[i])

    gross_sales = quantity * unit_price
    revenue = gross_sales * (1 - discount / 100)

    # Add modest variation to unit costs
    actual_unit_cost = unit_cost * np.random.uniform(0.97, 1.03)
    total_cost = quantity * actual_unit_cost

    profit = revenue - total_cost

    records.append({
        "OrderID": f"ORD{i + 1:05d}",
        "OrderDate": pd.Timestamp(order_dates[i]),
        "Product": product,
        "Category": product_info["category"],
        "Region": regions[i],
        "SalesChannel": channels[i],
        "Quantity": quantity,
        "UnitPrice": unit_price,
        "Discount": discount,
        "Revenue": round(revenue, 2),
        "Cost": round(total_cost, 2),
        "Profit": round(profit, 2),
        "CustomerID": (
            f"CUST{np.random.randint(1, N_CUSTOMERS + 1):04d}"
        ),
    })

# Create DataFrame and sort by date
df = pd.DataFrame(records)
df["OrderDate"] = pd.to_datetime(df["OrderDate"])
df = df.sort_values("OrderDate").reset_index(drop=True)

# Save raw dataset
output_path = RAW_DATA_DIR / "sales_data.csv"
df.to_csv(output_path, index=False)

# Display a summary
print("Sales dataset created successfully!")
print(f"Number of transactions: {len(df):,}")
print(f"Number of unique customers: {df['CustomerID'].nunique():,}")
print(f"Date range: {df['OrderDate'].min().date()} to "
      f"{df['OrderDate'].max().date()}")
print(f"Total revenue: {df['Revenue'].sum():,.2f}")
print(f"Total profit: {df['Profit'].sum():,.2f}")
print(f"Saved to: {output_path}")

print("\nFirst 5 rows:")
print(df.head().to_string(index=False))

print("\nMissing values:")
print(df.isnull().sum())