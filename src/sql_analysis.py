
from pathlib import Path

import duckdb


# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "sales_cleaned.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Connect to DuckDB
con = duckdb.connect(database=":memory:")

# Load CSV as a SQL table
con.execute(
    """
    CREATE TABLE sales AS
    SELECT *
    FROM read_csv_auto(?)
    """,
    [str(DATA_PATH)]
)

print("SQL ANALYSIS RESULTS")
print("=" * 60)


# 1. Overall sales KPIs
print("\n1. OVERALL SALES KPIs")

kpis = con.execute(
    """
    SELECT
        COUNT(*) AS total_orders,
        COUNT(DISTINCT CustomerID) AS unique_customers,
        ROUND(SUM(Revenue), 2) AS total_revenue,
        ROUND(SUM(Profit), 2) AS total_profit,
        ROUND(AVG(Revenue), 2) AS average_order_revenue
    FROM sales
    """
).df()

print(kpis.to_string(index=False))


# 2. Revenue by product
print("\n2. REVENUE BY PRODUCT")

product_revenue = con.execute(
    """
    SELECT
        Product,
        SUM(Quantity) AS units_sold,
        ROUND(SUM(Revenue), 2) AS total_revenue
    FROM sales
    GROUP BY Product
    ORDER BY total_revenue DESC
    """
).df()

print(product_revenue.to_string(index=False))
product_revenue.to_csv(
    OUTPUT_DIR / "sql_product_revenue.csv",
    index=False
)


# 3. Profit by region
print("\n3. PROFIT BY REGION")

region_profit = con.execute(
    """
    SELECT
        Region,
        ROUND(SUM(Revenue), 2) AS total_revenue,
        ROUND(SUM(Profit), 2) AS total_profit
    FROM sales
    GROUP BY Region
    ORDER BY total_profit DESC
    """
).df()

print(region_profit.to_string(index=False))
region_profit.to_csv(
    OUTPUT_DIR / "sql_region_profit.csv",
    index=False
)


# 4. Monthly revenue
print("\n4. MONTHLY REVENUE")

monthly_revenue = con.execute(
    """
    SELECT
        DATE_TRUNC('month', OrderDate) AS month,
        ROUND(SUM(Revenue), 2) AS total_revenue,
        ROUND(SUM(Profit), 2) AS total_profit
    FROM sales
    GROUP BY DATE_TRUNC('month', OrderDate)
    ORDER BY month
    """
).df()

print(monthly_revenue.head(12).to_string(index=False))
monthly_revenue.to_csv(
    OUTPUT_DIR / "sql_monthly_revenue.csv",
    index=False
)


# 5. Profit margin by category
print("\n5. PROFIT MARGIN BY CATEGORY")

category_margin = con.execute(
    """
    SELECT
        Category,
        ROUND(SUM(Revenue), 2) AS total_revenue,
        ROUND(SUM(Profit), 2) AS total_profit,
        ROUND(
            100.0 * SUM(Profit) / NULLIF(SUM(Revenue), 0),
            2
        ) AS profit_margin_percentage
    FROM sales
    GROUP BY Category
    ORDER BY profit_margin_percentage DESC
    """
).df()

print(category_margin.to_string(index=False))
category_margin.to_csv(
    OUTPUT_DIR / "sql_category_margin.csv",
    index=False
)


# 6. Top 10 customers by revenue
print("\n6. TOP 10 CUSTOMERS")

top_customers = con.execute(
    """
    SELECT
        CustomerID,
        COUNT(*) AS total_orders,
        ROUND(SUM(Revenue), 2) AS total_revenue,
        ROUND(SUM(Profit), 2) AS total_profit
    FROM sales
    GROUP BY CustomerID
    ORDER BY total_revenue DESC
    LIMIT 10
    """
).df()

print(top_customers.to_string(index=False))
top_customers.to_csv(
    OUTPUT_DIR / "sql_top_customers.csv",
    index=False
)


print("\nSQL analysis completed successfully.")
print("Reports saved in:", OUTPUT_DIR)

con.close()