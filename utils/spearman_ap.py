import pandas as pd
from scipy.stats import spearmanr
import argparse

#python spearman_ap.py file1.csv file2.csv --keys layer unit


def compute_spearman(
    file1: str,
    file2: str,
    value_col: str = "ap",
    key_cols: list[str] | None = None,
) -> None:
    """
    Compute Spearman correlation between `value_col` in two CSV files.

    If key_cols is provided, rows are aligned by those columns before correlation.
    Otherwise, correlation is computed by row order.
    """

    if key_cols is None:
        # Fast path: assume same row order
        df1 = pd.read_csv(file1, usecols=[value_col])
        df2 = pd.read_csv(file2, usecols=[value_col])

        if len(df1) != len(df2):
            raise ValueError(
                f"Row count mismatch: {file1} has {len(df1)} rows, "
                f"but {file2} has {len(df2)} rows. "
                "Use key-based alignment with --keys."
            )

        x = df1[value_col].to_numpy()
        y = df2[value_col].to_numpy()

    else:
        # Safer path: align by key columns
        cols1 = key_cols + [value_col]
        cols2 = key_cols + [value_col]

        df1 = pd.read_csv(file1, usecols=cols1)
        df2 = pd.read_csv(file2, usecols=cols2)

        df1 = df1.rename(columns={value_col: f"{value_col}_1"})
        df2 = df2.rename(columns={value_col: f"{value_col}_2"})

        merged = df1.merge(df2, on=key_cols, how="inner")

        if merged.empty:
            raise ValueError("No matching rows found after merge.")

        x = merged[f"{value_col}_1"].to_numpy()
        y = merged[f"{value_col}_2"].to_numpy()

        print(f"Merged rows: {len(merged):,}")

    corr, pval = spearmanr(x, y)

    print(f"Spearman correlation ({value_col}): {corr:.8f}")
    print(f"P-value: {pval:.8e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file1", help="First CSV file")
    parser.add_argument("file2", help="Second CSV file")
    parser.add_argument(
        "--value-col",
        default="ap",
        help="Column to correlate (default: ap)",
    )
    parser.add_argument(
        "--keys",
        nargs="*",
        default=None,
        help="Key columns for alignment, e.g. --keys uuid or --keys layer unit",
    )

    args = parser.parse_args()

    compute_spearman(
        file1=args.file1,
        file2=args.file2,
        value_col=args.value_col,
        key_cols=args.keys,
    )