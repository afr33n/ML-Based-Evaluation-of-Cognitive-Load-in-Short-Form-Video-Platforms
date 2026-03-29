import pandas as pd
from pathlib import Path


CLI_1FPS = Path("outputs/features1.csv")
CLI_3FPS = Path("outputs/features.csv")


def main():

    df1 = pd.read_csv(CLI_1FPS)
    df3 = pd.read_csv(CLI_3FPS)

    print("\n--- First 5 rows of features1.csv ---")
    print(df1.head())

    print("\n--- First 5 rows of features.csv ---")
    print(df3.head())


if __name__ == "__main__":
    main()
