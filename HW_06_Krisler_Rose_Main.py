# main function for agglomeration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    filename = "HW_CLUSTERING_SHOPPING_CART_v2245a.csv"
    df = pd.read_csv(filename)

    print(df.head())