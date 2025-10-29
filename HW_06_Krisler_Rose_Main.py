# main function for agglomeration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_strong_corr(cross_corrdf):
    strongest_corr = {}

    for feature, col in cross_corrdf.items():
        best_corr = -1000 # set to impossibly small
        best_feature = ""
        for label, val in col.items():
            abs_val = abs(val)

            if abs_val != 1 and abs_val > best_corr:
                best_corr = abs_val
                best_feature = label

        strongest_corr[feature] = [best_feature, best_corr]

    return strongest_corr


def fetch_distmtx(cross_corrdf):
    dist_mtx = cross_corrdf.copy(deep=True)
    for row_idx, row in cross_corrdf.iterrows():
        for col_label, val in row.items():
            # use pearsons dist to convert from cross corr to distance
            new_dist = 1 - val
            # add value to new mtx
            dist_mtx.loc[row_idx, col_label] = new_dist

    return dist_mtx

# TODO: find irrelevant attributes --> i dont think we need to code for this?
# TODO: implement agglomerative clustering
    # cross correlation provies a similarity between data points for
    # agglmeration

    # the datapoints with smallest CC can be merged together first

# def agglomerate(df, cross_corrdf, strong_corrs):


if __name__ == "__main__":
    filename = "HW_CLUSTERING_SHOPPING_CART_v2245a.csv"
    df_withID = pd.read_csv(filename)
    df = pd.read_csv(filename)
    # remove ID col from df for cross correlation
    del df['ID']
    # use pandas cross correlation library for N*N mtx
    cross_corrdf = df.corr()
    strong_corrs = find_strong_corr(cross_corrdf)
    fetch_distmtx(cross_corrdf)