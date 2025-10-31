# main function for agglomeration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

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


# this function just answers all the questions that are part of the reporting section of Part 2
def part2(cross_corrdf, strong_corrs):
    # part a: which two categories are most strongly correlated
    max_corr = -1000  # set to impossibly small
    max_pair = ("", "")
    for feature, col in cross_corrdf.items():
        for label, val in col.items():
            abs_val = abs(val)

            if abs_val != 1 and abs_val > max_corr:
                max_corr = abs_val
                max_pair = (feature, label)
    print(f'Part a: The most strongly correlated pair is {max_pair} at {max_corr}\n\n')
    
    # finding what customer is most likely to buy if they buy lots of manga books
    # find the strongest absolute value correlation to manga books
    manga_corr = strong_corrs['Manga']
    # find correlation between Manga and Horror, as well as Gifts
    horror_corr = cross_corrdf.loc['Manga', 'Horror']
    gifts_corr = cross_corrdf.loc['Manga', 'Gifts']
    print(f'Part b: Manga strongest corr is with {manga_corr[0]} at {manga_corr[1]}\n')
    print(f'Part b: Manga and Horror corr is {horror_corr}, Manga and Gifts corr is {gifts_corr}\n')
    # how likely are they to buy thrillers
    thriller_corr = cross_corrdf.loc['Manga', 'Thrillers']
    print(f'Part b: Manga and Thriller corr is {thriller_corr}\n\n')

    # what other category is fiction most strongly correlated with
    fiction_corr = strong_corrs['Fiction']
    print(f'Part c: Fiction strongest corr is with {fiction_corr[0]} at {fiction_corr[1]}\n\n')

    # what other category is self improvement most strongly correlated with
    selfimp_corr = strong_corrs['SelfImprov']
    print(f'Part d: Self Improvement strongest corr is with {selfimp_corr[0]} at {selfimp_corr[1]}')

    # what are the the top 3 strongest correlations with cookbooks
    cook_corrs = cross_corrdf['Cooking'].abs().sort_values(ascending=False)
    top3_cook = cook_corrs[1:4]
    bottom3_cook = cook_corrs[-3:]
    print(f'Part e: Top 3 strongest correlations with Cookbookss are:\n{top3_cook}\n')
    print(f'Part e: Bottom 3 strongest correlations with Cookbookss are:\n{bottom3_cook}\n\n')

    # what are the top 3 strongest correlations with classic novels
    classic_corrs = cross_corrdf['Classics'].abs().sort_values(ascending=False)
    top3_classic = classic_corrs[1:4]
    bottom3_classic = classic_corrs[-3:]
    print(f'Part f: Top 3 strongest correlations with Classic Novels are:\n{top3_classic}\n')
    print(f'Part f: Bottom 3 strongest correlations with Classic Novels are:\n{bottom3_classic}\n\n')

    # what are the top 3 strongest correlations with NEWS
    news_corrs = cross_corrdf['News'].abs().sort_values(ascending=False)
    top3_news = news_corrs[1:4]
    bottom3_news = news_corrs[-3:]
    print(f'Part g: Top 3 strongest correlations with News are:{top3_news}\n')
    print(f'Part g: Bottom 3 strongest correlations with News are:{bottom3_news}\n\n')

    # what are the top 3 strongest correlations with HairyPottery
    hp_corrs = cross_corrdf['HairyPottery'].abs().sort_values(ascending=False)
    top3_hp = hp_corrs[1:4]
    bottom3_hp = hp_corrs[-3:]
    print(f'Part h: Top 3 strongest correlations with HairyPottery are:{top3_hp}\n')
    print(f'Part h: Bottom 3 strongest correlations with HairyPottery are:{bottom3_hp}\n\n')

    # what are the top 3 and bottom 3 strongest correlations with Thrillers
    thriller_corrs = cross_corrdf['Thrillers'].abs().sort_values(ascending=False)
    top3_thriller = thriller_corrs[1:4]
    bottom3_thriller = thriller_corrs[-3:]
    print(f'Part i: Top 3 strongest correlations with Thrillers are:{top3_thriller}\n')
    print(f'Part i: Bottom 3 strongest correlations with Thrillers are:{bottom3_thriller}\n\n')

    # what are the top 3 and bottom 3 strongest correlations with Art&Hist
    arthist_corrs = cross_corrdf['Art&Hist'].abs().sort_values(ascending=False)
    top3_arthist = arthist_corrs[1:4]
    bottom3_arthist = arthist_corrs[-3:]
    print(f'Part j: Top 3 strongest correlations with Art&Hist are:{top3_arthist}\n')
    print(f'Part j: Bottom 3 strongest correlations with Art&Hist are:{bottom3_arthist}\n\n')

    # find the three attributes with the lowest overall correlation to other attributes. We
    # want to find which three attributes could possibly be removed from the dataset with minimal loss of information.
    avg_corrs = cross_corrdf.abs().mean().sort_values(ascending=True)
    lowest3_avg = avg_corrs[0:3]
    print(f'Question 3: The three attributes with the lowest overall correlation to other attributes are:\n{lowest3_avg}\n\n')


# helper function for finding the euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def agglomerate(data_df):
    """perform agglomerative clustering using center linkage method"""
    # convert the data to numpy arrays for easier processing
    data_array = data_df.to_numpy()
    num_records = len(data_array)
    print(f"Starting agglomeration with {num_records} records.\n")

    # initialize each record as its own cluster
    # we create a dictionary to use as a data structure for clusters
    # cluster_members: key=cluster_id, value=list of record indices in that cluster
    cluster_members = {i: [i] for i in range(num_records)}

    # using dictionary again to use as a data structure for cluster centers
    # cluster_centers: key=cluster_id, value=center coordinates
    cluster_centers = {i: data_array[i].copy() for i in range(num_records)}

    # creating the set of active clusters (those that havent been merged)
    active_clusters = set(range(num_records))

    # list to track the merge history for the last 20 merges
    merge_history = []

    # iteration counter to be used in debugging print statements
    iteration = 0

    # continue merging until only one cluster remains
    while len(active_clusters) > 1:
        iteration += 1
        # every 50th iteration, print status
        if iteration % 50 == 0:
            print(f"Iteration {iteration}: {len(active_clusters)} clusters remaining.")
        
        #find the two closest clusters
        min_distance = float('inf')
        closest_pair = (None, None) # (cluster_id1, cluster_id2)
        
        # converting active_clusters to a list for indexing
        active_list = list(active_clusters)

        # only computer the upper triangle of the distance matrix so you don't double compute distances
        for i in range(len(active_list)):
            for j in range(i + 1, len(active_list)): # by adding 1 to the index, we skip calculating bottom triangle
                cluster_i = active_list[i]
                cluster_j = active_list[j]

                # calculating the distance between cluster centers
                dist = euclidean_distance(cluster_centers[cluster_i], cluster_centers[cluster_j])
                if dist < min_distance:
                    min_distance = dist
                    closest_pair = (cluster_i, cluster_j)
        
        # we merge the two clusters that are the closest
        cluster_1, cluster_2 = closest_pair

        # keep the smaller ID as the ID of the newly merged cluster
        keep_id = min(cluster_1, cluster_2)
        remove_id = max(cluster_1, cluster_2)

        # here we track the size of the smallest cluster being merged
        size_cluster_1 = len(cluster_members[cluster_1])
        size_cluster_2 = len(cluster_members[cluster_2])
        smallest_cluster_size = min(size_cluster_1, size_cluster_2)

        # record the merge info into the merge history using dict for structure
        merge_history.append({
            'iteration': iteration,
            'cluster_1': cluster_1,
            'cluster_2': cluster_2,
            'size_1': size_cluster_1,
            'size_2': size_cluster_2,
            'smallest_size': smallest_cluster_size,
            'distance': min_distance
        })

        # merge the members
        cluster_members[keep_id].extend(cluster_members[remove_id])
        # compute the new center of mass (avg of all points in merged cluster)
        merged_indices = cluster_members[keep_id]
        cluster_centers[keep_id] = np.mean(data_array[merged_indices], axis=0)

        # removed the merged cluster
        del cluster_members[remove_id]
        del cluster_centers[remove_id]
        active_clusters.remove(remove_id)

    # print status update
    print(f"Agglomeration complete after {iteration} iterations.\n")

    # getting the last 20 merges for analysis
    last_20_merges = merge_history[-20:] if len(merge_history) >= 20 else merge_history
    print("Last 20 merges (or fewer if less than 20 total merges):")
    for merge in last_20_merges:
        print(f"Iteration {merge['iteration']}: "
              f"Merged clusters {merge['cluster_1']} (size {merge['size_1']}) "
              f"and {merge['cluster_2']} (size {merge['size_2']}) - "
              f"smallest size = {merge['smallest_size']}")

    return cluster_members, merge_history, cluster_centers


def graph_dendrogram(merge_history, num_samples):
    # create a linkage mtx array to track the agglomeration process for dendrogram graphing
    linkage_mtx = []
    # get the number of clusters for dendrogram tracking
    cur_cluster = num_samples

    # because the agglomeration deletes and reuses idx names, ids are not unique
    # dendrograms need unique cluster ids to graph, so we must create a dict
    # to hold unique cluster id values
    id_map = {i: i for i in range(num_samples)}

    # iterate through all the merges tracked during agglomeration
    for merge in merge_history:
        # get the tracked clusters that were merged
        c1 = merge['cluster_1']
        c2 = merge['cluster_2']
        # get this distance btwn clusters
        distance = merge['distance']
        # create the new cluster size by adding c1 and c2 sizes
        new_size = merge['size_1'] + merge['size_2']

        # get the unique ids for each cluster from the dict created
        c1_uniqueID = id_map[c1]
        c2_uniqueID = id_map[c2]

        # append all values to the linkage mtx for dendrogram plotting with scikit-learn
        linkage_mtx.append([c1_uniqueID, c2_uniqueID, distance, new_size])

        # assign new cluster id to the first cluster merged together
        id_map[c1] = cur_cluster

        # remove the second merged cluster from the unique id dict so it is not reused
        if c2 in id_map:
            del id_map[c2]

        # increment to the next cluster
        cur_cluster += 1

    # convert the array to a numpy array
    linkage_mtx = np.array(linkage_mtx)

    # plot the denogram using scikit-learn
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_mtx)
    plt.title("Shopping Cart Dendrogram")
    plt.show()




if __name__ == "__main__":
    filename = "TEST_DATA.csv"
    df_withID = pd.read_csv(filename)
    df = pd.read_csv(filename)
    df = df.rename(columns=lambda x: x.strip())  # remove leading/trailing spaces in col names
    # remove ID col from df for cross correlation
    del df['ID']
    # use pandas cross correlation library for N*N mtx
    cross_corrdf = df.corr()
    strong_corrs = find_strong_corr(cross_corrdf)
    fetch_distmtx(cross_corrdf)
    
    # this portion only necessary for the report section of part 2. hence, commented out 
    # since all the values have been recorded already in the report
    # part2(cross_corrdf, strong_corrs)

    # perform agglomerative clustering
    cluster_members, merge_history, cluster_centers = agglomerate(df)

    num_samples = df.shape[0] # pass in the number of rows not columns
    # columns  = # of features not number of samples
    graph_dendrogram(merge_history, num_samples)