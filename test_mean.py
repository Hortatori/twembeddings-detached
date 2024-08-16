import numpy as np
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)


def update_index(y_pred, previous_y) :
    nb_cluster = previous_y.max()
    logging.info("\nvalue max of cluster from previous batch {}\n\nvalue min from current batch {}".format(previous_y.max(), y_pred.min()))
    y_pred += nb_cluster
    logging.info("\nvalue max of cluster from previous batch {}\n\nvalue min from updated current batch {}".format(previous_y.max(), y_pred.min()))
    # besoin d update uniquement les numéros de clusters pcq les index (=tweets id) vont retomber sur le nombre de tweets après toutes les concats
    return y_pred

"""
input : matrix of embeddings, shape = [tweets,size embedding]
input : linking matrix of tweets - clusters, shape = [,tweets]
turn vectors of a cluster into one averaged vector for this cluster, ready to be use for global clustering
"""
def mean_vectors (X, y_pred) :
    logging.info("starting averaging the vectors...")
    _ndx = np.argsort(y_pred)
    logging.info("ndx :\n {}".format(_ndx))
    _id, _pos, g_count  = np.unique(y_pred[_ndx], 
                                    return_index=True, 
                                    return_counts=True)
    g_sum = np.add.reduceat(X[_ndx], _pos, axis=0) # sum of each element by position for each y index
    g_mean = g_sum / g_count[:,None] # mean for each point
    mean_vectors = np.array([i for i in g_mean])

    return mean_vectors

"""
inputs : 
cluster_y = linking matrix for tweets-daily clusters, 
mean_y = linking matrix for clusters - global clusters 
ouput : linking matrix tweets index - global clusters
"""
def mean_to_tweets(cluster_y, mean_y) :
    logging.info("link_tweets_clusters shape : {}".format(cluster_y.shape)) #95796
    cluster_y-=1 # we substract 1 on all indexes because clusters numeration start at 1 with scipy
    mapping = {key: value for key, value in enumerate(mean_y)}
    logging.info("mapping len: {} in mean_to_tweets function".format(len(mapping)))
    # tw_vs_bigclusters = [[i, mapping[j]] for i, j in enumerate(cluster_y[:-1])]
    tw_vs_bigclusters = [[i, mapping[j]] for i, j in enumerate(cluster_y)]

    logging.info("list of lists before npy : {} shape".format(len(tw_vs_bigclusters)))  #95796 !!
    selected = np.array(tw_vs_bigclusters)
    logging.info("selected shape : {} in mean_to_tweets function".format(selected.shape))
    output = selected[:,1]

    return output

def main_mean(X, y_pred) :
    # INDEX, CONCAT et SAVE
    # retrieve linking matrix between tweets index and clusters id
    try :
        previous_ypred = np.load("ids_tweets_clusters.npy")
        updated_ypred = update_index(y_pred, previous_ypred)
        # concat updated_ypred with previous IDS
        concat_matrices = np.concatenate((previous_ypred, updated_ypred), axis = 0)
        # save IDS
        np.save("ids_tweets_clusters", concat_matrices)
    except FileNotFoundError:
        #first run
        np.save("ids_tweets_clusters",y_pred)


    logging.info("updating and saving daily clusters succeeded...  start averaging vectors for daily clusters...")
    current_mean = mean_vectors (X, y_pred)

    try :
        previous_mean = np.load("mean_by_clusters.npy")
        updated_mean = np.concatenate((previous_mean, current_mean), axis = 0)
        np.save("mean_by_clusters", updated_mean)
    except FileNotFoundError :
        np.save("mean_by_clusters", current_mean)