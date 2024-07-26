import numpy as np
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)


def update_index(y_pred, previous_y) :
    logging.info("starting UPDATE_INDEX")
    nb_cluster = previous_y.max()
    logging.info("index max of cluster from previous batch {}\n \ t head from previous batch {}\n \t max index from current batch {}\n \t head from current bacth{}".format(previous_y.max(), previous_y[:10], y_pred.max(), y_pred[:10]))

    y_pred += nb_cluster
    logging.info("updated y pred shape : {}\nupdated y pred head : \n{}\n ENDED update infex".format(y_pred.shape,y_pred[:10]))
    # besoin d update uniquement les numéros de clusters pcq les index (=tweets id) vont retomber sur le nombre de tweets après toutes les concats
    return y_pred



# selectionner tous les id de tweets pour un cluster
# selectionner tous les vecteurs embedded corrspondant aux ids de tweets
# faire la moyenne de chaque groupe de vecteur embedded 

# dans param_tets dans clustering.py
# turn vectors of a cluster into one averaged vetcor for this cluster, ready to be use for clustering
def mean_vectors (X, y_pred) :
    logging.info("\nstarting MEAN VECTORS\ncurrent matrix X shape : {}\ncurrent prediction y shape {}".format(X.shape, y_pred.shape))

    _ndx = np.argsort(y_pred)
    logging.info("ndx :\n {}".format(_ndx))
    _id, _pos, g_count  = np.unique(y_pred[_ndx], 
                                    return_index=True, 
                                    return_counts=True)
    logging.info("\n_id, list of each label \n{}\n _pos pos of each label on the list \n{}\n g_count nb of each label \n{}\n ".format(_id,_pos,g_count))

    g_sum = np.add.reduceat(X[_ndx], _pos, axis=0) # sum of each element by position for each y index
    g_mean = g_sum / g_count[:,None] # mean for each point

    mean_vectors = np.array([i for i in g_mean])
    logging.info("\nmean_vectors output shape is {}, \nend of MEAN_VECTORS function".format(mean_vectors.shape))

    return mean_vectors

# input : 
# cluster_y = linking matrix for tweets-clusters, 
# mean_y = linking matrix for clusters-mean clusters 
# return linking matrix tweets index - mean clusters
def mean_to_tweets(cluster_y, mean_y) :
    # tweet-cluster y pred
    logging.info("cluster y, number of predicted clusters {}".format(cluster_y.shape))

    # cluster-mean y pred
    logging.info("mean_y {}".format(mean_y))

    mapping = {key: value for key, value in mean_y}
    logging.info("mapping {}".format(mapping))

    output = np.array([[i, mapping[j]] for i, j in cluster_y])
    logging.info("output {}".format(output))

    return output

def main_mean(X, y_pred) :
    # INDEX, CONCAT et SAVE
    logging.info("\nstarting MAIN MEAN. \ncurrent y_pred shape : {}".format(y_pred.shape))
    logging.info("\n STARTING INDEX matrix\ncurrent y_pred head \n{}".format(y_pred[:10]))
    # retrieve previous IDS
    try :
        previous_ypred = np.load("ids_tweets_clusters.npy")
        logging.info("\n-previous y pred shape : {}\n-head of previous y pred\n{}".format(previous_ypred.shape, previous_ypred[:10]))
        updated_ypred = update_index(y_pred, previous_ypred)
        logging.info("\nypred shape with new indexes {}\ny_pred with new indexes head :\n{}".format(updated_ypred.shape,updated_ypred[:10]))
    # concat updated_ypred with previous IDS
        concat_matrices = np.concatenate((previous_ypred, updated_ypred), axis = 0)
    # save IDS
        logging.info("\n-shape of concatenated matrix {}\n-head of concatenated matrix : \n{}\n-saving concatenated matrix here".format(concat_matrices.shape, concat_matrices[:10]))
        np.save("ids_tweets_clusters", concat_matrices)
    except FileNotFoundError:
        #first pred of IDS
        logging.info("SAVING matrix of ids cluster-tweets for FIRST TIME")
        np.save("ids_tweets_clusters",y_pred)


    logging.info("\nENDED INDEX tweets updates\nCOMPUTING MEAN VECTORS")
    current_mean = mean_vectors (X, y_pred)
    logging.info("current_mean shape : {}".format(current_mean.shape))
    logging.info("current_mean values :\n{}".format(current_mean[:10]))
    try :
        previous_mean = np.load("mean_by_clusters.npy")
        logging.info("\nloaded previous mean vectors, shape is :".format(previous_mean.shape))
        updated_mean = np.concatenate((previous_mean, current_mean), axis = 0)
        logging.info("concatenated mean vectors shape : {}".format(type(updated_mean), updated_mean.shape))
        logging.info("concatenated mean vectors values : \n{}".format(updated_mean[:10]))
        logging.info("SAVING concatenated mean vectors")

        np.save("mean_by_clusters", updated_mean)



    except FileNotFoundError :
        logging.info("SAVING mean vectors FOR FIRST TIME")
        np.save("mean_by_clusters", current_mean)

# test
# print(mean_vectors(X = np.array([[0,2,3],[1002,5000,3000],[8,6,9],[59, 88, 22]]), y_pred = np.array([10,11,11,12])))
# print()
# cluster_y = [[0,10],[1,11],[2,11]]
# mean_y = [[10, 99], [11, 123]]
# mean_to_tweets(cluster_y, mean_y)