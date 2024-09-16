from twembeddings import build_matrix, load_dataset
from twembeddings import ClusteringAlgo, ClusteringAlgoSparse
from twembeddings import general_statistics, cluster_event_match, mcminn_eval
import test_mean

from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score
import pandas as pd
import logging
import yaml
import argparse
import csv
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as sp_linkage
import os
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)
text_embeddings = ['tfidf_dataset', 'tfidf_all_tweets', 'w2v_gnews_en', "elmo", "bert", "sbert", "use"]
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--model',
                    nargs='+',
                    required=True,
                    choices=text_embeddings,
                    help="""
                    One or several text embeddings
                    """
                    )          
parser.add_argument('--dataset',
                    required=True,
                    help="""
                    Path to the dataset
                    """
                    )

parser.add_argument('--lang',
                    required=True,
                    choices=["en", "fr"])

parser.add_argument('--annotation',
                    required=False,
                    choices=["examined", "annotated", "no"])

parser.add_argument('--threshold',
                    nargs='+',
                    required=False
                    )

parser.add_argument('--batch_size',
                    required=False,
                    type=int
                    )

parser.add_argument('--remove_mentions',
                    action='store_true'
                    )

parser.add_argument('--window',
                    required=False,
                    default=24,
                    type=int
                    )
parser.add_argument('--sub-model',
                    required=False,
                    type=str
                    )
# ADDED clustering type
parser.add_argument('--clustering', 
                    required=True, 
                    choices=["FSD", "agglomerative", "DBSCAN", "spy_fcluster"], 
                    help="""
                    A clustering algorithm :
                    """
                    )
parser.add_argument('--daily',
                    required = False
                    )
# ADDED : only used in the script, not in commands for now
parser.add_argument("--global_clustering",
                    required=False) 

def main(args):
    with open("options.yaml", "r") as f:
        options = yaml.safe_load(f)
    for model in args["model"]:
        # load standard parameters
        params = options["standard"]
        logging.info("Clustering with {} model".format(model))
        if model in options:
            # change standard parameters for this specific model
            for opt in options[model]:
                params[opt] = options[model][opt]
        for arg in args:
            if args[arg] is not None:
                # params from command line overwrite options.yaml file
                params[arg] = args[arg]
        params["daily"] = True

        logging.info("dataset parameter is {}".format(params["dataset"]))
        params["model"] = model

        # delete daily saves from last run
        if os.path.exists("ids_tweets_clusters.npy") :
            os.remove("ids_tweets_clusters.npy")
        if os.path.exists("mean_by_clusters.npy") :
            os.remove("mean_by_clusters.npy")
        # apply clustering() to each file day
        thresholds = params.pop("threshold")
        params["global_clustering"] = True

        for t in thresholds:
            for filename in os.listdir("data/dailytweets_event2018/"):
                file_path = os.path.join("data/dailytweets_event2018/", filename)
                if os.path.isfile(file_path) and (filename.endswith("results.tsv") or filename.endswith("results_daily.tsv")):
                    os.remove(file_path)
                else :
                    params["dataset"] = file_path
                    params["threshold"] = t
                    test_params(**params)
            params["global_clustering"] = True
            global_X = np.load("mean_by_clusters.npy")
            logging.info("globalX shape : {}".format(global_X.shape))
            data = load_dataset("data/event2018.tsv", params["annotation"], params["text+"])
            logging.info("shape data : {}".format(data.shape))
            # params["clustering"] = "FSD"
            y_pred = clustering(global_X, data, t, **params)
            # if params["clustering"] == "spy_fcluster" :
            #     linking_matrix = sp_linkage(global_X, method = "average", metric = "cosine")
            #     y_pred = fcluster(linking_matrix, t, criterion='distance')
            stats = general_statistics(y_pred)
            logging.info("high level clustering succeded. stats : {}".format(stats))
            tweets_ids = np.load("ids_tweets_clusters.npy")
            matching = test_mean.mean_to_tweets(tweets_ids, y_pred)
            final_pred = np.array(matching)
            logging.info("post-matching btwn tweets & global cluster, shape of final pred : {}".format(final_pred.shape))
            logging.info("testing event match")
            p, r, f1 = cluster_event_match(data, final_pred)
            ami = adjusted_mutual_info_score(data.label, final_pred)
            ari = adjusted_rand_score(data.label, final_pred)
            logging.info("t {} p {} r {} f1 {} ami {} ari {} ".format(t, p, r, f1, ami, ari))


def clustering(X, data, t, **params) :
    logging.info("X shape is {}".format(X.shape))
    params["window"] = int(data.groupby("date").size().mean()*params["window"]/24// params["batch_size"] * params["batch_size"])
    logging.info("window size: {}".format(params["window"]))
    params["distance"] = "cosine"

    if params["model"].startswith("tfidf") and params["distance"] == "cosine":
        clustering = ClusteringAlgoSparse(threshold=float(t), window_size=params["window"],
                                            batch_size=params["batch_size"], intel_mkl=False)
        clustering.add_vectors(X)
    if params["clustering"] == "FSD" :# and params["global_clustering"] == False:
        clustering = ClusteringAlgo(threshold=float(t), window_size=params["window"],
                                    batch_size=params["batch_size"],
                                    distance=params["distance"])
        clustering.add_vectors(X)
        y_pred = clustering.incremental_clustering()
    # else:
    logging.info("testing other clustering than FSD : {}".format(params["clustering"]))
    if params["clustering"] == "spy_fcluster" :
        linking_matrix = sp_linkage(X, method = "average", metric = "cosine")
        y_pred = fcluster(linking_matrix, t, criterion='distance')
    if params["clustering"] == "DBSCAN":
        clustering = DBSCAN(eps=t, metric=params["distance"], min_samples=5).fit(X)        
        y_pred = clustering.labels_
    if params["clustering"] == "agglomerative":
        clustering = AgglomerativeClustering(n_clusters=None, metric= "cosine", linkage = 'average', distance_threshold = t).fit(X)
        y_pred = clustering.labels_
    logging.info("successed to test clustering {}".format(params["clustering"]))
    return y_pred


def test_params(**params):
    X, data = build_matrix(**params)
    logging.info("X shape : {}".format(X.shape))
    t = params.pop("threshold")

    y_pred = clustering(X, data, t, **params)

    stats = general_statistics(y_pred)
    p, r, f1 = cluster_event_match(data, y_pred)
    ami = adjusted_mutual_info_score(data.label, y_pred)
    ari = adjusted_rand_score(data.label, y_pred)
    data["pred"] = data["pred"].astype(int)
    data["id"] = data["id"].astype(int)
    candidate_columns = ["date", "time", "label", "pred", "user_id_str", "id"]
    result_columns = []
    for rc in candidate_columns:
        if rc in data.columns:
            result_columns.append(rc)

    # ADDED sending results/days in a dedicated file 
    # not usefull to save results for each days now? We dont need a mean on all daily results
    # data[result_columns].to_csv(params["dataset"].replace(".", "_results_daily."),
    #                             index=False,
    #                             sep="\t",
    #                             quoting=csv.QUOTE_NONE
    #                             )
    try:
        mcp, mcr, mcf1 = mcminn_eval(data, y_pred)
    except ZeroDivisionError as error:
        logging.error(error)
    stats.update({"t": t, "p": p, "r": r, "f1": f1, "mcp": mcp, "mcr": mcr, "mcf1": mcf1, "ami": ami, "ari": ari})
    stats.update(params)
    stats = pd.DataFrame(stats, index=[0])
    stats['datetime_of_run'] = pd.Timestamp.today().strftime('%Y-%m-%d-%H-%M')
    logging.info(stats[["t", "model", "tfidf_weights", "p", "r", "f1", "ami", "ari"]].iloc[0])
    # if params["save_results"]:
    #     # ADDED update a scores/day file with new daily stats
    #     try:
    #         results = pd.read_csv("results_daily_cluster_tests.csv")
    #     except FileNotFoundError:
    #         results = pd.DataFrame()
    #     stats = pd.concat([results, stats], ignore_index=True)
    #     stats.to_csv("results_daily_cluster_tests.csv", index=False)
    #     logging.info("Saved results to results_daily_cluster_tests.csv")

    test_mean.main_mean(X, y_pred)
    # # save results in a dedicated file for global dataset results
    # try:
    #     results = pd.read_csv("days_results_clustering.csv")
    # except FileNotFoundError:
    #     results = pd.DataFrame()
    # stats = pd.concat([results, stats], ignore_index=True)
    # stats.to_csv("days_results_clustering.csv", index=False)
    # logging.info("Saved results to days_results_clustering.csv")

if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(args)