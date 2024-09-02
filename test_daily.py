from twembeddings import build_matrix
from twembeddings import ClusteringAlgo, ClusteringAlgoSparse
from twembeddings import general_statistics, cluster_event_match, mcminn_eval
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
parser.add_argument('--daily',
                    required = False)
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
# ADDED : only used in the script, not in commands for now
parser.add_argument("--global_clustering",
                    required=False) 
parser.add_argument("--folder_name",
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

        # apply clustering() to each day
        thresholds = params.pop("threshold")
        params["global_clustering"] = True
        params['folder_name'] = params["dataset"]
        for t in thresholds:
            logging.info(f"{params['folder_name']}_daily_results.csv")
            if os.path.exists(f"{params['folder_name']}_daily_results.csv"):
                os.remove(f"{params['folder_name']}_daily_results.csv")

            for filename in os.listdir(params["folder_name"]):
                file_path = os.path.join(params["folder_name"], filename)
                if os.path.isfile(file_path) and (filename.endswith("results.tsv") or filename.endswith("results_daily.tsv")):
                    os.remove(file_path)
                else :
                    params["dataset"] = file_path
                    logging.info(params["dataset"])
                    params["threshold"] = t
                    test_params(**params)
            if params["save_results"]:
                try:
                    results = pd.read_csv(f"{params['folder_name']}_daily_results.csv")
                    temp = results.mean(numeric_only=True)
                    output = results.iloc[[0]].copy()
                    output.loc[0,"count"] = int(temp["count"])
                    output.loc[0,"mean":"max"] = temp["mean":"max"]
                    output.loc[0,"p":"mcf1"] = temp["p":"mcf1"]
                    output.loc[0,"datetime_of_run"] = pd.Timestamp.today().strftime("%Y-%m-%d-%H-%M")
                    logging.info("mean results on all dataset:\n{}".format(output))

                    try:
                        old_results = pd.read_csv("results_clustering.csv")
                    except FileNotFoundError:
                        logging.info("no file results_clustering.csv found")
                    stats = pd.concat([old_results, output], ignore_index=True)
                    stats.to_csv("results_clustering.csv", index=False)
                    logging.info("mean results successfully saved in results_clustering.csv")

                except FileNotFoundError:
                    logging.info(f"no file {params['folder_name']}_daily_results.csv found")


def clustering(X, data, t, **params) :
    logging.info("X shape is {}".format(X.shape))
    params["window"] = int(data.groupby("date").size().mean()*params["window"]/24// params["batch_size"] * params["batch_size"])
    logging.info("window size: {}".format(params["window"]))
    params["distance"] = "cosine"

    if params["model"].startswith("tfidf") and params["distance"] == "cosine":
        clustering = ClusteringAlgoSparse(threshold=float(t), window_size=params["window"],
                                            batch_size=params["batch_size"], intel_mkl=False)
        clustering.add_vectors(X)
    if params["clustering"] == "FSD" :
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
    if X.shape[0] < 2 :
        logging.info("this day has only one tweet, skipping it")
        return
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

    # sending results/days in a dedicated file 
    data[result_columns].to_csv(f"{params['folder_name']}_daily_results",
                                index=False,
                                sep="\t",
                                quoting=csv.QUOTE_NONE
                                )
    try:
        mcp, mcr, mcf1 = mcminn_eval(data, y_pred)
    except ZeroDivisionError as error:
        logging.error(error)
        return
    stats.update({"t": t, "p": p, "r": r, "f1": f1, "mcp": mcp, "mcr": mcr, "mcf1": mcf1, "ami": ami, "ari": ari})
    stats.update(params)
    stats = pd.DataFrame(stats, index=[0])
    stats.drop(columns=["folder_name","global_clustering"], axis = 1, inplace = True)
    logging.info(stats[["t", "model", "tfidf_weights", "p", "r", "f1", "ami", "ari"]].iloc[0])
    # ADDED update a scores/day file with new daily stats
    try:
        results = pd.read_csv(f"{params['folder_name']}_daily_results.csv")
    except FileNotFoundError:
        results = pd.DataFrame()
    stats = pd.concat([results, stats], ignore_index=True)
    stats.to_csv(f"{params['folder_name']}_daily_results.csv", index=False)
    logging.info("Saved results to results_daily_cluster_tests.csv")


if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(args)
