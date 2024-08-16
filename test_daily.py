import os
import subprocess
import pandas as pd
import logging
""""
to note, for this code, 
STEP 1 : you have to change daily from False to TRUE in clustering.py
STEP 2 : you have to change the value of cluster_name_daily for the name of the clustering algo you are testing

This code doesnt support to have multiple combination of options in the yaml file (not more than one threshold for ex.)
"""
cluster_name_daily = "fastcluster"

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)
def execute_command_on_files(directory):
    # delete daily results from last run
    if os.path.exists("results_daily_cluster_tests.csv") :
        os.remove("results_daily_cluster_tests.csv")

    if not os.path.isdir(directory):
        print(f"Le répertoire {directory} n'existe pas.")
        return
    # through all files of directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path) and not (filename.endswith("results.tsv") or filename.endswith("results_daily.tsv")):
            # create command with file path
            full_command = f"python clustering.py --dataset {file_path} --lang fr --model sbert --clustering {cluster_name_daily}"
            print(full_command)
            try:
                # Exécute la commande avec Popen pour capturer les sorties en temps réel
                process = subprocess.Popen(full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                # Lire les sorties en temps réel
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        print(output.strip())

                stderr_output = process.stderr.read()
                if stderr_output:
                    print("Erreur :", stderr_output.strip())

                rc = process.poll()
                print(f"Code de retour : {rc}")

            except Exception as e:
                print(f"Erreur lors de l'exécution de la commande sur {file_path} : {str(e)}")
                break

directory = "data/dailytweets_event2018/"
execute_command_on_files(directory)

try:
    results = pd.read_csv("results_daily_cluster_tests.csv")
    temp = results.mean(numeric_only=True)
    output = results.iloc[[0]].copy()
    output.loc[0,"count"] = int(temp["count"])
    output.loc[0,"mean":"max"] = temp["mean":"max"]
    output.loc[0,"p":"mcf1"] = temp["p":"mcf1"]
    output.loc[0,"dataset"] = "/".join(output.loc[0, "dataset"].split("/")[:-1])
    output.loc[0,"model"] = output.loc[0, "model"] + "_" + "daily"
    output.loc[0,"clustering"] = cluster_name_daily

    try:
        old_results = pd.read_csv("results_clustering.csv")
    except FileNotFoundError:
        old_results = pd.DataFrame()
    stats = pd.concat([old_results, output], ignore_index=True)
    stats.to_csv("results_clustering.csv", index=False)

except FileNotFoundError:
    print("pas de fichier results_daily_cluster_tests.csv trouvé")
