import os
import subprocess
import pandas as pd
import logging
""""
to note, for this code, 
I have modified clustering.py for the name of the csv file to save
I have modified clustering.py on for adding agglomerative_clustering directly
This code doesnt support to have multiple combination of options in the yaml file (not more than one threshold for ex.)
"""
logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)

def execute_command_on_files(directory):
    if not os.path.isdir(directory):
        print(f"Le répertoire {directory} n'existe pas.")
        return
    # Parcourt tous les fichiers du répertoire
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path) and not filename.endswith("results.tsv"):
            # Crée la commande complète en ajoutant le chemin du fichier
            full_command = f"python clustering.py --dataset {file_path} --lang fr --model sbert"
            print(full_command)
            try:
            # Exécute la commande
                result = subprocess.run(full_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"Commande exécutée avec succès sur {file_path} :\n{result.stdout.decode()}")
            except subprocess.CalledProcessError as e:
                print(f"Erreur lors de l'exécution de la commande sur {file_path} :\n{e.stderr.decode()}")


directory = "data/dailytweets_event2018/"
execute_command_on_files(directory)

try:
    results = pd.read_csv("results_daily_aggloclusters.csv")
    temp = results.mean(numeric_only=True)
    output = results.iloc[[0]].copy()
    output.loc[0,"count"] = int(temp["count"])
    output.loc[0,"mean":"max"] = temp["mean":"max"]
    output.loc[0,"p":"mcf1"] = temp["p":"mcf1"]
    output.loc[0,"dataset"] = "/".join(output.loc[0, "dataset"].split("/")[:-1])
    output.loc[0,"model"] = output.loc[0, "model"] + "_AggloC"
    print(output)

    try:
        old_results = pd.read_csv("results_clustering.csv")
    except FileNotFoundError:
        old_results = pd.DataFrame()
    stats = pd.concat([old_results, output], ignore_index=True)
    stats.to_csv("results_clustering.csv", index=False)

except FileNotFoundError:
    print("pas de fichier results_daily_aggloclusters.csv trouvé")
