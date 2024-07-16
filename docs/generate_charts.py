import csv
from cycler import cycler
import matplotlib.pyplot as plt
from collections import defaultdict
styling = ( #cycler('linestyle', ['-', '--', ':', '-.', '-', '--', ":"]) +
               cycler('marker', ['o', '+', '^', '.', "h", "v", "<",'o','o']) +
               cycler('color', ['c', 'm', 'darkorange', 'k', "b", "seagreen", "darkred", "teal", "goldenrod"])
)

def plot_chart(ax, results, title):
    ax.set_prop_cycle(styling)
    ax.set_ylim([0.1, 0.8])
    ax.grid()
    ax.title.set_text(title)
    for model, points in sorted(results.items()):
        ax.plot(points[0], points[1], label=model)
        ax.legend()

def append_row(row, results):
    if row["sub_model"]:
        model = row["model"] + " " + row["sub_model"]
    else:
        model = row["model"]
    results[model][0].append(float(row["t"]))
    results[model][1].append(float(row["f1"]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))

en_results, fr_results = defaultdict(lambda: ([], [])), defaultdict(lambda: ([], []))
with open("/data/mray/medialex/twembeddings/results_clustering.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["lang"] == "en":
            append_row(row, en_results)
        if row["lang"] == "fr":
            append_row(row, fr_results)

plot_chart(ax1, en_results, "Event2012 (English corpus)")
plot_chart(ax2, fr_results, "Event2018 (French corpus from B.Mazoyer)")
# plt.xlabel("t")
# plt.ylabel("f1 score")
# plt.savefig("/data/mray/medialex/twembeddings/docs/little_charts.jpg", bbox_inches="tight")
plt.savefig("/data/mray/medialex/twembeddings/docs/new_charts.jpg", bbox_inches="tight")