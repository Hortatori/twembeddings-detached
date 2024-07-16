# # from transformers import AutoModelForCausalLM, AutoTokenizer
# from angle_emb import AnglE
# from angle_emb.utils import cosine_similarity
# # model_path = "/data/llama-2/7b_hf"
# # tokenizer = AutoTokenizer.from_pretrained(model_path)
# # print(tokenizer)
# # model = AutoModelForCausalLM.from_pretrained(model_path)
# # text = "Hello my name is"
# # inputs = tokenizer(text, return_tensors="pt")
# # print(inputs)

# # init
# angle = AnglE.from_pretrained('/data/llama-2/7b_hf', pooling_strategy = 'cls')
# # encode text
# # for non-retrieval tasks, we don't need to specify prompt when using UAE-Large-V1.
# doc_vecs = angle.encode([
#     'The weather is great!',
#     'The weather is very good!',
#     'i am going to bed'
# ])

# for i, dv1 in enumerate(doc_vecs):
#     for dv2 in doc_vecs[i+1:]:
#         print(cosine_similarity(dv1, dv2))

"""test de timestamp pandas"""
import pandas as pd 

# df = pd.DataFrame([22,33])
# df['new_column'] = pd.Timestamp.today().strftime('%Y-%m-%d-%H-%M')
# print(df)

"""test de dicts"""
# features_names = ["truc", "machin"]
# r = dict({idx : term for idx, term in enumerate(features_names)})
# print(r)

"""test de dataframe"""
# df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
# print(df)
# temp = df.mean(numeric_only=True)
# print(temp["a":"b"])
# dfn = df.loc[[0]]
# print(dfn)
# dfn.loc[0,"a":"b"] = temp["a":"b"]
# print(dfn)

results = pd.read_csv("results_daily_aggloclusters.csv")
temp = results.mean(numeric_only=True)
output = results.iloc[[0]].copy()
output.loc[0,"count"] = int(temp["count"])
output.loc[0,"mean":"max"] = temp["mean":"max"]
output.loc[0,"p":"mcf1"] = temp["p":"mcf1"]
output.loc[0,"dataset"] = "/".join(output.loc[0, "dataset"].split("/")[:-1])
output.loc[0,"model"] = output.loc[0, "model"] + "_AggloC"
print(output)
# try:
#     results = pd.read_csv("results_daily_aggloclusters.csv")
#     temp = results.mean(numeric_only=True)
#     output = results.iloc[[0]].copy()
#     output.loc[0,"count":"mcf1"] = temp["count":"mcf1"]
#     output.loc[0,"dataset"] = "/".join(output.loc[0, "dataset"].split("/")[:-1])
#     output.loc[0,"model"] = output.loc[0, "model"] + "_AggloC"

#     try:
#         old_results = pd.read_csv("results_clustering.csv")
#     except FileNotFoundError:
#         old_results = pd.DataFrame()
#     stats = pd.concat([old_results, output], ignore_index=True)
#     stats.to_csv("results_clustering.csv", index=False)
    
# except FileNotFoundError:
#     print("no file results_daily_aggloclusters.csv")

