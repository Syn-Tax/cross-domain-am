import pathlib
import csv
import pandas as pd

from datastructs import Node

DATA_DIR = pathlib.Path("data")

output = {}

for path in DATA_DIR.rglob("**/argument_map.json"):
    with open(path, "r") as f:
        arg_map = Node.schema().loads(f.read(), many=True)
        scores = [round(n.audio_score, 4) for n in arg_map]

        output[str(path.parent).split("/")[-1]] = scores

scores_sub_qt = []

for k, v in output.items():
    if k == "Question Time": continue
    scores_sub_qt.extend(v)

output["total"] = scores_sub_qt

def pad_dict_list(dict_list, padel):
    lmax = 0
    for lname in dict_list.keys():
        lmax = max(lmax, len(dict_list[lname]))
    for lname in dict_list.keys():
        ll = len(dict_list[lname])
        if  ll < lmax:
            dict_list[lname] += [padel] * (lmax - ll)
    return dict_list

output = pad_dict_list(output, None)

df = pd.DataFrame.from_dict(output)
df.to_csv("scores.csv", index=False)