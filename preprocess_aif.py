import json
import pathlib
from datastructs import Node, Relation

DIR_PATH = "raw_data/Moral Maze/GreenBelt"
OUT_PATH = "data/Moral Maze/GreenBelt/argument_map.json"

TO_PROP_RELATIONS = ["YA"]
PROP_RELATIONS = ["RA", "CA", "MA"]
PROPOSITION_TYPE = "I"
LOCUTION_TYPE = "L"


def follow_edges(from_id: str, nodes, edges) -> list[dict]:
    indices = [i for i, x in enumerate(edges) if x["fromID"] == from_id]

    next_nodes = []

    for i in indices:
        next_node_id = edges[i]["toID"]
        next_nodes.append([x for x in nodes if x["nodeID"] == next_node_id][0])

    return next_nodes


def find_proposition(loc_id: str, nodes: list, edges: list) -> str:
    next_nodes = follow_edges(loc_id, nodes, edges)
    propositions = []

    for node in next_nodes:
        if node["type"] not in TO_PROP_RELATIONS:
            continue

        possible_propositions = follow_edges(node["nodeID"], nodes, edges)
        propositions.extend(
            [prop for prop in possible_propositions if prop["type"] == PROPOSITION_TYPE]
        )

    if len(propositions) == 0:
        return None, None

    return propositions[0]["text"], propositions[0]["nodeID"]


def find_relations(prop_id: str, nodes: list, edges: list) -> list[Relation]:
    rel_nodes = follow_edges(prop_id, nodes, edges)
    relations = []

    for node in rel_nodes:
        if node["type"] not in PROP_RELATIONS:
            continue

        possible_next_props = follow_edges(node["nodeID"], nodes, edges)
        relations.extend(
            [
                Relation(node["type"], int(x["nodeID"]))
                for x in possible_next_props
                if x["type"] == PROPOSITION_TYPE
            ]
        )

    return relations


def denormalise(obj: dict):
    output = []
    nodes = obj["nodes"]
    edges = obj["edges"]

    for node in nodes:
        if node["type"] != LOCUTION_TYPE:
            continue

        if node["text"].startswith("People :"):
            continue

        n = Node()

        n.locution = node["text"]

        n.proposition, prop_id = find_proposition(node["nodeID"], nodes, edges)

        # FIXME: this is going to be annoying
        if not prop_id:
            continue

        n.id = int(prop_id)

        n.relations = find_relations(prop_id, nodes, edges)

        output.append(n)

    return output


def get_files(dir: str) -> list[str]:
    dir_path = pathlib.Path(dir)
    all_files = [
        str(f) for f in dir_path.iterdir() if f.is_file() and f.name.endswith(".json")
    ]

    return all_files


if __name__ == "__main__":
    files = get_files(DIR_PATH)
    data = []
    for file in files:
        with open(file, "r") as f:
            obj = json.load(f)

        file_data = denormalise(obj)

        data.extend(file_data)
        print(f"Processed file: {file}")

    with open(OUT_PATH, "w") as f:
        out = Node.schema().dumps(data, many=True)
        f.write(out)
        print("Written output")
