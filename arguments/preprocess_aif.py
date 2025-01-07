import json
import pathlib
from datastructs import Node, Relation

DIR_PATH = "raw_data/Moral Maze/Welfare"
OUT_PATH = "data/Moral Maze/Welfare/argument_map.json"

TO_PROP_RELATIONS = ["YA"]
PROP_RELATIONS = ["RA", "CA", "MA"]
PROPOSITION_TYPE = "I"
LOCUTION_TYPE = "L"

REMOVE_PEOPLE = ["People", "Two of the Moral Maze's witnesses"]


def follow_edges(from_id: str, nodes, edges) -> list[dict]:
    """Method to follow the edges from a node

    Args:
        from_id (str): id of start node
        nodes (list): list of nodes
        edges (list): list of edges

    Returns:
        list[dict]: list of nodes connected to the start node
    """
    indices = [i for i, x in enumerate(edges) if x["fromID"] == from_id]

    next_nodes = []

    for i in indices:
        next_node_id = edges[i]["toID"]
        next_nodes.append([x for x in nodes if x["nodeID"] == next_node_id][0])

    return next_nodes


def find_proposition(loc_id: str, nodes: list, edges: list) -> str:
    """Method to find a proposition relating to a certain locution

    Args:
        loc_id (str): locution ID
        nodes (list): list of nodes
        edges (list): list of edges

    Returns:
        str: proposition text, proposition ID
    """

    # find nodes connected to the locution
    next_nodes = follow_edges(loc_id, nodes, edges)
    propositions = []

    # find propositions
    for node in next_nodes:
        if node["type"] not in TO_PROP_RELATIONS:
            continue

        possible_propositions = follow_edges(node["nodeID"], nodes, edges)
        propositions.extend(
            [prop for prop in possible_propositions if prop["type"] == PROPOSITION_TYPE]
        )

    # if no propositions found return to prevent "index out of range" issues
    if len(propositions) == 0:
        return None, None

    # return first proposition
    return propositions[0]["text"], propositions[0]["nodeID"]


def find_relations(prop_id: str, nodes: list, edges: list) -> list[Relation]:
    """Method to find the relation between two propositions

    Args:
        prop_id (str): proposition ID
        nodes (list): list of nodes
        edges (list): list of edges

    Returns:
        list[Relation]: list of relation objects
    """

    # find nodes connected to the proposition
    rel_nodes = follow_edges(prop_id, nodes, edges)
    relations = []

    # get all relations connected to the proposition
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
    """Method to denormalise AIF data into a more JSON friendly format

    Args:
        obj (dict): input json

    Returns:
        list[Node]: list of node objects
    """

    # initialise vars
    output = []
    nodes = obj["nodes"]
    edges = obj["edges"]

    for node in nodes:
        # find only the locutions
        if node["type"] != LOCUTION_TYPE:
            continue
        
        # remove accidental pronoun realisation
        if any(node["text"].startswith(x) for x in REMOVE_PEOPLE):
            continue
        
        # create node and add locution and proposition
        n = Node()
        n.locution = node["text"]
        n.proposition, prop_id = find_proposition(node["nodeID"], nodes, edges)

        # if no proposition is connected to the locution, ignore it
        if not prop_id:
            continue
            
        # set the node ID to the proposition ID
        n.id = int(prop_id)

        # get the relations relevant to the node
        n.relations = find_relations(prop_id, nodes, edges)

        # add it to the output
        output.append(n)

    ids = [n.id for n in output]

    # remove all relations going to non-existent nodes (for whatever reason)
    for node in output:
        for i, rel in enumerate([*node.relations]):
            if rel.to_node_id not in ids:
                del node.relations[i]


    return output


def get_files(dir: str) -> list[str]:
    """Method to get json files in a directory

    Args:
        dir (str): path to directory

    Returns:
        list[str]: list of paths to json files
    """
    dir_path = pathlib.Path(dir)
    all_files = [
        str(f) for f in dir_path.iterdir() if f.is_file() and f.name.endswith(".json")
    ]

    return all_files

def process_dir(dir_path: str, out_path: str) -> None:
    """Method to process an entire episode

    Args:
        dir_path (str): path to input directory
        out_path (str): path to output file
    """

    # loop through each json file in the directory
    files = get_files(dir_path)
    data = []
    for file in files:
        with open(file, "r") as f:
            obj = json.load(f)

        # get denormalised argument data
        file_data = denormalise(obj)

        data.extend(file_data)
        print(f"Processed file: {file}")

    # write output to output file
    with open(out_path, "w") as f:
        out = Node.schema().dumps(data, many=True)
        f.write(out)
        print()
        print("Written output")



if __name__ == "__main__":
    process_dir(DIR_PATH, OUT_PATH)