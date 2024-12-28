import pathlib
import tqdm

from datastructs import Node

QT_PATH = "data/Question Time"
OUT_PATH = "data/Question Time/argument_map.json"
QT_OMISSIONS = "qt_omissions.txt"

def main():
    output_map = []

    with open(QT_OMISSIONS, "r") as f:
        qt_omissions = [e.strip() for e in f.readlines()]

    paths = [p for p in pathlib.Path(QT_PATH).iterdir() if p.is_dir() and p.name not in qt_omissions]

    for path in tqdm.tqdm(paths):
        with open(path / "argument_map.json", "r") as f:
            nodes = Node.schema().loads(f.read(), many=True)

        for n in nodes:
            n.episode = path.name

        output_map.extend(nodes)

    print(len(output_map))

    with open(OUT_PATH, "w") as f:
        out = Node.schema().dumps(output_map, many=True)
        f.write(out)
        print("written output")


if __name__ == "__main__":
    main()