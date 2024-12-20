import re
import pathlib

INPUT_PATH = "raw_data/Question Time"
OUTPUT_PATH = "data/Question Time"

line_regex = r"^\[.{0,10}[0-9]+:[0-9]+:[0-9]+\].*$"
timestamp_regex = r"(\[.{0,10}[0-9]+:[0-9]+:[0-9]+\])"



def process_transcript(path, out_path):
    with open(path, "r") as f:
        raw = f.readlines()

    transcript = []
    for line in raw:
        if not re.match(line_regex, line): continue

        l = re.sub(timestamp_regex, "", line).strip()

        transcript.append(l)

    with open(out_path, "w+") as f:
        f.write("\n".join(transcript))

def process_concat(path, out_path):
    with open(path, "r") as f:
        lines = f.readlines()

    lines_dict = {}

    for line in lines:
        line_split = re.split(timestamp_regex, line)
        if len(line_split) < 3: continue
        timestamp = line_split[1][1:-1]
        timestamp_split = timestamp.split(":")
        try:
            time = int(timestamp_split[0]) * 3600 + int(timestamp_split[1]) * 60 + int(timestamp_split[2])
            lines_dict[time] = line_split[-1]
        except Exception as e:
            print(e)
            print(path)
            return

    sorted_lines = [l[1] for l in sorted(lines_dict.items(), key=lambda x: x[0])]
    string = "".join(sorted_lines)

    with open(out_path, "w") as f:
        f.write(string)


def main():
    with open("qt_omissions.txt", "r") as f:
        OMISSIONS = [s.strip() for s in f.readlines()]

    in_path = pathlib.Path(INPUT_PATH)
    out_path = pathlib.Path(OUTPUT_PATH)

    in_dirs = [f.name for f in pathlib.Path(INPUT_PATH).iterdir() if f.is_dir()]

    for dir in in_dirs:
        if dir in OMISSIONS: continue
        in_dir_path = in_path / dir
        out_dir_path = out_path / dir

        dir_files = [f.name for f in in_dir_path.iterdir() if f.is_file()]

        if "transcript_raw.txt" in dir_files:
            process_transcript(in_dir_path / "transcript_raw.txt", out_dir_path / "transcript.txt")
        elif "concat_processed.txt" in dir_files:
            process_concat(in_dir_path / "concat_processed.txt", out_dir_path / "transcript.txt")
        else:
            print(dir)


if __name__ == "__main__":
    #process_transcript("raw_data/Question Time/01.28May2020/transcript_raw.txt", "transcript.txt")
    main()
