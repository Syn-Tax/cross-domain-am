import preprocess_aif
import os

RAW_DATA = "raw_data/Question Time"
DATA = "data/Question Time"

def main():
    dirs = [d for d in os.listdir(RAW_DATA) if os.path.isdir(os.path.join(RAW_DATA, d))]
    print(dirs)
    # return
    for dir in dirs:
        if not os.path.exists(os.path.join(DATA, dir)):
            os.mkdir(os.path.join(DATA, dir))

        preprocess_aif.process_dir(os.path.join(RAW_DATA, dir), os.path.join(DATA, dir, "argument_map.json"))

if __name__ == "__main__":
    main()