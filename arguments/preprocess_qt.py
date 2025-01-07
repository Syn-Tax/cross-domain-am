import arguments.preprocess_aif as preprocess_aif
import os

""" File to process QT30 argument maps
"""

RAW_DATA = "raw_data/Question Time"
DATA = "data/Question Time"

def main():
    # get all directories in raw data dir
    dirs = [d for d in os.listdir(RAW_DATA) if os.path.isdir(os.path.join(RAW_DATA, d))]
    print(dirs)

    for dir in dirs:
        # if the required directory doesn't exist in processed data, create it
        if not os.path.exists(os.path.join(DATA, dir)):
            os.mkdir(os.path.join(DATA, dir))

        # process AIF data for this dataset
        preprocess_aif.process_dir(os.path.join(RAW_DATA, dir), os.path.join(DATA, dir, "argument_map.json"))

if __name__ == "__main__":
    main()