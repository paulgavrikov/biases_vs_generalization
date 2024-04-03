from glob import glob
import os

if __name__ == '__main__':

    for p in glob("/workspace/data/datasets/imagenetv2-matched-frequency-format-val/*"):
        parts = p.split("/")

        og = parts[-1]

        while len(parts[-1]) < 3:
            parts[-1] = "0" + parts[-1]

        # move folder
        os.rename("/".join(parts[:-1]) + "/" + og, "/".join(parts))
