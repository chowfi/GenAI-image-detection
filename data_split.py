import argparse
import glob
import json
import webdataset as wds


def split_dataset(path, n_train, n_val, n_test, label, domain_label):
    max_file_size = 1000
    input_files = glob.glob(path + "/*.tar")
    src = wds.WebDataset(input_files)

    train_path_prefix = path + "/train"
    val_path_prefix = path + "/val"
    test_path_prefix = path + "/test"

    def write_split(dataset, prefix, start, end):
        n_split = end - start
        output_files = [f"{prefix}_{i}.tar" for i in range(n_split // max_file_size + 1)]
        for i, output_file in enumerate(output_files):
            print(f"Writing {output_file}")
            with wds.TarWriter(output_file) as dst:
                for sample in dataset.slice(start + i * max_file_size, min(start + (i + 1) * max_file_size, end)):
                    new_sample = {
                        "__key__": sample["__key__"],
                        "jpg": sample["jpg"],
                        "label.cls": label,
                        "domain_label.cls": domain_label
                    }
                    dst.write(new_sample)
    
    write_split(src, train_path_prefix, 0, n_train)
    write_split(src, val_path_prefix, n_train, n_train + n_val)
    write_split(src, test_path_prefix, n_train + n_val, n_train + n_val + n_test)


def calculate_sizes(path):
    stat_files = glob.glob(path + "/*_stats.json")
    total = 0
    for f in stat_files:
        with open(f, "r") as stats:
            total += json.load(stats)["successes"]
    n_train = int(total * 0.8)
    n_val = int(total * 0.1)
    n_test = total - n_train - n_val

    return n_train, n_val, n_test


if __name__ == "__main__":

    paths = [
        "./data/laion400m_data",
        "./data/genai-images/StableDiffusion",
        "./data/genai-images/midjourney",
        "./data/genai-images/dalle2",
        "./data/genai-images/dalle3"
    ]

    sizes = []
    for p in paths:
        res = calculate_sizes(p)
        sizes.append(res)
    
    domain_labels = [0, 1, 4, 2, 3]

    for i, p in enumerate(paths):
        print(f"{p}: {sizes[i]}")
        label = 0 if i == 0 else 1
        print(label, domain_labels[i])
        split_dataset(p, *calculate_sizes(p), label, domain_labels[i])