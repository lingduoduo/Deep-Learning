import pandas as pd
import jsonlines,json
from tqdm import tqdm
import random,math
def generate_train_data(ratio_file, target_file):
    data_files = pd.read_excel(ratio_file)

    with jsonlines.open(target_file, 'w') as writer:
        for i in range(len(data_files)):
            filename = data_files.iloc[i]['filename']
            sample_rate = data_files.iloc[i]['sample_rate']
            with open(filename, 'r') as f:
                print(f"Start To Read Filename:{filename}")
                datas = []
                for line in tqdm(f):
                    datas.append(json.loads(line))
                print(f"Start To Sample {filename} Data, Sample Rate:{sample_rate}")
                if sample_rate<1:
                    sample_num = math.floor(len(datas) * sample_rate)
                    all_ids = [i for i in range(len(datas))]
                    ids = random.sample(all_ids, sample_num)
                else:
                    ratio_int = math.floor(sample_rate)
                    ratio_frac = sample_rate-ratio_int
                    all_ids = [i for i in range(len(datas))]
                    ids = random.sample(all_ids, math.floor(len(datas)*ratio_frac))
                    ids.extend(all_ids*ratio_int)
                print(f"From {filename} , Sampled {len(ids)} items")
                for index in ids:
                    writer.write(
                        {
                            "input":datas[index]["input"],
                            "instruction": "",
                            "output": datas[index]["output"]
                        }
                    )

if __name__ == '__main__':
    generate_train_data("train_1116_2.xlsx", "./data/longgen_20251129.jsonl")