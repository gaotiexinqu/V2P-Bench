import argparse
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('--data_path', type = str, help="data_path")
    args = parser.parse_args()
    return args

args = parse_arguments()
data = [json.loads(line) for line in open(args.data_path, "r", encoding="utf-8")]

durations = [0, 240, 1800, 7200]
dim_mapping = {1: "OA",2: "HA",3: "OD",4: "FM",5: "CR",6: "PU",7: "CI", 9: "FT",10: "RT",12: "AS",13: "SR",14: "GC",}

dim_nums = 16
dim_list_sum = [0] * dim_nums
dim_list_cor = [0] * dim_nums

short_cor, short_sum = 0, 0
medium_cor, medium_sum = 0, 0
long_cor, long_sum = 0, 0
correct_nums = total_nums = 0, 0

for line in data:
    dim = line["dimension"]
    dim_list_sum[dim - 1] += 1

    if line["duration"] < durations[1]:
        short_sum += 1
    elif line["duration"] < durations[2]:
        medium_sum += 1
    else:
        long_sum += 1

    if line["response"][0] == line["answer"]:
        dim_list_cor[dim - 1] += 1
        if line["duration"] < durations[1]:
            short_cor += 1
        elif line["duration"] < durations[2]:
            medium_cor += 1
        else:
            long_cor += 1

for index, (dim_cor, dim_sum) in enumerate(zip(dim_list_cor, dim_list_sum)):
    if index+1 not in [8, 11, 15, 16]:
        if dim_sum != 0:
            print(f"{dim_mapping[index + 1]}: {dim_cor / dim_sum:.3f}")
        else:
            raise ValueError(f"Dimension is zero: {dim_mapping[index + 1]}")

print("----------------------------------------------------------")

print(f"Short\nCorrect: {short_cor}, Total: {short_sum}, Accuracy: {short_cor / short_sum:.3f}")
print(f"Medium\nCorrect: {medium_cor}, Total: {medium_sum}, Accuracy: {medium_cor / medium_sum:.3f}")
print(f"Long\nCorrect: {long_cor}, Total: {long_sum}, Accuracy: {long_cor / long_sum:.3f}")

print("----------------------------------------------------------")
cor_data = sum(dim_list_cor)
all_data = sum(dim_list_sum)
print(f"Total Correct: {cor_data}")
print(f"Total Success: {all_data}")
print(f"Accuracy:      {cor_data / all_data:.3f}")

