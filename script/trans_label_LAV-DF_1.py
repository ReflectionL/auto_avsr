# 用于从LAV-DF数据集中选择一部分，保存路径文件和新json
import json
import random
import csv

def select_items_and_save(input_path, output_json_path, output_csv_path, prefix, zero_count=250, non_zero_count=750):
    with open(input_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    zero_items = [item for item in data if item['n_fakes'] == 0]
    non_zero_items = [item for item in data if item['n_fakes'] != 0]

    zero_count = min(zero_count, len(zero_items))
    non_zero_count = min(non_zero_count, len(non_zero_items))

    selected_zero_items = random.sample(zero_items, zero_count)
    selected_non_zero_items = random.sample(non_zero_items, non_zero_count)

    selected_items = selected_zero_items + selected_non_zero_items

    # 添加路径前缀并保存文件名到CSV
    filenames = []
    for item in selected_items:
        item['file'] = prefix + item['file']
        filenames.append([item['file']])

    with open(output_json_path, 'w', encoding='utf-8') as outfile:
        json.dump(selected_items, outfile, indent=4)

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(filenames)

# 使用示例
input_path = '/work/lixiaolou/program/auto_avsr/data/LAV-DF/test_all.json'  # 输入文件路径
output_json_path = '/work/lixiaolou/program/auto_avsr/data/LAV-DF/test_part.json'  # 输出JSON文件的路径
output_csv_path = '/work/lixiaolou/program/auto_avsr/data/LAV-DF/filenames.csv'  # 输出CSV文件的路径
prefix = '/ssd1/DF/LAV-DF/'  # 文件路径前缀
select_items_and_save(input_path, output_json_path, output_csv_path, prefix)
