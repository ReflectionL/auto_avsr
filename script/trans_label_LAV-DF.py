# 用于处理LAV-DF数据集的json文件，将split为'test'的项目保存到新文件
import json

def process_json(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    count_n_fakes_zero = 0
    count_n_fakes_non_zero = 0
    test_items = []

    for item in data:
        if item['split'] == 'test':
            test_items.append(item)
            if item['n_fakes'] == 0:
                count_n_fakes_zero += 1
            else:
                count_n_fakes_non_zero += 1

    # 保存split为'test'的项目到新文件
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(test_items, outfile, indent=4)

    return count_n_fakes_zero, count_n_fakes_non_zero

# 使用示例
file_path = '/ssd1/DF/LAV-DF/metadata.min.json'  # 这里替换成您的json文件路径
output_path = '/work/lixiaolou/program/auto_avsr/data/LAV-DF/test_all.json'  # 输出文件的路径
count_zero, count_non_zero = process_json(file_path, output_path)
print(f"Number of items with n_fakes as 0: {count_zero}")
print(f"Number of items with n_fakes non-zero: {count_non_zero}")
