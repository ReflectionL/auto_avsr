# 读取txt文件，移除重复行，写回文件

def remove_duplicates(file_path):
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 移除重复行
    unique_lines = list(set(lines))

    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(unique_lines)

# 调用函数
remove_duplicates('/work/lixiaolou/program/auto_avsr/script/success.txt')
