# 用于将琛哥写的test文件转换成我的格式

# 读取文件A
with open('/work/lixiaolou/program/DF/oot_syncnet/faceavceleb_fake.txt', 'r') as file_a:
    paths_a = [line.split('//')[-1].strip() for line in file_a]
    
# 读取文件D
with open('/work/lixiaolou/program/DF/oot_syncnet/faceavceleb_real.txt', 'r') as file_d:
    paths_d = [line.split('//')[-1].strip() for line in file_d]

# 读取文件B并检查是否包含文件A中的路径
matches = []
with open('/work/lixiaolou/program/auto_avsr/data/FakeAVCeleb/label.txt', 'r') as file_b:
    for line in file_b:
        if any(path in line for path in paths_a) or any(path in line for path in paths_d):
            matches.append(line.strip())

matches.sort()


# 将匹配结果保存到新文件
with open('/work/lixiaolou/program/auto_avsr/data/FakeAVCeleb/label_test.csv', 'w') as output_file:
    for match in matches:
        output_file.write(match + '\n')
