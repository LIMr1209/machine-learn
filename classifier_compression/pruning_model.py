import csv
from ruamel import yaml

with open('resnet152.schedule_sensitivity.yaml', 'r', encoding='utf-8') as f:
    data = yaml.load(f, Loader=yaml.RoundTripLoader)
# 模型敏感性分析后创建剪枝计划表  yaml
pruning_dict = {}

with open('sensitivity.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        low = ', '.join(row).split(',')
        if low[0] == 'parameter':
            continue
        if low[0] not in pruning_dict:
            pruning_dict[low[0]] = [{low[1]: low[2]}]
        else:
            pruning_dict[low[0]].append({low[1]: low[2]})

sensitivities_dict = {}
for key, value in pruning_dict.items():
    for i in range(len(value)):
        if float(list(value[i].values())[0]) < 90:
            sensitivities_dict[key] = round(float(list(value[i - 1].keys())[0]), 2)
            break
# 修改计划
data['pruners']['pruner1']['sensitivities'] = sensitivities_dict
# 保存写入文件
with open('resnet152.schedule_sensitivity.yaml', "w", encoding="utf-8") as f:
    yaml.dump(data, f, Dumper=yaml.RoundTripDumper)
