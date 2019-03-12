import csv

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
for key, value in pruning_dict.items():
    for i in range(len(value)):
        if float(list(value[i].values())[0]) < 90:
            print("'" + key + "': " + list(value[i - 1].keys())[0])
