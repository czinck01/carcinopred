import json

genes = []
with open('genes.2020-01-16.json') as json_file:
    data = json.load(json_file)
    for d in data:
        genes.append(d['gene_id'])

genes.sort()

with open('genes.txt', 'w') as file:
    for g in genes:
        file.write(g + "\n")
