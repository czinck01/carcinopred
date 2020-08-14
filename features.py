import os
import numpy as np
  
ROOT = 'data'
GENES = 'genes.txt'

def create_gene_dict(filename):
    dict = {}
    with open(filename) as file:
        genes = [x.strip('\n') for x in file.readlines()]
        for i in range(len(genes)):
           dict[genes[i]] = i
    return dict

def create_instance(gene_dict, genes):
    instance = np.zeros(len(gene_dict))
    for gene in genes:
        instance[gene_dict[gene]] = 1
    return instance

gene_dict = create_gene_dict(GENES)    

instances = []
for dir in sorted(os.listdir(ROOT)):
    for filename in sorted(os.listdir(os.path.join(ROOT, dir))):
        with open(os.path.join(ROOT, dir, filename)) as file:
            file.readline()
            genes = [x.strip('\n') for x in file.readlines()]
            instances.append(np.append(create_instance(gene_dict, genes), int(dir)))

mat = np.array(instances)

np.savetxt('mat.csv', mat, delimiter=',', fmt= '%1.0f')
