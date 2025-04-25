from util.io import write_file, read_file


data = read_file('../../../dataset/version3/msmarco/simplified-msmarco-full.json')
outdata = []

for index, dq in data:
    for i in range(10):
        outdata.append(f'[{index}]\t{dq[1][i]}')

with open('dataset/nq-data/corpus_pseudo_query.txt', 'w') as f:
    for item in outdata:
        f.write(f'{item}\n')


