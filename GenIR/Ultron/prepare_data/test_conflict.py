def read_file(file_path):
    lines = []
    with open(file_path, 'r') as f:
        for line in f:
            lines.append(line)
    lines = [line.strip().split('\t') for line in lines]
    return lines
def test_single_id(corpus_ids):
    # Get the number of unique ids
    id_count = {}
    for id in corpus_ids:
        s = id
        if s in id_count:
            id_count[s] += 1
        else:
            id_count[s] = 1
    count = 0
    for id in corpus_ids:
        s = id
        if id_count[s] == 1:
            count += 1
    return count

def get_impact(corpus_ids):
    id_count = {}
    for id in corpus_ids:
        s = id
        if s in id_count:
            id_count[s] += 1
        else:
            id_count[s] = 1
    count = 0
    for id in corpus_ids:
        s = id
        if id_count[s] > 1:
            count += id_count[s]

    res = 0
    for id in corpus_ids:
        s = id
        if id_count[s] > 1:
            res += id_count[s]/count * id_count[s]
    return res


lines = read_file('dataset/encoded_docid/t5_url_all.txt')
corpus_ids = [line[1] for line in lines]
count = test_single_id(corpus_ids)
im = get_impact(corpus_ids)
print(f'Conflict Rate for D{0}: {(len(corpus_ids) - count) / len(corpus_ids)}')
print(f'Impact for D{0}: {im}')

path = 'dataset/encoded_docid/t5_url_all_{}.txt'
for i in range(1, 6):
    lines = read_file(path.format(i))
    corpus_ids = [line[1] for line in lines]
    count = test_single_id(corpus_ids)
    print(f'Conflict Rate for D{i}: {(len(corpus_ids) - count) / len(corpus_ids)}')
    im = get_impact(corpus_ids)
    print(f'Impact for D{i}: {im}')




