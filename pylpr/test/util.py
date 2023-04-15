import requests
from os import path

def dot_to_ascii(dot: str, fancy: bool = True):
    """Source: https://github.com/ggerganov/dot-to-ascii"""

    url = 'https://dot-to-ascii.ggerganov.com/dot-to-ascii.php'
    boxart = 0

    # use nice box drawing char instead of + , | , -
    if fancy:
        boxart = 1

    params = {
        'boxart': boxart,
        'src': dot,
    }

    response = requests.get(url, params=params).text

    if response == '':
        raise SyntaxError('DOT string is not formatted correctly')

    return response

def dot_from_file(filepath: str):
    graph = 'digraph {'
    with open(filepath) as f:
        for line in f:
            splitted = line.strip().split()
            assert(len(splitted) == 3)
            graph += f'\n\t{splitted[0]} -> {splitted[2]} [label = "{splitted[1]}"]'

    graph += '\n}'

    return graph

def write_graphs(dataset_dir: str):
    train = dot_from_file(path.join(dataset_dir, "train.txt"))
    test = dot_from_file(path.join(dataset_dir, "test.txt"))
    valid = dot_from_file(path.join(dataset_dir, "valid.txt"))
    graphs_ascii = f'''
============================================
                    TRAIN
============================================

{dot_to_ascii(train)}


============================================
                    TEST
============================================

{dot_to_ascii(test)}


============================================
                   VALID
============================================

{dot_to_ascii(valid)}
'''

    with open(path.join(dataset_dir, "ascii"), 'w') as f:
        f.write(graphs_ascii)


if __name__ == "__main__":
    write_graphs("pylpr/test/testdataset")
