import numpy as np
import re
import itertools
from collections import Counter
import codecs
from mlxtend.preprocessing import one_hot


#TODO: label clean of none label and outliers


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    modified by Guoyin
    """
    # TODO: exam special cases in the doc notes, especially concatenate words
    # string = re.sub(r"[^A-Za-z0-9(),!?%\'\`\+/]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)

    string = re.sub(r"\s{2,}", " ", string)

    string = re.sub(r'(\d+)([^\s\d%+])', r'\1 \2', string)
    string = re.sub(r"-", "- ",string)
    return string.strip().lower()

def load_notes(note_file = 'pdata/tok_hpi'):
    """
    Loads doctor notes file, input is the doctor notes file name, default
    """

    with codecs.open(note_file, 'r', encoding='utf-8', errors='ignore') as nf:
        notes = nf.readlines()
    notes = [n.strip() for n in notes]
    notes = [clean_str(n) for n in notes]
    return notes

def load_labels(label_file = 'pdata/tok_disp'):
    """
    Load action on the patient
    Attention: have title need to remove
    """

    # TODO: remove empty labels

    with open(label_file, 'r', encoding='utf-8') as lf:
        labels = lf.readlines()
    labels = labels[1:] # remove header
    labels = [l.split(',')[1] for l in labels]
    return labels

def remove_empty_labels(notes, labels):
    non_empty_list = [[n,l]  for n, l in zip(notes, labels) if l is not ' ']
    notes = [l[0] for l in non_empty_list]
    labels = [l[1] for l in non_empty_list]
    assert(len(notes) == len(labels))
    return notes, labels

def binary_labels(labels):
    """
    generate binary labels: ' Tar-Treated/Released' and 'Unreleased'

    """
    labels = [l if l == ' Tar-Treated/Released' else 'Unreleased' for l in labels ]
    return labels

def one_hot_encoding(labels):
    """
    doing one hot encoding for all categories
    """

    label_counts = Counter(labels)
    label_num = len(label_counts)
    label_dict = { k : i  for i, k in enumerate(label_counts.keys())}
    reverse_label_dict = { i : k  for i, k in enumerate(label_counts.keys())}

    int_labels = list(map(lambda x: label_dict[x], labels))

    one_hot_labels = one_hot(int_labels)

    return one_hot_labels, int_labels



def shorten_notes(notes, length = 300):
    return [' '.join(n.split(" ")[:300]) for n in notes]





def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]
def generate_data_and_label(notes, one_hot_labels):
    return [notes, one_hot_labels]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



def load_embedding_vectors_word2vec(vocabulary, filename, binary):
    # load embedding_vectors from the word2vec
    encoding = 'utf-8'
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        # initial matrix with random uniform
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        if binary:
            binary_len = np.dtype('float32').itemsize * vector_size
            for line_no in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':
                        word.append(ch)
                word = str(b''.join(word), encoding=encoding, errors='strict')
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.seek(binary_len, 1)
        else:
            for line_no in range(vocab_size):
                line = f.readline()
                if line == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                parts = str(line.rstrip(), encoding=encoding, errors='strict').split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                word, vector = parts[0], list(map('float32', parts[1:]))
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = vector
        f.close()
        return embedding_vectors




def load_embedding_vectors_glove(vocabulary, filename, vector_size):
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors
if  __name__ == "__main__":
    labels =load_labels()
    notes = load_notes()
    notes, labels = remove_empty_labels(notes, labels)
    labels = binary_labels(labels)
    one_hot_labels, int_labels = one_hot_encoding(labels)
    generate_data_and_label(notes, labels)
