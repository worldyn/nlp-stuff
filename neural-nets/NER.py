#!/usr/bin/env python
# coding: utf-8
import argparse
import string
import codecs
import csv
from tqdm import tqdm
from terminaltables import AsciiTable
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

from GRU import GRU2

PADDING_WORD = '<PAD>'
UNKNOWN_WORD = '<UNK>'
CHARS = ['<UNK>', '<space>', '’', '—'] + list(string.punctuation) + list(string.ascii_letters) + list(string.digits)


def load_glove_embeddings(embedding_file, padding_idx=0, padding_word=PADDING_WORD, unknown_word=UNKNOWN_WORD):
    """
    The function to load GloVe word embeddings
    
    :param      embedding_file:  The name of the txt file containing GloVe word embeddings
    :type       embedding_file:  str
    :param      padding_idx:     The index, where to insert padding and unknown words
    :type       padding_idx:     int
    :param      padding_word:    The symbol used as a padding word
    :type       padding_word:    str
    :param      unknown_word:    The symbol used for unknown words
    :type       unknown_word:    str
    
    :returns:   (a vocabulary size, vector dimensionality, embedding matrix, mapping from words to indices)
    :rtype:     a 4-tuple
    """
    word2index, embeddings, N = {}, [], 0
    with open(embedding_file, encoding='utf8') as f:
        for line in f:
            data = line.split()
            word = data[0]
            vec = [float(x) for x in data[1:]]
            embeddings.append(vec)
            word2index[word] = N
            N += 1
    D = len(embeddings[0])
    
    if padding_idx is not None and type(padding_idx) is int:
        embeddings.insert(padding_idx, [0]*D)
        embeddings.insert(padding_idx + 1, [-1]*D)
        for word in word2index:
            if word2index[word] >= padding_idx:
                word2index[word] += 1
        word2index[padding_word] = padding_idx
        word2index[unknown_word] = padding_idx + 1
                
    return N, D, np.array(embeddings, dtype=np.float32), word2index


class NERDataset(Dataset):
    """
    A class loading NER dataset from a CSV file to be used as an input to PyTorch DataLoader.
    """
    def __init__(self, filename):
        reader = csv.reader(codecs.open(filename, encoding='ascii', errors='ignore'), delimiter=',')
        
        self.sentences = []
        self.labels = []
        
        sentence, labels = [], []
        for row in reader:
            if row:
                if row[0].strip():
                    if sentence and labels:
                        self.sentences.append(sentence)
                        self.labels.append(labels)
                    sentence = [row[1].strip()]
                    labels = [self.__bio2int(row[3].strip())]
                else:
                    sentence.append(row[1].strip())
                    labels.append(self.__bio2int(row[3].strip()))
                
    def __bio2int(self, x):
        return 0 if x == 'O' else 1
        
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]


class PadSequence:
    """
    A callable used to merge a list of samples to form a padded mini-batch of Tensor
    """
    def __call__(self, batch, pad_data=PADDING_WORD, pad_labels=0):
        batch_data, batch_labels = zip(*batch)
        max_len = max(map(len, batch_labels))
        padded_data = [[b[i] if i < len(b) else pad_data for i in range(max_len)] for b in batch_data]
        padded_labels = [[l[i] if i < len(l) else pad_labels for i in range(max_len)] for l in batch_labels]
        return padded_data, padded_labels


class NERClassifier(nn.Module):
    def __init__(self, word_emb_file, char_emb_size=16, char_hidden_size=25, word_hidden_size=100,
                 padding_word=PADDING_WORD, unknown_word=UNKNOWN_WORD, char_map=CHARS,
                 char_bidirectional=True, word_bidirectional=True):
        """
        Constructs a new instance.
        
        :param      word_emb_file:     The filename of the file with pre-trained word embeddings
        :type       word_emb_file:     str
        :param      char_emb_size:     The character embedding size
        :type       char_emb_size:     int
        :param      char_hidden_size:  The character-level BiRNN hidden size
        :type       char_hidden_size:  int
        :param      word_hidden_size:  The word-level BiRNN hidden size
        :type       word_hidden_size:  int
        :param      padding_word:      A token used to pad the batch to equal-sized tensor
        :type       padding_word:      str
        :param      unknown_word:      A token used for the out-of-vocabulary words 
        :type       unknown_word:      str
        :param      char_map:          A list of characters to be considered
        :type       char_map:          list
        """
        super(NERClassifier, self).__init__()
        self.padding_word = padding_word
        self.unknown_word = unknown_word
        self.char_emb_size = char_emb_size
        self.char_hidden_size = char_hidden_size
        self.word_hidden_size = word_hidden_size
        self.char_bidirectional = char_bidirectional
        self.word_bidirectional = word_bidirectional
        
        vocabulary_size, self.word_emb_size, embeddings, self.w2i = load_glove_embeddings(
            word_emb_file, padding_word=self.padding_word, unknown_word=self.unknown_word
        )
        
        self.word_emb = nn.Embedding(vocabulary_size, self.word_emb_size)
        self.word_emb.weight = nn.Parameter(torch.from_numpy(embeddings), requires_grad=False)
        
        if self.char_emb_size > 0:
            self.c2i = {c: i for i, c in enumerate(char_map)}
            self.char_emb = nn.Embedding(len(char_map), char_emb_size, padding_idx=0)
            self.char_birnn = GRU2(self.char_emb_size, self.char_hidden_size, bidirectional=char_bidirectional)
        else:
            self.char_hidden_size = 0

        multiplier = 2 if self.char_bidirectional else 1
        self.word_birnn = GRU2(
            self.word_emb_size + multiplier * self.char_hidden_size, # input size
            self.word_hidden_size,                          # hidden size
            bidirectional=word_bidirectional
        )
        
        # Binary classification - 0 if not part of the name, 1 if a name
        multiplier = 2 if self.word_bidirectional else 1
        self.final_pred = nn.Linear(multiplier * self.word_hidden_size, 2)
        
    def forward(self, x):
        """
        Performs a forward pass of a NER classifier
        Takes as input a 2D list `x` of dimensionality (B, T),
        where B is the batch size;
              T is the max sentence length in the batch (the sentences with a smaller length are already padded with a special token <PAD>)
        
        Returns logits, i.e. the output of the last linear layer before applying softmax.

        :param      x:    A batch of sentences
        :type       x:    list of strings
        """
        #
        # YOUR CODE HERE
        #
        B = len(x)
        MAXS = len(x[0])
        unknown_idx = self.w2i[UNKNOWN_WORD]

        # tokenize and get word embeddings
        MAXC = 0
        indexes = []
        for sentence in x:
            sen_l = []
            for word in sentence:
                sen_l.append(self.w2i.get(word, unknown_idx)) # word -> idx
                if len(word) > MAXC:
                    MAXC = len(word)
            indexes.append(sen_l)
        indexes = torch.IntTensor(indexes)
        word_embs = self.word_emb(indexes) # idxs -> word emb
        #print("glove word embs 3D: ",word_embs.shape)

        # get character embeddings
        char_embs = torch.empty(B, MAXS, MAXC, self.char_emb_size)
        for sidx in range(B): # sentence
            for widx in range(MAXS): # word
                word = x[sidx][widx]
                char_idxs = torch.empty(MAXC, dtype=torch.int) # init char idxs
                for cidx in range(MAXC): # chars
                    if cidx < len(word):
                        char = word[cidx] 
                    else:
                        char = '<UNK>'
                    char_idx = self.c2i[char] # char -> idx
                    char_idxs[cidx] = char_idx

                word_char_emb = self.char_emb(char_idxs) # char idxs -> embeddings
                char_embs[sidx,widx] = word_char_emb 
        #print("char embds 4D: ",char_embs.shape)
        
        # collapse dim for character GRU from 4D to 3D, (B x MAXS, MAXC, C)
        char_embs = torch.reshape(char_embs, (B * MAXS, MAXC, self.char_emb_size))
        #print("char embds 3D: ",char_embs.shape)

        # character GRU stuff, hid states , hid states (B x MAXS, C_hid)
        outs_char, h_fw_char, h_bw_char = self.char_birnn(char_embs)
        #print("hidden output char GRU (forward): ", h_fw_char.shape)

        # concat last hid, (B x MAXS, C_hid)
        hid_char_cat = torch.cat((h_fw_char,h_bw_char), dim=1)
        #print("concated cGRU hid: ", hid_char_cat.shape)

        # reshape back to be able to concat char and word lvl tensors
        hid_char_cat = torch.reshape(hid_char_cat, (B, MAXS, 2 * self.char_hidden_size))
        #print("cGRU hid 3D ", hid_char_cat.shape)
        #print("glove word embs 3D: ",word_embs.shape)

        # concat word and character tensors , (B, MAXS, 50 + 2*C_hid)
        concat = torch.cat((word_embs, hid_char_cat), dim=2)
        #print("word+char concat 3D: ",concat.shape)

        # word level GRU 
        outs, h_fw, h_bw = self.word_birnn(concat)

        # linear layer
        return self.final_pred(outs)
        #return torch.zeros((len(x), len(x[0]), 2), requires_grad=True)



#
# MAIN SECTION
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='word2vec embeddings toolkit')
    parser.add_argument('-tr', '--train', default='data/ner_training.csv',
                        help='A comma-separated training file')
    parser.add_argument('-t', '--test', default='data/ner_test.csv',
                        help='A comma-separated test file')
    parser.add_argument('-wv', '--word-vectors', default='glove.6B.50d.txt',
                        help='A txt file with word vectors')
    parser.add_argument('-c', '--char-emb-size', default=16, type=int,
                        help='A size of char embeddings, put 0 to switch off char embeddings')
    parser.add_argument('-cud', '--char-unidirectional', action='store_true')
    parser.add_argument('-wud', '--word-unidirectional', action='store_true')
    parser.add_argument('-lr', '--learning-rate', default=0.002, help='A learning rate')
    parser.add_argument('-e', '--epochs', default=5, type=int, help='Number of epochs')
    args = parser.parse_args()

    print("loading data...")
    training_data = NERDataset(args.train)
    training_loader = DataLoader(training_data, batch_size=128, collate_fn=PadSequence())
    print("setting up model...")

    ner = NERClassifier(
        args.word_vectors,
        char_emb_size=args.char_emb_size,
        char_bidirectional=not args.char_unidirectional,
        word_bidirectional=not args.word_unidirectional
    )


    optimizer = optim.Adam(ner.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    print("training started...")
    # Training
    for epoch in range(args.epochs):
        ner.train()
        for x, y in tqdm(training_loader, desc="Epoch {}".format(epoch + 1)):
            optimizer.zero_grad()
            logits = ner(x)
            logits_shape = logits.shape
            
            loss = criterion(logits.reshape(-1, logits_shape[2]), torch.tensor(y).reshape(-1,))
            loss.backward()
        
            clip_grad_norm_(ner.parameters(), 5)
            optimizer.step()
    
    # Evaluation
    ner.eval()
    confusion_matrix = [[0, 0],
                        [0, 0]]
    test_data = NERDataset(args.test)
    for x, y in test_data:
        pred = torch.argmax(ner([x]), dim=-1).detach().numpy().reshape(-1,)
        y = np.array(y)
        tp = np.sum(pred[y == 1])
        tn = np.sum(1 - pred[y == 0])
        fp = np.sum(1 - y[pred == 1])
        fn = np.sum(y[pred == 0])

        confusion_matrix[0][0] += tn
        confusion_matrix[1][1] += tp
        confusion_matrix[0][1] += fp
        confusion_matrix[1][0] += fn

    table = [['', 'Predicted no name', 'Predicted name'],
             ['Real no name', confusion_matrix[0][0], confusion_matrix[0][1]],
             ['Real name', confusion_matrix[1][0], confusion_matrix[1][1]]]

    t = AsciiTable(table)
    print(t.table)
    print("Accuracy: {}".format(
        round((confusion_matrix[0][0] + confusion_matrix[1][1]) / np.sum(confusion_matrix), 4))
    )
