import random
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from utils import PAD_ID, UNK_ID, SOS_ID, EOS_ID
import numpy as np


class DialogDataset(Dataset):
    def __init__(self, sentences, conversation_length, sentence_length, vocab, data=None):

        # [total_data_size, max_conversation_length, max_sentence_length]
        # tokenized raw text of sentences
        self.sentences = sentences
        self.vocab = vocab

        # conversation length of each batch
        # [total_data_size]
        self.conversation_length = conversation_length

        # list of length of sentences
        # [total_data_size, max_conversation_length]
        self.sentence_length = sentence_length
        self.data = data
        self.len = len(sentences)

    def __getitem__(self, index):
        """Return Single data sentence"""
        # [max_conversation_length, max_sentence_length]
        sentence = self.sentences[index]
        conversation_length = self.conversation_length[index]
        sentence_length = self.sentence_length[index]

        # word => word_ids
        sentence = self.sent2id(sentence)

        return sentence, conversation_length, sentence_length

    def __len__(self):
        return self.len

    def sent2id(self, sentences):
        """word => word id"""
        # [max_conversation_length, max_sentence_length]
        return [self.vocab.sent2id(sentence) for sentence in sentences]


def get_loader(sentences, conversation_length, sentence_length, vocab, batch_size=100, data=None, shuffle=True):
    """Load DataLoader of given DialogDataset"""

    def collate_fn(data):
        """
        Collate list of data in to batch

        Args:
            data: list of tuple(source, target, conversation_length, source_length, target_length)
        Return:
            Batch of each feature
            - source (LongTensor): [batch_size, max_conversation_length, max_source_length]
            - target (LongTensor): [batch_size, max_conversation_length, max_source_length]
            - conversation_length (np.array): [batch_size]
            - source_length (LongTensor): [batch_size, max_conversation_length]
        """
        # Sort by conversation length (descending order) to use 'pack_padded_sequence'
        data.sort(key=lambda x: x[1], reverse=True)

        # Separate
        sentences, conversation_length, sentence_length = zip(*data)

        # return sentences, conversation_length, sentence_length.tolist()
        return sentences, conversation_length, sentence_length

    dataset = DialogDataset(sentences, conversation_length,
                            sentence_length, vocab, data=data)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)

    return data_loader
