from solver import *
from data_loader import get_loader
from configs import get_config
from utils import Vocab
import os
import pickle
from models import VariationalModels

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    config = get_config(mode='train')
    val_config = get_config(mode='valid')
    print(config)
    with open(os.path.join(config.save_path, 'config.txt'), 'w') as f:
        print(config, file=f)

    print('Loading Vocabulary...')
    vocab = Vocab()
    vocab.load(config.word2id_path, config.id2word_path)
    print(f'Vocabulary size: {vocab.vocab_size}')

    config.vocab_size = vocab.vocab_size

    train_data_loader = get_loader(
        sentences=load_pickle(config.sentences_path),
        conversation_length=load_pickle(config.conversation_length_path),
        sentence_length=load_pickle(config.sentence_length_path),
        vocab=vocab,
        batch_size=config.batch_size)

    eval_data_loader = get_loader(
        sentences=load_pickle(val_config.sentences_path),
        conversation_length=load_pickle(val_config.conversation_length_path),
        sentence_length=load_pickle(val_config.sentence_length_path),
        vocab=vocab,
        batch_size=val_config.eval_batch_size,
        shuffle=False)

    # for testing
    # train_data_loader = eval_data_loader
    if config.model in VariationalModels:
        solver = VariationalSolver
    else:
        solver = Solver

    solver = solver(config, train_data_loader, eval_data_loader, vocab=vocab, is_train=True)

    solver.build()
    solver.train()
