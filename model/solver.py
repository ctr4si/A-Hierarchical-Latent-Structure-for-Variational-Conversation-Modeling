from itertools import cycle
import numpy as np
import torch
import torch.nn as nn
import models
from layers import masked_cross_entropy
from utils import to_var, time_desc_decorator, TensorboardWriter, pad_and_pack, normal_kl_div, to_bow, bag_of_words_loss, normal_kl_div, embedding_metric
import os
from tqdm import tqdm
from math import isnan
import re
import math
import pickle
import gensim

word2vec_path = "../datasets/GoogleNews-vectors-negative300.bin"

class Solver(object):
    def __init__(self, config, train_data_loader, eval_data_loader, vocab, is_train=True, model=None):
        self.config = config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.vocab = vocab
        self.is_train = is_train
        self.model = model

    @time_desc_decorator('Build Graph')
    def build(self, cuda=True):

        if self.model is None:
            self.model = getattr(models, self.config.model)(self.config)

            # orthogonal initialiation for hidden weights
            # input gate bias for GRUs
            if self.config.mode == 'train' and self.config.checkpoint is None:
                print('Parameter initiailization')
                for name, param in self.model.named_parameters():
                    if 'weight_hh' in name:
                        print('\t' + name)
                        nn.init.orthogonal_(param)

                    # bias_hh is concatenation of reset, input, new gates
                    # only set the input gate bias to 2.0
                    if 'bias_hh' in name:
                        print('\t' + name)
                        dim = int(param.size(0) / 3)
                        param.data[dim:2 * dim].fill_(2.0)

        if torch.cuda.is_available() and cuda:
            self.model.cuda()

        # Overview Parameters
        print('Model Parameters')
        for name, param in self.model.named_parameters():
            print('\t' + name + '\t', list(param.size()))

        if self.config.checkpoint:
            self.load_model(self.config.checkpoint)

        if self.is_train:
            self.writer = TensorboardWriter(self.config.logdir)
            self.optimizer = self.config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.learning_rate)

    def save_model(self, epoch):
        """Save parameters to checkpoint"""
        ckpt_path = os.path.join(self.config.save_path, f'{epoch}.pkl')
        print(f'Save parameters to {ckpt_path}')
        torch.save(self.model.state_dict(), ckpt_path)

    def load_model(self, checkpoint):
        """Load parameters from checkpoint"""
        print(f'Load parameters from {checkpoint}')
        epoch = re.match(r"[0-9]*", os.path.basename(checkpoint)).group(0)
        self.epoch_i = int(epoch)
        self.model.load_state_dict(torch.load(checkpoint))

    def write_summary(self, epoch_i):
        epoch_loss = getattr(self, 'epoch_loss', None)
        if epoch_loss is not None:
            self.writer.update_loss(
                loss=epoch_loss,
                step_i=epoch_i + 1,
                name='train_loss')

        epoch_recon_loss = getattr(self, 'epoch_recon_loss', None)
        if epoch_recon_loss is not None:
            self.writer.update_loss(
                loss=epoch_recon_loss,
                step_i=epoch_i + 1,
                name='train_recon_loss')

        epoch_kl_div = getattr(self, 'epoch_kl_div', None)
        if epoch_kl_div is not None:
            self.writer.update_loss(
                loss=epoch_kl_div,
                step_i=epoch_i + 1,
                name='train_kl_div')

        kl_mult = getattr(self, 'kl_mult', None)
        if kl_mult is not None:
            self.writer.update_loss(
                loss=kl_mult,
                step_i=epoch_i + 1,
                name='kl_mult')

        epoch_bow_loss = getattr(self, 'epoch_bow_loss', None)
        if epoch_bow_loss is not None:
            self.writer.update_loss(
                loss=epoch_bow_loss,
                step_i=epoch_i + 1,
                name='bow_loss')

        validation_loss = getattr(self, 'validation_loss', None)
        if validation_loss is not None:
            self.writer.update_loss(
                loss=validation_loss,
                step_i=epoch_i + 1,
                name='validation_loss')

    @time_desc_decorator('Training Start!')
    def train(self):
        epoch_loss_history = []
        for epoch_i in range(self.epoch_i, self.config.n_epoch):
            self.epoch_i = epoch_i
            batch_loss_history = []
            self.model.train()
            n_total_words = 0
            for batch_i, (conversations, conversation_length, sentence_length) in enumerate(tqdm(self.train_data_loader, ncols=80)):
                # conversations: (batch_size) list of conversations
                #   conversation: list of sentences
                #   sentence: list of tokens
                # conversation_length: list of int
                # sentence_length: (batch_size) list of conversation list of sentence_lengths

                input_conversations = [conv[:-1] for conv in conversations]
                target_conversations = [conv[1:] for conv in conversations]

                # flatten input and target conversations
                input_sentences = [sent for conv in input_conversations for sent in conv]
                target_sentences = [sent for conv in target_conversations for sent in conv]
                input_sentence_length = [l for len_list in sentence_length for l in len_list[:-1]]
                target_sentence_length = [l for len_list in sentence_length for l in len_list[1:]]
                input_conversation_length = [l - 1 for l in conversation_length]

                input_sentences = to_var(torch.LongTensor(input_sentences))
                target_sentences = to_var(torch.LongTensor(target_sentences))
                input_sentence_length = to_var(torch.LongTensor(input_sentence_length))
                target_sentence_length = to_var(torch.LongTensor(target_sentence_length))
                input_conversation_length = to_var(torch.LongTensor(input_conversation_length))

                # reset gradient
                self.optimizer.zero_grad()

                sentence_logits = self.model(
                    input_sentences,
                    input_sentence_length,
                    input_conversation_length,
                    target_sentences,
                    decode=False)

                batch_loss, n_words = masked_cross_entropy(
                    sentence_logits,
                    target_sentences,
                    target_sentence_length)

                assert not isnan(batch_loss.item())
                batch_loss_history.append(batch_loss.item())
                n_total_words += n_words.item()

                if batch_i % self.config.print_every == 0:
                    tqdm.write(
                        f'Epoch: {epoch_i+1}, iter {batch_i}: loss = {batch_loss.data[0] / n_words.data[0]:.3f}')

                # Back-propagation
                batch_loss.backward()

                # Gradient cliping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)

                # Run optimizer
                self.optimizer.step()

            epoch_loss = np.sum(batch_loss_history) / n_total_words
            epoch_loss_history.append(epoch_loss)
            self.epoch_loss = epoch_loss

            print_str = f'Epoch {epoch_i+1} loss average: {epoch_loss:.3f}'
            print(print_str)

            if epoch_i % self.config.save_every_epoch == 0:
                self.save_model(epoch_i + 1)

            print('\n<Validation>...')
            self.validation_loss = self.evaluate()

            if epoch_i % self.config.plot_every_epoch == 0:
                    self.write_summary(epoch_i)

        self.save_model(self.config.n_epoch)

        return epoch_loss_history

    def generate_sentence(self, input_sentences, input_sentence_length,
                          input_conversation_length, target_sentences):
        self.model.eval()

        # [batch_size, max_seq_len, vocab_size]
        generated_sentences = self.model(
            input_sentences,
            input_sentence_length,
            input_conversation_length,
            target_sentences,
            decode=True)

        # write output to file
        with open(os.path.join(self.config.save_path, 'samples.txt'), 'a') as f:
            f.write(f'<Epoch {self.epoch_i}>\n\n')

            tqdm.write('\n<Samples>')
            for input_sent, target_sent, output_sent in zip(input_sentences, target_sentences, generated_sentences):
                input_sent = self.vocab.decode(input_sent)
                target_sent = self.vocab.decode(target_sent)
                output_sent = '\n'.join([self.vocab.decode(sent) for sent in output_sent])
                s = '\n'.join(['Input sentence: ' + input_sent,
                               'Ground truth: ' + target_sent,
                               'Generated response: ' + output_sent + '\n'])
                f.write(s + '\n')
                print(s)
            print('')

    def evaluate(self):
        self.model.eval()
        batch_loss_history = []
        n_total_words = 0
        for batch_i, (conversations, conversation_length, sentence_length) in enumerate(tqdm(self.eval_data_loader, ncols=80)):
            # conversations: (batch_size) list of conversations
            #   conversation: list of sentences
            #   sentence: list of tokens
            # conversation_length: list of int
            # sentence_length: (batch_size) list of conversation list of sentence_lengths

            input_conversations = [conv[:-1] for conv in conversations]
            target_conversations = [conv[1:] for conv in conversations]

            # flatten input and target conversations
            input_sentences = [sent for conv in input_conversations for sent in conv]
            target_sentences = [sent for conv in target_conversations for sent in conv]
            input_sentence_length = [l for len_list in sentence_length for l in len_list[:-1]]
            target_sentence_length = [l for len_list in sentence_length for l in len_list[1:]]
            input_conversation_length = [l - 1 for l in conversation_length]

            with torch.no_grad():
                input_sentences = to_var(torch.LongTensor(input_sentences))
                target_sentences = to_var(torch.LongTensor(target_sentences))
                input_sentence_length = to_var(torch.LongTensor(input_sentence_length))
                target_sentence_length = to_var(torch.LongTensor(target_sentence_length))
                input_conversation_length = to_var(
                    torch.LongTensor(input_conversation_length))

            if batch_i == 0:
                self.generate_sentence(input_sentences,
                                       input_sentence_length,
                                       input_conversation_length,
                                       target_sentences)

            sentence_logits = self.model(
                input_sentences,
                input_sentence_length,
                input_conversation_length,
                target_sentences)

            batch_loss, n_words = masked_cross_entropy(
                sentence_logits,
                target_sentences,
                target_sentence_length)

            assert not isnan(batch_loss.item())
            batch_loss_history.append(batch_loss.item())
            n_total_words += n_words.item()

        epoch_loss = np.sum(batch_loss_history) / n_total_words

        print_str = f'Validation loss: {epoch_loss:.3f}\n'
        print(print_str)

        return epoch_loss

    def test(self):
        self.model.eval()
        batch_loss_history = []
        n_total_words = 0
        for batch_i, (conversations, conversation_length, sentence_length) in enumerate(tqdm(self.eval_data_loader, ncols=80)):
            # conversations: (batch_size) list of conversations
            #   conversation: list of sentences
            #   sentence: list of tokens
            # conversation_length: list of int
            # sentence_length: (batch_size) list of conversation list of sentence_lengths

            input_conversations = [conv[:-1] for conv in conversations]
            target_conversations = [conv[1:] for conv in conversations]

            # flatten input and target conversations
            input_sentences = [sent for conv in input_conversations for sent in conv]
            target_sentences = [sent for conv in target_conversations for sent in conv]
            input_sentence_length = [l for len_list in sentence_length for l in len_list[:-1]]
            target_sentence_length = [l for len_list in sentence_length for l in len_list[1:]]
            input_conversation_length = [l - 1 for l in conversation_length]

            with torch.no_grad():
                input_sentences = to_var(torch.LongTensor(input_sentences))
                target_sentences = to_var(torch.LongTensor(target_sentences))
                input_sentence_length = to_var(torch.LongTensor(input_sentence_length))
                target_sentence_length = to_var(torch.LongTensor(target_sentence_length))
                input_conversation_length = to_var(torch.LongTensor(input_conversation_length))

            sentence_logits = self.model(
                input_sentences,
                input_sentence_length,
                input_conversation_length,
                target_sentences)

            batch_loss, n_words = masked_cross_entropy(
                sentence_logits,
                target_sentences,
                target_sentence_length)

            assert not isnan(batch_loss.item())
            batch_loss_history.append(batch_loss.item())
            n_total_words += n_words.item()

        epoch_loss = np.sum(batch_loss_history) / n_total_words

        print(f'Number of words: {n_total_words}')
        print(f'Bits per word: {epoch_loss:.3f}')
        word_perplexity = np.exp(epoch_loss)

        print_str = f'Word perplexity : {word_perplexity:.3f}\n'
        print(print_str)

        return word_perplexity

    def embedding_metric(self):
        word2vec =  getattr(self, 'word2vec', None)
        if word2vec is None:
            print('Loading word2vec model')
            word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
            self.word2vec = word2vec
        keys = word2vec.vocab
        self.model.eval()
        n_context = self.config.n_context
        n_sample_step = self.config.n_sample_step
        metric_average_history = []
        metric_extrema_history = []
        metric_greedy_history = []
        context_history = []
        sample_history = []
        n_sent = 0
        n_conv = 0
        for batch_i, (conversations, conversation_length, sentence_length) \
                in enumerate(tqdm(self.eval_data_loader, ncols=80)):
            # conversations: (batch_size) list of conversations
            #   conversation: list of sentences
            #   sentence: list of tokens
            # conversation_length: list of int
            # sentence_length: (batch_size) list of conversation list of sentence_lengths

            conv_indices = [i for i in range(len(conversations)) if len(conversations[i]) >= n_context + n_sample_step]
            context = [c for i in conv_indices for c in [conversations[i][:n_context]]]
            ground_truth = [c for i in conv_indices for c in [conversations[i][n_context:n_context + n_sample_step]]]
            sentence_length = [c for i in conv_indices for c in [sentence_length[i][:n_context]]]

            with torch.no_grad():
                context = to_var(torch.LongTensor(context))
                sentence_length = to_var(torch.LongTensor(sentence_length))

            samples = self.model.generate(context, sentence_length, n_context)

            context = context.data.cpu().numpy().tolist()
            samples = samples.data.cpu().numpy().tolist()
            context_history.append(context)
            sample_history.append(samples)

            samples = [[self.vocab.decode(sent) for sent in c] for c in samples]
            ground_truth = [[self.vocab.decode(sent) for sent in c] for c in ground_truth]

            samples = [sent for c in samples for sent in c]
            ground_truth = [sent for c in ground_truth for sent in c]

            samples = [[word2vec[s] for s in sent.split() if s in keys] for sent in samples]
            ground_truth = [[word2vec[s] for s in sent.split() if s in keys] for sent in ground_truth]

            indices = [i for i, s, g in zip(range(len(samples)), samples, ground_truth) if s != [] and g != []]
            samples = [samples[i] for i in indices]
            ground_truth = [ground_truth[i] for i in indices]
            n = len(samples)
            n_sent += n

            metric_average = embedding_metric(samples, ground_truth, word2vec, 'average')
            metric_extrema = embedding_metric(samples, ground_truth, word2vec, 'extrema')
            metric_greedy = embedding_metric(samples, ground_truth, word2vec, 'greedy')
            metric_average_history.append(metric_average)
            metric_extrema_history.append(metric_extrema)
            metric_greedy_history.append(metric_greedy)

        epoch_average = np.mean(np.concatenate(metric_average_history), axis=0)
        epoch_extrema = np.mean(np.concatenate(metric_extrema_history), axis=0)
        epoch_greedy = np.mean(np.concatenate(metric_greedy_history), axis=0)

        print('n_sentences:', n_sent)
        print_str = f'Metrics - Average: {epoch_average:.3f}, Extrema: {epoch_extrema:.3f}, Greedy: {epoch_greedy:.3f}'
        print(print_str)
        print('\n')

        return epoch_average, epoch_extrema, epoch_greedy


class VariationalSolver(Solver):

    def __init__(self, config, train_data_loader, eval_data_loader, vocab, is_train=True, model=None):
        self.config = config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.vocab = vocab
        self.is_train = is_train
        self.model = model

    @time_desc_decorator('Training Start!')
    def train(self):
        epoch_loss_history = []
        kl_mult = 0.0
        conv_kl_mult = 0.0
        for epoch_i in range(self.epoch_i, self.config.n_epoch):
            self.epoch_i = epoch_i
            batch_loss_history = []
            recon_loss_history = []
            kl_div_history = []
            kl_div_sent_history = []
            kl_div_conv_history = []
            bow_loss_history = []
            self.model.train()
            n_total_words = 0

            self.evaluate()

            for batch_i, (conversations, conversation_length, sentence_length) \
                    in enumerate(tqdm(self.train_data_loader, ncols=80)):
                # conversations: (batch_size) list of conversations
                #   conversation: list of sentences
                #   sentence: list of tokens
                # conversation_length: list of int
                # sentence_length: (batch_size) list of conversation list of sentence_lengths

                target_conversations = [conv[1:] for conv in conversations]

                # flatten input and target conversations
                sentences = [sent for conv in conversations for sent in conv]
                input_conversation_length = [l - 1 for l in conversation_length]
                target_sentences = [sent for conv in target_conversations for sent in conv]
                target_sentence_length = [l for len_list in sentence_length for l in len_list[1:]]
                sentence_length = [l for len_list in sentence_length for l in len_list]

                sentences = to_var(torch.LongTensor(sentences))
                sentence_length = to_var(torch.LongTensor(sentence_length))
                input_conversation_length = to_var(torch.LongTensor(input_conversation_length))
                target_sentences = to_var(torch.LongTensor(target_sentences))
                target_sentence_length = to_var(torch.LongTensor(target_sentence_length))

                # reset gradient
                self.optimizer.zero_grad()

                sentence_logits, kl_div, _, _ = self.model(
                    sentences,
                    sentence_length,
                    input_conversation_length,
                    target_sentences)

                recon_loss, n_words = masked_cross_entropy(
                    sentence_logits,
                    target_sentences,
                    target_sentence_length)

                batch_loss = recon_loss + kl_mult * kl_div
                batch_loss_history.append(batch_loss.item())
                recon_loss_history.append(recon_loss.item())
                kl_div_history.append(kl_div.item())
                n_total_words += n_words.item()

                if self.config.bow:
                    bow_loss = self.model.compute_bow_loss(target_conversations)
                    batch_loss += bow_loss
                    bow_loss_history.append(bow_loss.item())

                assert not isnan(batch_loss.item())

                if batch_i % self.config.print_every == 0:
                    print_str = f'Epoch: {epoch_i+1}, iter {batch_i}: loss = {batch_loss.data[0] / n_words.data[0]:.3f}, recon = {recon_loss.data[0] / n_words.data[0]:.3f}, kl_div = {kl_div.data[0] / n_words.data[0]:.3f}'
                    if self.config.bow:
                        print_str += f', bow_loss = {bow_loss.data[0] / n_words.data[0]:.3f}'
                    tqdm.write(print_str)

                # Back-propagation
                batch_loss.backward()

                # Gradient cliping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)

                # Run optimizer
                self.optimizer.step()
                kl_mult = min(kl_mult + 1.0 / self.config.kl_annealing_iter, 1.0)

            epoch_loss = np.sum(batch_loss_history) / n_total_words
            epoch_loss_history.append(epoch_loss)

            epoch_recon_loss = np.sum(recon_loss_history) / n_total_words
            epoch_kl_div = np.sum(kl_div_history) / n_total_words

            self.kl_mult = kl_mult
            self.epoch_loss = epoch_loss
            self.epoch_recon_loss = epoch_recon_loss
            self.epoch_kl_div = epoch_kl_div

            print_str = f'Epoch {epoch_i+1} loss average: {epoch_loss:.3f}, recon_loss: {epoch_recon_loss:.3f}, kl_div: {epoch_kl_div:.3f}'
            if bow_loss_history:
                self.epoch_bow_loss = np.sum(bow_loss_history) / n_total_words
                print_str += f', bow_loss = {self.epoch_bow_loss:.3f}'
            print(print_str)

            if epoch_i % self.config.save_every_epoch == 0:
                self.save_model(epoch_i + 1)

            print('\n<Validation>...')
            self.validation_loss = self.evaluate()

            if epoch_i % self.config.plot_every_epoch == 0:
                    self.write_summary(epoch_i)

        return epoch_loss_history

    def generate_sentence(self, sentences, sentence_length,
                          input_conversation_length, input_sentences, target_sentences):
        """Generate output of decoder (single batch)"""
        self.model.eval()

        # [batch_size, max_seq_len, vocab_size]
        generated_sentences, _, _, _ = self.model(
            sentences,
            sentence_length,
            input_conversation_length,
            target_sentences,
            decode=True)

        # write output to file
        with open(os.path.join(self.config.save_path, 'samples.txt'), 'a') as f:
            f.write(f'<Epoch {self.epoch_i}>\n\n')

            tqdm.write('\n<Samples>')
            for input_sent, target_sent, output_sent in zip(input_sentences, target_sentences, generated_sentences):
                input_sent = self.vocab.decode(input_sent)
                target_sent = self.vocab.decode(target_sent)
                output_sent = '\n'.join([self.vocab.decode(sent) for sent in output_sent])
                s = '\n'.join(['Input sentence: ' + input_sent,
                               'Ground truth: ' + target_sent,
                               'Generated response: ' + output_sent + '\n'])
                f.write(s + '\n')
                print(s)
            print('')

    def evaluate(self):
        self.model.eval()
        batch_loss_history = []
        recon_loss_history = []
        kl_div_history = []
        bow_loss_history = []
        n_total_words = 0
        for batch_i, (conversations, conversation_length, sentence_length) \
                in enumerate(tqdm(self.eval_data_loader, ncols=80)):
            # conversations: (batch_size) list of conversations
            #   conversation: list of sentences
            #   sentence: list of tokens
            # conversation_length: list of int
            # sentence_length: (batch_size) list of conversation list of sentence_lengths

            target_conversations = [conv[1:] for conv in conversations]

            # flatten input and target conversations
            sentences = [sent for conv in conversations for sent in conv]
            input_conversation_length = [l - 1 for l in conversation_length]
            target_sentences = [sent for conv in target_conversations for sent in conv]
            target_sentence_length = [l for len_list in sentence_length for l in len_list[1:]]
            sentence_length = [l for len_list in sentence_length for l in len_list]

            with torch.no_grad():
                sentences = to_var(torch.LongTensor(sentences))
                sentence_length = to_var(torch.LongTensor(sentence_length))
                input_conversation_length = to_var(
                    torch.LongTensor(input_conversation_length))
                target_sentences = to_var(torch.LongTensor(target_sentences))
                target_sentence_length = to_var(torch.LongTensor(target_sentence_length))

            if batch_i == 0:
                input_conversations = [conv[:-1] for conv in conversations]
                input_sentences = [sent for conv in input_conversations for sent in conv]
                with torch.no_grad():
                    input_sentences = to_var(torch.LongTensor(input_sentences))
                self.generate_sentence(sentences,
                                       sentence_length,
                                       input_conversation_length,
                                       input_sentences,
                                       target_sentences)

            sentence_logits, kl_div, _, _ = self.model(
                sentences,
                sentence_length,
                input_conversation_length,
                target_sentences)

            recon_loss, n_words = masked_cross_entropy(
                sentence_logits,
                target_sentences,
                target_sentence_length)

            batch_loss = recon_loss + kl_div
            if self.config.bow:
                bow_loss = self.model.compute_bow_loss(target_conversations)
                bow_loss_history.append(bow_loss.item())

            assert not isnan(batch_loss.item())
            batch_loss_history.append(batch_loss.item())
            recon_loss_history.append(recon_loss.item())
            kl_div_history.append(kl_div.item())
            n_total_words += n_words.item()

        epoch_loss = np.sum(batch_loss_history) / n_total_words
        epoch_recon_loss = np.sum(recon_loss_history) / n_total_words
        epoch_kl_div = np.sum(kl_div_history) / n_total_words

        print_str = f'Validation loss: {epoch_loss:.3f}, recon_loss: {epoch_recon_loss:.3f}, kl_div: {epoch_kl_div:.3f}'
        if bow_loss_history:
            epoch_bow_loss = np.sum(bow_loss_history) / n_total_words
            print_str += f', bow_loss = {epoch_bow_loss:.3f}'
        print(print_str)
        print('\n')

        return epoch_loss

    def importance_sample(self):
        ''' Perform importance sampling to get tighter bound
        '''
        self.model.eval()
        weight_history = []
        n_total_words = 0
        kl_div_history = []
        for batch_i, (conversations, conversation_length, sentence_length) \
                in enumerate(tqdm(self.eval_data_loader, ncols=80)):
            # conversations: (batch_size) list of conversations
            #   conversation: list of sentences
            #   sentence: list of tokens
            # conversation_length: list of int
            # sentence_length: (batch_size) list of conversation list of sentence_lengths

            target_conversations = [conv[1:] for conv in conversations]

            # flatten input and target conversations
            sentences = [sent for conv in conversations for sent in conv]
            input_conversation_length = [l - 1 for l in conversation_length]
            target_sentences = [sent for conv in target_conversations for sent in conv]
            target_sentence_length = [l for len_list in sentence_length for l in len_list[1:]]
            sentence_length = [l for len_list in sentence_length for l in len_list]

            # n_words += sum([len([word for word in sent if word != PAD_ID]) for sent in target_sentences])
            with torch.no_grad():
                sentences = to_var(torch.LongTensor(sentences))
                sentence_length = to_var(torch.LongTensor(sentence_length))
                input_conversation_length = to_var(
                    torch.LongTensor(input_conversation_length))
                target_sentences = to_var(torch.LongTensor(target_sentences))
                target_sentence_length = to_var(torch.LongTensor(target_sentence_length))

            # treat whole batch as one data sample
            weights = []
            for j in range(self.config.importance_sample):
                sentence_logits, kl_div, log_p_z, log_q_zx = self.model(
                    sentences,
                    sentence_length,
                    input_conversation_length,
                    target_sentences)

                recon_loss, n_words = masked_cross_entropy(
                    sentence_logits,
                    target_sentences,
                    target_sentence_length)

                log_w = (-recon_loss.sum() + log_p_z - log_q_zx).data
                weights.append(log_w)
                if j == 0:
                    n_total_words += n_words.data[0]
                    kl_div_history.append(kl_div.data[0])

            # weights: [n_samples]
            weights = torch.stack(weights, 0)
            m = np.floor(weights.max())
            weights = np.log(torch.exp(weights - m).sum())
            weights = m + weights - np.log(self.config.importance_sample)
            weight_history.append(weights)

        print(f'Number of words: {n_total_words}')
        bits_per_word = -np.sum(weight_history) / n_total_words
        print(f'Bits per word: {bits_per_word:.3f}')
        word_perplexity = np.exp(bits_per_word)

        epoch_kl_div = np.sum(kl_div_history) / n_total_words

        print_str = f'Word perplexity upperbound using {self.config.importance_sample} importance samples: {word_perplexity:.3f}, kl_div: {epoch_kl_div:.3f}\n'
        print(print_str)

        return word_perplexity
