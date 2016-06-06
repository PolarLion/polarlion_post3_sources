#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import argparse
import cPickle
import traceback
import logging
import time
import sys

import numpy

import experiments.nmt
from experiments.nmt import\
    RNNEncoderDecoder,\
    prototype_state,\
    parse_input

from experiments.nmt.numpy_compat import argpartition

logger = logging.getLogger(__name__)

class Timer(object):

    def __init__(self):
        self.total = 0

    def start(self):
        self.start_time = time.time()

    def finish(self):
        self.total += time.time() - self.start_time

class BeamSearch(object):

    def __init__(self, enc_dec):
        self.enc_dec = enc_dec
        state = self.enc_dec.state
        self.eos_id = state['null_sym_target']
        self.unk_id = state['unk_sym_target']

    def compile(self):
        self.comp_repr = self.enc_dec.create_representation_computer()
        self.comp_init_states = self.enc_dec.create_initializers()
        self.comp_next_probs = self.enc_dec.create_next_probs_computer()
        self.comp_next_states = self.enc_dec.create_next_states_computer()

    def search(self, seq, n_samples, ignore_unk=False, minlen=1):
        # n_samples = 6
        c = self.comp_repr(seq)[0]
        states = map(lambda x : x[None, :], self.comp_init_states(c))
        dim = states[0].shape[1]

        num_levels = len(states)

        fin_trans = []
        fin_costs = []
        fin_trace = []
        alignment = []
        trans_trace =[]
        for i in range(n_samples):
            trans_trace.append([])

        trans = [[]]
        costs = [0.0]

        for k in range(3 * len(seq)):
            if n_samples == 0:
                break

            # Compute probabilities of the next words for
            # all the elements of the beam.
            beam_size = len(trans)
            last_words = (numpy.array(map(lambda t : t[-1], trans))
                    if k > 0
                    else numpy.zeros(beam_size, dtype="int64"))
            next_probs = self.comp_next_probs(c, k, last_words, *states)
            # log_probs = numpy.log(self.comp_next_probs(c, k, last_words, *states)[0])
            log_probs = numpy.log(next_probs[0])

            alignment.append(next_probs[1])


            # Adjust log probs according to search restrictions
            if ignore_unk:
                log_probs[:,self.unk_id] = -numpy.inf
            # TODO: report me in the paper!!!
            if k < minlen:
                log_probs[:,self.eos_id] = -numpy.inf

            # Find the best options by calling argpartition of flatten array
            next_costs = numpy.array(costs)[:, None] - log_probs
            flat_next_costs = next_costs.flatten()
            best_costs_indices = argpartition(
                    flat_next_costs.flatten(),
                    n_samples)[:n_samples]

            # Decypher flatten indices
            voc_size = log_probs.shape[1]
            trans_indices = best_costs_indices / voc_size
            word_indices = best_costs_indices % voc_size

            print "trans_indices", trans_indices

            costs = flat_next_costs[best_costs_indices]

            # Form a beam for the next iteration
            new_trans = [[]] * n_samples
            new_trans_trace = [[]] * n_samples
            new_costs = numpy.zeros(n_samples)
            new_states = [numpy.zeros((n_samples, dim), dtype="float32") for level
                    in range(num_levels)]
            inputs = numpy.zeros(n_samples, dtype="int64")
            for i, (orig_idx, next_word, next_cost) in enumerate(
                    zip(trans_indices, word_indices, costs)):
                new_trans[i] = trans[orig_idx] + [next_word]
                new_trans_trace[i] = trans_trace[orig_idx] + [i] 

                new_costs[i] = next_cost
                for level in range(num_levels):
                    new_states[level][i] = states[level][orig_idx]
                inputs[i] = next_word
            new_states = self.comp_next_states(c, k, inputs, *new_states)

            print "new_trans_trace", new_trans_trace

            # Filter the sequences that end with end-of-sequence character
            trans = []
            trans_trace = []
            costs = []
            indices = []
            for i in range(n_samples):
                if new_trans[i][-1] != self.enc_dec.state['null_sym_target']:
                    trans.append(new_trans[i])
                    costs.append(new_costs[i])
                    indices.append(i)
                    print i, "trans ", new_trans[i]
                    trans_trace.append(new_trans_trace[i])

                else:
                    n_samples -= 1
                    fin_trans.append(new_trans[i])
                    fin_costs.append(new_costs[i])
                    print i, "fin trans ", new_trans[i]
                    fin_trace.append(new_trans_trace[i])

            states = map(lambda x : x[indices], new_states)

        # Dirty tricks to obtain any translation
        if not len(fin_trans):
            if ignore_unk:
                logger.warning("Did not manage without UNK")
                return self.search(seq, n_samples, False, minlen)
            elif n_samples < 500:
                logger.warning("Still no translations: try beam size {}".format(n_samples * 2))
                return self.search(seq, n_samples * 2, False, minlen)
            else:
                logger.error("Translation failed")

        fin_trace = numpy.array(fin_trace)[numpy.argsort(fin_costs)]
        fin_trans = numpy.array(fin_trans)[numpy.argsort(fin_costs)]
        fin_costs = numpy.array(sorted(fin_costs))
        fin_aligment = []
        for i in range(len(fin_trace[0])):
            if i == 0:
                print alignment[i][0]
                fin_aligment.append(list(alignment[i][0]))
            else:
                print alignment[i][fin_trace[0][i]]
                fin_aligment.append(list(alignment[i][fin_trace[0][i]]))
        return fin_trans, fin_costs, fin_aligment

def indices_to_words(i2w, seq):
    sen = []
    for k in xrange(len(seq)):
        if i2w[seq[k]] == '<eol>':
            break
        sen.append(i2w[seq[k]])
    return sen

def sample(lm_model, seq, n_samples, ampler=None, beam_search=None, ignore_unk=False, normalize=False, alpha=1, verbose=False):
    if beam_search:
        sentences = []
        trans, costs, alignment = beam_search.search(seq, n_samples,
                ignore_unk=ignore_unk, minlen=len(seq) / 2)
        if normalize:
            counts = [len(s) for s in trans]
            costs = [co / cn for co, cn in zip(costs, counts)]
        for i in range(len(trans)):
            sen = indices_to_words(lm_model.word_indxs, trans[i])
            # sentences.append(" ".join(sen))
            sentences.append(sen)
        for i in range(len(costs)):
            if verbose:
                print "{}: {}".format(costs[i], sentences[i])
        return sentences, costs, alignment




languages_set = set(['zh-en', 'zh-zh', 'en-zh'])
lm_models = {}
states = {}
enc_dec = {}
beam_search = {}
indx_word = {}

for languages in languages_set:
  model_path = "static/"+languages+"/search_model.npz"
  state_file = "static/"+languages+"/search_state.pkl"
  states[languages] = prototype_state()
  with open(state_file) as src:
    states[languages].update(cPickle.load(src))
  states[languages].update(eval("dict({})".format("")))

  logging.basicConfig(level=getattr(logging, states[languages]['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

  rng = numpy.random.RandomState(states[languages]['seed'])
  enc_dec[languages] = RNNEncoderDecoder(states[languages], rng, skip_init=True)
  enc_dec[languages].build()

  lm_models[languages] =  enc_dec[languages].create_lm_model()
  lm_models[languages].load(model_path)
  indx_word[languages] = cPickle.load(open(states[languages]['word_indx'],'rb'))
  beam_search[languages] = None
  beam_search[languages] = BeamSearch(enc_dec[languages])
  beam_search[languages].compile()

  idict_src = cPickle.load(open(states[languages]['indx_word'],'r'))


def translate(languages, source):
    print "translate/", languages+"/"+source+"/"
    # source = "哈 哈 哈"
    # languages = "zh-en"
    start_time = time.time()

    # n_samples = args.beam_size
    n_samples = 12
    total_cost = 0.0
    logging.debug("Beam size: {}".format(n_samples))
    seqin = source.strip()
    print source
    seq, parsed_in = parse_input(states[languages], indx_word[languages], seqin, idx2word=idict_src)
    trans, costs, alignment = sample(lm_models[languages], seq, n_samples, beam_search=beam_search[languages], ignore_unk=False, normalize=False)
    best = numpy.argmin(costs)
    print type(trans[best])
    return trans[best], alignment

if __name__ == "__main__":
    print translate( "zh-en","我 的 家 在 沈阳")
