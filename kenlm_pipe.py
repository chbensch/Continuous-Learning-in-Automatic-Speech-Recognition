import torch
import kenlm
import ctcdecode
import pickle
import numpy as np

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
from ctcdecode.prefix import State
from datasets import load_from_disk, load_dataset

results51 = load_from_disk("res51_full_log")

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

model = Wav2Vec2ForCTC.from_pretrained("./wav2vec2-large-xlsr-ger-chris-cut/checkpoint-51000")

vocab_dict = tokenizer.get_vocab()
sort_vocab = sorted((value, key) for (key,value) in vocab_dict.items())
vocab = [x[1].replace("|", " ") if x[1] not in tokenizer.all_special_tokens else "_" for x in sort_vocab]

vocabulary = vocab
alpha = 2.5 # LM Weight
beta = 0.0 # LM Usage Reward
word_lm_scorer = ctcdecode.WordKenLMScorer('lm_data/train.arpa', alpha, beta) # use your own kenlm model

def get_pruned_vocab_indices(log_probs):
    """ Return vocab indices of pruned probabilities of a time step. """

    index_to_prob = [(k, log_probs[k]) for k in range(log_probs.shape[0])]
    index_to_prob = sorted(index_to_prob, key=lambda x: x[1], reverse=True)

    if 40 < len(index_to_prob):
        index_to_prob = index_to_prob[:40]

    if np.log(0.000001) < 1.0:
        filtered = []
        for x in index_to_prob:
            if x[1] >= np.log(0.000001):
                filtered.append(x)
        index_to_prob = filtered

    return [x[0] for x in index_to_prob]

def decode(probs):
    # Num time steps
    nT = probs.shape[0]

    # Initialize prefixes
    prefixes = State(
        scorers=[word_lm_scorer],
        size=128
    )

    # Iterate over timesteps
    for t in range(nT):
        step_probs = probs[t]
        pruned_step_probs = get_pruned_vocab_indices(step_probs)

        # Iterate over symbols
        for v in pruned_step_probs:
            symbol = vocabulary[v]
            symbol_prob = step_probs[v]

            # Iterate over prefixes
            for prefix in prefixes:

                # If there is a blank, we extend the existing prefix
                if symbol == '_':
                    prefix.add_p_blank(symbol_prob + prefix.score)

                else:

                    # If the last symbol is repeated
                    # update the existing prefix
                    if symbol == prefix.symbol:
                        p = symbol_prob + prefix.p_non_blank_prev
                        prefix.add_p_non_blank(p)

                    new_prefix = prefixes.get_prefix(prefix, symbol)

                    if new_prefix is not None:
                        p = -np.inf

                        if symbol == prefix.symbol and \
                                prefix.p_blank_prev > -np.inf:
                            p = prefix.p_blank_prev + symbol_prob

                        elif prefix.symbol != symbol:
                            p = prefix.score + symbol_prob

                        new_prefix.add_p_non_blank(p)

        prefixes.step()

    prefixes.finalize()

    return prefixes.best()


def infer_lm(batch):

    batch["lm_str"] = decode(np.asarray(batch["lm_raw"]))
    return batch

results51_lm = results51.map(infer_lm, num_proc=4)

results51_lm.save_to_disk("res51_lm")