import sys
import os
from collections import defaultdict
import copy
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
#import mkl
import time

import random
import ast

from Asgd import MyAsgd
import character_encoders as enc
from dropout import Dropout
import discodop_eval
from state import State
import corpus_reader
import tree as T

from token_encoder import StackedSupertagTokenEncoder
from layernorm import LayerNormalization
from neuralmodels import FeedForward

import multi_data_loader

import depccg_util as depccg

from utils import batch_stochastic_replacement, save_dict, load_dict, sentence_to_tensors

L1 = 0 #1e-8
L2 = 0 #1e-7
def mem(device):
    #return True
    v1 = torch.cuda.memory_allocated()
    v2 = torch.cuda.max_memory_allocated()
    return "{}\t{}\t{}\t{}".format(v1, v2, round(v1/1024), round(v2/1024))

def print_data_on_memory_size():
    l = np.array(State.memory_sizes)
    State.memory_sizes = None
    print("Num data points:", len(l))
    print("Mean size of mem:", np.mean(l))
    print("Max size of mem:", np.max(l))
    print("Std:", np.std(l))
    print("Distribution:")
    d = defaultdict(int)
    for i in l:
        d[i] += 1
    print("Size N %")
    cum = 0
    for k, v in sorted(d.items()):
        cum += v / len(l) * 100
        print("{}\t{}\t{:.2f}\t{}".format(k, v, v / len(l) * 100, cum))


class Structure(nn.Module):
    """Structure classifier batches """
    def __init__(self, input_size, dropout, hidden, atn, layernorm, final_bias, num_hidden, activation):
        super(Structure, self).__init__()
        self.input_size = input_size
        input_dim = input_size
        if atn:
            input_dim = (input_size // 2) * 3
        
        # new:
        self.structure = FeedForward(input_dim, hidden, 1, dropout, final_bias, activation, layernorm, num_hidden)
        self.softmax = nn.LogSoftmax(dim=0)
        self.atn = atn

    def forward(self, input_list):
        if type(input_list) == list:
            lengths = [len(input) for input in input_list]
            padded = torch.cat(input_list, dim=0)
            out_layer = self.structure(padded)
            split_out = out_layer.split(lengths)
            return [self.softmax(out).view(1, -1) for out in split_out]
        out_layer = self.structure(input_list)
        return self.softmax(out_layer).view(1, -1)
        

class Transducer(nn.Module):
    #memory_sizes = []
    """Stores sentence level bi-lstm and all other paramters of parser (struct and label classifiers)"""
    def __init__(self, args, depth, char_voc_size, word_voc_size, num_labels, num_tag_labels, num_task_labels, words2tensors, supertag_num = None):
        super(Transducer, self).__init__()
        
        act_fun_dict = {"tanh" : nn.Tanh, "ReLU" : nn.ReLU, "sigmoid" : nn.Sigmoid}
        activation = act_fun_dict[args.a]

        layernorm           = bool(args.L)
        initial_transform   = bool(args.it)
        final_bias_parser   = bool(args.pb)
        final_bias_tagger    = bool(args.tb)

        self.token_encoder = StackedSupertagTokenEncoder(   depth           = depth, 
                                                            dim_char_emb    = args.c,
                                                            dim_char_lstm   = args.C,
                                                            dim_word_emb    = args.w,
                                                            dim_lstm_stack  = args.W, 
                                                            char_voc_size   = char_voc_size, 
                                                            word_voc_size   = word_voc_size,
                                                            words2tensors   = words2tensors, 
                                                            emb_init        = args.I,
                                                            drop_lstm_in    = args.Q, 
                                                            drop_char_emb   = args.K,
                                                            supertag_num    = supertag_num, 
                                                            dim_supertag    = args.sup, 
                                                            drop_supertag   = args.Y,
                                                            residual_add    = args.Ra,
                                                            residual_gated  = args.Rg,
                                                            residual_add_gated  = args.Rga,
                                                            vardrop_i           = args.vi,
                                                            vardrop_h           = args.vh,
                                                            layernorm           = layernorm,
                                                            initial_transform   = initial_transform)

        # Structural actions
        self.structure = Structure( input_size  = args.W*8, 
                                    dropout     = args.D,
                                    hidden      = args.H, 
                                    atn         = args.A, 
                                    layernorm   = layernorm, 
                                    final_bias  = final_bias_parser, 
                                    num_hidden  = args.ph, 
                                    activation  = activation)
 
        self.label = nn.Sequential(
                        FeedForward(
                            d_in        = args.W*4, 
                            d_hid       = args.H, 
                            d_out       = num_labels, 
                            drop_in     = args.D, 
                            final_bias  = final_bias_parser, 
                            activation  = activation,
                            layer_norm  = layernorm,
                            n_hid       = args.ph),
                        nn.LogSoftmax(dim=1)
                    )

        tagger_args_dict = {"d_in"          : args.W, 
                            "d_hid"         : args.H, 
                            "drop_in"       : args.X,
                            "final_bias"    : final_bias_tagger,
                            "activation"    : activation,
                            "layer_norm"    : layernorm,
                            "n_hid"         : args.th}
        
        self.tagger = nn.Sequential(
                        FeedForward(d_out = num_tag_labels, 
                                    **tagger_args_dict),
                        nn.LogSoftmax(dim=1))
        # AUX tagging
        self.aux_taggers = nn.ModuleDict({task : nn.Sequential(
                                                    FeedForward(d_out = label_num, 
                                                                **tagger_args_dict),
                                                    nn.LogSoftmax(dim=1))
                                                for task, label_num in num_task_labels.items()
                                        })
        
        self.tagger_loss = nn.NLLLoss(reduction="sum")

        self.initialize_parameters()

    def initialize_parameters(self):
        # Xavier initialization for every layer
        # uniform initialization for embeddings
        for mod in [self.structure, self.label, self.tagger] + list(self.aux_taggers.values()):
            for p in mod.parameters():
                n_ids = sum(p.data.shape)
                m = (6 / n_ids)**0.5
                p.data.uniform_(-m, m)

    def forward(self, sentence, depth = -1, batch = False, parsing = False, supertags = None):

        output_list = self.token_encoder(sentence, depth, batch, parsing, supertags)

        if parsing:
            return output_list[1], output_list[-1]
        else:
            return output_list[-1], None
        

def get_vocabulary(corpus, *other_token_lists):
    """Extract vocabulary for characters, tokens, non-terminals, POS tags"""
    words = defaultdict(int)

    chars = defaultdict(int)
    chars["<START>"] += 2
    chars["<STOP>"] += 2

    # These should be treated as single characters
    chars["-LRB-"] += 2
    chars["-RRB-"] += 2
    chars["#LRB#"] += 2
    chars["#RRB#"] += 2

    tag_set = defaultdict(int)

    label_set = defaultdict(int)
    for tree in corpus:
        tokens = T.get_yield(tree)
        for tok in tokens:
            for char in tok.token:
                chars[char] += 1

            tag_set[tok.get_tag()] += 1

            words[tok.token] += 1

        constituents = T.get_constituents(tree)
        for label, _ in constituents:
            label_set[label] += 1

    for l in other_token_lists:
        for sentence in l:
            for word in sentence:
                words[word] += 1

            for char in word:
                chars[char] += 1

    return words, chars, label_set, tag_set


def set2tensor(device, iset, add=None, embeddings=None):
    # Returns a representation of a constituent, i.e. set of indexes as either
    # - a tensor of token indexes
    # - a tensor of token context-aware embeddings
    shift = 4     # shift bc first indexes of embedding matrix are reserved for additional parameters
    mini = min(iset)
    maxi = max(iset)
    gapi = {i for i in range(mini, maxi+1) if i not in iset}
    mingapi = 2
    maxgapi = 3
    if len(gapi) > 0:
        mingapi = min(gapi) + shift
        maxgapi = max(gapi) + shift
    
    iset = [mini+shift, maxi+shift, mingapi, maxgapi]

    # Deactivated for now
    if False and add is not None:
        iset.append(add)
    input_set_tensors = torch.tensor(iset, dtype=torch.long, device=device)
    if embeddings is None:
        return input_set_tensors
    return embeddings[input_set_tensors].view(-1)

def embed_and_parse_one(device, model, i2labels, i2tags, sentence, sentence_tensors, supertags = None):
    # Parse single sentence
    with torch.no_grad():
        state = State(sentence)
        embeddings_l1, embeddings_l2 = model(sentence_tensors, depth = -1, parsing = True, supertags = supertags)
        return parse(device, model, i2labels, i2tags, sentence, state, embeddings_l1, embeddings_l2)

def embed_and_parse_batch(device, model, i2labels, i2tags, sentences, sentences_tensors, supertags = None):
    # Parse batch of sentences
    with torch.no_grad():
        trees = []
        embeddings_l1, embeddings_l2 = model(sentences_tensors, depth = -1, batch=True, parsing = True, supertags = supertags)
        for el1, el2, sent in zip(embeddings_l1, embeddings_l2, sentences):
            state = State(sent)
            trees.append(parse(device, model, i2labels, i2tags, sent, state, el1, el2))
        return trees

def embed_and_extract_dyn_oracle(device, model, labels2i, i2labels, sentences, sentences_tensors, gold_constituents, supertags = None):
    with torch.no_grad():
        oracles = []
        _, embeddings_l2 = model(sentences_tensors, depth = -1, batch=True, parsing = True, supertags = supertags)
        for el2, sent, gconst in zip(embeddings_l2, sentences, gold_constituents):
            state = State(sent)
            oracles.append(extract_dyn_oracle(device, model, labels2i, i2labels, sent, state, el2, gconst))
        return oracles

def extract_dyn_oracle_from_corpus(device, model, labels2i, i2labels, sentences_copy, tensors, gold_constituents, p, supertags = None):
    # keep training example with probability p
    ps = np.random.rand(len(gold_constituents)) > (1-p)
    idxes = np.arange(len(gold_constituents))
    idxes = idxes[ps]

    sentences = [sentences_copy[i] for i in idxes]
    tensors = [tensors[i] for i in idxes]
    gold_constituents = [gold_constituents[i] for i in idxes]

    if supertags is None:
        idxes, sentences, tensors, gold_constituents = zip(*sorted(zip(
                                    idxes,
                                    sentences, 
                                    tensors,
                                    gold_constituents), 
                                   key = lambda x: len(x[1]),
                                   reverse=True))

        gold_constituents = [{tuple_idx:label for label, tuple_idx in gc} for gc in gold_constituents]

        oracles = []
        batch_size=100
        for i in range(0, len(sentences), batch_size):
            oracles.extend(embed_and_extract_dyn_oracle(device,
                                                         model, 
                                                         labels2i,
                                                         i2labels,
                                                         sentences[i:i+batch_size],
                                                         tensors[i:i+batch_size],
                                                         gold_constituents[i:i+batch_size]))
    else:
        supertags = [supertags[i] for i in idxes]

        idxes, sentences, tensors, gold_constituents, supertags = zip(*sorted(zip(
                                    idxes,
                                    sentences, 
                                    tensors,
                                    gold_constituents,
                                    supertags), 
                                   key = lambda x: len(x[1]),
                                   reverse=True))

        gold_constituents = [{tuple_idx:label for label, tuple_idx in gc} for gc in gold_constituents]

        oracles = []
        batch_size=100
        for i in range(0, len(sentences), batch_size):
            oracles.extend(embed_and_extract_dyn_oracle(device,
                                                         model, 
                                                         labels2i,
                                                         i2labels,
                                                         sentences[i:i+batch_size],
                                                         tensors[i:i+batch_size],
                                                         gold_constituents[i:i+batch_size],
                                                         supertags[i:i+batch_size]))
    
    assert(len(idxes) == len(oracles))
    return list(zip(idxes, oracles))


def extract_dyn_oracle(device, model, labels2i, i2labels, sentence, state, embeddings_l2, gold_constituents):

    oinput_struct  = []
    ooutput_struct = []
    oinput_labels  = []
    ooutput_labels = []

    #return embeddings[input_set_tensors].view(-1)

    while not state.is_final():
        next_type = state.next_action_type()
        if next_type == State.STRUCT:
            memory, focus, buf = state.get_structural_step_input()
            if focus is None or len(memory) == 0:
                state.shift()
                continue

            # append input
            focus_tensor = set2tensor(device, focus)
            memory_tensor = [torch.cat([set2tensor(device,
                                                input_set,
                                                add=0),
                                        focus_tensor])
                             for input_set in memory]
            if buf is not None:
                memory_tensor.append(torch.cat([set2tensor(device, buf, add=1), focus_tensor]))

            #mem = torch.stack(memory_tensor)
            #foc = focus_tensor
            struct_input = torch.stack(memory_tensor)
            oinput_struct.append(struct_input)

            em_input = embeddings_l2[struct_input].view(len(memory_tensor), -1)
            log_probs = model.structure(em_input).view(-1)

            
            best_action = state.dyn_oracle(gold_constituents)

            if best_action[0] == "combine":
                target = best_action[1]
            else:
                target = len(memory_tensor) - 1
            ooutput_struct.append(target)

            probs = torch.exp(log_probs)
            action_id = np.random.choice(len(memory_tensor), p=probs.cpu().numpy())
            if action_id == len(memory):
                state.shift()
            else:
                state.combine(action_id)
        else:
            assert(next_type == State.LABEL)
            input_set = state.get_labelling_step_input()
            #input_tensor = set2tensor(device, input_set, embeddings=embeddings_l2)
            input_tensor = set2tensor(device, input_set)

            oinput_labels.append(input_tensor)
            best_action = state.dyn_oracle(gold_constituents)
            ooutput_labels.append(labels2i[best_action[1]])

            em_input = embeddings_l2[input_tensor].view(1, -1)

            log_probs = model.label(em_input).view(-1)

            if state.is_prefinal(): # Forbid no-label for the constituent that spans the whole sentence (need a root label)
                prediction = torch.argmax(log_probs[1:]) + 1
            else:
                prediction = torch.argmax(log_probs)

            # For dynamic oracle training: the labelling action does not matter
            # (bc no error propagation from labels)
            if prediction == 0:
                state.nolabel()
            else:
                prediction_str = i2labels[prediction]
                state.labelX(prediction_str)

    r1, r2 = None, None
    if len(oinput_struct) > 0:
        #r1 = torch.stack(struct_input)
        #r2 = torch.cat(struct_output)
        r1 = oinput_struct
        r2 = torch.tensor(ooutput_struct, dtype=torch.long, device=device).view(-1, 1)
    r3 = torch.stack(oinput_labels)
    r4 = torch.tensor(ooutput_labels, dtype=torch.long, device=device)

    return r1, r2, r3, r4


def parse(device, model, i2labels, i2tags, sentence, state, embeddings_l1, embeddings_l2):
    # Predict POS tags and parse, returns a tree
    # Embeddings_l1, embeddings_l2: output of sentence level bi-LSTM

    # predict and assign pos tags
    tag_scores = model.tagger(embeddings_l1)
    tag_predictions = torch.argmax(tag_scores, dim=1).cpu().numpy()
    for tag, tok in zip(tag_predictions, sentence):
        tok.set_tag(i2tags[tag])

    # Parsing
    while not state.is_final():
        next_type = state.next_action_type()
        if next_type == State.STRUCT:
            memory, focus, buf = state.get_structural_step_input()
            #model.memory_sizes.append(len(memory))
            if focus is None or len(memory) == 0:
                state.shift()
                continue

            focus_tensor = set2tensor(device, focus, embeddings=embeddings_l2)
            memory_tensor = [set2tensor(device,
                                        input_set,
                                        add=0, 
                                        embeddings=embeddings_l2)
                             for input_set in memory]
            if buf is not None:
                memory_tensor.append(set2tensor(device, buf, add=1, embeddings=embeddings_l2))
            mem = torch.stack(memory_tensor)
            foc = focus_tensor
            struct_input = torch.cat([mem, foc.repeat(len(memory_tensor), 1)], dim=1)

            log_probs = model.structure(struct_input).view(-1)
            action_id = torch.argmax(log_probs)
            if action_id == len(memory):
                state.shift()
            else:
                state.combine(action_id)
        else:
            assert(next_type == State.LABEL)
            input_set = state.get_labelling_step_input()
            input_tensor = set2tensor(device, input_set, embeddings=embeddings_l2)

            log_probs = model.label(input_tensor.unsqueeze(0)).view(-1)

            if state.is_prefinal(): # Forbid no-label for the constituent that spans the whole sentence
                prediction = torch.argmax(log_probs[1:]) + 1
            else:
                prediction = torch.argmax(log_probs)

            if prediction == 0:
                state.nolabel()
            else:
                prediction_str = i2labels[prediction]
                state.labelX(prediction_str)
    return state.get_tree()


def train_sentence_batch(device, model, optimizer, sentence_tensors, features, batch=False, dynamic=None, supertags = None):
    # Computes the parsing loss for a sentence
    # in batch mode (batch all structure and label predictions)
    # args:
    #   sentence_tensors: representations of chars and tokens in sentence
    #   features: input representation for each action
    #   batch: True if the input consists of several sentences
    # returns:
    #   loss: the negative log likelihood loss node

    optimizer.zero_grad()
    if not batch:
        struct_input, struct_output, labels_input, labels_output = features
        if dynamic is not None:
            struct_input, struct_output, labels_input, labels_output = dynamic
#            ds_input, ds_output, dl_input, dl_output = dynamic
#            if ds_input is not None:
#                struct_input = struct_input + ds_input
#                struct_output = torch.cat([struct_output, ds_output])
#            labels_input = torch.cat([labels_input, dl_input])
#            labels_output = torch.cat([labels_output, dl_output])

        input_tensors = sentence_tensors
        _, embeddings = model(input_tensors, depth = -1, parsing = True, supertags = supertags)

        #labels_input_tensors = embeddings[labels_input].view(len(labels_input), -1)
        labels_input_tensors = F.embedding(labels_input, embeddings).view(len(labels_input), -1)

        labels_output_tensors = model.label(labels_input_tensors)
        loss = F.nll_loss(labels_output_tensors, labels_output, reduction="sum")

        if struct_input is not None:
            # ++ 
            #struct_input_tensors = [embeddings[str_in].view(len(str_in), -1) for str_in in struct_input]
            struct_input_tensors = [F.embedding(str_in, embeddings).view(len(str_in), -1) for str_in in struct_input]
            # ++ 
            
            struct_output_tensors = model.structure(struct_input_tensors)
            for i, so in enumerate(struct_output_tensors):
                loss += F.nll_loss(so, struct_output[i], reduction="sum")
        # --

        for param in model.parameters():
           loss += L1 * param.abs().sum() + L2 * (param**2).sum()

        loss.backward()
        return loss.float().detach()
    else:
        input_tensors = sentence_tensors

        _, embeddings = model(input_tensors, depth = -1, batch=True, parsing = True, supertags = supertags)

        # construct batch input / output for each sentence
        batch_label_input = []
        batch_label_output = []
        batch_struct_input = []
        batch_struct_output = []
        for i, (e, feat) in enumerate(zip(embeddings, features)): # iterates on output of bi-lstm for each sentence
            if dynamic is None:
                struct_input, struct_output, labels_input, labels_output = feat
                batch_label_input.append(e[labels_input].view(len(labels_input), -1))
                batch_label_output.append(labels_output)
                if struct_input is not None:
                    batch_struct_input.extend([e[str_in].view(len(str_in), -1) for str_in in struct_input])
                    batch_struct_output.append(struct_output)
            #if dynamic is not None:
            else:
                ds_input, ds_output, dl_input, dl_output = dynamic
                batch_label_input.append(e[dl_input].view(len(dl_input), -1))
                batch_label_output.append(dl_output)
                if ds_input is not None:
                    batch_struct_input.extend([e[str_in].view(len(str_in), -1) for str_in in ds_input])
                    batch_struct_output.append(ds_output)

        batch_label_input = torch.cat(batch_label_input, dim=0)
        batch_label_output = torch.cat(batch_label_output)

        labels_output_tensors = model.label(batch_label_input)
        loss = F.nll_loss(labels_output_tensors, batch_label_output, reduction="sum")

        if len(batch_struct_input) > 0:
            #batch_struct_output = torch.cat(batch_struct_output)
            struct_output_tensors = model.structure(batch_struct_input)
            struct_targets = torch.cat(batch_struct_output, dim=0)
            for i, so in enumerate(struct_output_tensors):
                loss += F.nll_loss(so, struct_targets[i], reduction="sum")

        loss /= len(features)

        for param in model.parameters():
           loss += L1 * param.abs().sum() + L2 * (param**2).sum()
           

        loss.backward()
        return loss.float().detach()



def train_generic_tagging(device, model, optimizer, sentence_tensors, tags, depth, task, batch=False):
    # Returns loss node for tagging
    optimizer.zero_grad()
    if not batch:
        input_tensors = sentence_tensors
        embeddings, _ = model(input_tensors, depth = depth)
        output = model.aux_taggers[task](embeddings)
        loss = torch.sum(model.tagger_loss(output, tags))
        
        for param in model.parameters():
           loss += L1 * param.abs().sum() + L2 * (param**2).sum()
        loss.backward()
        return loss.float().detach()
    else:
        input_tensors = sentence_tensors
        embeddings, _ = model(input_tensors, depth = depth, batch=True)
        
        batch_input = torch.cat(embeddings, dim=0)
        batch_tags = torch.cat(tags)
        output = model.aux_taggers[task](batch_input)
        loss = torch.sum(model.tagger_loss(output, batch_tags))
        loss /= len(tags)
        for param in model.parameters():
           loss += L1 * param.abs().sum() + L2 * (param**2).sum()
        loss.backward()
        return loss.float().detach()
    
def train_sentence_tagging(device, model, optimizer, sentence_tensors, tags, depth, batch=False, supertags = None):
    # Returns loss node for tagging
    optimizer.zero_grad()
    if not batch:
        input_tensors = sentence_tensors
        embeddings, _ = model(input_tensors, depth = depth, supertags = supertags)
        output = model.tagger(embeddings)
        loss = torch.sum(model.tagger_loss(output, tags))
        for param in model.parameters():
           loss += L1 * param.abs().sum() + L2 * (param**2).sum()
        loss.backward()
        return loss.float().detach()
    else:
        input_tensors = sentence_tensors
        embeddings, _ = model(input_tensors, depth = depth, batch=True, supertags = supertags)
        
        batch_input = torch.cat(embeddings, dim=0)
        batch_tags = torch.cat(tags)
        output = model.tagger(batch_input)
        loss = torch.sum(model.tagger_loss(output, batch_tags))
        loss /= len(tags)
        for param in model.parameters():
           loss += L1 * param.abs().sum() + L2 * (param**2).sum()
        loss.backward()
        return loss.float().detach()

def extract_features(device, labels2i, sentence):

    state = State(sentence)

    struct_input = []
    struct_output = []

    labels_input = []
    labels_output = []

    #labels_input_sets = set()

    while not state.is_final():
        next_type = state.next_action_type()
        if next_type == State.STRUCT:
            gold_action, (memory, focus, buf) = state.oracle()
        
            if focus is None or len(memory) == 0:
                continue

            focus_tensor = set2tensor(device, focus)
            input_tensors = [torch.cat([set2tensor(device, input_set, add=0), focus_tensor]) for input_set in memory]

            if buf is not None:
                input_tensors.append(torch.cat([set2tensor(device, buf, add=1), focus_tensor]))

            struct_input.append(torch.stack(input_tensors))

            if gold_action[0] == "combine":
                target = gold_action[1] 
            else:
                target = len(input_tensors) - 1

            struct_output.append(target)
        else:
            assert(next_type == State.LABEL)
            gold_action, input_set = state.oracle()
            input_tensor = set2tensor(device, input_set)
            labels_input.append(input_tensor)
            target = labels2i[gold_action[1]]
            labels_output.append(target)

            #labels_input_sets.add(tuple(input_set))

#    if len(struct_input) > 0:
#        struct_input, struct_output = zip(*sorted(list(zip(struct_input, struct_output)), key=lambda x: len(x), reverse=True))

    r1, r2 = None, None
    if len(struct_input) > 0:
        #r1 = torch.stack(struct_input)
        #r2 = torch.cat(struct_output)
        r1 = struct_input
        r2 = torch.tensor(struct_output, dtype=torch.long, device=device).view(-1, 1)
    r3 = torch.stack(labels_input)
    r4 = torch.tensor(labels_output, dtype=torch.long, device=device)

    return r1, r2, r3, r4


def extract_tags(device, tags2i, sentence):
    # Returns a tensor for tag ids for a single sentence
    idxes = [tags2i[tok.get_tag()] for tok in sentence]
    return torch.tensor(idxes, dtype=torch.long, device=device)

def compute_f(TPs, total_golds, total_preds):
    p, r, f = 0, 0, 0
    if total_preds > 0:
        p = TPs / total_preds
    if total_golds > 0:
        r = TPs / total_golds
    if (p, r) != (0, 0):
        f = 2*p*r / (p+r)
    return p, r, f

def Fscore_corpus(golds, preds):
    TPs = 0
    total_preds = 0
    total_golds = 0
    UTPs = 0
    for gold, pred in zip(golds, preds):
        TPs += len([c for c in gold if c in pred])
        total_golds += len(gold)
        total_preds += len(pred)

        ugold = defaultdict(int)
        for _, span in gold:
            ugold[span] += 1
        upred = defaultdict(int)
        for _, span in pred:
            upred[span] += 1
        for span in upred:
            UTPs += min(upred[span], ugold[span])

    p, r, f = compute_f(TPs, total_golds, total_preds)

    up, ur, uf = compute_f(UTPs, total_golds, total_preds)
    return p*100, r*100, f*100, up*100, ur*100, uf*100

def predict_corpus(device, model, i2labels, i2tags, sentences_copy, tensors, batch=True, supertags = None):

    trees = []

    if supertags is None:
        if not batch:
            for toks, tens in zip(sentences_copy, tensors):
                tree = embed_and_parse_one(device, model, i2labels, i2tags, toks, tens)
                tree.expand_unaries()
                trees.append(tree)
            return trees
        else:
            # sort by length for lstm batching
            indices, sentences_copy, tensors = zip(*sorted(zip(range(len(sentences_copy)), 
                                                               sentences_copy,
                                                               tensors), 
                                                           key = lambda x: len(x[1]),
                                                           reverse=True))
            batch_size=200
            for i in range(0, len(sentences_copy), batch_size):
                tree_batch = embed_and_parse_batch(device, 
                                                   model,
                                                   i2labels, i2tags,
                                                   sentences_copy[i:i+batch_size],
                                                   tensors[i:i+batch_size])
                for t in tree_batch:
                    t.expand_unaries()
                    trees.append(t)
            # reorder
            _, trees = zip(*sorted(zip(indices, trees), key = lambda x:x[0]))
            return trees
    else:
        if not batch:
            for toks, tens, sup in zip(sentences_copy, tensors, supertags):
                tree = embed_and_parse_one(device, model, i2labels, i2tags, toks, tens, supertags = sup)
                tree.expand_unaries()
                trees.append(tree)
            return trees
        else:
            # sort by length for lstm batching
            indices, sentences_copy, tensors, supertags = zip(*sorted(zip(range(len(sentences_copy)), 
                                                               sentences_copy,
                                                               tensors, supertags), 
                                                           key = lambda x: len(x[1]),
                                                           reverse=True))
            batch_size=200
            for i in range(0, len(sentences_copy), batch_size):
                tree_batch = embed_and_parse_batch(device, 
                                                   model,
                                                   i2labels, i2tags,
                                                   sentences_copy[i:i+batch_size],
                                                   tensors[i:i+batch_size],
                                                   supertags[i:i+batch_size])
                for t in tree_batch:
                    t.expand_unaries()
                    trees.append(t)
            # reorder
            _, trees = zip(*sorted(zip(indices, trees), key = lambda x:x[0]))
            return trees


def prepare_corpus(corpus, words2i, device):
    sentences = [T.get_yield(corpus[i]) for i in range(len(corpus))]
    raw_sentences = [[tok.token for tok in sentence] for sentence in sentences]
    sentences_copy = [[T.Token(tok.token, i, [feat for feat in tok.features])
                        for i, tok in enumerate(sent)] for sent in sentences]
    tensors = [sentence_to_tensors(sent, words2i, device) for sent in raw_sentences]
    return sentences, raw_sentences, sentences_copy, tensors


def eval_tagging(gold, pred):
    # Returns accuracy for tag predictions
    acc = 0
    tot = 0
    assert(len(gold) == len(pred))
    for sent_g, sent_p in zip(gold, pred):
        assert(len(sent_g) == len(sent_p))
        for tok_g, tok_p in zip(sent_g, sent_p):
            if tok_g.get_tag() == tok_p.get_tag():
                acc += 1
        tot += len(sent_g)
    return acc * 100 / tot

def eval_generic(model, gold, prediction_sentences, i2t, task, level):

    acc = 0
    tot = 0
    assert(len(gold) == len(prediction_sentences))
    for sent_g, sent_p in zip(gold, prediction_sentences):
        
        embeddings, _ = model(sent_p, depth = level)
        scores = model.aux_taggers[task](embeddings)
        predictions = torch.argmax(scores, dim=1).cpu().numpy()
        #print(sent_g, supertag_predictions)
        assert(len(sent_g) == len(predictions))
        for tok_g, tok_p in zip(sent_g, predictions):
            if tok_g == i2t[tok_p]:
                acc += 1
        tot += len(sent_g)
    return acc * 100 / tot


def main_train(args, logger, device):

    logger.info("Loading corpora...")
    train_corpus = corpus_reader.read_ctbk_corpus(args.train)
    dev_corpus = corpus_reader.read_ctbk_corpus(args.dev)

    for tree in train_corpus:
        tree.merge_unaries()

    # Aux tasks
    logger.info("Auxiliary corpora extraction...")

    corpus_dirs = {"ccg" : args.ccg, "depptb" : args.depptb, "lcfrsptb" : args.lcfrs, "LTAGspinalTB" : args.LTAGspinal}
    
    tasks = ast.literal_eval(args.T)
    task2depth = {task : n + 1 for n, level in enumerate(tasks) for task in level}

    with open("{}/task2depth".format(args.model), "w") as f:
        for k, v in task2depth.items():
            f.write("{}:{}\n".format(k, v))

    tagging_tasks_flat = [task for level in tasks for task in level if task != "parsing" and task != "tag"]
    
    # Initialisation of DataLoaders to manage train and dev auxiliary corpus data.
    # The datasets necessary for the tasks specified in tagging_tasks_flat are automatically loaded.
    aux_train_loader    = multi_data_loader.DataLoader(tagging_tasks_flat, corpus_dirs, "train", "train", args.S, device = device, verbose = True)
    aux_dev_loader      = multi_data_loader.DataLoader(tagging_tasks_flat, corpus_dirs, "dev", "eval", args.S, vocab = aux_train_loader.vocab)

    # Vocab
    # The tokens retrieved for auxiliary tasks are also treated as part
    # of the input alphabet. Currently, this would not be necessary since
    # the auxiliary train corpora do not transcend the PTB/WSJ sections 2-21.
    logger.info("Vocabulary extraction...")
    words, vocabulary, label_set, tag_set = get_vocabulary(train_corpus, *aux_train_loader.token_lists)

    i2chars = ["<PAD>", "<UNK>"] + sorted(vocabulary, key = lambda x: vocabulary[x], reverse=True)
    chars2i = {k:i for i, k in enumerate(i2chars)}

    i2labels = ["nolabel"] + sorted(label_set)
    labels2i = {k: i for i, k in enumerate(i2labels)}

    i2tags = sorted(tag_set)
    tags2i = {k:i for i, k in enumerate(i2tags)}

    i2words = ["<PAD>", "<UNK>"] + sorted(words, key=lambda x: words[x], reverse=True)
    words2i = {k:i for i, k in enumerate(i2words)}

    save_dict(i2chars, "{}/i2chars".format(args.model))
    save_dict(i2labels, "{}/i2labels".format(args.model))
    save_dict(i2tags, "{}/i2tags".format(args.model))
    save_dict(i2words, "{}/i2words".format(args.model))

    aux_train_loader.save_vocab(args.model)

    # Apparently, when pytorch manages training examples of various
    # sizes, it allocates fresh memory without freeing stale memory
    # Hence -> cuda memory error
    # https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/11
    # fix: on first pass optimize largest examples first
    # Nope: just a memory leak on pytorch Dropout
#    train_corpus.sort(key=lambda x: len(T.get_yield(x)), reverse=True)
#    Nl = 20000
#    if len(train_corpus) > Nl:
#        train_corpus, rest = train_corpus[:Nl], train_corpus[Nl:]
#        random.shuffle(rest)
#        train_corpus.extend(rest)
    random.shuffle(train_corpus)

    if args.S is not None:
        train_corpus    = train_corpus[:args.S]
        dev_corpus      = dev_corpus[:args.S]

    #print(aux_train_data_tensors.keys(), aux_train_data_tensors["supertag"])

    logger.info("Model initialization...")

    words2tensors = enc.Words2Tensors(device, chars2i, words2i, pchar=None)

    # Provide number of supertags in depCCG to the Transducer in the case
    # of a pipeline-approach. The number of supertags specifies the output
    # dimension of depCCG.
    if args.sup > 0:
        model = Transducer(args, len(tasks), len(i2chars), len(i2words), len(labels2i), len(tags2i), aux_train_loader.num_labels, words2tensors, supertag_num = depccg.CCG_CATS[args.sM])
    else:
        model = Transducer(args, len(tasks), len(i2chars), len(i2words), len(labels2i), len(tags2i), aux_train_loader.num_labels, words2tensors)
    
    model.to(device)

    if args.O == "asgd":
        optimizer = MyAsgd(model.parameters(), lr=args.l, 
                           momentum=args.m, weight_decay=0,
                           dc=args.d)
    elif args.O == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.l)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.l, momentum=args.m, weight_decay=0)


    logger.info("Constructing training examples...")
    corpus_tmp = prepare_corpus(train_corpus, words2i, device)
    train_sentences, train_raw_sentences, train_sentences_copy, train_tensors = corpus_tmp
    #TODO all_gold_train_constituents = []

    # initialise auxiliary task token tensors and tensors_copy
    aux_train_loader.init_tensors(words2i, device)
    aux_dev_loader.init_tensors(words2i, device)

    dev_sentences, dev_raw_sentences, dev_sentences_copy, dev_tensors = prepare_corpus(dev_corpus, words2i, device)
    gold_dev_constituents = [T.get_constituents(tree, filter_root=True) for tree in dev_corpus]

    sample_train_corpus = train_corpus[:len(dev_sentences)//4]

    corpus_tmp = prepare_corpus(sample_train_corpus, words2i, device)
    sample_train_sentences, sample_train_raw_sentences, sample_train_sentences_copy, sample_train_tensors = corpus_tmp

    # create auxiliary task train sample for evaluation
    aux_train_sample_loader = aux_train_loader.get_sample(len(dev_sentences)//4, device)
    aux_train_sample_loader.init_tensors(words2i, device)

    for t in sample_train_corpus:
        t.expand_unaries()
    sample_gold_train_constituents = [T.get_constituents(tree, filter_root=True) for tree in sample_train_corpus]
    for t in sample_train_corpus:
        t.merge_unaries()

    gold_train_constituents = [T.get_constituents(tree, filter_root=False) for tree in train_corpus]

    features = [extract_features(device, labels2i, sentence) for sentence in train_sentences]

    print_data_on_memory_size()

    dynamic_features = [None for _ in features]

    logger.info("Deep copying train tensors...")
    train_tensors_copy = copy.deepcopy(train_tensors)

    tag_features = [extract_tags(device, tags2i, sentence) for sentence in train_sentences]

    num_tag_tokens          = sum([len(sentence) for sentence in train_sentences])
    num_parsing_sentences   = len(train_sentences)

    # depCCG Pipeline data
    # Supertag the train and dev corpus pre-training to use supertags
    # as input features.
    if args.sup > 0:
        ccg_model = depccg.load_model(-1 if args.gpu is None else args.gpu, variant = args.sM)

        train_ccg_corpus    = depccg.supertag_distribution(train_raw_sentences, device, ccg_model)
        dev_ccg_corpus      = depccg.supertag_distribution(dev_raw_sentences, device, ccg_model)

    if args.sup > 0:
        sample_train_ccg_corpus = train_ccg_corpus[:len(dev_sentences)//4]
        
    print("Training sentences: {}".format(len(train_corpus)))
    print("Dev set sentences: {}".format(len(dev_corpus)))

    # create a list of (task, sentence_num) pairs. In each iteration, all sentences 
    # from all tasks are trained once in random order.
    idxs = [(task, sentence_num) for task in tagging_tasks_flat for sentence_num in range(aux_train_loader.num_sentences[task])]
    idxs += [(task, sentence_num) for task in ("tag", "parsing") for sentence_num in range(len(train_sentences))]

    outlogger = open("{}/learning_log".format(args.model), "w", buffering=1)
    outlogger.write(f"Epoch pl   tl  {'   '.join([(task[:2] if len(task) > 1 else  task) + 'l' for task in tagging_tasks_flat])}   tp   tr   tf   tup  tur  tuf  tt  {'  '.join(['t' + (task[:2] if len(task) > 1 else  task) for task in tagging_tasks_flat])}  dp   dr   df   dup  dur  duf  dt  {'  '.join(['d' + (task[:2] if len(task) > 1 else  task) for task in tagging_tasks_flat])} discoF disco2F\n")

    logger.info("Starting training")
    best_dev_f = 0
    for epoch in range(1, args.i+1):
        epoch_loss = 0
        tag_eloss = 0

        losses = {task : 0 for task in tagging_tasks_flat}

        if args.v > 0:
            logger.info("Stochastic replacement started")
        batch_stochastic_replacement(device,
                                    train_tensors,
                                    train_tensors_copy,
                                    words2i,
                                    pword=0.3)
        
        aux_train_loader.batch_stochastic_replacement(0.3, device)
        
        if args.v > 0:
            logger.info("Stochastic replacement done")

        model.train()

        grad_norms = {task : 0 for task in tagging_tasks_flat}

        grad_norm_p = 0
        grad_norm_t = 0

        if args.B == 1:
            for i, (task, ex) in enumerate(idxs):

                if i % 100 == 0 and args.v > 0:
                    logger.info("Epoch {} Training sent {} / {}".format(epoch, i, len(idxs)))

                if task == "parsing":
                    
                    if args.sup > 0:                                                        # provide supertags as input features
                        epoch_loss += train_sentence_batch(device, model, optimizer,
                                                       train_tensors_copy[ex],
                                                       features[ex],
                                                       dynamic=dynamic_features[ex],
                                                       supertags=train_ccg_corpus[ex])
                    else:
                        epoch_loss += train_sentence_batch(device, model, optimizer,
                                                       train_tensors_copy[ex],
                                                       features[ex],
                                                       dynamic=dynamic_features[ex])
                    #print("a", mem(device))
                    if args.G is not None:
                        grad_norm_p += torch.nn.utils.clip_grad_norm_(model.parameters(), args.G)
                    optimizer.step()

                elif task == "tag":
                    if args.sup > 0:
                        tag_eloss += train_sentence_tagging(device, model, optimizer, train_tensors_copy[ex], tag_features[ex], task2depth[task], supertags=train_ccg_corpus[ex])
                    else:
                        tag_eloss += train_sentence_tagging(device, model, optimizer, train_tensors_copy[ex], tag_features[ex], task2depth[task])

                    if args.G is not None:
                        grad_norm_t += torch.nn.utils.clip_grad_norm_(model.parameters(), args.G)
                    optimizer.step()
                
                else:
                    losses[task] += train_generic_tagging(device, model, optimizer, aux_train_loader.tensors_copy[task][ex], aux_train_loader.tensor_features[task][ex], task2depth[task], task)
                    if args.G is not None:
                        grad_norms[task] += torch.nn.utils.clip_grad_norm_(model.parameters(), args.G)
                    optimizer.step()

            for task in grad_norms.keys():
                grad_norms[task] /= aux_train_loader.num_sentences[task]
            grad_norm_p /= len(train_sentences)
            grad_norm_t /= len(train_sentences)

        else:
            raise Exception("batch training not implemented")

        random.shuffle(idxs)


        if args.dyno is not None:
            if args.sup > 0:
                oracle_examples = extract_dyn_oracle_from_corpus(
                                               device, model,
                                               labels2i, i2labels,
                                               train_sentences_copy,
                                               train_tensors,
                                               gold_train_constituents, args.dyno,
                                               supertags = train_ccg_corpus)
            else:
                oracle_examples = extract_dyn_oracle_from_corpus(
                                               device, model,
                                               labels2i, i2labels,
                                               train_sentences_copy,
                                               train_tensors,
                                               gold_train_constituents, args.dyno)
            dynamic_features = [None for _ in features]
            for i, all_feats in oracle_examples:
                dynamic_features[i] = all_feats


        epoch_loss /= num_parsing_sentences
        tag_eloss /= num_tag_tokens

        for task in losses.keys():
            losses[task] /= aux_train_loader.num_tokens[task]

        if epoch % args.E != 0:
            summary = "Ep{} lr={:.5f} Tr l={:.5f} tl={:.5f} " + " ".join([f"{task}l=" + "{:.5f}" for task in losses.keys()]) + " normp={:.3f} normt={:.3f} " + " ".join([f"norm{task}=" + "{:.3f}" for task in grad_norms.keys()])
            print(summary.format(epoch, optimizer.param_groups[0]['lr'], 
                                 epoch_loss, tag_eloss, *losses.values(),
                                 grad_norm_p, grad_norm_t, *grad_norms.values()), flush=True)
            continue

        optimizer.zero_grad()
        model.eval()

        if args.O == "asgd":
            optimizer.average()

        if args.sup > 0:
            dpred_trees = predict_corpus(device, model, i2labels, i2tags, dev_sentences_copy, dev_tensors, batch=True, supertags = dev_ccg_corpus)
        else:
            dpred_trees = predict_corpus(device, model, i2labels, i2tags, dev_sentences_copy, dev_tensors, batch=True)

        dcorpus_pred_constituents = [T.get_constituents(tree, filter_root=True) for tree in dpred_trees]
        p, r, f, up, ur, uf = Fscore_corpus(gold_dev_constituents, dcorpus_pred_constituents)
        dtag = eval_tagging(dev_sentences, dev_sentences_copy)

        daux = [eval_generic(model, aux_dev_loader.features[task], aux_dev_loader.tensors[task], aux_dev_loader.vocab[task][0], task, task2depth[task]) for task in tagging_tasks_flat]

        outfile = "{}/tmp_dev.discbracket".format(args.model)
        with open(outfile, "w") as fstream:
            for tree in dpred_trees:
                fstream.write("{}\n".format(str(tree)))

        discop, discor, discodop_f = discodop_eval.call_eval(args.dev.replace(".ctbk", ".discbracket"), outfile)
        disco2p, disco2r, discodop2_f = discodop_eval.call_eval(args.dev.replace(".ctbk", ".discbracket"),
                                                                outfile, disconly=True)

        if args.sup > 0:
            tpred_trees = predict_corpus(device, model,
                                     i2labels, i2tags,
                                     sample_train_sentences_copy,
                                     sample_train_tensors, batch=True,
                                     supertags = sample_train_ccg_corpus)
        else:
            tpred_trees = predict_corpus(device, model,
                                     i2labels, i2tags,
                                     sample_train_sentences_copy,
                                     sample_train_tensors, batch=True)
        

        tcorpus_pred_constituents = [T.get_constituents(tree, filter_root=True) for tree in tpred_trees]
        tp, tr, tf, tup, tur, tuf = Fscore_corpus(sample_gold_train_constituents, tcorpus_pred_constituents)
        ttag = eval_tagging(sample_train_sentences, sample_train_sentences_copy)
        
        taux = [eval_generic(model, aux_train_sample_loader.features[task], aux_train_sample_loader.tensors[task], aux_train_sample_loader.vocab[task][0], task, task2depth[task]) for task in tagging_tasks_flat]
        

        summary = "Ep{} lr={:.5f} Tr l={:.5f} tl={:.5f} " + " ".join([task + "loss={:.5f}" for task in losses.keys()]) + " pr{}/{} f={:.2f} u={}/{}/{:.1f} t={:.2f} " + " ".join([task + "={:.2f}" for task in tagging_tasks_flat]) + " Dev pr{}/{} f={:.2f} ({:.2f}) u={}/{}/{:.1f} t={:.2f} " + " ".join([task + "={:.2f}" for task in tagging_tasks_flat])
        print(summary.format(epoch, optimizer.param_groups[0]['lr'],
                             epoch_loss, tag_eloss, *losses.values(),
                             int(tp), int(tr), tf,
                             int(tup), int(tur), tuf, ttag, *taux,
                             int(p), int(r), f, discodop_f,
                             int(up), int(ur), uf, dtag, *daux), flush=True)

        summary = "{} " + " ".join(["{:.3f}" for _ in range(2 + len(losses))]) + " " + " ".join(["{:.1f}" for i in range(16 + 2*len(tagging_tasks_flat))]) + "\n"
        outlogger.write(summary.format(epoch, epoch_loss, tag_eloss, *losses.values(),
                                       tp, tr, tf, tup, tur, tuf, ttag, *taux,
                                       p, r, f, up, ur, uf, dtag, *daux, discodop_f, discodop2_f))

        if discodop_f > best_dev_f:
            logger.info("Saving model")
            best_dev_f = discodop_f
            model.cpu()
            torch.save(model, "{}/model".format(args.model))
            model.to(device)

        if args.O == "asgd":
            optimizer.cancel_average()

    if args.O == "asgd":
        optimizer.average()

def read_raw_corpus(filename):
    sentences = []
    with open(filename) as f:
        for line in f:
            # For negra
            line = line.replace("(", "#LRB#").replace(")", "#RRB#")
            line = line.strip().split()
            if len(line) > 0:
                sentences.append(line)
    return sentences

def main_eval(args, logger, device):
    
    model = torch.load("{}/model".format(args.model), map_location=device)
    model.to(device)

    model.eval()
    logger.info("Model loaded")
    
    i2labels, labels2i = load_dict("{}/i2labels".format(args.model))
    i2tags, tags2i = load_dict("{}/i2tags".format(args.model))
    i2chars, chars2i = load_dict("{}/i2chars".format(args.model))
    i2words, words2i = load_dict("{}/i2words".format(args.model))

    sentences = read_raw_corpus(args.corpus)
    sentence_toks = [[T.Token(token, i, [None]) for i, token in enumerate(sentence)] for sentence in sentences]
    
    # args.ctbk is specified if an evaluation of tagging 
    # and multi-task tagging is requested
    if args.ctbk is not None:
        test_corpus = corpus_reader.read_ctbk_corpus(args.ctbk)
        test_sentences, test_raw_sentences, _, _ = prepare_corpus(test_corpus, words2i, device)

        corpus_dirs = {"ccg" : args.ccg, "depptb" : args.depptb, "lcfrsptb" : args.lcfrs, "LTAGspinalTB" : args.LTAGspinal}

        if args.pipeline == 1:
            depccg_model = depccg.load_model(-1 if args.gpu is None else args.gpu, args.sM)
            start_supertag = time.time()
            test_ccg_corpus = depccg.supertag_distribution(test_raw_sentences, tensor_device = device, model = depccg_model)
            end_supertag = time.time()
            p_time_supertag = end_supertag - start_supertag

        task2depth = {}
        with open("{}/task2depth".format(args.model), "r") as f:
            for l in f:
                kv = l.split(":")
                k = kv[0].strip()
                l = int(kv[1].strip())
                task2depth[k] = l

        tasks = list(task2depth.keys())

        flat_task_list = [task for task in tasks if task != "parsing" and task != "tag"]

        aux_loader = multi_data_loader.DataLoader(flat_task_list, corpus_dirs, args.split, "eval", None, words2i, device)
        aux_loader.load_vocab(args.model)

    with torch.no_grad():
        sent_tensors = [sentence_to_tensors(sent, words2i, device) for sent in sentences]

        start = time.time()
        if args.pipeline == 1:
            trees = predict_corpus(device, model, i2labels, i2tags, sentence_toks, sent_tensors, batch=True, supertags = test_ccg_corpus)
        else:
            trees = predict_corpus(device, model, i2labels, i2tags, sentence_toks, sent_tensors, batch=True)
        end = time.time()

        n_sentences = len(test_sentences)
        n_tokens = sum([len(sent) for sent in test_sentences])
        p_time = end - start
        if args.pipeline == 1:
            p_time += p_time_supertag

        logger.info("parsing time: {:.2f} seconds for {} sentences ({} tokens)".format(p_time, n_sentences, n_tokens))
        logger.info("parsing time: {:.2f} sentences per second, {:.2f} tokens per second".format(n_sentences / p_time, n_tokens / p_time))

        if args.ctbk is not None:
            tag = eval_tagging(test_sentences, sentence_toks)

            for task in flat_task_list:
                for t, f in zip(aux_loader.tensors[task], aux_loader.features[task]):
                    if len(t) == 0 or len(f) == 0:
                        print(task, t, f)
            aux = [eval_generic(model, aux_loader.features[task], aux_loader.tensors[task], aux_loader.vocab[task][0], task, task2depth[task]) for task in flat_task_list]

        if args.output is None:
            for tree in trees:
                print(tree)
        else:
            with open(args.output, "w") as f:
                for tree in trees:
                    f.write("{}\n".format(str(tree)))

            if args.gold is not None:
                p, r, f = discodop_eval.call_eval(args.gold, args.output, disconly=False)
                dp, dr, df = discodop_eval.call_eval(args.gold, args.output, disconly=True)

                print("precision={}".format(p))
                print("recall={}".format(r))
                print("fscore={}".format(f))
                print("disc-precision={}".format(dp))
                print("disc-recall={}".format(dr))
                print("disc-fscore={}".format(df))

                # print tagging and auxiliary-task tagging accuracies
                if args.ctbk is not None:
                    print("tag-accuracy={:.2f}".format(tag))

                    for task, accuracy in zip(flat_task_list, aux):
                        print("{}-accuracy={:.2f}".format(task, accuracy))

#    counts = defaultdict(int)
#    for m in model.memory_sizes:
#        counts[m] += 1
#    print()
#    for k, v in sorted(counts.items()):
#        print("{}\t{}".format(k, v))
#    print()

def main(args, logger, device):
    """
        Discoparset: transition based discontinuous constituency parser

    warning: CPU trainer is non-deterministic (due to multithreading approximation)
    """
    if args.mode == "train":
        main_train(args, logger, device)
    else:
        main_eval(args, logger, device)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    import argparse

    usage = main.__doc__

    parser = argparse.ArgumentParser(description = usage, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    subparsers = parser.add_subparsers(dest="mode", description="Execution modes", help='train: training, eval: test')
    subparsers.required = True

    train_parser = subparsers.add_parser("train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    eval_parser = subparsers.add_parser("eval", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # train corpora
    train_parser.add_argument("model", help="Directory (to be created) for model exportation")
    train_parser.add_argument("train", help="Training corpus")
    train_parser.add_argument("dev",   help="Dev corpus")

    # general options
    train_parser.add_argument("--gpu", type=int, default=None, help="Use GPU if available")
    train_parser.add_argument("-t", type=int, default=1, help="Number of threads for torch cpu")
    train_parser.add_argument("-S", type=int, default=None, help="Use only X first training examples")
    train_parser.add_argument("-v", type=int, default=1, choices=[0,1], help="Verbosity level")

    # training options
    train_parser.add_argument("-i", default=100, type=int, help="Number of epochs")
    train_parser.add_argument("-l", default=0.01, type=float, help="Learning rate")
    train_parser.add_argument("-m", default=0, type=float, help="Momentum (for sgd and asgd)")
    train_parser.add_argument("-d", default=1e-7, type=float, help="Decay constant for learning rate")
    train_parser.add_argument("-E", default=10, type=int, help="Evaluate on dev every E epoch")

    train_parser.add_argument("-A", action="store_true", help="Use attention-based context for structure classifier")
    train_parser.add_argument("-I", type=float, default=0.1, help="Embedding initialization uniform on [-I, I]")
    train_parser.add_argument("-G", default=100, type=float, help="Max norm for gradient clipping")

    train_parser.add_argument("-B", type=int, default=1, help="Size of batch")
    train_parser.add_argument("-O", default="asgd", choices=["adam", "sgd", "asgd"], help="Optimizer")
    train_parser.add_argument("-s", type=int, default=10, help="Random seed")

    train_parser.add_argument("-K", default=0.2, type=float, help="Dropout for character embedding layer")
    train_parser.add_argument("-Q", default=0, type=float, help="Dropout for char bi-LSTM output")
    train_parser.add_argument("-D", default=0.5, type=float, help="Dropout for parser output layers")
    train_parser.add_argument("-X", default=0.5, type=float, help="Dropout for tagger")

    train_parser.add_argument("-H", type=int, default=200, help="Dimension of hidden layer for FF nets")
    train_parser.add_argument("-C", type=int, default=100, help="Dimension of char bilstm")
    train_parser.add_argument("-c", type=int, default=100, help="Dimension of char embeddings")

    train_parser.add_argument("-w", type=int, default=None, help="Use word embeddings with dim=w")
    train_parser.add_argument("-W", type=int, default=400, help="Dimension of sentence bi-LSTM")
    train_parser.add_argument("-a", default="tanh", choices=["tanh","ReLu","sigmoid"], help="Choice of activatio functions for hidden layers.")
    train_parser.add_argument("-Ra", default=1, type=int, help="Layer to use element-wise addition residual connection from. 0 is none, 1 is the previous layer, 2 is the second previous, ...")
    train_parser.add_argument("-Rg", default=0, type=int, help="Layer to use gated residual connection from. 0 is none, 1 is the previous layer, 2 is the second previous, ...")
    train_parser.add_argument("-Rga", default=0, type=int, help="Layer to use gated residual connection at output from. 0 is none, 1 is the previous layer, 2 is the second previous, ...")
    train_parser.add_argument("-vi", type=float, default=0, help="Variational LSTM dropout input")
    train_parser.add_argument("-vh", type=float, default=0, help="Variational LSTM dropout hidden layer")
    train_parser.add_argument("-it", type=int, default=0, choices=[0,1], help="Transform LSTM stack input dimension into hidden dimension to enable residual connection from input.")
    train_parser.add_argument("-L", type=int, default=0, choices=[0,1], help="Use layer normalisation between LSTMs and in final FF networks")
    train_parser.add_argument("-ph", type=int, default=2, help="Number of parser (label, struct) feed forward hidden layers")
    train_parser.add_argument("-th", type=int, default=2, help="Number of tagger feed forward hidden layers")
    train_parser.add_argument("-pb", type=int, default=0, choices=[0,1], help="Use of bias in parser (label, struct) feed forward last linear transformation")
    train_parser.add_argument("-tb", type=int, default=0, choices=[0,1], help="Use of bias in tagger feed forward last linear transformation")

    
    train_parser.add_argument("-T", type=str, default="[['tag'],['parsing']]", help="Multi-task hierarchy; only allowed to contain tag and parsing if pipeline model is used")
    train_parser.add_argument("-ccg", type=str, default="../CCGrebank/data", help="CCGrebank directory")
    train_parser.add_argument("-depptb", type=str, default="../DepPTB/treebank.conllu", help="depPTB directory")
    train_parser.add_argument("-lcfrs", type=str, default="../LCFRS", help="LCFRS directory")
    train_parser.add_argument("-LTAGspinal", type=str, default="../LTAGspinal", help="LTAG-spinal directory")

    train_parser.add_argument("-sup", type=int, default=0, help="Supertag pipeline hidden dimension; if 0 then the pipeline model is not used; must be set to 0 if auxiliary tasks are used")
    train_parser.add_argument("-Y", type=float, default=0, help="Supertag pipeline drop")
    train_parser.add_argument("-sM", type=str, default="standard", choices=["standard", "elmo", "rebank", "elmo_rebank"], help="DepCCG model to use for supertagging.")    

    train_parser.add_argument("--dyno", type=float, default=None, help="Use the dynamic oracle")

    # test corpus
    eval_parser.add_argument("model", help="Pytorch model")
    eval_parser.add_argument("corpus", help="Test corpus, 1 tokenized sentence per line")
    eval_parser.add_argument("output", help="Outputfile")
    eval_parser.add_argument("--gold", default=None, help="Gold corpus (discbracket). If provided, eval with discodop")

    # test options
    eval_parser.add_argument("--gpu", type=int, default=None, help="Use GPU <int> if available")
    eval_parser.add_argument("-v", type=int, default=1, choices=[0,1], help="Verbosity level")
    eval_parser.add_argument("-t", type=int, default=1, help="Number of threads for torch cpu")

    eval_parser.add_argument("-ccg", type=str, default="../CCGrebank/data", help="CCGrebank directory")
    eval_parser.add_argument("-depptb", type=str, default="../DepPTB/treebank.conllu", help="CCGrebank directory")
    eval_parser.add_argument("-lcfrs", type=str, default="../LCFRS", help="LCFRS directory")
    eval_parser.add_argument("-LTAGspinal", type=str, default="../LTAGspinal", help="LTAG-spinal directory")

    eval_parser.add_argument("-pipeline", type=int, default=0, choices=[0,1], help="Use depCCG supertagger as input feature for parser. Must be set to the same value as in model training. 0 is false, 1 is true.")
    eval_parser.add_argument("-sM", type=str, default="standard", choices=["standard", "elmo", "rebank", "elmo_rebank"], help="DepCCG model to use for supertagging.")    
    eval_parser.add_argument("-ctbk", type=str, default=None, help="Corpus in ctbk format if tag and auxiliary tag eval is desired")
    eval_parser.add_argument("-split", type=str, default="test", help="Split for auxiliary eval data loading")

    args = parser.parse_args()

    for k, v in vars(args).items():
        print(k, v)
    
    #mkl.set_num_threads(args.t)
    torch.set_num_threads(args.t)

    logger = logging.getLogger()
    logger.info("Mode={}".format(args.mode))

    if args.mode == "train":
        os.makedirs(args.model, exist_ok = True)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    use_cuda = torch.cuda.is_available()
    if use_cuda and args.gpu is not None:
        logger.info("Using gpu {}".format(args.gpu))
        device = torch.device("cuda".format(args.gpu))
    else:
        logger.info("Using cpu")
        device = torch.device("cpu")
    
    SEED = 0
    if args.mode == "train":
        SEED = args.s
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    main(args, logger, device)




