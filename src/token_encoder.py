import torch
import torch.nn as nn

import character_encoders as enc
from dropout import Dropout

from lstmstack import LSTMStack

def unpack(tensor, lengths):
    output = [t.squeeze(0) for t in tensor.split([1 for l in lengths], dim=0)]
    output = [t[:l,:] for t, l in zip(tensor, lengths)]
    return output

class TokenEncoder(nn.Module):
    """Stores sentence level bi-lstm and all other paramters of parser (struct and label classifiers)"""
    def __init__(self, num_lstm, dim_char_emb, dim_char_lstm, dim_word_emb, dim_lstm_stack, char_voc_size, word_voc_size, words2tensors, drop_lstm_in, drop_char_emb, emb_init):
        super(TokenEncoder, self).__init__()

        # Using word embeddings is optional
        self.word_embeddings = None
        dim_lstm_stack_in = dim_char_lstm
        if dim_word_emb is not None:
            self.word_embeddings = nn.Embedding(word_voc_size, dim_word_emb, padding_idx=0)
            dim_lstm_stack_in += dim_word_emb

        self.char_encoder = enc.CharacterLstmLayer(
                                emb_dim=dim_char_emb,
                                voc_size=char_voc_size,
                                out_dim=dim_char_lstm,
                                embed_init=emb_init,
                                words2tensors=words2tensors,
                                dropout=drop_char_emb)

        self.word_transducer_l1 = nn.LSTM(input_size=dim_lstm_stack_in,
                                       hidden_size=dim_lstm_stack //2,
                                       num_layers=1,
                                       batch_first=True,
                                       bidirectional=True)

        assert(num_lstm >= 2)
        self.word_transducer_l2 = nn.LSTM(input_size=dim_lstm_stack,
                                       hidden_size=dim_lstm_stack_in //2,
                                       num_layers=num_lstm -1,
                                       batch_first=True,
                                       bidirectional=True)

        self.dropout = Dropout(drop_lstm_in)

        # "Padding" parameters for constituents with no gaps
        self.default_values = nn.Parameter(torch.Tensor(4, dim_lstm_stack))
        self.default_values.data.uniform_(-0.01, 0.01)

        self.initialize_parameters(emb_init)

    def initialize_parameters(self, emb_init):
        # Xavier initialization for every layer
        # uniform initialization for embeddings
        if self.word_embeddings is not None:
            self.word_embeddings.weight.data.uniform_(-emb_init, emb_init)
        for mod in [self.word_transducer_l1, self.word_transducer_l2]:
            for p in mod.parameters():
                n_ids = sum(p.data.shape)
                m = (6 / n_ids)**0.5
                p.data.uniform_(-m, m)

    def forward(self, sentence, l1_only=False, batch=False):
        """
        l1_only: only computes first layer of bilstm (for training the tagger)
        batch: true if sentence is a list of sentences"""
        if not batch:
            char_based_embeddings = self.char_encoder(sentence[0])
            #char_based_embeddings = self.norm_char(char_based_embeddings)
            char_based_embeddings = self.dropout(char_based_embeddings)
            if self.word_embeddings is not None:
                embeds = self.word_embeddings(sentence[1])
                char_based_embeddings = torch.cat([char_based_embeddings, embeds], dim=1)
            
            output_l1, (h_n, c_n) = self.word_transducer_l1(char_based_embeddings.unsqueeze(0))
            #output_l1 = self.norm1(output_l1)
            #output_l1 = self.dropout(output_l1)
            if l1_only:
                return output_l1.squeeze(0), None

            output_l2, (h_n, c_n) = self.word_transducer_l2(output_l1)
            output_l2 = output_l2 + output_l1 # residual connections
            #output_l2 = self.norm2(output_l2)
            #output_l2 = self.dropout(output_l2)
            output_l2 = torch.cat([self.default_values, output_l2.squeeze(0)])

            return output_l1.squeeze(0), output_l2
        else:
            sentence, all_words = zip(*sentence)
            lengths = [len(s) for s in sentence]

            all_tokens = [tok for s in sentence for tok in s]
            char_based_embeddings = self.char_encoder(all_tokens)
            #char_based_embeddings = self.norm_char(char_based_embeddings)
            char_based_embeddings = self.dropout(char_based_embeddings)
            char_based_embeddings = char_based_embeddings.split(lengths)

            if self.word_embeddings is not None:
                embeds = [self.word_embeddings(words) for words in all_words]
                char_based_embeddings = [torch.cat([cb_es, w_e], dim=1)
                                         for cb_es, w_e in zip(char_based_embeddings, embeds)]

            

            padded_char_based_embeddings = torch.nn.utils.rnn.pad_sequence(char_based_embeddings, batch_first=True)
            packed_padded_char_based_embeddings = torch.nn.utils.rnn.pack_padded_sequence(
                                    padded_char_based_embeddings, lengths, batch_first=True)

            output_l1, (h_n, c_n) = self.word_transducer_l1(packed_padded_char_based_embeddings)
            output_l1, _ = torch.nn.utils.rnn.pad_packed_sequence(output_l1, batch_first=True)
            #output_l1 = self.norm1(output_l1)
            #output_l1 = self.dropout(output_l1)

            if l1_only:
                output_l1 = [t.squeeze(0) for t in output_l1.split([1 for l in lengths], dim=0)]
                output_l1 = [t[:l,:] for t, l in zip(output_l1, lengths)]
                return output_l1, None

            packed_l1 = torch.nn.utils.rnn.pack_padded_sequence(output_l1, lengths, batch_first=True)

            output_l2, (h_n, c_n) = self.word_transducer_l2(packed_l1)
            unpacked_l2, _ = torch.nn.utils.rnn.pad_packed_sequence(output_l2, batch_first=True)

            output_l2 = unpacked_l2 + output_l1 # residual connections
            #output_l2 = self.norm2(output_l2)
            #output_l2 = self.dropout(output_l2)

            output_l1 = [t.squeeze(0) for t in output_l1.split([1 for l in lengths], dim=0)]
            output_l1 = [t[:l,:] for t, l in zip(output_l1, lengths)]
            output_l2 = [t.squeeze(0) for t in output_l2.split([1 for l in lengths], dim=0)]
            output_l2 = [t[:l,:] for t, l in zip(output_l2, lengths)]

            output_l2 = [torch.cat([self.default_values, ol2], dim=0) for ol2 in output_l2]
            return output_l1, output_l2
            


class StackedTokenEncoder(nn.Module):
    """multi-task LSTM stack"""
    def __init__(self, num_tasks, dim_char_emb, dim_char_lstm, dim_word_emb, dim_lstm_stack, char_voc_size, word_voc_size, words2tensors, drop_lstm_in, drop_char_emb, emb_init):
        super(StackedTokenEncoder, self).__init__()

        # Using word embeddings is optional
        self.depth = num_tasks

        self.word_embeddings = None
        dim_lstm_stack_in = dim_char_lstm
        if dim_word_emb is not None:
            self.word_embeddings = nn.Embedding(word_voc_size, dim_word_emb, padding_idx=0)
            dim_lstm_stack_in += dim_word_emb

        self.char_encoder = enc.CharacterLstmLayer(
                                emb_dim=dim_char_emb,
                                voc_size=char_voc_size,
                                out_dim=dim_char_lstm,
                                embed_init=emb_init,
                                words2tensors=words2tensors,
                                dropout=drop_char_emb)

        self.lstm_stack = LSTMStack(num_tasks, dim_lstm_stack_in, dim_lstm_stack, emb_init)

        self.dropout = Dropout(drop_lstm_in)

        # "Padding" parameters for constituents with no gaps
        self.default_values = nn.Parameter(torch.Tensor(4, dim_lstm_stack))
        self.default_values.data.uniform_(-0.01, 0.01)

        self.initialize_parameters(emb_init)

    def initialize_parameters(self, emb_init):
        # Xavier initialization for every layer
        # uniform initialization for embeddings
        if self.word_embeddings is not None:
            self.word_embeddings.weight.data.uniform_(-emb_init, emb_init)

    def forward(self, sentence, depth = -1, batch=False, parsing=False):
        """
        depth: level of stack to return; 0 is LSTM input
        batch: true if sentence is a list of sentences
        
        if depth equals max depth of the stack, the final vector is appended with default values
        return [l_0, ..., l_depth]
        """

        if not batch:
            char_based_embeddings = self.char_encoder(sentence[0])
            char_based_embeddings = self.dropout(char_based_embeddings)
            if self.word_embeddings is not None:
                embeds = self.word_embeddings(sentence[1])
                char_based_embeddings = torch.cat([char_based_embeddings, embeds], dim=1)
            
            output_list = self.lstm_stack(char_based_embeddings, depth)

            if parsing:
                output_list[-1] = torch.cat([self.default_values, output_list[-1]])

            return output_list
        
        else:
            sentence, all_words = zip(*sentence)
            lengths = [len(s) for s in sentence]

            all_tokens = [tok for s in sentence for tok in s]
            char_based_embeddings = self.char_encoder(all_tokens)
            
            char_based_embeddings = self.dropout(char_based_embeddings)
            char_based_embeddings = char_based_embeddings.split(lengths)

            if self.word_embeddings is not None:
                embeds = [self.word_embeddings(words) for words in all_words]
                char_based_embeddings = [torch.cat([cb_es, w_e], dim=1)
                                         for cb_es, w_e in zip(char_based_embeddings, embeds)]

            
            output_list = self.lstm_stack(char_based_embeddings, depth, batch, lengths)
            
            if parsing:
                output_list[-1] = [torch.cat([self.default_values, ol2], dim=0) for ol2 in output_list[-1]]

            return output_list
        

class StackedSupertagTokenEncoder(nn.Module):
    """multi-task LSTM stack"""
    def __init__(self, num_tasks, dim_char_emb, dim_char_lstm, dim_word_emb, dim_lstm_stack, 
                    char_voc_size, word_voc_size, words2tensors, drop_lstm_in, drop_char_emb, 
                    emb_init, supertag_num = None, dim_supertag = None, drop_supertag = None):
        super(StackedSupertagTokenEncoder, self).__init__()

        # Using word embeddings is optional
        self.depth = num_tasks

        self.word_embeddings = None
        dim_lstm_stack_in = dim_char_lstm
        if dim_word_emb is not None:
            self.word_embeddings = nn.Embedding(word_voc_size, dim_word_emb, padding_idx=0)
            dim_lstm_stack_in += dim_word_emb

        self.supertag_encoder = None
        if supertag_num != None:
            self.supertag_encoder = nn.Linear(supertag_num, dim_supertag, bias = False)
            self.supertag_dropout = Dropout(drop_supertag)
            dim_lstm_stack_in += dim_supertag

        self.char_encoder = enc.CharacterLstmLayer(
                                emb_dim=dim_char_emb,
                                voc_size=char_voc_size,
                                out_dim=dim_char_lstm,
                                embed_init=emb_init,
                                words2tensors=words2tensors,
                                dropout=drop_char_emb)

        self.lstm_stack = LSTMStack(num_tasks, dim_lstm_stack_in, dim_lstm_stack, emb_init)

        self.dropout = Dropout(drop_lstm_in)

        # "Padding" parameters for constituents with no gaps
        self.default_values = nn.Parameter(torch.Tensor(4, dim_lstm_stack))
        self.default_values.data.uniform_(-0.01, 0.01)

        self.initialize_parameters(emb_init)

    def initialize_parameters(self, emb_init):
        # Xavier initialization for every layer
        # uniform initialization for embeddings
        if self.word_embeddings is not None:
            self.word_embeddings.weight.data.uniform_(-emb_init, emb_init)
        if self.supertag_encoder is not None:
            self.supertag_encoder.weight.data.uniform_(-emb_init, emb_init)

    def forward(self, sentence, depth = -1, batch=False, parsing=False, supertags = None):
        """
        depth: level of stack to return; 0 is LSTM input
        batch: true if sentence is a list of sentences
        
        if depth equals max depth of the stack, the final vector is appended with default values
        return [l_0, ..., l_depth]
        """

        if not batch:
            char_based_embeddings = self.char_encoder(sentence[0])
            char_based_embeddings = self.dropout(char_based_embeddings)

            if self.word_embeddings is not None:
                embeds = self.word_embeddings(sentence[1])
                char_based_embeddings = torch.cat([char_based_embeddings, embeds], dim=1)
            
            if supertags is not None:
                supertag_embeddings = self.supertag_encoder(self.supertag_dropout(supertags.to(char_based_embeddings.device)))

                char_based_embeddings = torch.cat([char_based_embeddings, supertag_embeddings], dim=1)

            output_list = self.lstm_stack(char_based_embeddings, depth)

            if parsing:
                output_list[-1] = torch.cat([self.default_values, output_list[-1]])

            return output_list
        
        else:
            sentence, all_words = zip(*sentence)
            lengths = [len(s) for s in sentence]

            all_tokens = [tok for s in sentence for tok in s]
            char_based_embeddings = self.char_encoder(all_tokens)
            
            char_based_embeddings = self.dropout(char_based_embeddings)
            char_based_embeddings = char_based_embeddings.split(lengths)

            if self.word_embeddings is not None:
                embeds = [self.word_embeddings(words) for words in all_words]
                char_based_embeddings = [torch.cat([cb_es, w_e], dim=1)
                                         for cb_es, w_e in zip(char_based_embeddings, embeds)]

            if supertags is not None:
                supertag_embeddings = [self.supertag_encoder(self.supertag_dropout(sup.to(char_based_embeddings[0].device))) for sup in supertags]
                char_based_embeddings = [torch.cat([cb_es, s_e], dim=1)
                                         for cb_es, s_e in zip(char_based_embeddings, supertag_embeddings)]
            
            output_list = self.lstm_stack(char_based_embeddings, depth, batch, lengths)
            
            if parsing:
                output_list[-1] = [torch.cat([self.default_values, ol2], dim=0) for ol2 in output_list[-1]]

            return output_list