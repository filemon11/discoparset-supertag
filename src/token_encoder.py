"""
This module contains encoders
that produce context-aware token
representations for the use in parsing.

Classes
----------
TokenEncoder
    Original implementation of word embedding,
    character embedding and two-level biLSTM
    for parsing and intermediate POS-tagging.

StackedSupertagTokenEncoder
    Token encoder with word embedding, character
    embedding, optional supertagger feature input
    and LSTM stack with variable level output
    for multi-task hierarchical learning.

"""

import torch
import torch.nn as nn

import character_encoders as enc
from dropout import Dropout

from lstmstack import LSTMStack, VardropLSTMStack

from neuralmodels import FeedForward

class TokenEncoder(nn.Module):
    """Original implementation of word embedding, character embedding and 
    two-level biLSTM from https://gitlab.com/mcoavoux/discoparset
    for parsing and intermediate POS-tagging as a separate encoder class.
    Stores sentence level bi-lstm and all other paramters of parser (struct and label classifiers)"""
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
            

class StackedSupertagTokenEncoder(nn.Module):
    """
    Encoder module featuring character-aware
    embeddings and word-embeddings as well
    as a vertically stacked biLSTM architecture
    for level-variable task prediction and
    support for supertag feature input.

    Attributes
    ----------
    depth : int
        Total number of biLSTM stack levels.
    self.word_embeddings : None | nn.Embedding
        Word embeddings module.
    self.supertag_encoder : None | nn.Linear
        Transformation from supertag input dimension
        to a specified dimensionality. 
    self.char_encoder : character_encoders.CharacterLstmLayer
        Encoder module for character-aware
        embeddings.
    self.lstm_stack : lstmstack.LSTMstack
        Module containing the biLSTM stacks.
    self.dropout : nn.Dropout
        Dropout for LSTM input.
    default_values : nn.Paramter
        Values to prepend to parser layer output.

    Methods
    -------
    initialize_parameters
        Initialisation for word emeddings and suprtag_encoder.
    forward
        Apply the module.
    """
    def __init__(self, depth, dim_char_emb, dim_char_lstm, dim_word_emb, dim_lstm_stack, 
                    char_voc_size, word_voc_size, words2tensors, emb_init, drop_lstm_in = 0,
                    drop_char_emb = 0, supertag_num = None, dim_supertag = None, drop_supertag = 0,
                    residual_add = 1, residual_gated = 0, residual_add_gated = 0, vardrop_i = 0, 
                    vardrop_h = 0, layernorm = False, initial_transform = False):
        """
        Initialising method for the ``StackedSupertagTokenEncoder`` module.
        Includes a call to ``initialize_parameters`` to initialise weights.

        Parameters
        ----------
        depth : int
            Total number of biLSTM stack levels.
        dim_char_emb : int
            Dimensionality for character embeddings.
        dim_char_lstm : int
            Dimensionality for character biLSTM output.
        dim_word_emb : int
            Dimensionality for word embeddings.
        dim_lstm_stack : int
            Dimensionality for sentence-level biLSTM output.
        char_voc_size : int
            Number of distinct characters in training set.
        word_vor_size : int
            Number of distinct words in training set.
        words2tensors : ``character_encoders.Words2Tensors``
            module that stores characters as indices
            for each word in the training set.
        emb_init : float
            Embedding initialization uniform on [-emb_init, emb_init].
        drop_lstm_int : float, default = 0, meaning no dropout
            Dropout for LSTM input.
        drop_char_emb : float, default = 0, meaning no dropout
            Dropout for ``char_encoder``.
        supertag_num : None | int, default = None, meaning no supertag input
            Dimensionality of the supertag tensor provided
            at ``forward``.
        dim_supertag : None | int, default = None
            Dimensionality to convert ``supertag_num`` to
            before concatenating with the word embedding and
            the character-aware embedding.
        drop_supertag : float, default = 0, meaning no dropout
            Dropout for ``supertag_encoder`` input.
        residual_add : int, default = 1
            Which preceding LSTM stack layer to add as a residual
            connection to each LSTM output. 0 means no residual
            connection, 1 means the previous layer, 2 means layer n-2
            and so forth. If ``initial_transform`` is False,
            no residual connection is added from the input dimension.
        residual_gated : int, default = 0
            Which preceding LSTM stack layer to add as a gated
            internal LSTM residual connection to each LSTM.
            0 means no residual connection, 1 means the previous layer, 
            2 means layer n-2 and so forth. If the input dimension does not
            match, a linear transformation is performed at the residual
            connection.
        residual_add_gated : int, default = 0
            Which preceding LSTm stack layer to add as a simple gated 
            residual connection to each LSTM output. 0 means no residual
            connection, 1 means the previous layer, 2 means layer n-2
            and so forth. If ``initial_transform`` is False,
            no residual connection is added from the input dimension.
        vardrop_i : float, default = 0, meaning no dropout
            Value of variational LSTM cell input dropout. 
        vardrop_h : float, default = 0, meaning no dropout
            Value of variational LSTM cell hidden dropout.
        layernorm : bool, default = False
            Whether to use layer normalization between biLSTM layers
            and as a final operation after the last biLSTM in the stack.
        initial_transform : bool, default = False
            Whether to perform a simple linear transformation
            at input level to assign all LSTMs uniform input
            and hidden dimensionality. If residual connections
            are used, this adds a residual connection from
            the input level.
        """
        super(StackedSupertagTokenEncoder, self).__init__()

        # Using word embeddings is optional
        self.depth = depth

        self.word_embeddings = None
        dim_lstm_stack_in = dim_char_lstm
        if dim_word_emb is not None:
            self.word_embeddings = nn.Embedding(word_voc_size, dim_word_emb, padding_idx=0)
            dim_lstm_stack_in += dim_word_emb

        self.supertag_encoder = None
        if supertag_num != None:
            self.supertag_encoder = FeedForward(d_in        = supertag_num,
                                                d_hid       = dim_supertag,
                                                d_out       = dim_supertag,
                                                drop_in     = 0,
                                                final_bias  = True,
                                                activation  = nn.Tanh,
                                                layer_norm  = False,
                                                n_hid       = 1,
                                                drop_hid    = drop_supertag)
            dim_lstm_stack_in += supertag_num

        self.char_encoder = enc.CharacterLstmLayer(
                                emb_dim=dim_char_emb,
                                voc_size=char_voc_size,
                                out_dim=dim_char_lstm,
                                embed_init=emb_init,
                                words2tensors=words2tensors,
                                dropout=drop_char_emb)

        # Use fast LSTM implementation if the internal structure does not need to be redefined.
        if residual_gated > 0:        
            self.lstm_stack = VardropLSTMStack(depth                = self.depth, 
                                               dim_lstm_in          = dim_lstm_stack_in, 
                                               dim_lstm_stack       = dim_lstm_stack, 
                                               vardrop_i            = vardrop_i, 
                                               vardrop_h            = vardrop_h, 
                                               residual_add         = residual_add,
                                               residual_gated       = residual_gated, 
                                               residual_add_gated   = residual_add_gated,
                                               layernorm            = layernorm,
                                               initial_transform    = initial_transform
                                               )
        else:
            self.lstm_stack = LSTMStack(depth               = self.depth,
                                        dim_lstm_in         = dim_lstm_stack_in, 
                                        dim_lstm_stack      = dim_lstm_stack, 
                                        vardrop_i           = vardrop_i, 
                                        vardrop_h           = vardrop_h, 
                                        residual            = residual_add, 
                                        residual_add_gated  = residual_add_gated,
                                        layernorm           = layernorm,
                                        initial_transform   = initial_transform
                                        )

        self.dropout = Dropout(drop_lstm_in)

        # "Padding" parameters for constituents with no gaps
        self.default_values = nn.Parameter(torch.Tensor(4, dim_lstm_stack))
        self.default_values.data.uniform_(-0.01, 0.01)

        self.initialize_parameters(emb_init)

    def initialize_parameters(self, emb_init):
        """
        Initialises the word embeddings uniformly
        from [-emb_init, emb_init] and the
        ``supertag_encoder`` weights with Xavier
        initalisation. The ``char_encoder``
        and the biLSTM are initialised in their
        respective module classes when calling
        ``__init__``.

        Parameters
        ----------
        emb_init : float
            Boundaries for uniform initialisation
            of word embeddings.

        Returns
        -------
        None
        """

        # uniform initialization for embeddings
        if self.word_embeddings is not None:
            self.word_embeddings.weight.data.uniform_(-emb_init, emb_init)
        # Xavier initialization
        if self.supertag_encoder is not None:
            for p in self.supertag_encoder.parameters():
                n_ids = sum(p.data.shape)
                m = (6 / n_ids)**0.5
                p.data.uniform_(-m, m)

    def forward(self, sentence, depth = -1, batch=False, parsing=False, supertags = None):
        """
        Forward method for applying the model on input
        features. If the model is in training mode, the
        computation graph is constructed automatically.

        Parameters
        ----------
        sentence : List[Tuple[List[str], torch.Tensor]]]] | Tuple[List[str], torch.Tensor]]]
            The input sentence(s) where a sentence has the form
            (sen, tensor), where sen is a list of the tokens in
            string form and tensor is the index tensor representation.
        depth : int, default = -1, meaning maximum depth
            Level up to which the biLSTM stack should be computed.
            n means computing the output up to (and including)
            the nth LSTM. 
        batch : bool, default = False
            True if the input is a list of sentences.
        parsing : bool, default = False
            Whether to prepend ``default_values`` to output tensor.
        supertags : None | List[torch.Tensor] | torch.Tensor
            Supertag input features.
            
        Returns
        -------
        List[torch.Tensor] | List[List[torch.Tensor]]
            The list of all biLSTM outputs. The first element
            is the first LSTM's input, i.e. concatenation of
            character-aware embeddings, words-embeddings and
            optionally supertag-features.
        """


        if not batch:
            char_based_embeddings = self.char_encoder(sentence[0])
            char_based_embeddings = self.dropout(char_based_embeddings)

            if self.word_embeddings is not None:
                embeds = self.word_embeddings(sentence[1])
                char_based_embeddings = torch.cat([char_based_embeddings, embeds], dim=1)
            
            if supertags is not None:
                supertag_embeddings = self.supertag_encoder(supertags)

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
                supertag_embeddings = [self.supertag_encoder(sup) for sup in supertags]
                char_based_embeddings = [torch.cat([cb_es, s_e], dim=1)
                                         for cb_es, s_e in zip(char_based_embeddings, supertag_embeddings)]
            
            output_list = self.lstm_stack(char_based_embeddings, depth, batch)
            
            if parsing:
                output_list[-1] = [torch.cat([self.default_values, ol2], dim=0) for ol2 in output_list[-1]]

            return output_list