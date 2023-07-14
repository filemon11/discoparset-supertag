import torch
import torch.nn as nn

def unpack(tensor, lengths):
    output = [t.squeeze(0) for t in tensor.split([1 for l in lengths], dim=0)]
    output = [t[:l,:] for t, l in zip(tensor, lengths)]
    return output

class LSTMStack(nn.Module):
    """multi-task LSTM stack"""
    def __init__(self, num_tasks, dim_lstm_in, dim_lstm_stack, emb_init):
        super(LSTMStack, self).__init__()

        self.lstm_list = nn.ModuleList([nn.LSTM(input_size   = dim_lstm_in if n == 0 else dim_lstm_stack,
                                                        hidden_size     = dim_lstm_stack //2,
                                                        num_layers      = 1,
                                                        batch_first     = True,
                                                        bidirectional   = True) 
                                                    for n in range(num_tasks)])

        self.initialize_parameters(emb_init)

    def initialize_parameters(self, emb_init):
        # Xavier initialization for every layer
        # uniform initialization for embeddings
        for mod in self.lstm_list:
            for p in mod.parameters():
                n_ids = sum(p.data.shape)
                m = (6 / n_ids)**0.5
                p.data.uniform_(-m, m)

    def forward(self, X, depth = -1, batch=False, lengths=None):
        """
        depth: level of stack to return; 0 is LSTM input
        batch: true if sentence is a list of sentences
        """
        output_list = []

        if depth == -1:
            depth = len(self.lstm_list)

        if not batch:
            output_list.append(X)
            
            current_input = X.unsqueeze(0)

            for lstm, n in zip(self.lstm_list, range(depth)):
                output, _ = lstm(current_input)

                if n != 0:
                    current_input = current_input + output
                else: 
                    current_input = output

                output_list.append(current_input.squeeze(0))

            return output_list
        
        else:
            assert(lengths != None)

            output_list.append(X)
            
            current_input = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)

            for lstm, n in zip(self.lstm_list, range(depth)):
                output, _ = lstm(torch.nn.utils.rnn.pack_padded_sequence(
                                    current_input, lengths, batch_first=True))

                output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
                
                if n != 0:
                    current_input = current_input + output
                else: 
                    current_input = output

                output_list.append(unpack(current_input, lengths))

            return output_list