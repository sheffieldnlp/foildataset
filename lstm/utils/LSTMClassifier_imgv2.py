import torch.nn as nn
#import ipdb
#import torch.nn.functional as F
import torch
from torch.autograd import Variable


class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, lin_dim, vocab_size, label_size, batch_size, use_gpu):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.lin_dim = lin_dim
        self.linlayer = nn.Linear(self.lin_dim, embedding_dim)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=0.5)
        self.nonlinearity = nn.Tanh()
        #self.hidden2label = nn.Linear(hidden_dim+lin_dim, label_size)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, sentence, image):
        embeds = self.word_embeddings(sentence)
        (max_len,batch_size, emb_dim) = embeds.size()
        mullin = self.linlayer(image)
        mulrep = mullin.unsqueeze(1).repeat(1, max_len, 1)
        mulrep = mulrep.view(len(sentence), self.batch_size, -1)
        src_emb = self.nonlinearity(embeds + mulrep)
        x = src_emb.view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
#        y  = self.hidden2label(torch.cat((lstm_out[-1], image ) ,1))
        y  = self.hidden2label(lstm_out[-1])

        #ipdb.set_trace()
        #import ipdb
        #ipdb.set_trace()
        return y
