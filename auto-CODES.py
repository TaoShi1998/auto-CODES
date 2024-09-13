import torch
import torch.nn as nn



class LE_SupCon(nn.Module):
    def __init__(self, temp, n_class):
        super().__init__()
        self.temp = temp
        self.n_class = n_class
    
    def forward(self, batch_feature_emb, batch_label, label_emb):
        batch_label_emb = label_emb[batch_label]
        expand_batch_feature_emb = batch_feature_emb.unsqueeze(1).repeat(1, self.n_class, 1)
        expand_batch_label_emb = label_emb.unsqueeze(0).repeat(batch_label.shape[0], 1, 1)
        numerator = torch.exp(torch.sum(batch_feature_emb * batch_label_emb / self.temp, dim = -1))
        denominator = torch.sum(torch.exp(torch.sum(expand_batch_feature_emb * expand_batch_label_emb / self.temp, dim = -1)), dim = -1)
        log_prob = - torch.log(numerator / denominator)
        loss = torch.mean(log_prob)

        return loss
        


class LE(nn.Module):
    def __init__(self, n_class, hidden_dim, output_dim):
        super().__init__()
        self.embed = nn.Embedding(n_class, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, labels):
        return self.dropout(self.linear(self.relu(self.embed(labels))))



if __name__ == '__main__':
    batch_feature_emb = torch.randn(32, 300)
    batch_label = torch.randint(0, 7, (32,))
    label_emb = torch.randn(7, 300)
    temp = 1
    n_class = 7

    LE_SupCon_loss = LE_SupCon(temp = temp, n_class = n_class)
    loss = LE_SupCon_loss(batch_feature_emb, batch_label, label_emb)
    print(loss)