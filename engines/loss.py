import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)

def inter_class_relation(y_s, y_t, mask=None):
    if mask is not None:
        return ((1 - pearson_correlation(y_s, y_t)) * mask).sum() / max(mask.sum(), 1)
    else:
        return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DIST(nn.Module):
    def __init__(self, temp=1.0):
        super(DIST, self).__init__()
        self.temp = temp

    def forward(self, z_s, z_t, mask=None):
        y_s = (z_s / self.temp).softmax(dim=1)
        y_t = (z_t / self.temp).softmax(dim=1)
        # intra_loss = self.temp**2 * intra_class_relation(y_s, y_t)
        intra_loss = 0
        inter_loss = self.temp**2 * inter_class_relation(y_s, y_t, mask)

        return intra_loss + inter_loss

    def inter_loss(self, z_s, z_t):
        y_s = (z_s / self.temp).softmax(dim=1)
        y_t = (z_t / self.temp).softmax(dim=1)
        return self.temp**2 * inter_class_relation(y_s, y_t)
    
    def intra_loss(self, z_s, z_t):
        y_s = (z_s / self.temp).softmax(dim=1)
        y_t = (z_t / self.temp).softmax(dim=1)
        return self.temp**2 * intra_class_relation(y_s, y_t)
    

class CriterionIFV(nn.Module):
    def __init__(self):
        super(CriterionIFV, self).__init__()

    def forward(self, z_s, z_t, mask=None):
        """
        Args:
        z_s: student features
        z_t: teacher features
        mask: whether 2 samples are from the same class
        """

        # Compute center
        center_feat_S = torch.matmul(mask, z_s) / mask.sum(dim=1, keepdim=True)
        center_feat_T = torch.matmul(mask, z_t) / mask.sum(dim=1, keepdim=True)

        # cosinesimilarity along C
        cos = nn.CosineSimilarity(dim=1)
        pcsim_feat_S = cos(z_s, center_feat_S)
        pcsim_feat_T = cos(z_t, center_feat_T)

        # mseloss
        mse = nn.MSELoss()
        loss = mse(pcsim_feat_S, pcsim_feat_T)
        return loss