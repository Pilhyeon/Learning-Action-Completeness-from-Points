import torch
import torch.nn as nn


class Cls_Module(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(Cls_Module, self).__init__()
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=num_classes+1, kernel_size=1,
                      stride=1, padding=0, bias=False)
        )
        self.drop_out = nn.Dropout(p=0.7)

    def forward(self, x):
        # x: (B, T, F)
        out = x.permute(0, 2, 1)
        # out: (B, F, T)
        out = self.conv_1(out)
        feat = out.permute(0, 2, 1)
        out = self.drop_out(out)
        cas = self.classifier(out)

        cas = cas.permute(0, 2, 1)
        # out: (B, T, C + 1)
        return feat, cas

        
class Model(nn.Module):
    def __init__(self, len_feature, num_classes, r_act):
        super(Model, self).__init__()
        self.len_feature = len_feature
        self.num_classes = num_classes
        self.r_act = r_act

        self.cls_module = Cls_Module(len_feature, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, vid_labels=None):
        num_segments = x.shape[1]
        k_act = num_segments // self.r_act

        features, cas = self.cls_module(x)

        cas_sigmoid = self.sigmoid(cas)

        cas_sigmoid_fuse = cas_sigmoid[:,:,:-1] * (1 - cas_sigmoid[:,:,-1].unsqueeze(2))
        cas_sigmoid_fuse = torch.cat((cas_sigmoid_fuse, cas_sigmoid[:,:,-1].unsqueeze(2)), dim=2)

        value, _ = cas_sigmoid.sort(descending=True, dim=1)
        topk_scores = value[:,:k_act,:-1]

        if vid_labels is None:
            vid_score = torch.mean(topk_scores, dim=1)
        else:
            vid_score = (torch.mean(topk_scores, dim=1) * vid_labels) + (torch.mean(cas_sigmoid[:,:,:-1], dim=1) * (1 - vid_labels))

        return vid_score, cas_sigmoid_fuse, features
