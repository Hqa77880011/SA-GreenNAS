import torch
from torch import nn
from torch.nn import functional as F
from .layers import classification_demo


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;w=c.get('width',24);k=c.get('kernel',3);depth=c.get('depth',3)
        channels=c.get('input_shape',[3,32,32])[0];layers=[]
        for i in range(depth):
            layers.extend([nn.Conv2d(channels,w,3,stride=2,padding=1),nn.SiLU(),nn.Conv2d(w,w,k,padding=k//2,groups=w),nn.Conv2d(w,w,1),nn.SiLU()]);channels=w
        self.encoder=nn.Sequential(*layers);self.attention=nn.Conv2d(2,1,5,padding=2);self.head=nn.Linear(w,c.get('classes',3))
        self.register_buffer('counts',torch.ones(c.get('classes',3)));self.register_buffer('confusion',torch.eye(c.get('classes',3)))
    def prepare(self,data):
        self.counts.copy_(torch.bincount(torch.as_tensor(data['y']).long(),minlength=len(self.counts)).to(self.counts))
    def forward(self,b):
        h=self.encoder(b['x'].float());attention=self.attention(torch.cat([h.mean(1,keepdim=True),h.max(1,keepdim=True).values],1)).sigmoid()
        if self.c['variant'] not in {'baseline','no_attention'}:h=h*(1+attention)
        logits=self.head(h.mean((2,3)));return {'probability':logits.softmax(-1),'logits':logits,'attention':attention}
    def objective(self,b,p):
        labels=b['y'].long()
        if self.c['variant'] in {'baseline','no_fuzzy'}:return F.cross_entropy(p['logits'],labels)
        if self.training:
            with torch.no_grad():
                for label in labels.unique():
                    self.confusion[label].lerp_(p['probability'][labels==label].mean(0),.05)
        entropy=-(self.confusion.clamp_min(1e-8)*self.confusion.clamp_min(1e-8).log()).sum(-1)/torch.log(torch.tensor(float(len(self.counts)),device=labels.device))
        effective=(1-.999)/(1-.999**self.counts).clamp_min(1e-8)
        weight=effective*(1+torch.sigmoid((entropy-.35)/.08));weight=weight/weight.mean()
        return F.cross_entropy(p['logits'],labels,weight=weight)


def demo(seed,n,c):
    import numpy as np
    d=classification_demo(seed,n,classes=c.get('classes',3));rng=np.random.default_rng(seed)
    x=rng.normal(size=(n,3,32,32)).astype('float32');x[:,0]+=d['y'][:,None,None]/3;d['x']=x;return d
