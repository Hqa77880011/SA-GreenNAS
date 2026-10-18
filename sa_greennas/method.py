import torch
from torch import nn
from torch.nn import functional as F



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
    from scipy.ndimage import gaussian_filter
    if n<16 or n%4:raise ValueError('Condition demo count must be a multiple of four and at least 16')
    rng=np.random.default_rng(seed);sources=n//4;train=min(max(1,int(.6*sources)),sources-3);val=min(max(1,int(.15*sources)),sources-train-2);cal=min(max(1,int(.1*sources)),sources-train-val-1)
    images=[];labels=[];conditions=[];groups=[];splits=[]
    for source in range(sources):
        role=0 if source<train else 1 if source<train+val else 2 if source<train+val+cal else 3
        label=source%c.get('classes',3);clean=rng.uniform(.1,.3,(3,32,32)).astype('float32');clean[label%3]+=.4
        for condition in (0,1,2,3 if role==3 else 0):
            image=clean if condition==0 else np.clip(clean+.15,0,1) if condition==1 else gaussian_filter(clean,(0,1,1)) if condition==2 else .5+(clean-.5)*.4
            images.append(image.copy());labels.append(label);conditions.append(condition);groups.append(source);splits.append(role)
    return {'x':np.array(images),'y':np.array(labels),'condition':np.array(conditions),'group':np.array(groups),'split':np.array(splits)}
