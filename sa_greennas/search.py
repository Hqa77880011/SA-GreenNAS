import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def ranks(records):
    vectors=np.asarray([r['objectives'] for r in records]);remaining=set(range(len(records)));fronts=[]
    while remaining:
        front=[i for i in remaining if not any(np.all(vectors[j]<=vectors[i]) and np.any(vectors[j]<vectors[i]) for j in remaining if j!=i)]
        fronts.append(front);remaining.difference_update(front)
    return fronts


def crowding(records,front):
    distance={i:0. for i in front}
    for axis in range(len(records[0]['objectives'])):
        order=sorted(front,key=lambda i:records[i]['objectives'][axis]);distance[order[0]]=distance[order[-1]]=float('inf')
        extent=records[order[-1]]['objectives'][axis]-records[order[0]]['objectives'][axis]
        for k in range(1,len(order)-1):distance[order[k]]+=(records[order[k+1]]['objectives'][axis]-records[order[k-1]]['objectives'][axis])/max(extent,1e-12)
    return distance


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',required=True);parser.add_argument('--output',required=True);parser.add_argument('--population',type=int,default=24);parser.add_argument('--generations',type=int,default=10);parser.add_argument('--epochs',type=int,default=12);parser.add_argument('--seed',type=int,default=13);parser.add_argument('--latency-cap',type=float,default=15.)
    args=parser.parse_args();package=__package__;root=Path(args.output);root.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(args.seed);space=[[16,24,32,48],[3,5,7],[2,3,4,5]];cache={};evaluations=[]
    def evaluate(gene):
        key=tuple(gene)
        if key in cache:return cache[key]
        path=root/('candidate_'+str(len(cache)));path.mkdir();config={'width':gene[0],'kernel':gene[1],'depth':gene[2]};(path/'config.json').write_text(json.dumps(config))
        subprocess.run([sys.executable,'-m',package+'.cli','train','--input',args.data,'--output',str(path),'--config',str(path/'config.json'),'--epochs',str(args.epochs),'--seed',str(args.seed)],check=True)
        subprocess.run([sys.executable,'-m',package+'.cli','evaluate','--input',str(Path(args.data)/'validation.npz'),'--model',str(path/'model.pt'),'--output',str(path/'validation'),'--batch-size','1'],check=True)
        metric=json.loads((path/'validation/metrics.json').read_text());objectives=[1-metric['macro_f1'],1-metric.get('worst_group_f1',metric['macro_f1']),metric['ece'],np.log(max(metric['p95_latency_ms'],1e-6)),np.log(max(metric['parameters'],1))]
        record={'gene':gene,'objectives':list(map(float,objectives)),'metrics':metric,'feasible':metric['p95_latency_ms']<=args.latency_cap,'path':str(path)};cache[key]=record;evaluations.append(record);return record
    population=[[int(rng.choice(axis)) for axis in space] for _ in range(args.population)]
    for generation in range(args.generations):
        records=[evaluate(gene) for gene in population];children=[]
        fronts=ranks(records);rank={i:r for r,front in enumerate(fronts) for i in front};distance={i:d for front in fronts for i,d in crowding(records,front).items()}
        def tournament():
            first,second=rng.integers(len(records),size=2);return population[min([first,second],key=lambda i:(not records[i]['feasible'],rank[i],-distance[i]))]
        while len(children)<args.population:
            a,b=tournament(),tournament();child=[a[k] if rng.random()<.5 else b[k] for k in range(3)]
            if rng.random()<.1:
                k=int(rng.integers(3));child[k]=int(rng.choice(space[k]))
            children.append(child)
        combined=records+[evaluate(gene) for gene in children];selected=[]
        for front in ranks(combined):
            ordering=crowding(combined,front);front=sorted(front,key=lambda i:(not combined[i]['feasible'],-ordering[i]));selected.extend(front[:args.population-len(selected)])
            if len(selected)==args.population:break
        population=[combined[i]['gene'] for i in selected]
        (root/'evaluations.json').write_text(json.dumps(evaluations,indent=2))
    feasible=[r for r in evaluations if r['feasible']];pareto=[feasible[i] for i in ranks(feasible)[0]] if feasible else []
    selected=min(pareto,key=lambda r:(r['objectives'][1],r['objectives'][2],r['objectives'][3])) if pareto else None
    (root/'selection.json').write_text(json.dumps({'selected':selected,'pareto':pareto},indent=2))


if __name__=='__main__':main()
