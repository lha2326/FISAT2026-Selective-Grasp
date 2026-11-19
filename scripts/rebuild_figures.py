from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from selective_grasp import SCORE_KEYS, aurc, ltt_calibrate_single, verify_risk_guarantee


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Compute a figure from released score caches; figures in the submitted PDF
# are the authoritative publication versions.
if __name__=='__main__':
    fig,ax=plt.subplots(figsize=(7.5,4.6))
    alphas=[0.15,0.10,0.05,0.02,0.01]
    for f in sorted((ROOT/'data/scores').glob('*.npz')):
        with np.load(f,allow_pickle=False) as d:
            loss=1-d['success'].astype(float)
            rank=sorted((aurc(d[k],loss),k) for k,_ in SCORE_KEYS)
            k=rank[0][1]
            cov=[ltt_calibrate_single(d[k],loss,a,0.10).cal_coverage for a in alphas]
            ax.plot(alphas,cov,'o--',label=f.stem.replace('_',' / '))
    ax.set(xscale='log',xlabel='Risk budget alpha',ylabel='LTT certified coverage (on cache)',ylim=(-.02,1.05))
    ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout()
    out=ROOT/'results/summary/coverage_vs_alpha_from_caches.png'
    fig.savefig(out,dpi=200);print(out)
