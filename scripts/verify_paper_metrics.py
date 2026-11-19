from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from selective_grasp import SCORE_KEYS, aurc, ltt_calibrate_single, verify_risk_guarantee


EXPECTED = {
    'cornell_grconvnet': (300,0.6866666667,'s_mc',0.1954),
    'cornell_ggcnn':     (300,0.6300000000,'s_mc',0.1990),
    'cornell_unet':      (300,0.8233333333,'s_mc',0.0746),
    'jacquard_grconvnet':(375,0.9280000000,'s_conf',0.0163),
    'jacquard_ggcnn':    (375,0.9200000000,'s_conf',0.0205),
    'jacquard_unet':     (375,0.9600000000,'s_topk',0.0106),
}

def check():
    rows=[]
    for label,(expected_n,expected_succ,expected_best,expected_aurc) in EXPECTED.items():
        with np.load(ROOT/'data/scores'/f'{label}.npz',allow_pickle=False) as data:
            n=len(data['success']); success=float(np.mean(data['success']))
            losses=1-data['success'].astype(float)
            rank=sorted((aurc(data[k],losses),k) for k,_ in SCORE_KEYS)
            best_aurc,best_key=rank[0]
            assert (n==expected_n and abs(success-expected_succ)<1e-7 and
                best_key==expected_best and abs(best_aurc-expected_aurc)<0.0006), label
            rows.append((label,n,success,best_key,best_aurc))
            print(f'{label:23} N={n:3} success={success:.4f} best={best_key:6} AURC={best_aurc:.4f} PASS')
    with np.load(ROOT/'data/scores/jacquard_grconvnet.npz',allow_pickle=False) as data:
        l=1-data['success'].astype(float);s=data['s_conf']
        r=ltt_calibrate_single(s,l,0.05,0.10)
        assert r.valid and abs(r.cal_coverage-232/375)<1e-10
        print(f'Jacquard GR-ConvNet LTT alpha=0.05: coverage={r.cal_coverage:.4f}, accepted={r.n_accepted}, PASS')
        if '--full' in sys.argv:
            for a,target in [(0.1,1.0),(0.05,0.99)]:
                v=verify_risk_guarantee(s,l,a,0.1,n_trials=300,seed=0)
                assert v['n_certified']==300 and abs(v['empirical_coverage']-target)<1e-12,(a,v)
                print(f'300-split verification alpha={a:.2f} hold={v["empirical_coverage"]:.3f} PASS')
    return rows

if __name__ == '__main__': check()
