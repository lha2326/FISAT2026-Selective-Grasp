## Table 1

| score | AURC | E-AURC | AUSE | R@90 | R@70 | C*(a=0.15) | C*(a=0.1) |
|---|---|---|---|---|---|---|---|
| S1 negative peak quality | 0.2642 | 0.2085 | 0.2085 | 0.2926 | 0.2667 | 0.0000 | 0.0000 |
| S2 quality-map entropy | 0.2395 | 0.1838 | 0.1838 | 0.2926 | 0.2857 | 0.2033 | 0.0000 |
| S3 MC-dropout dispersion | 0.1954 | 0.1397 | 0.1397 | 0.2704 | 0.2524 | 0.4333 | 0.1667 |
| S4 evidential (epistemic) | 0.2364 | 0.1807 | 0.1807 | 0.3111 | 0.2524 | 0.0000 | 0.0000 |
| S4 evidential (aleatoric) | 0.2462 | 0.1905 | 0.1905 | 0.3111 | 0.2619 | 0.0000 | 0.0000 |
| S5 top-k disagreement | 0.2577 | 0.2020 | 0.2043 | 0.2852 | 0.2524 | 0.0000 | 0.0000 |
| Oracle | 0.0557 | 0.0000 | 0.0000 | 0.2370 | 0.0190 | 0.8067 | 0.7600 |

## Table 2

| rule | scores | AURC | C*(a=0.15) |
|---|---|---|---|
| R1 best single | s_mc | 0.1954 | 0.4333 |
| R2 conjunctive | s_conf AND s_mc | 0.1895 | 0.4467 |
| R2 conjunctive | s_conf AND s_topk | 0.2396 | 0.0000 |
| R2 conjunctive | s_conf AND s_ev_ep | 0.2463 | 0.0000 |
| R2 conjunctive | s_conf AND s_ent | 0.2235 | 0.2433 |
| R3 logistic fusion | s_conf+s_ent+s_mc+s_ev_ep+s_ev_al+s_topk | 0.1977 | 0.0000 |

## Table 3

| rule | alpha | delta | mean_eval_risk | violation_rate | mean_coverage | coverage_sd |
|---|---|---|---|---|---|---|
| naive (score=s_mc) | 0.1500 | - | 0.2220 | 0.7600 | 0.3244 | 0.2171 |
| ltt (score=s_mc) | 0.1500 | 0.1000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| coverage lost to validity | 0.1500 | 0.1000 | - | - | 0.3244 | - |
| naive (score=s_mc) | 0.1000 | - | 0.3225 | 0.5400 | 0.0554 | 0.1323 |
| ltt (score=s_mc) | 0.1000 | 0.1000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| coverage lost to validity | 0.1000 | 0.1000 | - | - | 0.0554 | - |
