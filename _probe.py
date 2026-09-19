import pandas as pd, numpy as np
from data_pipeline import spec
f=pd.read_parquet(spec.PROCESSED_DIR/"order_features.parquet")
o=pd.read_parquet(spec.PROCESSED_DIR/"order_outcomes.parquet")
d=f.merge(o,on="order_id")
neg = d.days_handover_to_estimate < 0
print(f"handed to carrier AFTER promised date: {neg.sum():,} ({neg.mean():.2%}); late rate {d.loc[neg,'is_late'].mean():.4%}")
print(f"slack >= 0: {(~neg).sum():,}, late rate {d.loc[~neg,'is_late'].mean():.3%}")
print()
for s in ['train','validation','test']:
    sub=d[d.split==s]; sp=sub[sub.days_handover_to_estimate>=0]
    print(f"  {s:<11s} all n={len(sub):>6,} late={sub.is_late.sum():>5,} | slack>=0: n={len(sp):>6,} late={sp.is_late.sum():>5,} ({sp.is_late.mean():.3%})")
print()
m=pd.read_parquet(spec.PROCESSED_DIR/"snapshot_members.parquet")
pre=m[~m.is_overdue].merge(d,on="order_id")
print(f"pre-deadline snapshot orders: {len(pre):,}; negative handover slack: {(pre.days_handover_to_estimate<0).sum():,}")
print("\nslack distribution among slack>=0 (test):")
t=d[(d.split=='test')&(d.days_handover_to_estimate>=0)]
print(t.days_handover_to_estimate.describe().round(2).to_string())
