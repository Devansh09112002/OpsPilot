import os, gc, psutil
p=psutil.Process(os.getpid())
def rss(): return p.memory_info().rss/1024/1024
from app.core.config import get_settings
from app.ml.predictor import predictor
import pandas as pd
from data_pipeline.features import MODEL_FEATURES
print(f"before load            {rss():7.1f} MB")
predictor.load(get_settings().artifact_dir)
print(f"after load             {rss():7.1f} MB")
gc.collect()
print(f"after gc.collect()     {rss():7.1f} MB")
row={f:1.0 for f in MODEL_FEATURES}
for c in ["customer_state","seller_state","product_category"]: row[c]="SP"
for _ in range(50): predictor.predict_one(row)
gc.collect()
print(f"after 50 inferences+gc {rss():7.1f} MB")
m=predictor._model
clf=m.named_steps["clf"]
booster=clf.get_booster()
print("n_trees:", len(booster.get_dump()))
pre=m.named_steps["pre"]
print("expanded features:", len(pre.get_feature_names_out()))
