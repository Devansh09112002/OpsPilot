import os, gc, psutil
p=psutil.Process(os.getpid())
def rss(): return p.memory_info().rss/1024/1024
print("OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS"))
from app.core.config import get_settings
from app.ml.predictor import predictor
from data_pipeline.features import MODEL_FEATURES
from app.main import app
print(f"app imported           {rss():7.1f} MB")
predictor.load(get_settings().artifact_dir)
print(f"after model load       {rss():7.1f} MB")
m=predictor._model; m.named_steps["clf"].set_params(n_jobs=1)
row={f:1.0 for f in MODEL_FEATURES}
for c in ["customer_state","seller_state","product_category"]: row[c]="SP"
import time
for _ in range(20): predictor.predict_one(row)
t=time.perf_counter()
for _ in range(100): predictor.predict_one(row)
ms=(time.perf_counter()-t)/100*1000
gc.collect()
print(f"after 120 inferences   {rss():7.1f} MB")
print(f"single-row latency     {ms:7.2f} ms")
