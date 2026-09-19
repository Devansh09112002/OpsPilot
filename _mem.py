import os, psutil
p = psutil.Process(os.getpid())
def rss(): return p.memory_info().rss / 1024 / 1024
print(f"baseline python        {rss():7.1f} MB")
from app.core.config import get_settings
import pandas, numpy, sklearn
print(f"+ pandas/numpy/sklearn {rss():7.1f} MB")
import xgboost
print(f"+ xgboost              {rss():7.1f} MB")
from app.main import app
print(f"+ fastapi app          {rss():7.1f} MB")
from app.ml.predictor import predictor
predictor.load(get_settings().artifact_dir)
print(f"+ model artifact       {rss():7.1f} MB")
from app.agent.graph import build_graph
build_graph()
print(f"+ langgraph            {rss():7.1f} MB")
from google import genai
print(f"+ google-genai         {rss():7.1f} MB")
# score once to force lazy allocations
import pandas as pd
from data_pipeline.features import MODEL_FEATURES
row = {f: 1.0 for f in MODEL_FEATURES}
for c in ["customer_state","seller_state","product_category"]: row[c]="SP"
predictor.predict_one(row)
print(f"after first inference  {rss():7.1f} MB")
