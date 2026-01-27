from pathlib import Path
import pickle



path = Path(__file__).resolve().parent.parent.parent / "results" / "cma_run_20260122T105743Z" / "run_0001.pkl"

with open(path,"rb") as f:
    data = pickle.load(f)
print(data)