import sys
import torch

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
print("TOP-LEVEL KEYS:", list(ck.keys())[:10])
sd = next((ck[k] for k in ("model_state_dict", "generator_state_dict", "state_dict") if k in ck), ck)
for i, (k, v) in enumerate(sd.items()):
    if i < 30:
        print(k, tuple(v.shape))
print("total tensors:", len(sd))