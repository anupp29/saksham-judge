import torch
from model import load_model
from feature_extractor import SEQUENCE_LENGTH, FEATURE_DIM

device = torch.device("cpu")
model = load_model("best_boxing_model.pth", device)
model.eval()
dummy = torch.randn(1, SEQUENCE_LENGTH, FEATURE_DIM)
traced = torch.jit.trace(model, (dummy,))
torch.jit.save(traced, "boxing_traced.pt")
print("Warm-start cache written -> boxing_traced.pt")