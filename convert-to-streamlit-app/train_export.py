# train_export.py
import json, joblib
import numpy as np
import torch
import torch.nn as nn

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

SEED = 42

# 1) Data
data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
X_raw, y = data.data, data.target
label_names = data.target_names
num_classes = len(label_names)

# 2) Vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,
    lowercase=True,
    stop_words="english",
    strip_accents="unicode",
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
)
X = vectorizer.fit_transform(X_raw).toarray().astype(np.float32)

# 3) Split
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# 4) Model
class NewsMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NewsMLP(Xtr.shape[1], num_classes).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
yt = torch.tensor(ytr, dtype=torch.long, device=device)

# 5) Train
for _ in range(8):
    model.train()
    logits = model(Xt)
    loss = crit(logits, yt)
    opt.zero_grad()
    loss.backward()
    opt.step()

# 6) Export
torch.save(model.state_dict(), "model_state_dict.pt")
joblib.dump(vectorizer, "vectorizer.pkl")
with open("label_names.json", "w", encoding="utf-8") as f:
    json.dump(label_names, f)

with open("meta.json", "w", encoding="utf-8") as f:
    json.dump(
        {"input_dim": int(Xtr.shape[1]), "num_classes": int(num_classes)},
        f
    )

print("Exported: model_state_dict.pt, vectorizer.pkl, label_names.json, meta.json")
