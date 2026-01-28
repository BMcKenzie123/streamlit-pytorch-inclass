# streamlit_app.py
import numpy as np
import torch
import torch.nn as nn
import streamlit as st

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="20 Newsgroups Classifier", layout="wide")


# ----------------------------
# Model
# ----------------------------
class NewsMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Train once, cache in memory
# ----------------------------
@st.cache_resource
def train_and_cache():
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

    # 3) Split (we only really need train for this demo)
    Xtr, _, ytr, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # 4) Train (CPU)
    device = torch.device("cpu")
    model = NewsMLP(input_dim=Xtr.shape[1], num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.long, device=device)

    for _ in range(8):
        model.train()
        logits = model(Xt)
        loss = crit(logits, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    return vectorizer, label_names, model


vectorizer, label_names, model = train_and_cache()


# ----------------------------
# UI
# ----------------------------
st.title("20 Newsgroups Text Classifier (PyTorch + TF-IDF)")
st.caption("Enter text, get predicted topic with probabilities.")

with st.form("predict"):
    text = st.text_area("Paste text", height=200, placeholder="Type or paste an email/article…")
    submitted = st.form_submit_button("Classify")


def predict(texts):
    X = vectorizer.transform(texts).toarray().astype(np.float32)
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
    return preds, probs


if submitted:
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        pred, probs = predict([text])
        pred_idx = int(pred[0])
        st.subheader(f"Prediction: {label_names[pred_idx]}")

        top = np.argsort(-probs[0])[:5]
        st.write({label_names[i]: float(probs[0][i]) for i in top})
