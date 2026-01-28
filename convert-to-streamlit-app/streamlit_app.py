# streamlit_app.py
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

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
# Caching layers (clean + faster iteration)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_dataset():
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    return data.data, data.target, data.target_names


@st.cache_data(show_spinner=False)
def vectorize_texts(X_raw, max_features: int):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        lowercase=True,
        stop_words="english",
        strip_accents="unicode",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    X = vectorizer.fit_transform(X_raw).toarray().astype(np.float32)
    return vectorizer, X


@st.cache_resource(show_spinner=True)
def train_model(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    epochs: int,
    lr: float,
    seed: int,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")
    model = NewsMLP(input_dim=X.shape[1], num_classes=num_classes).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)

    for _ in range(epochs):
        model.train()
        logits = model(Xt)
        loss = crit(logits, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    return model


def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


def plot_confusion_matrix(cm: np.ndarray, labels, normalize: str | None):
    # normalize: None, "true", "pred", "all"
    cm_plot = cm.astype(np.float64)
    title = "Confusion Matrix"

    if normalize is not None:
        with np.errstate(all="ignore"):
            if normalize == "true":
                cm_plot = cm_plot / cm_plot.sum(axis=1, keepdims=True)
                title += " (Normalized by True)"
            elif normalize == "pred":
                cm_plot = cm_plot / cm_plot.sum(axis=0, keepdims=True)
                title += " (Normalized by Pred)"
            elif normalize == "all":
                cm_plot = cm_plot / cm_plot.sum()
                title += " (Normalized Overall)"
        cm_plot = np.nan_to_num(cm_plot)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_plot, interpolation="nearest")
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    # Labels can be long; rotate + smaller font
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    fig.tight_layout()
    return fig


# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Settings")

SEED = st.sidebar.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)
MAX_FEATURES = st.sidebar.select_slider("TF-IDF max_features", options=[2000, 5000, 10000], value=5000)
TEST_SIZE = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
EPOCHS = st.sidebar.slider("Epochs", 1, 25, 8, 1)
LR = st.sidebar.select_slider("Learning rate", options=[1e-4, 3e-4, 1e-3, 3e-3], value=1e-3)

TOP_K = st.sidebar.slider("Top-K classes to show", 3, 10, 5, 1)

cm_norm = st.sidebar.selectbox(
    "Confusion matrix normalization",
    options=["none", "true", "pred", "all"],
    index=1,
)

show_report = st.sidebar.checkbox("Show classification report (text)", value=False)
show_per_class = st.sidebar.checkbox("Show per-class metrics table", value=False)

st.sidebar.divider()
st.sidebar.caption("Tip: Changing settings retrains because caches key off these values.")


# ----------------------------
# Load + vectorize + split + train
# ----------------------------
X_raw, y, label_names = load_dataset()
num_classes = len(label_names)

vectorizer, X_all = vectorize_texts(X_raw, max_features=MAX_FEATURES)

Xtr, Xte, ytr, yte = train_test_split(
    X_all, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
)

model = train_model(Xtr, ytr, num_classes=num_classes, epochs=EPOCHS, lr=LR, seed=SEED)


# ----------------------------
# Layout
# ----------------------------
st.title("20 Newsgroups Text Classifier")
st.caption("TF-IDF → PyTorch MLP. Type/paste text to classify, and view evaluation metrics.")

tab_predict, tab_eval = st.tabs(["🔮 Predict", "📊 Evaluation"])


# ----------------------------
# Predict tab
# ----------------------------
with tab_predict:
    left, right = st.columns([2, 1], vertical_alignment="top")

    with left:
        with st.form("predict_form"):
            text = st.text_area(
                "Input text",
                height=220,
                placeholder="Paste an email/article/post…",
            )
            submitted = st.form_submit_button("Classify")

    with right:
        st.subheader("Quick info")
        st.metric("Classes", num_classes)
        st.metric("Train samples", len(ytr))
        st.metric("Test samples", len(yte))

    if submitted:
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            X_one = vectorizer.transform([text]).toarray().astype(np.float32)
            probs = predict_proba(model, X_one)[0]
            pred_idx = int(np.argmax(probs))

            st.subheader(f"Prediction: {label_names[pred_idx]}")

            # Top-K table
            top_idx = np.argsort(-probs)[:TOP_K]
            rows = [{"class": label_names[i], "probability": float(probs[i])} for i in top_idx]
            st.dataframe(rows, use_container_width=True, hide_index=True)

            # Bar chart
            st.bar_chart(
                {label_names[i]: float(probs[i]) for i in top_idx},
                horizontal=True,
            )


# ----------------------------
# Evaluation tab (metrics + confusion matrix)
# ----------------------------
with tab_eval:
    st.subheader("Test-set performance")

    probs_te = predict_proba(model, Xte)
    yhat = probs_te.argmax(axis=1)

    # Core metrics
    acc = accuracy_score(yte, yhat)

    # Macro averages: treats each class equally (good when classes are imbalanced)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        yte, yhat, average="macro", zero_division=0
    )

    # Weighted averages: weights by class support (more "overall" feel)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        yte, yhat, average="weighted", zero_division=0
    )

    # Display metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("Macro Precision", f"{prec_m:.4f}")
    c3.metric("Macro Recall", f"{rec_m:.4f}")
    c4.metric("Macro F1", f"{f1_m:.4f}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Weighted Precision", f"{prec_w:.4f}")
    c2.metric("Weighted Recall", f"{rec_w:.4f}")
    c3.metric("Weighted F1", f"{f1_w:.4f}")

    # Confusion matrix
    cm = confusion_matrix(yte, yhat, labels=np.arange(num_classes))
    norm_arg = None if cm_norm == "none" else cm_norm
    fig = plot_confusion_matrix(cm, label_names, normalize=norm_arg)
    st.pyplot(fig, clear_figure=True)

    # Optional: per-class metrics table
    if show_per_class:
        prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(
            yte, yhat, average=None, zero_division=0
        )
        rows = [
            {
                "class": label_names[i],
                "precision": float(prec_c[i]),
                "recall": float(rec_c[i]),
                "f1": float(f1_c[i]),
                "support": int(sup_c[i]),
            }
            for i in range(num_classes)
        ]
        st.subheader("Per-class metrics")
        st.dataframe(rows, use_container_width=True, hide_index=True)

    # Optional: sklearn's full report text
    if show_report:
        rep = classification_report(yte, yhat, target_names=label_names, digits=4, zero_division=0)
        st.text(rep)
