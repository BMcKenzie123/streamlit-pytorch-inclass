# streamlit_app.py
import json, joblib, numpy as np, torch, torch.nn as nn
import streamlit as st

st.set_page_config(page_title="20 Newsgroups Classifier", layout="wide")

@st.cache_resource  # load once per process
def load_resources():
    vectorizer = joblib.load("vectorizer.pkl")
    with open("label_names.json") as f:
        label_names = json.load(f)

    class NewsMLP(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.0),
                nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.0),
                nn.Linear(256, num_classes)
            )
        def forward(self, x): return self.net(x)

    model = NewsMLP(input_dim=vectorizer.max_features or len(vectorizer.vocabulary_), 
                    num_classes=len(label_names))
    model.load_state_dict(torch.load("model_state_dict.pt", map_location="cpu"))
    model.eval()
    return vectorizer, label_names, model

vectorizer, label_names, model = load_resources()

st.title("20 Newsgroups Text Classifier (PyTorch + TF-IDF)")
st.caption("Enter text, get predicted topic with probabilities.")

with st.form("predict"):
    text = st.text_area("Paste text", height=200, placeholder="Type or paste an email/article…")
    submitted = st.form_submit_button("Classify")

def predict(texts):
    X = vectorizer.transform(texts).toarray()  # small batch OK; stays CPU
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).numpy()
        preds = probs.argmax(axis=1)
    return preds, probs

if submitted:
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        pred, probs = predict([text])
        label = label_names[int(pred[0])]
        st.subheader(f"Prediction: {label}")
        # Show top-5 classes
        top = np.argsort(-probs[0])[:5]
        st.write({label_names[i]: float(probs[0][i]) for i in top})
