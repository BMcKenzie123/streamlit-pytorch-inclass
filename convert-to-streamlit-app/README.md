# 20 Newsgroups Classifier (PyTorch + Streamlit)

This project demonstrates how to transform a plain PyTorch training script into a fully interactive **Streamlit app**.  
The app classifies user-provided text into one of the 20 Newsgroups categories, using a TF-IDF vectorizer and a simple feed-forward neural network (MLP).

---

## 📂 Project Structure

your-app/  
|-- `train_export.py` *# Script to train once and export artifacts*  
|-- `streamlit_app.py` *# Main Streamlit app (inference only)*  
|-- `model_state_dict.pt` *# Saved PyTorch model weights*  
|-- `vectorizer.pkl` *# Trained TF-IDF vectorizer*  
|-- `label_names.json` *# List of newsgroup labels*  
|-- `requirements.txt` *# Python dependencies*  
|-- `README.md` *# This file*  


### Files
- **`train_export.py`**: trains the model locally, exports artifacts (model weights, vectorizer, labels).
- **`streamlit_app.py`**: lightweight inference app that loads artifacts and provides a UI.
- **Artifacts (`.pt`, `.pkl`, `.json`)**: pre-trained components needed at inference.
- **`requirements.txt`**: ensures consistent environment for local/dev/deploy.

---

## 🔄 Converting the Original Script → Streamlit App

1. **Start from a raw training script**:
   - Data loading, preprocessing, training, and evaluation are all in one file.

2. **Split responsibilities**:
   - Keep training in **`train_export.py`** (run offline, once).
   - Move inference/UI to **`streamlit_app.py`**.

3. **Export artifacts**:
   - Save model with `torch.save(model.state_dict(), "model_state_dict.pt")`.
   - Save vectorizer with `joblib.dump(vectorizer, "vectorizer.pkl")`.
   - Save labels with `json.dump(label_names, f)`.

4. **Load in the app**:
   - Add a function with `@st.cache_resource` to load model/vectorizer/labels only once.
   - Define the same model class in `streamlit_app.py` and load weights.

5. **Build the UI**:
   - `st.text_area()` for input text.
   - `st.form_submit_button()` for prediction.
   - Display predicted label + top probabilities.

---

## ▶️ Run Locally

1. **Install dependencies**:

   pip install -r requirements.txt

2. **Train and export artifacts** (first time only):
   
   python train_export.py

3. **Start the app**:

   streamlit run streamlit_app.py

4. **Open the browser**:

    By default: http://localhost:8501

## ☁️ Deploy to Streamlit Community Cloud

1. **Push repo to GitHub** with:

* `streamlit_app.py`
* artifacts (`model_state_dict.pt`, `vectorizer.pkl`, `label_names.json`)
* `requirements.txt`
* `README.md`

2. **Go to** share.streamlit.io (Streamlit Cloud).

3. **Connect GitHub**:
* Select your repository.
* Set the entry point to **`streamlit_app.py`**.

4. **Deploy**:
* The app builds automatically.
* Share the URL provided by Streamlit Cloud.