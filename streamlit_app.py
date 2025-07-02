# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from transformers import AutoTokenizer
import re

@st.cache_resource
def load_models():
    model_dnabert = tf.keras.models.load_model(
        "models/best_model_phase2_2406.keras",
        custom_objects={
            "extract_bert_outputs": lambda x: x,
            "hybrid_loss": lambda y_true, y_pred: tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred))
        }, compile=False
    )
    model_lgbm = joblib.load("models/lightgbm_final_model.pkl")
    meta_model = joblib.load("models/meta_model_stacking_lr.pkl")
    return model_dnabert, model_lgbm, meta_model

@st.cache_resource
def load_encoders_tokenizer():
    with open("models/encoders_201nt_meta.pkl", "rb") as f:
        enc = joblib.load(f)
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6")
    return enc, tokenizer

def kmer_tokenizer(seq, k=6):
    return [seq[i:i+k] for i in range(len(seq)-k+1)]

def compute_gc(seq):
    seq = seq.upper()
    return round((seq.count("G") + seq.count("C")) / len(seq), 4)

def encode_input(seq201, chrom, ref, alt, pos, variant_class, variant_type, hugo_symbol, enc, tokenizer):
    seq201 = seq201.upper()
    kmer_ids = tokenizer.convert_tokens_to_ids(kmer_tokenizer(seq201))
    x_dict = {
        "seq_in": tf.constant([kmer_ids], dtype=tf.int32),
        "chr_in": tf.constant([enc['le_chr'].transform([chrom])[0]]),
        "ref_in": tf.constant([enc['le_ref'].transform([ref])[0]]),
        "alt_in": tf.constant([enc['le_alt'].transform([alt])[0]]),
        "pos_in": tf.constant([enc['scaler_pos'].transform(np.log1p([[pos]]))[0][0]]),
        "gc_in": tf.constant([enc['scaler_gc'].transform([[compute_gc(seq201)]])[0][0]]),
        "gc3_in": tf.constant(enc['scaler_gc3'].transform([[compute_gc(seq201[:67]), compute_gc(seq201[67:134]), compute_gc(seq201[134:])]])[0]),
        "vc_in": tf.constant([enc['le_vc'].transform([variant_class])[0]]),
        "vt_in": tf.constant([enc['le_vt'].transform([variant_type])[0]]),
        "hugo_in": tf.constant([enc['le_hugo'].transform([hugo_symbol])[0]]),
    }
    return x_dict

# === Interface Streamlit ===
st.title("🧬 Prédiction du type de cancer à partir d'une séquence ADN")
st.markdown("Ce système utilise DNABERT + LightGBM + Stacking pour une classification précise.")

with st.form("dna_form"):
    seq_input = st.text_area("Saisir une séquence ADN de 201 nucléotides :", max_chars=201)
    col1, col2 = st.columns(2)
    chrom = col1.text_input("Chromosome", value="1")
    ref = col1.text_input("Allèle Référence", value="A")
    alt = col2.text_input("Allèle Mutant", value="T")
    pos = col2.number_input("Position (Start_Position)", min_value=1, value=1234567)

    variant_class = st.selectbox("Variant Classification", ["Missense_Mutation", "Nonsense_Mutation", "Silent"])
    variant_type = st.selectbox("Variant Type", ["SNP", "INS", "DEL"])
    hugo_symbol = st.text_input("Nom du gène (HUGO)", value="TP53")
    submitted = st.form_submit_button("Prédire")

if submitted:
    if not re.fullmatch(r"[ACGTacgt]{201}", seq_input or ""):
        st.error("❌ La séquence doit contenir exactement 201 nucléotides (A, C, G, T).")
    else:
        with st.spinner("Chargement des modèles..."):
            model_dnabert, model_lgbm, meta_model = load_models()
            enc, tokenizer = load_encoders_tokenizer()

        x_dict = encode_input(seq_input, chrom, ref, alt, pos, variant_class, variant_type, hugo_symbol, enc, tokenizer)
        probs_dnabert = model_dnabert(x_dict, training=False).numpy()

        extractor = tf.keras.Model(inputs=model_dnabert.input, outputs=model_dnabert.get_layer("dense_18").output)
        features = extractor(x_dict, training=False).numpy()
        probs_lgbm = model_lgbm.predict_proba(features)

        x_stack = np.hstack([probs_dnabert, probs_lgbm])
        y_pred = meta_model.predict(x_stack)[0]
        cancer_type = enc['le_lbl'].inverse_transform([y_pred])[0]
        st.success(f"🧬 Type de cancer prédit : **{cancer_type}**")

        st.subheader("Probabilités par classe :")
        probas = meta_model.predict_proba(x_stack)[0]
        df_probs = pd.DataFrame({
            "Classe": enc['le_lbl'].inverse_transform(np.arange(len(probas))),
            "Probabilité": np.round(probas * 100, 2)
        })
        st.dataframe(df_probs.sort_values("Probabilité", ascending=False), use_container_width=True)
