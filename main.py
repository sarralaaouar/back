# # main.py
# from fastapi import FastAPI
# from pydantic import BaseModel
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GATConv
# from torch_geometric.data import Data
# import joblib
# import json
# import numpy as np
# from sentence_transformers import SentenceTransformer

# # --- 1) Your GATNet definition ---


# class GATNet(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.6):
#         super(GATNet, self).__init__()
#         self.conv1 = GATConv(in_channels, hidden_channels,
#                              heads=heads, dropout=dropout)
#         self.conv2 = GATConv(hidden_channels * heads, hidden_channels,
#                              heads=1, concat=False, dropout=dropout)
#         self.lin = nn.Linear(hidden_channels, out_channels)
#         self.dropout = dropout

#     def forward(self, x, edge_index):
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.conv1(x, edge_index)
#         x = F.elu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.conv2(x, edge_index)
#         x = F.elu(x)
#         x = self.lin(x)
#         return x
# # --- end GATNet ---


# app = FastAPI()


# class PatientInput(BaseModel):
#     subject_id: int
#     age: float
#     GENDER: str
#     LANGUAGE: str
#     INSURANCE: str
#     RELIGION: str
#     MARITAL_STATUS: str
#     ETHNICITY: str
#     Maladie_chronique: str
#     Symptômes: str
#     Allergies: str
#     Traitement_régulier: str


# # Globals
# model = None
# scaler = None
# label_encoders = None
# drug_encoder = None
# node_map = None
# edge_index = None
# edge_attr = None
# x_base = None
# sentence_model = None
# feature_dim = None
# num_drugs = None


# @app.on_event("startup")
# def load_artifacts():
#     global model, scaler, label_encoders, drug_encoder
#     global node_map, edge_index, edge_attr, x_base, sentence_model
#     global feature_dim, num_drugs

#     # 1) Preprocessing artifacts
#     scaler = joblib.load("scaler.pkl")
#     label_encoders = joblib.load("label_encoders.pkl")
#     drug_encoder = joblib.load("drug_encoder.pkl")

#     # 2) Base node features
#     x_base = np.load("node_features.npy")
#     feature_dim = x_base.shape[1]
#     num_drugs = len(drug_encoder.classes_)

#     # 3) Graph topology
#     with open("node_map.json") as f:
#         node_map = json.load(f)
#     edge_index = torch.load("edge_index.pt")
#     edge_attr = torch.load("edge_attr.pt")

#     # 4) Model
#     model = GATNet(
#         in_channels=feature_dim,
#         hidden_channels=128,
#         out_channels=num_drugs,
#         heads=4,
#         dropout=0.6
#     )
#     model.load_state_dict(torch.load("best_gat_model.pth", map_location="cpu"))
#     model.eval()

#     # 5) Text embedder
#     sentence_model = SentenceTransformer('all-MiniLM-L6-v2')


# @app.post("/predict")
# def predict(input: PatientInput):
#     # A) Safe categorical encoding with fallback
#     cat_cols = ["GENDER", "LANGUAGE", "INSURANCE", "RELIGION",
#                 "MARITAL_STATUS", "ETHNICITY", "Maladie_chronique"]
#     cat_feats = []
#     for col in cat_cols:
#         raw = getattr(input, col)
#         le = label_encoders[col]
#         # If unseen, pick the encoder's first known class
#         if raw not in le.classes_:
#             raw = le.classes_[0]
#         enc = le.transform([raw])[0]
#         cat_feats.append(int(enc))

#     # B) Age + categorical array
#     raw_feats = np.array([input.age] + cat_feats).reshape(1, -1)

#     # C) Text embedding
#     txt = f"{input.Symptômes} {input.Allergies} {input.Traitement_régulier}"
#     emb = sentence_model.encode([txt])  # shape (1, embed_dim)

#     # D) Combine + scale
#     combined = np.hstack([raw_feats, emb])         # (1, feature_dim)
#     scaled = scaler.transform(combined)          # (1, feature_dim)

#     # E) Append new node features
#     new_node_id = len(node_map)
#     node_map[f"patient_{input.subject_id}"] = new_node_id
#     x_all = np.vstack([x_base, scaled])            # (N_base+1, feature_dim)
#     x_tensor = torch.tensor(x_all, dtype=torch.float)

#     # F) Build graph data
#     data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr)

#     # G) Inference
#     with torch.no_grad():
#         out = model(data.x, data.edge_index)    # [num_nodes, num_drugs]
#         logits = out[new_node_id]                  # [num_drugs]
#         probs = torch.softmax(logits, dim=0).cpu().numpy()

#     # H) Top‑5 drugs
#     topk_idx = probs.argsort()[-5:][::-1]
#     drugs = drug_encoder.inverse_transform(topk_idx)

#     return {
#         "recommended_drugs": drugs.tolist(),
#         "probabilities":     probs[topk_idx].tolist()
#     }

# # Run with:
# #    uvicorn main:app --reload --port 8000
# main.py

# main.py
from dotenv import load_dotenv
import os
import json
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import faiss
import math
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI

app = FastAPI()

# Autoriser uniquement ton domaine frontend
origins = [
    "https://front-theta-pied-26.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # ou ["*"] pour tout autoriser (non recommandé en prod)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment vars
load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1) GATNet ---


class GATNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.6):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels,
                             heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels,
                             heads=1, concat=False, dropout=dropout)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        return self.lin(x)
# --- end GATNet ---


class PatientInput(BaseModel):
    subject_id:        int
    age:               float
    GENDER:            str
    LANGUAGE:          str
    INSURANCE:         str
    RELIGION:          str
    MARITAL_STATUS:    str
    ETHNICITY:         str
    Maladie_chronique: str
    Symptômes:         str
    Allergies:         str
    Traitement_régulier: str
    narrative:         str


# Globals
model = scaler = label_encoders = drug_encoder = None
x_base = edge_index = edge_attr = sentence_model = None
feature_dim = num_drugs = None

df_mimic = None
faiss_index = None
tokenizer_rag = None
model_rag = None

client: OpenAI

# === RAG helpers ===


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)


def embed_query(query: str) -> np.ndarray:
    encoded = tokenizer_rag(query, padding=True,
                            truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model_rag(**encoded)
    emb = mean_pooling(out, encoded["attention_mask"])
    return F.normalize(emb, p=2, dim=1).cpu().numpy()


def semantic_search(query: str, top_k: int = 3) -> pd.DataFrame:
    q_emb = embed_query(query)
    _, indices = faiss_index.search(q_emb, top_k)
    return df_mimic.iloc[indices[0]].reset_index(drop=True)


@app.on_event("startup")
def load_artifacts():
    global model, scaler, label_encoders, drug_encoder
    global x_base, edge_index, edge_attr, sentence_model
    global feature_dim, num_drugs, client
    global df_mimic, faiss_index, tokenizer_rag, model_rag

    # 1) GAT artifacts
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    drug_encoder = joblib.load("drug_encoder.pkl")

    x_base = np.load("node_features.npy")
    feature_dim = x_base.shape[1]
    num_drugs = len(drug_encoder.classes_)

    with open("node_map.json") as f:
        # node_map no longer needed for indexing
        _ = json.load(f)
    edge_index = torch.load("edge_index.pt")
    edge_attr = torch.load("edge_attr.pt")

    model = GATNet(
        in_channels=feature_dim,
        hidden_channels=128,
        out_channels=num_drugs,
        heads=4,
        dropout=0.6
    )
    model.load_state_dict(torch.load("best_gat_model.pth", map_location="cpu"))
    model.eval()

    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

    # 2) RAG artifacts
    df_mimic = pd.read_csv("mimic_data_with_combined_text.csv")
    faiss_index = faiss.read_index("mimic_faiss.index")
    tokenizer_rag = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2")
    model_rag = AutoModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2")
    model_rag.eval()

    # 3) OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    client = OpenAI(api_key=api_key)


@app.post("/predict")
def predict(input: PatientInput):
    # A) Tabular + text features for GAT
    cat_cols = ["GENDER", "LANGUAGE", "INSURANCE", "RELIGION",
                "MARITAL_STATUS", "ETHNICITY", "Maladie_chronique"]
    cat_feats = []
    for col in cat_cols:
        raw = getattr(input, col)
        le = label_encoders[col]
        if raw not in le.classes_:
            raw = le.classes_[0]
        cat_feats.append(int(le.transform([raw])[0]))

    raw_feats = np.array([input.age] + cat_feats).reshape(1, -1)
    txt = f"{input.Symptômes} {input.Allergies} {input.Traitement_régulier}"
    emb = sentence_model.encode([txt])
    combined = np.hstack([raw_feats, emb])
    scaled = scaler.transform(combined)

    # E) Append new node at fixed index base_n
    base_n = x_base.shape[0]
    new_node_id = base_n
    x_all = np.vstack([x_base, scaled])
    x_tensor = torch.tensor(x_all, dtype=torch.float)
    data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr)

    # G) Inference
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.softmax(out[new_node_id], dim=0).cpu().numpy()

    topk_idx = probs.argsort()[-5:][::-1]
    drugs = drug_encoder.inverse_transform(topk_idx)
    scores = probs[topk_idx]
    safe_scores = [float(v) if math.isfinite(v) else 0.0 for v in scores]

    # B) RAG retrieval
    rag_query = f"{input.Maladie_chronique} | {input.Symptômes} | {input.Allergies} | {input.Traitement_régulier}"
    sims = semantic_search(rag_query, top_k=3)
    cases = sims[["Maladie_chronique", "Symptômes", "Allergies",
                  "Traitement_régulier"]].to_dict(orient="records")

    # C) Build prompt
    top_drug = drugs[0]
    sim_text = ""
    for i, rec in enumerate(cases, 1):
        sim_text += (
            f"- Cas {i} :\n"
            f"    • Maladie chronique   : {rec['Maladie_chronique']}\n"
            f"    • Symptômes           : {rec['Symptômes']}\n"
            f"    • Allergies           : {rec['Allergies']}\n"
            f"    • Traitement régulier : {rec['Traitement_régulier']}\n\n"
        )

    prompt = (
        "Voici trois cas similaires extraits du dataset :\n\n"
        f"{sim_text}"
        "Patient cible :\n"
        f"- Sujet ID           : {input.subject_id}\n"
        f"- Âge                 : {input.age}\n"
        f"- Sexe                : {input.GENDER}\n"
        f"- Maladie chronique   : {input.Maladie_chronique}\n"
        f"- Symptômes           : {input.Symptômes}\n"
        f"- Allergies           : {input.Allergies}\n"
        f"- Traitement régulier : {input.Traitement_régulier}\n\n"
        f"- Narrative           : {input.narrative}\n\n"
        f"Le système recommande **{top_drug}**.\n"
        "Explique pourquoi ce médicament est adapté à ce patient, "
        "et si le système a recommandé un médicament auquel le patient est allergique, "
        "tu peux modifier la recommandation en te basant sur les cas similaires."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system",  "content": "Tu es un assistant médical expert."},
                {"role": "user",    "content": prompt}
            ],
            temperature=0.7
        )
        explanation = resp.choices[0].message.content.strip()
    except Exception as e:
        explanation = f"Erreur d'explication LLM: {e}"

    return {
        "recommended_drugs": drugs.tolist(),
        "probabilities":     safe_scores,
        "similar_cases":     cases,
        "explanation":       explanation
    }
