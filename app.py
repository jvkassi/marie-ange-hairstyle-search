import streamlit as st
import os
import tempfile
import time
import json
import pickle
import numpy as np

from google import genai
from google.genai import types
from PIL import Image

st.set_page_config(page_title="Happee Beauty - Hairstyle Search", page_icon="💇‍♀️", layout="wide")

st.title("💇‍♀️ Happee Beauty - Hairstyle Search")
st.write("Trouvez la coiffure idéale en un clic ! Cherchez par **image** ou par **texte**.")

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.warning("Veuillez renseigner votre clé API Gemini :")
    api_key = st.text_input("Clé API :", type="password")
    if not api_key:
        st.stop()

# Initialize the new SDK client
client = genai.Client(api_key=api_key)

# -----------------
# DATA MANAGEMENT
# -----------------
DATA_DIR = "/data"
DB_PATH = os.path.join(DATA_DIR, "hairstyles.pkl")

# Use a local folder fallback if /data doesn't exist (e.g. for local dev)
if not os.path.exists(DATA_DIR):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
    except Exception:
        DATA_DIR = "./data"
        os.makedirs(DATA_DIR, exist_ok=True)
        DB_PATH = os.path.join(DATA_DIR, "hairstyles.pkl")

def load_db():
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Erreur lors du chargement de la base de données : {e}")
    return []

def save_db(db):
    try:
        with open(DB_PATH, "wb") as f:
            pickle.dump(db, f)
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde : {e}")

if "db" not in st.session_state:
    st.session_state["db"] = load_db()

def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

# -----------------
# SIDEBAR: ADD HAIRSTYLE
# -----------------
st.sidebar.header("➕ Ajouter une coiffure")
uploaded_file = st.sidebar.file_uploader(
    "Photo ou Vidéo de la coiffure", 
    type=["jpg", "jpeg", "png", "mp4", "mov"],
)
description = st.sidebar.text_area("Description (ex: Tresses courtes blondes avec perles)")

if st.sidebar.button("Enregistrer la coiffure") and uploaded_file and description:
    with st.spinner("Analyse et vectorisation par Gemini..."):
        # Save file to /data
        timestamp = int(time.time())
        file_ext = uploaded_file.name.split('.')[-1]
        local_filename = f"{timestamp}_{uploaded_file.name}"
        local_filepath = os.path.join(DATA_DIR, local_filename)
        
        with open(local_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())

        try:
            # 1. Upload to Gemini
            st.toast("Upload sécurisé vers Gemini...")
            g_file = client.files.upload(file=local_filepath, config={"display_name": local_filename})
            
            # Wait for processing if video
            if uploaded_file.type.startswith("video"):
                st.toast("Traitement vidéo en cours par Google...")
                while True:
                    state_val = getattr(g_file, 'state', None)
                    if state_val and 'PROCESSING' not in str(state_val):
                        break
                    time.sleep(3)
                    g_file = client.files.get(name=g_file.name)

            # 2. Embed media + text
            st.toast("Création de l'empreinte sémantique (Embedding 2.0)...")
            # We embed BOTH the image/video and the text description together as a single multimodal prompt
            result = client.models.embed_content(
                model='gemini-embedding-2-preview',
                contents=[g_file, description]
            )
            embedding_vector = result.embeddings[0].values
            
            # 3. Save to our database
            new_entry = {
                "id": timestamp,
                "filename": local_filename,
                "filepath": local_filepath,
                "type": uploaded_file.type,
                "description": description,
                "embedding": embedding_vector,
                "g_file_uri": g_file.uri
            }
            st.session_state["db"].append(new_entry)
            save_db(st.session_state["db"])
            
            st.sidebar.success("Coiffure ajoutée avec succès ! ✅")
            st.balloons()
            
        except Exception as e:
            st.sidebar.error(f"Erreur : {e}")

st.sidebar.markdown("---")
st.sidebar.write(f"📚 **Total en base :** {len(st.session_state['db'])} coiffures")

# -----------------
# MAIN PAGE: SEARCH
# -----------------
st.subheader("🔍 Moteur de recherche")

search_mode = st.radio("Mode de recherche", ["Texte", "Image/Vidéo"], horizontal=True)

search_embedding = None
search_query_text = ""
search_query_file = None

if search_mode == "Texte":
    search_query_text = st.text_input("Décrivez la coiffure recherchée (ex: 'Chignon de mariage élégant')")
else:
    search_query_file = st.file_uploader("Uploadez une image/vidéo de référence", type=["jpg", "jpeg", "png", "mp4", "mov"])

if st.button("Rechercher", type="primary"):
    if not st.session_state["db"]:
        st.warning("La base de données est vide. Ajoutez des coiffures depuis le menu latéral !")
    elif search_mode == "Texte" and not search_query_text:
        st.warning("Veuillez saisir un texte de recherche.")
    elif search_mode == "Image/Vidéo" and not search_query_file:
        st.warning("Veuillez uploader un fichier de référence.")
    else:
        with st.spinner("Recherche des meilleures correspondances..."):
            try:
                # Get embedding for the search query
                if search_mode == "Texte":
                    result = client.models.embed_content(
                        model='gemini-embedding-2-preview',
                        contents=search_query_text
                    )
                    search_embedding = result.embeddings[0].values
                else:
                    # Upload the search reference file temporarily to Gemini
                    with tempfile.NamedTemporaryFile(delete=False, suffix="."+search_query_file.name.split('.')[-1]) as tmp:
                        tmp.write(search_query_file.getvalue())
                        tmp_path = tmp.name
                    
                    g_search_file = client.files.upload(file=tmp_path)
                    
                    if search_query_file.type.startswith("video"):
                        while True:
                            state_val = getattr(g_search_file, 'state', None)
                            if state_val and 'PROCESSING' not in str(state_val):
                                break
                            time.sleep(3)
                            g_search_file = client.files.get(name=g_search_file.name)
                            
                    result = client.models.embed_content(
                        model='gemini-embedding-2-preview',
                        contents=g_search_file
                    )
                    search_embedding = result.embeddings[0].values
                    
                    # Cleanup search file
                    os.remove(tmp_path)
                    try:
                        client.files.delete(name=g_search_file.name)
                    except:
                        pass
                
                # Compute similarities
                similarities = []
                for item in st.session_state["db"]:
                    sim = cosine_similarity(search_embedding, item["embedding"])
                    similarities.append((sim, item))
                    
                similarities.sort(key=lambda x: x[0], reverse=True)
                
                st.success("Résultats trouvés !")
                st.markdown("---")
                
                # Display top 3 results
                cols = st.columns(3)
                for i, (sim, item) in enumerate(similarities[:3]):
                    with cols[i]:
                        st.write(f"**Top {i+1}** (Pertinence: {sim*100:.1f}%)")
                        if item["type"].startswith("image"):
                            try:
                                img = Image.open(item["filepath"])
                                st.image(img, use_container_width=True)
                            except:
                                st.write("*(Image indisponible sur le disque)*")
                        elif item["type"].startswith("video"):
                            try:
                                st.video(item["filepath"])
                            except:
                                st.write("*(Vidéo indisponible sur le disque)*")
                        
                        st.info(item["description"])
                        
            except Exception as e:
                st.error(f"Erreur lors de la recherche : {e}")

