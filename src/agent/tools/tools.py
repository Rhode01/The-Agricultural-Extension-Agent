from langchain_core.tools import tool
from pathlib import Path
import tensorflow as tf
import numpy as np
from src.data.vector_store.store import LLMvector_store
from src.models.LLM.embeddings import LLMEmbedding

@tool
def disease_analysis_tool(image_path: Path) -> list[str]:
    """
    Uses the custom maize-disease model to detect diseases in the given leaf image.
    Returns a list of disease identifiers, e.g. ["gray_leaf_spot"].
    """
    
    model_dir   = Path(__file__).resolve().parents[1] / "models" / "saved_model"
    model_path  = model_dir / "best_model.keras"
    model       = tf.keras.models.load_model(model_path)

    _, h, w, _ = model.input_shape
    img        = tf.keras.utils.load_img(image_path, target_size=(h, w))
    arr        = tf.keras.utils.img_to_array(img)[None, ...]
    preprocess = getattr(tf.keras.applications, "EfficientNetV2SPreprocessInput", None)
    arr        = preprocess(arr) if preprocess else arr / 255.0

    preds      = model.predict(arr)[0]
    idxs       = np.where(preds > 0.5)[0]
    class_names_file = model_dir / "class_names.txt"
    with open(class_names_file) as f:
        class_names = [line.strip() for line in f]
    return [class_names[i] for i in idxs]


@tool
def get_treatment_advice_tool(disease_name: str) -> str:
    """
    Queries the RAG-backed knowledge base for treatment advice
    for the specified disease, e.g. "gray_leaf_spot".
    Returns a concise recommendation string.
    """
    
    persist_dir = Path(__file__).resolve().parents[1] / "knowledge_store"

    embeddings  = LLMEmbedding().client
    vectordb    = LLMvector_store(embeddings).vector_store

    docs = vectordb.similarity_search(disease_name, k=3)

    advice = "\n\n".join(doc.page_content for doc in docs)
    return advice
