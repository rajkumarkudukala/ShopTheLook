import os
import json
import numpy as np
import faiss
from offline_indexer import ARTIFACTS_DIR, precompute_attributes

def finish():
    emb_file = os.path.join(ARTIFACTS_DIR, "checkpoint_embeddings.npy")
    ids_file = os.path.join(ARTIFACTS_DIR, "checkpoint_ids.json")
    
    if not os.path.exists(emb_file) or not os.path.exists(ids_file):
        print("Checkpoints not found!")
        return
        
    embeddings = list(np.load(emb_file))
    valid_ids = json.load(open(ids_file))
    print(f"Loaded {len(valid_ids)} items from checkpoint.")
    
    embeddings_array = np.array(embeddings, dtype=np.float32)
    np.save(os.path.join(ARTIFACTS_DIR, "catalog_embeddings.npy"), embeddings_array)
    json.dump(valid_ids, open(os.path.join(ARTIFACTS_DIR, "catalog_ids.json"), "w"))
    
    precompute_attributes(valid_ids, embeddings_array)
    
    dim = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_array)
    faiss.write_index(index, os.path.join(ARTIFACTS_DIR, "catalog.index"))
    
    print(f"FAISS index built: {index.ntotal} vectors.")

if __name__ == "__main__":
    finish()
