from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from tqdm import tqdm

# Load ESM-2 model (650M) and tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
model.eval().to("cuda")  # Move model to GPU

# Tokenize a batch of sequences
def batch_tokenize(seqs, device="cuda"):
    return tokenizer(
        seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True
    ).to(device)

# Compute mean embeddings for a batch
def esm2_embed_batch(seqs, device="cuda"):
    inputs = batch_tokenize(seqs, device)
    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = []
    for i, length in enumerate(inputs["attention_mask"].sum(dim=1)):
        token_embeddings = outputs.last_hidden_state[i, 1:length-1, :]  # Exclude [CLS], [EOS]
        mean_embedding = token_embeddings.mean(dim=0)
        embeddings.append(mean_embedding.cpu().numpy().tolist())
    return embeddings


final_subset["esm2"] = None  # Create column if not present
batch_size = 32

for i in tqdm(range(0, len(final_subset), batch_size)):
    batch = final_subset.iloc[i:i + batch_size]
    seqs = batch["seq"].tolist()

    try:
        embeddings = esm2_embed_batch(seqs)
        for j, emb in enumerate(embeddings):
            final_subset.at[i + j, "esm2"] = emb
        if i % 256 == 0:
            print(f"Processed rows {i} to {i + len(batch) - 1}")
    except Exception as e:
        print(f"Failed at batch {i}: {e}")

    if i % 5000 == 0:
      filename = f"esm2_partial_{i}.pkl"
      final_subset.to_pickle(filename)
      print(f"💾 Saved partial checkpoint to {filename}")

final_subset.to_csv("bindingdb_esm2_embedded_100k.csv", index=False)
