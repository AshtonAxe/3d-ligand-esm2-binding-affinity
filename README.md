# AI Drug Discovery: Protein-Ligand Affinity Prediction

This project implements a deep learning pipeline for predicting protein-ligand binding affinity using 3D molecular graph embeddings and protein language model embeddings (ESM-2). 

- **Dataset:** BindingDB
- **Goal:** Predict pKi with low RMSE across held-out test pairs
- **Result:** Achieved 1.05 RMSE, outperforming baselines such as DeepAtom (1.23 RMSE)
- **Tech Stack:** PyTorch, PyTorch Geometric, RDKit, ESM-2, Cross-Attention Fusion

*This project was conducted as part of independent research on AI-driven drug discovery at MIT.*

[📄 Read the full paper (PDF)](ligand_protein_affinity_prediction.pdf)

---

## How to run the pipeline

```bash
git clone https://github.com/Ashton_Axe/3d-ligand-esm2-binding-affinity.git
cd 3d-ligand-esm2-binding-affinity
conda create -n ligand_protein_prediction python=3.11
conda activate ligand_protein_prediction
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 torch-geometric==2.5.3 torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu118.html numpy pandas rdkit
python scripts/run_experiment.py
