# AI Drug Discovery: Protein-Ligand Affinity Prediction

This project implements a deep learning pipeline for predicting protein-ligand binding affinity using 3D molecular graph embeddings and protein language model embeddings (ESM-2). 

- **Dataset:** BindingDB
- **Goal:** Predict pKi with low RMSE across held-out test pairs
- **Result:** Achieved 1.05 RMSE, outperforming baselines such as DeepAtom (1.23 RMSE)
- **Tech Stack:** PyTorch, PyTorch Geometric, RDKit, ESM-2, Cross-Attention Fusion

*This project was conducted as part of independent research on AI-driven drug discovery at MIT.*

[📄 Read the full paper (PDF)](ligand_protein_affinity_prediction.pdf)
