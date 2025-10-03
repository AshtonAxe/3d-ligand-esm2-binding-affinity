# Predicted vs Actual pKi (Best Model on Test Set)
save_path = 'best_model.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProteinLigandAffinityModel(
    esm2_dim=1280, ligand_in_channels=108, embedding_dim=128
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

model.load_state_dict(torch.load("new_best_model.pt"))
model.eval()

all_preds = []
all_targets = []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) or isinstance(v, torch_geometric.data.Batch) else v for k, v in batch.items()}
        batch['ligand'] = batch['ligand'].to(device)
        pred = model(batch)
        pred_actual = pred * std_pKi + mean_pKi
        target_actual = batch['pKi'] * std_pKi + mean_pKi
        all_preds.append(pred_actual.cpu())
        all_targets.append(target_actual.cpu())

# Combine and convert to lists for plotting
all_preds = torch.cat(all_preds).tolist()
all_targets = torch.cat(all_targets).tolist()

# Plot
plt.scatter(all_targets, all_preds, alpha=0.5)
plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], color='red', linestyle='--')
plt.xlabel("Actual pKi")
plt.ylabel("Predicted pKi")
plt.title("Predicted vs. Actual pKi on Test Set")
plt.show()

# Compute RMSE
test_rmse = torch.sqrt(torch.mean((all_preds_tensor - all_targets_tensor) ** 2)).item()

# Compute R²
ss_res = torch.sum((all_targets_tensor - all_preds_tensor) ** 2)
ss_tot = torch.sum((all_targets_tensor - torch.mean(all_targets_tensor)) ** 2)
test_r2 = (1 - ss_res / ss_tot).item()

residuals = all_preds_tensor - all_targets_tensor

# Plot residuals
plt.hist(residuals.tolist(), bins=50, alpha=0.7)
plt.xlabel("Prediction Error (pKi)")
plt.ylabel("Frequency")
plt.title("Residuals Histogram on Test Set")
plt.show()

print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test R²: {test_r2:.4f}")
