from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProteinLigandAffinityModel(
    esm2_dim=1280, ligand_in_channels=108, embedding_dim=128
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

best_val_loss = float('inf')
patience = 7        # Number of epochs to wait after no improvement
patience_counter = 0
min_delta = 1e-4    # Minimum change in val loss to be considered an improvement
save_path = 'new_best_model.pt'

train_losses = []
val_losses = []
val_rmses = []
val_r2s = []

def compute_rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()

def compute_r2(pred, target):
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()

num_epochs = 150
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_batches = 0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)

    for batch in train_bar:
        try:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) or isinstance(v, torch_geometric.data.Batch) else v for k, v in batch.items()}
            batch['ligand'] = batch['ligand'].to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch['pKi'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
            train_bar.set_postfix(batch_loss=loss.item())
        except Exception as e:
            print(f"[SKIPPED BATCH] due to error: {e}")
            continue

    avg_train_loss = train_loss / max(train_batches, 1)
    train_bar.set_postfix(avg_train_loss=avg_train_loss)

    model.eval()
    val_loss = 0
    val_batches = 0

    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in val_bar:
            try:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) or isinstance(v, torch_geometric.data.Batch) else v for k, v in batch.items()}
                batch['ligand'] = batch['ligand'].to(device)
                pred = model(batch)
                pred_actual = pred * std_pKi + mean_pKi
                target_actual = batch['pKi'] * std_pKi + mean_pKi
                loss = loss_fn(pred_actual, target_actual)
                val_loss += loss.item()
                val_batches += 1
                val_bar.set_postfix(batch_loss=loss.item())

                all_preds.append(pred_actual.cpu())
                all_targets.append(target_actual.cpu())

            except Exception as e:
                print(f"[SKIPPED VAL BATCH] due to error: {e}")
                continue

    avg_val_loss = val_loss / max(val_batches, 1)
    val_bar.set_postfix(avg_val_loss=avg_val_loss)


    # End of epoch validation loss check
    if avg_val_loss < best_val_loss - min_delta:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), save_path)
        print(f"Saved new best model with val_loss = {best_val_loss:.4f}")
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter} epochs")

    # Check if early stopping condition met
    if patience_counter >= patience:
        print("Early stopping triggered")
        break

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    # Compute RMSE and R² from unstandardized values
    val_rmse = compute_rmse(all_preds, all_targets)
    val_r2 = compute_r2(all_preds, all_targets)

    val_rmses.append(val_rmse)
    val_r2s.append(val_r2)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
