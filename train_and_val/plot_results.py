# Training & Validation Loss over Epochs
import matplotlib.pyplot as plt
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (pKi)")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()

# Validation RMSE over Epochs
plt.plot(val_rmses, label='Validation RMSE')
plt.xlabel("Epoch")
plt.ylabel("RMSE (pKi)")
plt.title("Validation RMSE")
plt.legend()
plt.show()

# Validation R² Score over Epochs
plt.plot(val_r2s, label='Validation R²')
plt.xlabel("Epoch")
plt.ylabel("R² Score")
plt.title("Validation R² Score")
plt.legend()
plt.show()
