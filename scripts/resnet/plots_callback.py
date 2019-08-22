import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import keras
from sklearn.metrics import roc_curve, auc

class Plots_callback(keras.callbacks.Callback):
    def __init__(self, path, x_val, y_val):
        self.path = path
        self.x_val = x_val
        self.y_val = y_val
    
    def on_train_begin(self, logs={}):
        self.x = []
        self.accuracies = []
        self.val_accuracies = []
        self.losses = []
        self.val_losses = []
        self.val_aucs = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(epoch)
        self.accuracies.append(logs.get('acc'))
        self.val_accuracies.append(logs.get('val_acc'))
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        # Calculate AUC
        y_pred = self.model.predict(self.x_val)
        np.save(os.path.join(self.path, "predictions", "y_pred"), y_pred[:, 1])
        y_compare = np.zeros((self.y_val.shape[0], 2))
        y_compare[:, 0] = y_pred[:, 1]
        y_compare[:, 1] = self.y_val[:, 1]
        y_compare_df = pd.DataFrame(data=y_compare, columns=["Predicted Probabilities", "Ground Truth"])
        y_compare_df.to_csv(os.path.join(self.path, "predictions", f"comparisons_{epoch}.csv"), index=None, header=True)
        
        fpr, tpr, thresholds = roc_curve(self.y_val[:, 1], y_pred[:, 1])
        auc_value = auc(fpr, tpr)
        self.val_aucs.append(auc_value)

        # Plot figures
        fig = plt.figure(figsize=(20, 20))
        fig.patch.set_facecolor('white')
        plt.suptitle(f'Accuracies, Losses, ROC and AUC epoch({epoch})')

        # Accuracies and losses
        gs = gridspec.GridSpec(2, 2)
        plt.subplot(gs[0, :])
        plt.title(f"Accuracies, losses and AUC")
        plt.plot(self.x, self.losses,
                 label=f"loss {logs.get('loss'): .4}")
        plt.plot(self.x, self.val_losses,
                 label=f"val_loss {logs.get('val_loss'): .4}")
        plt.plot(self.x, self.accuracies,
                 label=f"acc {logs.get('acc'): .4}")
        plt.plot(self.x, self.val_accuracies,
                 label=f"val_acc {logs.get('val_acc'): .4}")
        plt.plot(self.x, self.val_aucs,
                 label=f"val_auc {auc_value: .4}")
        
        y_loss_m, y_acc_M, y_auc_M = min(self.val_losses), max(self.val_accuracies), max(self.val_aucs)
        x_loss_m, x_acc_M, x_auc_M = self.val_losses.index(y_loss_m), self.val_accuracies.index(y_acc_M), self.val_aucs.index(y_auc_M)

        plt.plot(x_loss_m, y_loss_m ,'o')
        plt.annotate(f"Min val_loss: {y_loss_m: .4}", xy=(x_loss_m, y_loss_m))
        plt.plot(x_acc_M, y_acc_M ,'o')
        plt.annotate(f"Max val_acc: {y_acc_M: .4}", xy=(x_acc_M, y_acc_M))
        plt.plot(x_auc_M, y_auc_M ,'o')
        plt.annotate(f"Max val_auc: {y_auc_M: .4}", xy=(x_auc_M, y_auc_M))
        plt.legend()
        
        # ROC Curve
        plt.subplot(gs[1, 0])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='Area = {:.3f}'.format(auc_value))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')

        plt.subplot(gs[1, 1])
        plt.xlim(0, 0.3)
        plt.ylim(0.7, 1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='Area = {:.3f}'.format(auc_value))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve (zoomed in at top left)')
        plt.legend(loc='best')
        fig.savefig(os.path.join(self.path, "figures", "snapshots", f"figures{epoch}"))
        fig.savefig(os.path.join(self.path, "figures", "figures"))
