# from sklearn.metrics import roc_curve, auc

# y_pred = model.predict(x_test)
# fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_pred[:, 1])
# auc_value = auc(fpr, tpr)

# fig1 = plt.figure()
# fig1.patch.set_facecolor('white')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr, tpr, label='Area = {:.3f}'.format(auc_value))
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
# plt.legend(loc='best')

# fig2 = plt.figure()
# fig2.patch.set_facecolor('white')
# plt.xlim(0, 0.3)
# plt.ylim(0.7, 1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr, tpr, label='Area = {:.3f}'.format(auc_value))
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve (zoomed in at top left)')
# plt.legend(loc='best')

# def plot_confusion_matrix(cm,
#                           target_names,
#                           title='Confusion matrix',
#                           cmap=None,
#                           normalize=True):
#     """
#     given a sklearn confusion matrix (cm), make a nice plot

#     Arguments
#     ---------
#     cm:           confusion matrix from sklearn.metrics.confusion_matrix

#     target_names: given classification classes such as [0, 1, 2]
#                   the class names, for example: ['high', 'medium', 'low']

#     title:        the text to display at the top of the matrix

#     cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
#                   see http://matplotlib.org/examples/color/colormaps_reference.html
#                   plt.get_cmap('jet') or plt.cm.Blues

#     normalize:    If False, plot the raw numbers
#                   If True, plot the proportions

#     Usage
#     -----
#     plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
#                                                               # sklearn.metrics.confusion_matrix
#                           normalize    = True,                # show proportions
#                           target_names = y_labels_vals,       # list of names of the classes
#                           title        = best_estimator_name) # title of graph

#     Citiation
#     ---------
#     http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import itertools

#     accuracy = np.trace(cm) / float(np.sum(cm))
#     misclass = 1 - accuracy

#     if cmap is None:
#         cmap = plt.get_cmap('Blues')

#     fig = plt.figure(figsize=(8, 6))
#     fig.patch.set_facecolor('white')
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()

#     if target_names is not None:
#         tick_marks = np.arange(len(target_names))
#         plt.xticks(tick_marks, target_names, rotation=45)
#         plt.yticks(tick_marks, target_names)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


#     thresh = cm.max() / 1.5 if normalize else cm.max() / 2
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         if normalize:
#             plt.text(j, i, "{:0.4f}".format(cm[i, j]),
#                      horizontalalignment="center",
#                      color="white" if cm[i, j] > thresh else "black")
#         else:
#             plt.text(j, i, "{:,}".format(cm[i, j]),
#                      horizontalalignment="center",
#                      color="white" if cm[i, j] > thresh else "black")


#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))


# no_steps = 10000.
# for p in range(1, 20):
#     y_pred_thresh = y_pred[:, 1] > p/no_steps
#     cm = confusion_matrix(y_test[:, 1], y_pred_thresh)
#     plot_confusion_matrix(cm=cm, target_names=["Healthy", "Defective"], normalize=False)
#     negatives_array = y_pred_thresh == 0
#     false_array = y_pred_thresh != y_test[:,1]
#     false_negatives_array = negatives_array * false_array
#     no_negatives = negatives_array.sum()
#     no_false_negatives = false_negatives_array.sum()
#     print(f"Threshold p = {p/no_steps}: {no_false_negatives}/{no_negatives} = {no_false_negatives/no_negatives * 100: .4}%")
