import pdb

def confusion_matrix(x, y, prediction):
  true_positive = 0
  true_negative = 0
  false_positive = 0
  false_negative = 0
  for i in range(0, len(x)):
    p = 1 if prediction[i] > 0.5 else 0
    if p == 1 and y[i] == 1:
        true_positive += 1
    if p == 0 and y[i] == 0:
        true_negative += 1
    if p == 1 and y[i] == 0:
        false_positive += 1
    if p == 0 and y[i] == 1:
        false_negative += 1
  return (true_positive, false_positive, false_negative, true_negative)

def precision_recall(x, y, prediction):
  (true_positive, false_positive, false_negative, true_negative) = confusion_matrix(x, y, prediction)
  precision = float(true_positive)/(true_positive + false_positive)
  recall = float(true_positive)/(true_positive + false_negative)
  return (precision, recall)
