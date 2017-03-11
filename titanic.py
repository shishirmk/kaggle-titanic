import csv
import pdb
import logistic_regression
import model_analysis

# Fields

# 0 PassengerId: 2
# 1 Pclass: 1
# 2 Name: "Cumings"
# 3 Sex: "female"
# 4 Age: 38
# 5 SibSp: 1
# 6 Parch: 0
# 7 Ticket: PC 17599
# 8 Fare: 71.2833
# 9 Cabin: C85
# 10 Embarked: C

def get_features(row):
  passenger_id = int(row[0])
  pclass = float(row[1])
  if row[3] == "female":
    sex = 1
  else:
    sex = 0
  if row[5]:
    siblings = float(row[6])
  else:
    siblings = 0
  if row[6]:
    parents = float(row[6])
  else:
    parents = 0
  if row[8]:
    fare = float(row[8])
  else:
    fare = 0
  return [
      passenger_id,
      pclass,
      sex,
      fare,
      siblings,
      parents
  ]

def extract_features(filename, has_training_examples = False):
  y = []
  x = []
  index = 0
  with open(filename, 'rb') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in csv_reader:
        try:
          if index == 0:
            index += 1
            continue
          if has_training_examples:
            y.append(float(row[1]))
            row.pop(1)
          x.append(get_features(row))
          index += 1
        except ValueError as err:
          print(row)
          raise err

  return x, y

def print_predictions(x, y, prediction):
  for i in range(0, len(x)):
    print("ID: %d, P: %d, Y: %d" % (x[i][0], 1 if prediction[i] > 0.5 else 0, y[i]))

def output_predictions(x, prediction):
  for i in range(0, len(x)):
    with open('submission.csv', 'w') as csvfile:
      csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"')
      for i in range(0, len(x)):
        csv_writer.writerow([x[i][0], 1 if prediction[i] > 0.5 else 0])

def run_training_examples():
  train_filename = 'train.csv'
  x, y = extract_features(train_filename, True)
  theta = logistic_regression.train(x, y)
  prediction = logistic_regression.hypothesis(x, theta)
  (precision, recall) = model_analysis.precision_recall(x, y, prediction)
  print("Precision: %f, Recall %f" % (precision, recall))
  return theta

def run_test_examples(theta):
  test_filename = 'test.csv'
  x_test, y_test = extract_features(test_filename)
  prediction_test = logistic_regression.hypothesis(x_test, theta)
  output_predictions(x_test, prediction_test)

theta = run_training_examples()
run_test_examples(theta)
