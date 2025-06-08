from Utils.treat_data import treat_data_set
from Utils.split_data_set import split_dataset
from CNN.generate_model import gen_model
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay





test_path = "archive/splitted_data/test"
test_generator = treat_data_set(path=test_path, shuffle_value=False)

model = load_model("Model/dog_vs_cat_model_final.keras")

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

y_pred_probs = model.predict(test_generator)
y_pred = (y_pred_probs > 0.5).astype(int)

y_true = test_generator.classes

print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

cm = confusion_matrix(y_true, y_pred)

# Nomes das classes (ordenados pelo índice 0, 1)
labels = list(test_generator.class_indices.keys())

# Exibir matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

plt.figure(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix - Test Set")
plt.show()


