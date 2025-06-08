from Utils.treat_data import treat_data_set
from Utils.split_data_set import split_dataset
from CNN.generate_model import gen_model
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

# Caminho do conjunto de teste
test_path = "archive/splitted_data/test"
test_generator = treat_data_set(path=test_path, shuffle_value=False)

# Carrega o modelo treinado
model = load_model("Model/dog_vs_cat_model_final.keras")

# Avaliação do modelo
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

# Geração de predições
y_pred_probs = model.predict(test_generator)
y_pred = (y_pred_probs > 0.5).astype(int)
y_true = test_generator.classes

# Relatório de classificação
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

# Matriz de confusão
cm = confusion_matrix(y_true, y_pred)
labels = list(test_generator.class_indices.keys())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax)
plt.title("Confusion Matrix - Test Set")

# Salvar a matriz de confusão
os.makedirs("Visualization", exist_ok=True)
plt.savefig("Visualization/Confusion_Matrix.jpeg", dpi=300, bbox_inches='tight')
plt.close()

# 🔍 Classificação de imagens individuais de uma pasta
predict_path = "teste"  # Pasta com imagens para classificação
save_file_path = "Visualization/"

if os.path.exists(predict_path):
    for file in os.listdir(predict_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(predict_path, file)

            # Carrega e processa a imagem
            img = image.load_img(img_path, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Faz a predição
            pred_prob = model.predict(img_array)[0][0]
            pred_class = "Dog" if pred_prob > 0.5 else "Cat"
            confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob

            # Exibe a imagem com a predição
            plt.imshow(np.array(img).astype("uint8"))
            plt.axis('off')
            plt.title(f"{file} → {pred_class} ({confidence:.2%})")
            plt.savefig(save_file_path + file, bbox_inches='tight', dpi=150)
            plt.show()
else:
    print(f"⚠️ Pasta '{predict_path}' não encontrada. Crie a pasta e adicione imagens para testar predições visuais.")
