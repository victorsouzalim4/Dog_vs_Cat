# Classificador de Gatos e Cachorros com CNN 🐱🐶

Este projeto implementa uma rede neural convolucional (CNN) para classificar imagens de gatos e cachorros utilizando TensorFlow/Keras.

Repositório GitHub: [victorsouzalim4/Dog_vs_Cat](https://github.com/victorsouzalim4/Dog_vs_Cat)

---

## 🛠️ Requisitos e Instalação

### 1. Clone o repositório e entre na pasta:

```bash
git clone https://github.com/victorsouzalim4/Dog_vs_Cat.git
cd Dog_vs_Cat
```

### 2. Crie e ative um ambiente virtual:

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências:

```bash
pip install -r requirements.txt
```

---

## 📦 Estrutura de dados

As imagens do dataset original estão disponíveis via Kaggle:

🔗 [Dataset no Kaggle - Dogs vs Cats](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)

1. Baixe o arquivo zip do data set
2. Extraia o conteúdo dentro da pasta `archive/PetImages/`
3. A estrutura final deverá ficar assim:

```
archive/
└── PetImages/
    ├── Cat/
    └── Dog/
```

---

## ✂️ Divisão dos dados

Antes de treinar, execute a função `split_dataset()` presente em `Utils/split_data_set.py` para dividir o conjunto original em:

- Treinamento (70%)
- Validação (15%)
- Teste (15%)

Isso criará a pasta `archive/splitted_data/` automaticamente.

---

## 🚀 Treinando o modelo

Execute o script de treinamento:

- Lê os dados de `archive/splitted_data/`
- Treina uma CNN com camadas de dropout e early stopping
- Salva o melhor modelo em `Model/melhor_modelo.keras`

---

## 🧪 Testando o modelo

Para testar o modelo salvo:

```bash
python main.py
```

Esse script:

- Carrega o modelo salvo
- Avalia o desempenho no conjunto de teste
- Gera o `classification_report`
- Plota e salva a matriz de confusão

---

## 🧠 Modelos pré-treinados

Os modelos treinados (`.keras`) estarão disponíveis via Google Drive:

🔗 [Link do modelo no Google Drive](https://drive.google.com/file/d/1-cbuJWmsAYYVPqwHyKYnhQZYTWgTkwCN/view?usp=sharing)

---

## 📌 Observações

- Evite subir os modelos `.keras` diretamente no GitHub. Use `.gitignore`.
- Modelos acima de 100MB exigem [Git LFS](https://git-lfs.com/) para versionamento.
- O dataset original **não é incluso no repositório**, baixe pelo link da Kaggle acima.

---

## 👨‍💻 Autor

Projeto desenvolvido por [victorsouzalim4]
