# Classificador de Gatos e Cachorros com CNN

Este projeto implementa uma rede neural convolucional (CNN) para classificar imagens de gatos e cachorros utilizando TensorFlow/Keras.

---

## 🛠️ Requisitos e Instalação

### 1. Clone o repositório e entre na pasta:

```bash
git clone <URL_DO_REPOSITORIO>
cd nome-do-repositorio
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

Os dados originais estão disponibilizados via Google Drive. Após baixar o arquivo `.zip`:

🔗 [Link para o dataset no Google Drive](https://drive.google.com/...)

1. Extraia o conteúdo (ex: `PetImages/`) dentro da pasta `archive/`
2. A estrutura deve ficar assim:

```
archive/
└── PetImages/
    ├── Cat/
    └── Dog/
```

---

## 🚀 Treinando o modelo

O script principal treina o modelo com validação e salva o melhor resultado.

```bash
python train.py
```

Esse script:

- Divide os dados em treino, validação e teste
- Treina uma CNN com dropout e early stopping
- Salva o melhor modelo em `Model/melhor_modelo.keras`

---

## 🧪 Testando o modelo

Para testar o desempenho em imagens novas ou no conjunto de teste:

```bash
python evaluate.py
```

Esse script carrega o modelo salvo e:

- Avalia o desempenho em teste
- Gera relatório de classificação
- Plota a matriz de confusão

---

## 🧠 Modelos pré-treinados

Os arquivos `.keras` contendo os modelos treinados serão disponibilizados via Google Drive:

🔗 [Link do modelo no Google Drive](https://drive.google.com/...)

---

## 📥 Dataset original

O conjunto de imagens **PetImages** está disponível via Google Drive (ver link acima). Após baixar:

1. Extraia o conteúdo em `archive/`
2. Mantenha a estrutura de pastas original (`Cat/`, `Dog/`)

---

## 📌 Observações

- Evite subir os modelos `.keras` diretamente no GitHub. Use `.gitignore`.
- Os modelos maiores que 100MB exigem [Git LFS](https://git-lfs.com/) caso precise versionar.

---

## 👨‍💻 Autor

Projeto desenvolvido por [Seu Nome].
