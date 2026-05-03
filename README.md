# 📧 Classificação de E-mails — Spam ou Não Spam

**Disciplina:** Inteligência Artificial — IFPI Campus Paulistana  
**Docente:** Maíla de Lima Claro  
**Discentes:**
- Kayky Terles Ferreira Feitosa
- Raili Saweni de Sousa Reis
- Giselly Carvalho Cavalcante
- Josimar Rodrigues de Macedo Filho
- Pedro Ravy Teixeira de Sousa

---

## 📌 Descrição do Problema

O objetivo deste projeto é desenvolver um sistema de Machine Learning capaz de classificar e-mails automaticamente como **Spam** ou **Não Spam**, utilizando três algoritmos de aprendizado supervisionado: Naive Bayes, Regressão Logística e Random Forest.

O dataset utilizado é numérico no formato Bag of Words, onde cada coluna representa a frequência de uma palavra no e-mail. Foi obtido na plataforma Kaggle.

---

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Finalidade |
|---|---|
| Python 3.12 | Linguagem principal |
| Pandas & NumPy | Manipulação de dados |
| Scikit-learn | Algoritmos de ML e métricas |
| Matplotlib & Seaborn | Visualização de resultados |
| Joblib | Persistência dos modelos treinados |

---

## 📁 Estrutura do Projeto

```
classificacao_spam/
├── data/
│   └── raw/
│       └── emails/
│           └── emails.csv        # Dataset (não versionado)
├── notebooks/
│   └── email-spam-classification.ipynb
├── results/
│   ├── figures/                  # Gráficos gerados
│   └── models/                   # Modelos .pkl treinados
├── data_loader.py
├── train.py
├── evaluate.py
├── predict.py
├── compare_models.py
├── requirements.txt
└── README.md
```

---

## ▶️ Como Executar

**1. Clone o repositório e ative o ambiente virtual:**
```bash
.\venv\Scripts\activate
```

**2. Instale as dependências:**
```bash
pip install -r requirements.txt
```

**3. Baixe o dataset** no [Kaggle — Spam Email Dataset](https://www.kaggle.com) e coloque o arquivo `emails.csv` em `data/raw/emails/`.

**4. Execute os scripts na ordem:**
```bash
python train.py      # Treina e salva os modelos
python evaluate.py   # Avalia e gera os gráficos
python predict.py    # Testa a classificação de e-mails individuais
```

Ou abra o notebook `email-spam-classification.ipynb` para ver o pipeline completo de forma interativa.

---

## 📊 Resultados Obtidos

| Modelo | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Naive Bayes | 95,46% | 89,03% | 95,95% | 92,36% |
| Regressão Logística | 96,72% | 90,36% | 98,32% | 94,18% |
| **Random Forest** | **97,78%** | **95,96%** | **96,28%** | **96,12%** |

O **Random Forest** obteve o melhor desempenho geral. A **Regressão Logística** se destacou pelo maior Recall (98,32%), sendo ideal para cenários onde detectar o máximo de spams é prioritário.
