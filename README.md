# Projeto de Classificação de E-mails (Spam vs Ham)
**Disciplina:** Inteligência Artificial - IFPI Campus Paulistana  
**Docente:** Maíla de Lima Claro  
**Discente:** Kayky Terles Ferreira Feitosa, RAILI SAWENI DE SOUSA REIS, 
Giselly Carvalho Cavalcante, Josimar Rodrigues de Macedo Filho, 
Pedro Ravy Teixeira de Sousa

## 1. Descrição do Problema
O objetivo deste projeto é desenvolver um modelo de Machine Learning capaz de classificar e-mails automaticamente entre "Spam" e "Não Spam", utilizando algoritmos de Naive Bayes e Regressão Logística.

## 2. Tecnologias Utilizadas
- Python 3.12
- Pandas & NumPy (Manipulação de dados)
- Scikit-learn (Algoritmos de ML e métricas)
- Matplotlib & Seaborn (Visualização de resultados)
- Joblib (Persistência do modelo treinado)

## 3. Estrutura do Projeto
- `data/raw/`: Base de dados original (Spam Email Dataset).
- `src/`: Scripts de carregamento, treino, avaliação e predição.
- `results/models/`: Arquivos `.pkl` dos modelos treinados.
- `results/figures/`: Gráficos de Matriz de Confusão gerados.

## 4. Como Executar
1. Ative o ambiente virtual: `.\venv\Scripts\activate`
2. Instale as dependências: `pip install -r requirements.txt`
3. Para treinar: `python src/train.py`
4. Para avaliar: `python src/evaluate.py`
5. Para testar uma previsão: `python src/predict.py`

## 5. Resultados Obtidos
O modelo de **Regressão Logística** apresentou o melhor desempenho com **97.2% de Acurácia**.