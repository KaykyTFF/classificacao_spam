import os
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from data_loader import carregar_e_limpar_dados, separar_dados

def treinar_modelos(X_treino, y_treino):
    # Treina o Naive Bayes
    print("-> Ensinando o modelo Naive Bayes...")
    modelo_nb = MultinomialNB()
    modelo_nb.fit(X_treino, y_treino)

    # Treina a Regressão Logística com StandardScaler dentro de um Pipeline
    # O scaler normaliza os dados antes do modelo, melhorando a convergência
    print("-> Ensinando o modelo de Regressão Logística (com normalização)...")
    modelo_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000))
    ])
    modelo_lr.fit(X_treino, y_treino)

    # Treina o Random Forest, conjunto de várias árvores de decisão
    print("-> Ensinando o modelo Random Forest...")
    modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)  # 100 árvores, semente fixa para reprodutibilidade
    modelo_rf.fit(X_treino, y_treino)

    return modelo_nb, modelo_lr, modelo_rf

def salvar_modelos(modelo_nb, modelo_lr, modelo_rf, pasta_destino="results/models/"):
    # Cria a pasta se ela ainda não existir
    os.makedirs(pasta_destino, exist_ok=True)

    # Salva cada modelo em um arquivo .pkl para reutilizar depois sem retreinar
    joblib.dump(modelo_nb, os.path.join(pasta_destino, "modelo_nb.pkl"))
    joblib.dump(modelo_lr, os.path.join(pasta_destino, "modelo_lr.pkl"))
    joblib.dump(modelo_rf, os.path.join(pasta_destino, "modelo_rf.pkl"))

    print(f"\n✓ Modelos salvos com sucesso na pasta: {pasta_destino}")

if __name__ == "__main__":
    caminho_csv = "data/raw/emails/emails.csv"

    print("=======================================")
    print(" FASE DE TREINAMENTO DA INTELIGÊNCIA")
    print("=======================================\n")

    # Carrega e limpa os dados do CSV
    dados = carregar_e_limpar_dados(caminho_csv)

    # Divide em 80% treino e 20% teste
    X_tr, X_te, y_tr, y_te = separar_dados(dados)

    # Treina os três modelos
    modelo_nb, modelo_lr, modelo_rf = treinar_modelos(X_tr, y_tr)

    # Salva os modelos treinados
    salvar_modelos(modelo_nb, modelo_lr, modelo_rf)

    # Guarda os dados de teste para usar na avaliação
    joblib.dump((X_te, y_te), "results/models/dados_teste.pkl")
    print("✓ Dados de teste guardados para a etapa de avaliação.")