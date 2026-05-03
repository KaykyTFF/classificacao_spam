import os
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Importamos as funções que criamos no passo anterior!
from data_loader import carregar_e_limpar_dados, separar_dados

def treinar_modelos(X_treino, y_treino):
    """
    Recebe os dados de estudo e treina os dois algoritmos propostos.
    """
    print("-> Ensinando o modelo Naive Bayes...")
    modelo_nb = MultinomialNB()
    modelo_nb.fit(X_treino, y_treino) # O .fit() é onde a IA aprende os padrões

    print("-> Ensinando o modelo de Regressão Logística...")
    # max_iter=1000 garante que o modelo tenha tempo (tentativas) suficiente para aprender
    modelo_lr = LogisticRegression(max_iter=1000)
    modelo_lr.fit(X_treino, y_treino)

    return modelo_nb, modelo_lr

def salvar_modelos(modelo_nb, modelo_lr, pasta_destino="../results/models/"):
    """
    Salva os modelos treinados em arquivos .pkl para não precisarmos treinar de novo depois.
    """
    # Garante que a pasta de destino existe
    os.makedirs(pasta_destino, exist_ok=True)

    # Definimos os nomes dos arquivos
    caminho_nb = os.path.join(pasta_destino, "modelo_nb.pkl")
    caminho_lr = os.path.join(pasta_destino, "modelo_lr.pkl")

    # A função dump "congela" a inteligência e salva no HD
    joblib.dump(modelo_nb, caminho_nb)
    joblib.dump(modelo_lr, caminho_lr)

    print(f"\n✓ Modelos salvos com sucesso na pasta: {pasta_destino}")

# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================
if __name__ == "__main__":
    caminho_csv = "../data/raw/emails/emails.csv"
    
    print("=======================================")
    print(" FASE DE TREINAMENTO DA INTELIGÊNCIA")
    print("=======================================\n")
    
    # 1. Busca e prepara os dados usando o nosso data_loader
    dados = carregar_e_limpar_dados(caminho_csv)
    X_tr, X_te, y_tr, y_te = separar_dados(dados)
    
    # 2. Treina os algoritmos
    modelo_nb_treinado, modelo_lr_treinado = treinar_modelos(X_tr, y_tr)
    
    # 3. Salva a IA treinada
    salvar_modelos(modelo_nb_treinado, modelo_lr_treinado)
    
    #  salvar os dados da (X_te, y_te)
    joblib.dump((X_te, y_te), "../results/models/dados_teste.pkl")
    print("✓ Dados de teste guardados para a etapa de avaliação.")