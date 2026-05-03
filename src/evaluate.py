import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def avaliar_modelo(nome_modelo, y_verdadeiro, y_previsto):
    # Calcula as metricas de classificacao
    acc = accuracy_score(y_verdadeiro, y_previsto)
    prec = precision_score(y_verdadeiro, y_previsto)
    rec = recall_score(y_verdadeiro, y_previsto)
    f1 = f1_score(y_verdadeiro, y_previsto)
    
    # Exibe os resultados no terminal
    print(f"\nResultados para o modelo: {nome_modelo}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    return [acc, prec, rec, f1]

def gerar_matriz_confusao(y_verdadeiro, y_previsto, nome_modelo):
    # Gera a matriz de confusao
    cm = confusion_matrix(y_verdadeiro, y_previsto)
    
    # Configura o layout do grafico
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matriz de Confusao - {nome_modelo}')
    plt.ylabel('Gabarito (Real)')
    plt.xlabel('Previsao do Modelo')
    
    # Salva a imagem no diretorio correspondente
    caminho_imagem = f"../results/figures/matriz_{nome_modelo.replace(' ', '_').lower()}.png"
    plt.savefig(caminho_imagem)
    plt.close()

if __name__ == "__main__":
    print("=======================================")
    print(" FASE DE AVALIACAO DOS MODELOS")
    print("=======================================")

    # Carrega os modelos e os dados de teste em formato .pkl
    modelo_nb = joblib.load("../results/models/modelo_nb.pkl")
    modelo_lr = joblib.load("../results/models/modelo_lr.pkl")
    X_teste, y_teste = joblib.load("../results/models/dados_teste.pkl")
    
    # Executa a predicao
    previsoes_nb = modelo_nb.predict(X_teste)
    previsoes_lr = modelo_lr.predict(X_teste)
    
    # Avalia e imprime as metricas
    metricas_nb = avaliar_modelo("Naive Bayes", y_teste, previsoes_nb)
    metricas_lr = avaliar_modelo("Regressao Logistica", y_teste, previsoes_lr)
    
    # Gera e salva os graficos
    gerar_matriz_confusao(y_teste, previsoes_nb, "Naive Bayes")
    gerar_matriz_confusao(y_teste, previsoes_lr, "Regressao Logistica")
    
    print("\nGraficos gerados e salvos na pasta 'results/figures/'.")