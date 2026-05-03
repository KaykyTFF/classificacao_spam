import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

# Cria as pastas de saída se ainda não existirem
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/models", exist_ok=True)

def avaliar_modelo(nome_modelo, y_verdadeiro, y_previsto):
    # Calcula as quatro métricas de classificação
    acc  = accuracy_score(y_verdadeiro, y_previsto)
    prec = precision_score(y_verdadeiro, y_previsto)
    rec  = recall_score(y_verdadeiro, y_previsto)
    f1   = f1_score(y_verdadeiro, y_previsto)

    print(f"\nResultados para o modelo: {nome_modelo}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Retorna lista para usar no gráfico comparativo
    return [acc, prec, rec, f1]

def gerar_matriz_confusao(y_verdadeiro, y_previsto, nome_modelo):
    # Monta a matriz com os acertos e erros do modelo
    cm = confusion_matrix(y_verdadeiro, y_previsto)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matriz de Confusão - {nome_modelo}')
    plt.ylabel('Gabarito (Real)')
    plt.xlabel('Previsão do Modelo')

    # Nome do arquivo gerado a partir do nome do modelo
    caminho = f"results/figures/matriz_{nome_modelo.replace(' ', '_').lower()}.png"
    plt.savefig(caminho)
    plt.close()
    print(f"✓ Matriz de confusão salva: {caminho}")

def gerar_grafico_comparativo(metricas_nb, metricas_lr, metricas_rf):
    modelos = ['Naive Bayes', 'Regressão Logística', 'Random Forest']
    labels  = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    valores = [metricas_nb, metricas_lr, metricas_rf]
    cores   = ['skyblue', 'salmon', 'lightgreen']

    x     = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(12, 6))

    # Cria um grupo de barras por modelo para cada métrica
    for i, (modelo, metrica, cor) in enumerate(zip(modelos, valores, cores)):
        plt.bar(x + i * width, metrica, width, label=modelo, color=cor)

    plt.xticks(x + width, labels, fontsize=12)
    plt.ylim(0.8, 1.05)
    plt.title('Comparação Completa entre Modelos', fontsize=14)
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()

    caminho = "results/figures/comparativo_modelos.png"
    plt.savefig(caminho)
    plt.close()
    print(f"✓ Gráfico comparativo salvo: {caminho}")

if __name__ == "__main__":
    print("=======================================")
    print(" FASE DE AVALIAÇÃO DOS MODELOS")
    print("=======================================")

    # Carrega os modelos salvos na etapa de treinamento
    modelo_nb = joblib.load("results/models/modelo_nb.pkl")
    modelo_lr = joblib.load("results/models/modelo_lr.pkl")
    modelo_rf = joblib.load("results/models/modelo_rf.pkl")
    X_teste, y_teste = joblib.load("results/models/dados_teste.pkl")

    # Gera as previsões de cada modelo sobre os dados de teste
    prev_nb = modelo_nb.predict(X_teste)
    prev_lr = modelo_lr.predict(X_teste)
    prev_rf = modelo_rf.predict(X_teste)

    # Avalia e imprime as métricas dos três modelos
    metricas_nb = avaliar_modelo("Naive Bayes", y_teste, prev_nb)
    metricas_lr = avaliar_modelo("Regressão Logística", y_teste, prev_lr)
    metricas_rf = avaliar_modelo("Random Forest", y_teste, prev_rf)

    # Gera as matrizes de confusão individuais
    gerar_matriz_confusao(y_teste, prev_nb, "Naive Bayes")
    gerar_matriz_confusao(y_teste, prev_lr, "Regressão Logística")
    gerar_matriz_confusao(y_teste, prev_rf, "Random Forest")

    # Gera o gráfico comparando os três modelos em todas as métricas
    gerar_grafico_comparativo(metricas_nb, metricas_lr, metricas_rf)

    print("\n✓ Avaliação completa! Gráficos salvos em 'results/figures/'.")