import joblib
import pandas as pd

def simular_previsao(indice_amostra=0):
    # Carrega o modelo de Regressao Logistica (melhor desempenho)
    modelo = joblib.load("../results/models/modelo_lr.pkl")
    
    # Carrega a base de dados de teste isolada na etapa de treinamento
    X_teste, y_teste = joblib.load("../results/models/dados_teste.pkl")
    
    # Extrai uma unica linha (amostra) com base no indice fornecido
    dados_email = X_teste.iloc[[indice_amostra]]
    gabarito_real = y_teste.iloc[indice_amostra]
    
    # Executa a inferencia do modelo sobre a amostra
    previsao_numerica = modelo.predict(dados_email)[0]
    
    # Converte os outputs binarios para strings compreensiveis
    status_previsao = "SPAM" if previsao_numerica == 1 else "NAO SPAM (Normal)"
    status_real = "SPAM" if gabarito_real == 1 else "NAO SPAM (Normal)"
    
    # Imprime o relatorio de inferencia no terminal
    print(f"Analisando e-mail (Indice {indice_amostra} da base de testes)...")
    print(f"-> O modelo classificou como: {status_previsao}")
    print(f"-> O gabarito original era:   {status_real}")
    
    if previsao_numerica == gabarito_real:
        print("-> Resultado: ACERTO")
    else:
        print("-> Resultado: ERRO")

if __name__ == "__main__":
    print("=======================================")
    print(" SISTEMA DE CLASSIFICACAO ATIVO")
    print("=======================================\n")
    
    # Loop para demonstrar a classificacao de 5 e-mails diferentes
    indices_para_testar = [0, 10, 42, 100, 250]
    
    for idx in indices_para_testar:
        simular_previsao(idx)
        print("-" * 40)