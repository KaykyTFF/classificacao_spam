import pandas as pd
from sklearn.model_selection import train_test_split

def carregar_e_limpar_dados(caminho_arquivo):
    """
    Lê o dataset numérico de e-mails, remove a coluna de ID e prepara o alvo.
    """
    # 1. Carregamento da base de dados
    df = pd.read_csv(caminho_arquivo, encoding='latin-1')

    print(f"-> Total de colunas (palavras analisadas): {len(df.columns)}")

    # 2. Remoção de dados que não ajudam o modelo. 
    # A coluna 'Email No.' é apenas um identificador (ex: Email 1, Email 2)
    if 'Email No.' in df.columns:
        df = df.drop('Email No.', axis=1)

    # 3. Renomeamos a coluna 'Prediction' para 'label_num' para padronizar o projeto
    if 'Prediction' in df.columns:
        df = df.rename(columns={'Prediction': 'label_num'})

    # 4. Tratamento de valores ausentes
    df = df.dropna()

    return df

def separar_dados(df, tamanho_teste=0.2, semente=42):
    """
    Separa os dados matriciais em conjuntos de Treinamento e Teste.
    """
    # O gabarito (y) é a nossa coluna de classificação
    y = df['label_num']
    
    # As características (X) são todas as outras colunas (as palavras),
    X = df.drop('label_num', axis=1)

    # Divisão de 80% para treino e 20% para teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=tamanho_teste, random_state=semente
    )

    return X_treino, X_teste, y_treino, y_teste

# ==========================================
# BLOCO DE TESTE DO MÓDULO
# ==========================================
if __name__ == "__main__":
    caminho_csv = "data/raw/emails/emails.csv"
    
    print("Iniciando o carregamento do dataset numérico...\n")
    
    try:
        dados_limpos = carregar_e_limpar_dados(caminho_csv)
        print("✓ Dados carregados e estruturados com sucesso!")
        print(f"Total de e-mails processados: {len(dados_limpos)}")
        
        X_tr, X_te, y_tr, y_te = separar_dados(dados_limpos)
        print("\n✓ Divisão Treino/Teste concluída!")
        print(f"E-mails separados para TREINAR a IA: {len(X_tr)}")
        print(f"E-mails separados para TESTAR a IA: {len(X_te)}\n")
        
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em '{caminho_csv}'.")