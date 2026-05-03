import matplotlib.pyplot as plt
import os

# 1. Definir os dados extraídos dos testes anteriores
modelos = ['Naive Bayes', 'SVM (SVC)', 'Random Forest']
acuracias = [0.938, 0.901, 0.976]

# 2. Configurar o gráfico
plt.figure(figsize=(10, 6))
cores = ['skyblue', 'salmon', 'lightgreen']
barras = plt.bar(modelos, acuracias, color=cores)

# 3. Adicionar detalhes visuais
plt.title('Comparação de Acurácia entre Modelos - Classificação de Spam', fontsize=14)
plt.ylabel('Acurácia (0.0 a 1.0)', fontsize=12)
plt.ylim(0.8, 1.0)  # Foca no intervalo de interesse para melhor visualização

# Adicionar os valores em cima das barras
for barra in barras:
    yval = barra.get_height()
    plt.text(barra.get_x() + barra.get_width()/2, yval + 0.005, f'{yval*100:.1f}%', ha='center', va='bottom')

# 4. Salvar o gráfico na pasta de resultados
output_path = 'results/figures/comparativo_modelos.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
print(f"Gráfico salvo com sucesso em: {output_path}")

plt.show()