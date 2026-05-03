import matplotlib.pyplot as plt
import numpy as np

# Nomes dos modelos treinados no projeto
modelos = ['Naive Bayes', 'Regressão Logística', 'Random Forest']

# Métricas reais obtidas na avaliação
metricas = {
    'Accuracy':  [0.9546, 0.9720, 0.9778],
    'Precision': [0.8903, 0.9435, 0.9596],
    'Recall':    [0.9595, 0.9595, 0.9628],
    'F1-Score':  [0.9236, 0.9514, 0.9612]
}

x = np.arange(len(modelos))
width = 0.2
cores = ['skyblue', 'salmon', 'lightgreen', 'plum']

plt.figure(figsize=(12, 6))

# Cria uma barra por métrica para cada modelo
for i, (metrica, valores) in enumerate(metricas.items()):
    plt.bar(x + i * width, valores, width, label=metrica, color=cores[i])

# Centraliza os nomes dos modelos no eixo X
plt.xticks(x + width * 1.5, modelos)
plt.ylim(0.8, 1.05)
plt.title('Comparação Completa entre Modelos', fontsize=14)
plt.ylabel('Score')
plt.legend()
plt.tight_layout()

# Salva o gráfico na pasta de resultados
plt.savefig('results/figures/comparativo_modelos.png')
plt.show()