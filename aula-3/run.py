# Importa as bibliotecas necessárias para criar e treinar o modelo
import torch
import torch.nn as nn
import torch.optim as optim

# Define os dados de entrada (x) como tensores de uma única feature
x = torch.tensor([[5.0], [10.0], [10.0], [5.0], [10.0],
                  [5.0], [10.0], [10.0], [5.0], [10.0],
                  [5.0], [10.0], [10.0], [5.0], [10.0],
                  [5.0], [10.0], [10.0], [5.0], [10.0]], dtype=torch.float32)

# Define os dados de saída (y) correspondentes aos valores de entrada
y = torch.tensor([[30.5], [63.0], [67.0], [29.0], [62.0],
                  [30.5], [63.0], [67.0], [29.0], [62.0],
                  [30.5], [63.0], [67.0], [29.0], [62.0],
                  [30.5], [63.0], [67.0], [29.0], [62.0]], dtype=torch.float32)

# Define a arquitetura da rede neural
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Camada totalmente conectada com 1 entrada e 5 neurônios
        self.fc1 = nn.Linear(1, 5)
        # Camada totalmente conectada com 5 entradas e 1 saída
        self.fc2 = nn.Linear(5, 1)

    # Define o fluxo de dados pela rede (forward pass)
    def forward(self, x):
        # Aplica a função de ativação ReLU na primeira camada
        x = torch.relu(self.fc1(x))
        # Passa os dados pela segunda camada
        x = self.fc2(x)
        return x

# Instancia o modelo da rede neural
model = Net()

# Define a função de perda como o erro quadrático médio (MSE)
criterion = nn.MSELoss()

# Define o otimizador como o Gradiente Descendente Estocástico (SGD) com taxa de aprendizado de 0.01
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Loop de treinamento para ajustar os pesos da rede
for epoch in range(1000):
    # Zera os gradientes acumulados do otimizador
    optimizer.zero_grad()
    # Faz a previsão (forward pass) com os dados de entrada
    outputs = model(x)
    # Calcula a perda entre as previsões e os valores reais
    loss = criterion(outputs, y)
    # Calcula os gradientes (backward pass)
    loss.backward()
    # Atualiza os pesos da rede com base nos gradientes
    optimizer.step()

    # Exibe a perda a cada 100 épocas
    if epoch % 100 == 99:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Faz previsões com o modelo treinado sem calcular gradientes
with torch.no_grad():
    # Faz a previsão para um novo valor de entrada (10.0)
    predicted = model(torch.tensor([[10.0]], dtype=torch.float32))
    # Exibe o resultado da previsão
    print(f'Previsão de tempo de conclusão: {predicted.item()} minutos')