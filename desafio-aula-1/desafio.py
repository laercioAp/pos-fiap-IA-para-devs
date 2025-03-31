# 1. Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# 2. Carregar os dados
url = "https://raw.githubusercontent.com/pnferreira/fiap-ia-devs/main/dropout-inaugural.csv"
df = pd.read_csv(url)

# 3. Explorar os dados
print(df.head())
print(df.info())
print(df['Target'].value_counts())

# 4. Preprocessamento
label_encoder = LabelEncoder()
df['Target'] = label_encoder.fit_transform(df['Target'])  # 'Graduate' -> 0, 'Dropout' -> 1

# Separar features e target
X = df.drop(columns=['Target'])  # Removendo a variável alvo
y = df['Target']

# Transformar variáveis categóricas em numéricas
X = pd.get_dummies(X, drop_first=True)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Treinar modelo XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# 6. Avaliação do modelo
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)  # Recall para Dropout (1)
print(f"Acurácia: {accuracy:.2%}")
print(f"Recall (Dropout): {recall:.2%}")
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred))

# Matriz de Confusão
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.show()

# 7. Criar um dataset para previsão (Forecast)
dados_novos = pd.DataFrame({
    'Feature1': [valor1],  
    'Feature2': [valor2],
    'Feature3': [valor3],
    # Adicione todas as colunas necessárias conforme o dataset original
})

# Transformar variáveis categóricas em numéricas
dados_novos = pd.get_dummies(dados_novos, drop_first=True)

# Garantir que tenha as mesmas colunas que o modelo treinado
faltantes = set(X.columns) - set(dados_novos.columns)
for coluna in faltantes:
    dados_novos[coluna] = 0  # Adiciona colunas ausentes com valor zero

# Reordenar colunas para corresponder ao modelo
dados_novos = dados_novos[X.columns]

# Normalizar os novos dados
dados_novos = scaler.transform(dados_novos)

# Fazer previsão
previsao = model.predict(dados_novos)

# Converter para 'Graduate' ou 'Dropout'
resultado = label_encoder.inverse_transform(previsao)

print("Previsão para novos alunos:", resultado)
