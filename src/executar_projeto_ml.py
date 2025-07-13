#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projeto Completo de Machine Learning: Classificação de Reclamações de Consumidores
Execução automatizada do pipeline completo
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import pickle
import json
import os
from datetime import datetime

print("🤖 PROJETO COMPLETO DE MACHINE LEARNING")
print("=" * 50)
print("📋 Classificação de Reclamações de Consumidores")
print(f"📅 Execução: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

# 1. CRIAÇÃO DO DATASET
print("\n📊 1. CRIANDO DATASET SINTÉTICO...")
np.random.seed(42)

categorias = {
    'Produto Defeituoso': [
        'Produto chegou com defeito de fábrica',
        'Aparelho parou de funcionar após poucos dias',
        'Produto não corresponde à descrição',
        'Defeito na tela do celular',
        'Produto veio danificado na entrega'
    ],
    'Atendimento': [
        'Atendimento muito demorado no telefone',
        'Funcionário foi grosseiro e mal educado',
        'Não consegui resolver meu problema',
        'Atendente não soube me informar',
        'Fui mal atendido na loja física'
    ],
    'Cobrança Indevida': [
        'Cobraram valor a mais na fatura',
        'Taxa não informada na contratação',
        'Cobrança duplicada no cartão',
        'Valor diferente do acordado',
        'Cobrança após cancelamento do serviço'
    ],
    'Entrega': [
        'Produto não foi entregue no prazo',
        'Entrega foi feita no endereço errado',
        'Produto chegou muito atrasado',
        'Não recebi o produto comprado',
        'Entregador foi mal educado'
    ]
}

empresas = ['TechMart', 'SuperCompras', 'MegaStore', 'FastDelivery', 'EletroMax']
estados = ['SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'GO', 'PE', 'CE']

dados = []
n_samples = 2000

for i in range(n_samples):
    categoria = np.random.choice(list(categorias.keys()))
    texto = np.random.choice(categorias[categoria])
    empresa = np.random.choice(empresas)
    estado = np.random.choice(estados)
    
    if categoria == 'Produto Defeituoso':
        valor = np.random.uniform(50, 2000)
        tempo_resposta = np.random.uniform(1, 15)
    elif categoria == 'Cobrança Indevida':
        valor = np.random.uniform(20, 500)
        tempo_resposta = np.random.uniform(1, 10)
    elif categoria == 'Entrega':
        valor = np.random.uniform(30, 800)
        tempo_resposta = np.random.uniform(1, 7)
    else:  # Atendimento
        valor = np.random.uniform(0, 100)
        tempo_resposta = np.random.uniform(1, 20)
    
    dados.append({
        'texto_reclamacao': texto,
        'categoria': categoria,
        'empresa': empresa,
        'estado': estado,
        'valor_envolvido': round(valor, 2),
        'tempo_resposta_dias': round(tempo_resposta, 1)
    })

df = pd.DataFrame(dados)
print(f"✅ Dataset criado: {df.shape[0]} linhas, {df.shape[1]} colunas")
print(f"📊 Classes: {df['categoria'].value_counts().to_dict()}")

# 2. FEATURE ENGINEERING
print("\n🔧 2. FEATURE ENGINEERING...")
df_processed = df.copy()

# Criando bins para categorização
valor_bins = pd.qcut(df_processed['valor_envolvido'], q=3, retbins=True)[1]
tempo_bins = pd.qcut(df_processed['tempo_resposta_dias'], q=3, retbins=True)[1]

df_processed['categoria_valor'] = pd.cut(
    df_processed['valor_envolvido'], 
    bins=valor_bins, 
    labels=['Baixo', 'Médio', 'Alto'],
    include_lowest=True
)

df_processed['categoria_tempo'] = pd.cut(
    df_processed['tempo_resposta_dias'], 
    bins=tempo_bins, 
    labels=['Rápido', 'Médio', 'Lento'],
    include_lowest=True
)

df_processed['tamanho_texto'] = df_processed['texto_reclamacao'].str.len()
df_processed['num_palavras'] = df_processed['texto_reclamacao'].str.split().str.len()

print("✅ Features criadas: categoria_valor, categoria_tempo, tamanho_texto, num_palavras")

# 3. DIVISÃO DOS DADOS
print("\n📊 3. DIVISÃO TREINO/TESTE...")
X = df_processed.drop('categoria', axis=1)
y = df_processed['categoria']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")

# 4. PRÉ-PROCESSAMENTO
print("\n⚙️ 4. PRÉ-PROCESSAMENTO...")
numeric_features = ['valor_envolvido', 'tempo_resposta_dias', 'tamanho_texto', 'num_palavras']
categorical_features = ['empresa', 'estado', 'categoria_valor', 'categoria_tempo']
text_feature = 'texto_reclamacao'

numeric_transformer = Pipeline([('scaler', StandardScaler())])
categorical_transformer = Pipeline([('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))])
text_transformer = Pipeline([('tfidf', TfidfVectorizer(max_features=200, ngram_range=(1, 2), min_df=2, max_df=0.95))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('text', text_transformer, text_feature)
    ],
    remainder='drop'
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"✅ Dados processados: {X_train_processed.shape[1]} features")

# 5. ENCODING TARGET
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print(f"✅ Classes codificadas: {dict(enumerate(label_encoder.classes_))}")

# 6. TREINAMENTO DE MODELOS
print("\n🤖 5. TREINAMENTO DE MODELOS...")
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'SVM': SVC(kernel='rbf', C=1.0, random_state=42, probability=True),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"🔄 Treinando {name}...")
    start_time = datetime.now()
    
    model.fit(X_train_processed, y_train_encoded)
    y_test_pred = model.predict(X_test_processed)
    
    test_accuracy = accuracy_score(y_test_encoded, y_test_pred)
    f1 = f1_score(y_test_encoded, y_test_pred, average='weighted')
    training_time = (datetime.now() - start_time).total_seconds()
    
    results[name] = {
        'test_accuracy': test_accuracy,
        'f1_score': f1,
        'training_time': training_time
    }
    
    trained_models[name] = model
    print(f"   ✅ Acurácia: {test_accuracy:.4f} | F1: {f1:.4f} | Tempo: {training_time:.2f}s")

# 7. SELEÇÃO DO MELHOR MODELO
best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
best_model = trained_models[best_model_name]
best_f1 = results[best_model_name]['f1_score']
best_accuracy = results[best_model_name]['test_accuracy']

print(f"\n🏆 MELHOR MODELO: {best_model_name}")
print(f"   Acurácia: {best_accuracy:.4f}")
print(f"   F1-Score: {best_f1:.4f}")

# 8. VALIDAÇÃO CRUZADA
print("\n🔍 6. VALIDAÇÃO CRUZADA...")
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_train_processed, y_train_encoded, 
                           cv=cv_strategy, scoring='f1_weighted', n_jobs=-1)

print(f"✅ CV F1-Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# 9. OTIMIZAÇÃO DE HIPERPARÂMETROS (simplificada)
print("\n⚙️ 7. OTIMIZAÇÃO DE HIPERPARÂMETROS...")
if 'Random Forest' in best_model_name:
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    model_for_grid = RandomForestClassifier(random_state=42, n_jobs=-1)
else:
    param_grid = {'C': [0.1, 1, 10]}
    model_for_grid = LogisticRegression(random_state=42, max_iter=1000)

grid_search = GridSearchCV(model_for_grid, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train_processed, y_train_encoded)

optimized_model = grid_search.best_estimator_
y_test_pred_optimized = optimized_model.predict(X_test_processed)
test_accuracy_optimized = accuracy_score(y_test_encoded, y_test_pred_optimized)
test_f1_optimized = f1_score(y_test_encoded, y_test_pred_optimized, average='weighted')

print(f"✅ Modelo otimizado - Acurácia: {test_accuracy_optimized:.4f} | F1: {test_f1_optimized:.4f}")
print(f"✅ Melhores parâmetros: {grid_search.best_params_}")

# 10. AVALIAÇÃO DETALHADA
print("\n📊 8. AVALIAÇÃO DETALHADA...")
classification_rep = classification_report(y_test_encoded, y_test_pred_optimized, 
                                         target_names=label_encoder.classes_, output_dict=True)

print("📋 Relatório de Classificação:")
for classe in label_encoder.classes_:
    precision = classification_rep[classe]['precision']
    recall = classification_rep[classe]['recall']
    f1 = classification_rep[classe]['f1-score']
    support = int(classification_rep[classe]['support'])
    print(f"   {classe}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} (n={support})")

# 11. TESTE EM NOVOS DADOS
print("\n🔮 9. TESTE EM NOVOS DADOS...")
novos_dados = [
    {
        'texto_reclamacao': 'Produto chegou com defeito de fábrica',
        'empresa': 'TechMart',
        'estado': 'SP',
        'valor_envolvido': 800.0,
        'tempo_resposta_dias': 3.0
    },
    {
        'texto_reclamacao': 'Atendimento muito demorado e mal educado',
        'empresa': 'SuperCompras',
        'estado': 'RJ',
        'valor_envolvido': 0.0,
        'tempo_resposta_dias': 7.0
    }
]

df_novos = pd.DataFrame(novos_dados)

# Feature engineering nos novos dados
df_novos['categoria_valor'] = pd.cut(df_novos['valor_envolvido'], bins=valor_bins, 
                                   labels=['Baixo', 'Médio', 'Alto'], include_lowest=True)
df_novos['categoria_tempo'] = pd.cut(df_novos['tempo_resposta_dias'], bins=tempo_bins, 
                                   labels=['Rápido', 'Médio', 'Lento'], include_lowest=True)
df_novos['tamanho_texto'] = df_novos['texto_reclamacao'].str.len()
df_novos['num_palavras'] = df_novos['texto_reclamacao'].str.split().str.len()

X_novos_processed = preprocessor.transform(df_novos)
predicoes = optimized_model.predict(X_novos_processed)
probabilidades = optimized_model.predict_proba(X_novos_processed)
predicoes_labels = label_encoder.inverse_transform(predicoes)

print("🎯 Predições em novos dados:")
for i, (pred, prob_max) in enumerate(zip(predicoes_labels, probabilidades.max(axis=1))):
    print(f"   Exemplo {i+1}: {pred} (confiança: {prob_max:.2%})")

# 12. SALVAMENTO DOS MODELOS
print("\n💾 10. SALVAMENTO DOS MODELOS...")
model_dir = 'modelos_treinados'
os.makedirs(model_dir, exist_ok=True)

# Salvando componentes
with open(os.path.join(model_dir, 'modelo_classificacao_reclamacoes.pkl'), 'wb') as f:
    pickle.dump(optimized_model, f)

with open(os.path.join(model_dir, 'preprocessor.pkl'), 'wb') as f:
    pickle.dump(preprocessor, f)

with open(os.path.join(model_dir, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)

# Metadados
metadata = {
    'modelo_tipo': best_model_name,
    'data_treinamento': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'acuracia_teste': float(test_accuracy_optimized),
    'f1_score_teste': float(test_f1_optimized),
    'cv_score_medio': float(grid_search.best_score_),
    'melhores_parametros': grid_search.best_params_,
    'classes': label_encoder.classes_.tolist(),
    'total_features': int(X_train_processed.shape[1]),
    'total_amostras_treino': int(X_train_processed.shape[0])
}

with open(os.path.join(model_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# Bins para feature engineering
bins_data = {
    'valor_bins': valor_bins.tolist(),
    'tempo_bins': tempo_bins.tolist()
}

with open(os.path.join(model_dir, 'feature_bins.json'), 'w') as f:
    json.dump(bins_data, f, indent=2)

print(f"✅ Modelos salvos em: {model_dir}")

# Salvando dataset e resultados
df.to_csv('data/reclamacoes_dataset.csv', index=False)

resultados_finais = pd.DataFrame({
    'Modelo': list(results.keys()),
    'Acurácia': [results[m]['test_accuracy'] for m in results.keys()],
    'F1-Score': [results[m]['f1_score'] for m in results.keys()],
    'Tempo (s)': [results[m]['training_time'] for m in results.keys()]
}).sort_values('F1-Score', ascending=False)

resultados_finais.to_csv('data/comparacao_modelos.csv', index=False)

print("\n🎉 PROJETO CONCLUÍDO COM SUCESSO!")
print("=" * 50)
print(f"🏆 Melhor modelo: {best_model_name}")
print(f"📊 Acurácia final: {test_accuracy_optimized:.4f} ({test_accuracy_optimized*100:.2f}%)")
print(f"📊 F1-Score final: {test_f1_optimized:.4f}")
print(f"📁 Arquivos gerados:")
print(f"   📄 reclamacoes_dataset.csv")
print(f"   📄 comparacao_modelos.csv")
print(f"   📁 modelos_treinados/ (5 arquivos)")
print(f"\n💡 O modelo está pronto para uso em produção!")
