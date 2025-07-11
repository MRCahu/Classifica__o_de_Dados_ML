# 📊 Classificação de Reclamações com Machine Learning

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue?logo=python) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Build](https://img.shields.io/badge/build-passing-brightgreen)

Automatize a triagem de reclamações de clientes por meio de técnicas de **Machine Learning** e **NLP**. Este repositório fornece um pipeline completo – desde a preparação de dados até o deploy do modelo – 100 % reproduzível em **Python**.

> "Transforme texto bruto em insights acionáveis e acelere a tomada de decisão."

---

## 📚 Sumário

1. [Visão Geral](#visão-geral)
2. [Destaques do Projeto](#destaques-do-projeto)
3. [Estrutura de Pastas](#estrutura-de-pastas)
4. [Requisitos](#requisitos)
5. [Instalação Rápida](#instalação-rápida)
6. [Execução Passo a Passo](#execução-passo-a-passo)
7. [Resultados](#resultados)
8. [Contribuindo](#contribuindo)
9. [Contato](#contato)
10. [Licença](#licença)

---

## Visão Geral

O objetivo é **classificar automaticamente** categorias de reclamações a partir de variáveis textuais e estruturadas. O pipeline engloba:

* **Limpeza & EDA** do dataset (`reclamacoes_dataset.csv`).
* **Pré-processamento** com `ColumnTransformer` & `Pipeline` do *scikit‑learn*.
* Teste de múltiplos algoritmos (RandomForest, XGBoost, Logistic Regression) com *cross‑validation*.
* **Comparação de métricas** (Accuracy, F1, ROC‑AUC) registrada em `comparacao_modelos.csv`.
* Serialização do **modelo campeão** e *preprocessor* em `modelos_treinados/`.
* Geração de relatórios HTML interativos para correlação, distribuição de classes e variáveis.

## Destaques do Projeto

* 📈 **Performance:** F1‑Score ≥ 0.91 no conjunto hold‑out.
* 🏷️ **Explainability:** `permutation_importance` para features top‑10.
* ⚙️ **Automação:** Script `executar_projeto_ml.py` executa todo o fluxo em um único comando.
* 🚀 **Pronto para Deploy:** Artefatos `.pkl` podem ser servidos via FastAPI/Flask.

## Estrutura de Pastas

```text
Classificacao_de_Dados_ML/
├── data/                       # (<1 GB) dados brutos & processados
│   └── reclamacoes_dataset.csv
├── notebooks/
│   └── classificacao_reclamacoes_ml.ipynb
├── src/                        # scripts reutilizáveis
│   └── executar_projeto_ml.py
├── reports/                    # relatórios HTML + métricas
├── modelos_treinados/          # modelos & encoders salvos
├── requirements.txt
└── README.md
```

> 🔒 **Nota:** Dados sensíveis foram removidos; use variáveis de ambiente para credenciais.

## Requisitos

* Python >= 3.10
* Dependências listadas em `requirements.txt` (<50 MB de download).

## Instalação Rápida

```bash
# clone
git clone https://github.com/<seu-usuario>/Classificacao_de_Dados_ML.git
cd Classificacao_de_Dados_ML

# ambiente virtual (opcional)
python -m venv .venv && source .venv/bin/activate

# dependências
pip install -r requirements.txt
```

## Execução Passo a Passo

```bash
# 1. roda pipeline completo (pré‑processa, treina, avalia, salva artefatos)
python src/executar_projeto_ml.py

# 2. (opcional) abre notebook para exploração
jupyter notebook notebooks/classificacao_reclamacoes_ml.ipynb
```

## Resultados

| Modelo              | Accuracy | F1 (macro) | ROC‑AUC  |
| ------------------- | -------- | ---------- | -------- |
| RandomForest        | **0.93** | **0.91**   | 0.95     |
| XGBoost             | 0.92     | 0.90       | **0.96** |
| Logistic Regression | 0.88     | 0.85       | 0.90     |

Relatórios completos disponíveis em `reports/`. Exemplo gráfico:
![Matriz de Confusão](reports/matriz_confusao.png)

## Contribuindo

1. Faça um *fork* do projeto.
2. Crie uma *feature branch*: `git checkout -b feature/sua-feature`.
3. *Commit* & *push*: `git push origin feature/sua-feature`.
4. Abra um *Pull Request* 📝.

Antes de enviar, execute `pre-commit run --all-files` para garantir estilo e qualidade.

## Contato

**Mauro Roberto Barbosa Cahu**  ·  [LinkedIn](https://www.linkedin.com/in/mauro-cahu-159a05273/)  ·  [GitHub](https://github.com/MRCahu)

✉️ Email: [maurocahu@gmail.com](mailto:maurocahu@gmail.com) · 📱 +55 81 99292‑2415

---

## Licença

Distribuído sob a licença **MIT**. Consulte `LICENSE` para mais informações.

---

# 📊 Complaint Classification with Machine Learning

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue?logo=python) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Build](https://img.shields.io/badge/build-passing-brightgreen)

Automate customer complaint triage with **Machine Learning** & **NLP**. This repository ships a fully reproducible pipeline – from data prep to model deploy – in **Python**.

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Project Layout](#project-layout)
4. [Requirements](#requirements)
5. [Quick Setup](#quick-setup)
6. [Step‑by‑Step Usage](#step-by-step-usage)
7. [Results](#results-en)
8. [Contributing](#contributing)
9. [Contact](#contact)
10. [License](#license)

---

## Overview

The goal is to **automatically classify** complaint categories from text & structured variables. The pipeline covers:

* **Data cleaning & EDA** (`reclamacoes_dataset.csv`).
* **Pre‑processing** via `ColumnTransformer` & `Pipeline` (scikit‑learn).
* Grid‑search across algorithms (RandomForest, XGBoost, Logistic Regression) with cross‑validation.
* **Metric comparison** stored in `comparacao_modelos.csv`.
* Serialization of the **best model** and pre‑processor to `modelos_treinados/`.
* Generation of interactive HTML reports for correlation, class distribution & variables.

## Key Features

* 📈 **Performance:** F1‑Score ≥ 0.91 on hold‑out set.
* 🏷️ **Explainability:** top‑10 feature importance with permutation tests.
* ⚙️ **Automation:** `executar_projeto_ml.py` runs the entire flow with one command.
* 🚀 **Deploy‑ready:** `.pkl` artifacts can be served via FastAPI/Flask.

## Project Layout

```text
Classificacao_de_Dados_ML/
├── data/
│   └── reclamacoes_dataset.csv
├── notebooks/
│   └── classificacao_reclamacoes_ml.ipynb
├── src/
│   └── executar_projeto_ml.py
├── reports/
├── modelos_treinados/
├── requirements.txt
└── README.md
```

## Requirements

* Python >= 3.10
* Dependencies in `requirements.txt` (<50 MB download).

## Quick Setup

```bash
# clone
git clone https://github.com/<your-user>/Classificacao_de_Dados_ML.git
cd Classificacao_de_Dados_ML

# virtual env
python -m venv .venv && source .venv/bin/activate

# deps
pip install -r requirements.txt
```

## Step‑by‑Step Usage

```bash
# full pipeline
python src/executar_projeto_ml.py

# notebook exploration
jupyter notebook notebooks/classificacao_reclamacoes_ml.ipynb
```

## Results {#results-en}

| Model               | Accuracy | F1 (macro) | ROC‑AUC  |
| ------------------- | -------- | ---------- | -------- |
| RandomForest        | **0.93** | **0.91**   | 0.95     |
| XGBoost             | 0.92     | 0.90       | **0.96** |
| Logistic Regression | 0.88     | 0.85       | 0.90     |

Full reports are available in `reports/`. Example plot:
![Confusion Matrix](reports/matriz_confusao.png)

## Contributing

1. Fork the project.
2. Create your feature branch: `git checkout -b feature/awesome-feature`.
3. Commit & push: `git push origin feature/awesome-feature`.
4. Open a Pull Request 📝.

Run `pre-commit run --all-files` before pushing to ensure consistency.

## Contact

**Mauro Roberto Barbosa Cahu**  ·  [LinkedIn](https://www.linkedin.com/in/mauro-cahu-159a05273/) · [GitHub](https://github.com/MRCahu)

✉️ Email: [maurocahu@gmail.com](mailto:maurocahu@gmail.com) · 📱 +55 81 99292‑2415

---

## License

Distributed under the **MIT** License – see `LICENSE` for details.
