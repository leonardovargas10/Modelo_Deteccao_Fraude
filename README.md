---

## Modelo de Detecção de Fraude — Transações de Cartão de Crédito

<p align="center"><img src="./img01.jpeg" width="50%"></p>

> **Autor:** Leonardo Aderaldo Vargas  
> **Fonte:** [Kaggle — Credit Card Transactions Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)  
> **Período:** janeiro de 2019 a dezembro de 2020

<p align="center"><img src="https://img.shields.io/static/v1?label=STATUS&message=CONCLUIDO&color=GREEN&style=for-the-badge"/></p>

---

## Sumário

1. [Contexto de Negócio](#1-contexto-de-negócio)
2. [Objetivos](#2-objetivos)
3. [Fundamentação Teórica](#3-fundamentação-teórica)
4. [Fonte de Dados](#4-fonte-de-dados)
5. [Arquitetura da Solução](#5-arquitetura-da-solução)
6. [Definição da Target](#6-definição-da-target)
7. [Estratégia de Amostragem](#7-estratégia-de-amostragem)
8. [Análise Exploratória](#8-análise-exploratória)
9. [Feature Engineering](#9-feature-engineering)
10. [Pré-Processamento e Feature Selection](#10-pré-processamento-e-feature-selection)
11. [Modelagem Supervisionada](#11-modelagem-supervisionada)
12. [Calibração, Threshold e Retorno Financeiro](#12-calibração-threshold-e-retorno-financeiro)
13. [Explicabilidade](#13-explicabilidade)
14. [Resultados Consolidados](#14-resultados-consolidados)
15. [Artefatos Gerados](#15-artefatos-gerados)
16. [Referências](#16-referências)

---

## 1. Contexto de Negócio

Este projeto desenvolve uma solução de detecção de fraude em transações de cartão. O modelo não é uma regra isolada de “fraude ou não fraude”: ele integra uma política que equilibra fraudes capturadas, clientes legítimos impactados, capacidade operacional e retorno financeiro.

Cada transação recebe um score de risco. Um **alerta** ocorre quando o score ultrapassa o cutoff e a transação é encaminhada para revisão, autenticação adicional ou outra ação. O `Alert Rate` é a proporção das transações que gera alerta e traduz o modelo em carga operacional.

**Capacidade** é o volume máximo que a operação consegue tratar. Como não é possível revisar tudo, o modelo deve concentrar fraudes nos primeiros percentuais da fila. Um Rating de `A` a `E` complementa a política, organizando as transações da menor para a maior concentração de risco.

---

## 2. Objetivos

- Detectar fraudes usando a target real `is_fraud`.
- Preservar a ordem temporal e impedir vazamento de informação futura.
- Criar features transacionais, comportamentais, geográficas e relacionais.
- Comparar um LightGBM regularizado com uma versão otimizada por HyperOpt.
- Controlar overfitting por gaps de Gini e PR-AUC.
- Calibrar probabilidades em período separado.
- Escolher cutoff e Rating em amostra exclusiva de política.
- Comparar Treino, Validação, Teste e OOT com as mesmas regras congeladas.
- Monitorar métricas, prevalência, capacidade, Rating e retorno financeiro.
- Explicar o modelo com SHAP.

---

## 3. Fundamentação Teórica

- [x] Classificação binária desbalanceada
- [x] Validação temporal e OOT
- [x] Feature engineering causal
- [x] Grafo bipartido simples
- [x] LightGBM e HyperOpt
- [x] Gini e PR-AUC
- [x] Calibração de probabilidades
- [x] Cutoff, alerta e capacidade
- [x] Rating de risco
- [x] Retorno financeiro incremental
- [x] Explicabilidade com SHAP

---

## 4. Fonte de Dados

A base foi gerada pelo simulador Sparkov e contém transações legítimas e fraudulentas de aproximadamente mil clientes e centenas de estabelecimentos.

| Arquivo | Registros | Fraudes | Taxa de fraude | Período |
|---|---:|---:|---:|---|
| `fraudTrain.csv` | 1.296.675 | 7.506 | 0,579% | jan/2019 a 21/jun/2020 |
| `fraudTest.csv` | 555.719 | 2.145 | 0,386% | 21/jun/2020 a dez/2020 |
| **Total** | **1.852.394** | **9.651** | **0,521%** | jan/2019 a dez/2020 |

Os dados incluem data, cartão, estabelecimento, categoria, valor, localização do cliente e do lojista e a flag `is_fraud`. Nomes, endereço, gênero, profissão e identificadores textuais não são entregues ao modelo. Por ser simulada, a base demonstra metodologia, mas não garante o mesmo desempenho em produção.

---

## 5. Arquitetura da Solução

```text
fraudTrain.csv + fraudTest.csv
              |
      ordenação cronológica
              |
       EDA e auditoria
              |
 feature engineering causal + grafo simples
              |
 Treino / Validação / Calibração / Política / Teste / OOT
              |
 LightGBM Benchmark -> HyperOpt -> LightGBM final
              |
 Calibração -> Cutoff -> Rating
              |
 Métricas temporais -> Retorno -> SHAP
```

Modelo, calibrador e política são componentes separados. Teste e OOT somente são consultados depois que hiperparâmetros, calibração, cutoff e Rating estão congelados.

---

## 6. Definição da Target

| `is_fraud` | Interpretação |
|---:|---|
| `0` | Transação legítima |
| `1` | Transação fraudulenta |

Fraudes representam somente 0,521% da base. Por isso, acurácia não é métrica principal: prever tudo como legítimo produziria mais de 99% de acurácia sem detectar uma única fraude.

---

## 7. Estratégia de Amostragem

| Amostra | Período | Registros | Fraudes | Taxa | Uso |
|---|---|---:|---:|---:|---|
| Treino | jan/2019 a dez/2019 | 924.850 | 5.220 | 0,564% | Ajuste do modelo |
| Validação | jan/2020 a mar/2020 | 172.843 | 1.123 | 0,650% | HyperOpt e overfitting |
| Calibração | abr/2020 a mai/2020 | 141.235 | 829 | 0,587% | Ajuste do calibrador |
| Política | 1/jun/2020 a 21/jun/2020 | 57.747 | 334 | 0,578% | Cutoff e Rating |
| Teste | 21/jun/2020 a set/2020 | 274.198 | 1.209 | 0,441% | Avaliação final |
| OOT | out/2020 a dez/2020 | 281.521 | 936 | 0,333% | Generalização futura |

A queda da prevalência ao longo de 2020 é analisada junto das métricas mensais. Assim, é possível separar mudanças da taxa-base de uma perda real de ordenação.

---

## 8. Análise Exploratória

A EDA cobre qualidade, prevalência, tempo, geografia, variáveis numéricas e categorias:

- evolução mensal do volume e da fraude;
- distribuição do valor em escala original e logarítmica;
- comparação de valor, idade, população e distância entre classes;
- taxa de fraude por categoria e estado acompanhada do volume;
- comportamento por hora, dia da semana e fim de semana;
- distância cliente–estabelecimento pela fórmula de Haversine;
- cardinalidade, duplicidades e dados ausentes.

A prevalência cai para 0,333% no OOT. Essa mudança afeta especialmente PR-AUC, Precision e carga operacional, mas não explica sozinha todo gap entre treino e futuro.

---

## 9. Feature Engineering

Todas as estatísticas históricas respeitam a ordem cronológica e usam somente eventos anteriores, por meio de `cumcount`, `shift` e agregações acumuladas deslocadas.

### 9.1 Transação, tempo e geografia

- valor e logaritmo do valor;
- idade na data da transação;
- hora, dia da semana, fim de semana, seno e cosseno da hora;
- população e distância cliente–estabelecimento.

### 9.2 Histórico do cartão e do estabelecimento

- quantidade de transações anteriores;
- tempo desde a última transação;
- média, desvio padrão e z-score histórico do valor;
- frequência acumulada por dia;
- volume e quantidade de cartões distintos por estabelecimento.

### 9.3 Grafo bipartido simples

Na relação `cartão -> estabelecimento`, foram criados grau histórico do cartão, grau do estabelecimento, interações anteriores do par, flag de primeiro contato e participação do lojista no histórico do cartão. `merchant` e `cc_num` são apenas chaves técnicas; seus valores não são features diretas.

---

## 10. Pré-Processamento e Feature Selection

O modelo utiliza **26 features**. `category` e `state` são categorias nativas do LightGBM, que não exige escalonamento das variáveis numéricas.

Foram excluídos target, identificadores únicos, PII, nome, endereço, `gender`, `job`, `merchant` e `cc_num` como entradas diretas, além de coordenadas brutas após a criação da distância. A retirada de `job` decorre de IV excessivo e risco de proxy artificial do simulador.

A seleção prioriza disponibilidade no momento da decisão, causalidade temporal, justificativa de negócio e estabilidade. Ganho e SHAP servem como diagnóstico, não como regra automática de exclusão.

---

## 11. Modelagem Supervisionada

### 11.1 Experimentos

| Experimento | Papel |
|---|---|
| LightGBM Benchmark | Referência simples e regularizada |
| LightGBM + HyperOpt | Modelo principal otimizado em 20 avaliações |
| Calibração sigmoide | Candidata paramétrica |
| Calibração isotônica | Candidata não paramétrica e método vencedor |

A parte não supervisionada foi removida. O projeto combina um classificador supervisionado auditável com política operacional e Rating.

### 11.2 Métricas

| Métrica | Interpretação |
|---|---|
| Gini | Poder de ordenação: `2 × AUC − 1` |
| PR-AUC | Qualidade do ranking quando fraude é rara |
| Precision | Fraudes entre os alertas |
| Recall | Fraudes capturadas entre todas as fraudes |
| F1 | Equilíbrio entre Precision e Recall |
| Alert Rate | Percentual enviado para ação |

### 11.3 Benchmark e HyperOpt

| Modelo | Etapa | Gini | PR-AUC |
|---|---|---:|---:|
| LightGBM Benchmark | Treino | 0,9947 | 0,8996 |
| LightGBM Benchmark | Validação | 0,9926 | 0,8801 |
| LightGBM + HyperOpt | Treino | 0,9982 | 0,9481 |
| LightGBM + HyperOpt | Validação | 0,9955 | 0,9053 |

O HyperOpt maximiza PR-AUC de validação e penaliza gaps de Gini, PR-AUC e complexidade. O ganho veio acompanhado de gap de PR-AUC de 0,0428, sinal de overfitting que exige confirmação em Teste, OOT e estabilidade mensal.

---

## 12. Calibração, Threshold e Retorno Financeiro

### 12.1 Calibração

| Método | Brier Score | LogLoss |
|---|---:|---:|
| Isotônica | 0,0014 | 0,0056 |
| Sigmoide | 0,0018 | 0,0096 |

A regressão isotônica foi escolhida. Probabilidades calibradas não precisam ocupar todo o intervalo de 0 a 1: em fraude rara, valores concentrados abaixo de 0,20 podem ser coerentes. Calibração busca correspondência entre probabilidade e frequência observada, não espalhamento visual.

### 12.2 Cutoff e capacidade

O cutoff de score foi definido somente na amostra de Política:

```text
cutoff = 0,036098
```

Na Política, ele produziu Alert Rate de 0,95%, Precision de 53,85%, Recall de 88,02% e F1 de 0,6682.

| Capacidade | Alertas | Precision | Recall |
|---:|---:|---:|---:|
| 0,10% | 58 | 100,00% | 17,37% |
| 0,25% | 145 | 100,00% | 43,41% |
| 0,50% | 289 | 86,51% | 74,85% |
| 1,00% | 578 | 50,87% | 88,02% |
| 2,00% | 1.155 | 27,71% | 95,81% |

### 12.3 Rating

Os cortes `A` a `E` são congelados na Política e reaplicados em todas as amostras. No OOT, o Rating `E` contém 0,34% das transações e taxa de fraude de 77,05%; no treino, contém 0,59% e taxa de fraude de 85,54%. Há drift, mas a faixa continua concentrando fortemente o risco.

### 12.4 Retorno financeiro

```text
Retorno incremental = fraude evitada − custo de revisão e atrito
```

Com recuperação de 75% e custo/atrito de R$ 5 por alerta legítimo:

| Etapa | Perda sem modelo | Fraude evitada | Revisão/atrito | Retorno incremental |
|---|---:|---:|---:|---:|
| Teste | R$ 643.430,84 | R$ 467.273,13 | R$ 10.615,00 | **R$ 456.658,13** |
| OOT | R$ 489.893,84 | R$ 358.831,42 | R$ 10.615,00 | **R$ 348.216,42** |

Como não existem custos reais na base, esses valores são cenários. O notebook inclui sensibilidade para recuperação de 50%, 75% e 100% e custo de R$ 1, R$ 5 e R$ 10.

---

## 13. Explicabilidade

SHAP é usado para importância global, direção dos efeitos e explicações individuais. Ele mostra como as features deslocam o score, mas não demonstra causalidade. A análise também verifica se históricos ou categorias atuam como proxies artificiais do simulador.

---

## 14. Resultados Consolidados

### 14.1 Modelo final

| Indicador | Resultado |
|---|---|
| Target | `is_fraud` |
| Modelo | LightGBM + HyperOpt |
| Features | 26 |
| Calibração | Isotônica |
| Cutoff | 0,036098 |
| Política | Capacidade próxima de 1% |
| Rating | A a E, com cortes congelados |
| Explicabilidade | SHAP |

### 14.2 Mesma política nas quatro amostras

| Etapa | Gini | PR-AUC | Prevalência | Precision | Recall | F1 | Alert Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Treino | 0,9982 | 0,9481 | 0,56% | 49,33% | 96,80% | 0,6535 | 1,11% |
| Validação | 0,9955 | 0,9053 | 0,65% | 56,54% | 91,27% | 0,6982 | 1,05% |
| Teste | 0,9925 | 0,8499 | 0,44% | 46,68% | 87,84% | 0,6096 | 0,83% |
| OOT | 0,9899 | 0,8604 | 0,33% | 39,86% | 90,28% | 0,5530 | 0,75% |

O Gini permanece alto, mas não é lido isoladamente. A queda de PR-AUC evidencia otimismo no desenvolvimento; ainda assim, Teste e OOT mantêm forte concentração de fraude, Recall entre 87,84% e 90,28% e Alert Rate abaixo de 1%. A menor Precision no OOT é parcialmente coerente com a menor prevalência.

### 14.3 Limitações

- Dados simulados podem conter padrões excessivamente fáceis.
- Não há atraso real de chargeback.
- Custos e capacidade são premissas.
- Gini alto não garante probabilidades perfeitas nem desempenho real equivalente.
- Fraudes mudam; modelo, calibração, cutoff e Rating exigem monitoramento.

---

## 15. Artefatos Gerados

| Artefato | Localização | Descrição |
|---|---|---|
| `Modelo_Deteccao_Fraude.ipynb` | raiz | Notebook completo e executado |
| `modelo_fraude_sparkov.pkl` | `models/` | Pipeline e decisões congeladas |

Os CSVs não são versionados devido ao tamanho. Para executar o notebook, disponibilize `fraudTrain.csv` e `fraudTest.csv` no diretório indicado na célula de leitura.

---

## 16. Referências

- [Kaggle — Credit Card Transactions Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- [Sparkov Data Generation](https://github.com/namebrandon/Sparkov_Data_Generation)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [Hyperopt](https://hyperopt.github.io/hyperopt/)
- [SHAP](https://shap.readthedocs.io/)

---
