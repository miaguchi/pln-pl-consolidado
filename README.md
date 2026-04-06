# PLN + Sucesso Legislativo: Projetos de Lei na Camara dos Deputados

Pipeline completo de **Processamento de Linguagem Natural (PLN)**, **modelagem estatistica** e **machine learning** para analise de projetos de lei (PLs) na Camara dos Deputados do Brasil, com foco em corporacoes de seguranca publica e defesa.

## Objetivo

Investigar os determinantes do sucesso legislativo de projetos de lei, combinando:

- **NLP**: tokenizacao, frequencias, bigramas/trigramas, LDA (topic modeling)
- **Inferencia estatistica**: logit, probit, modelos de contagem, multinivel, Firth, Cox
- **Machine learning**: Random Forest, XGBoost, Hist-GBT com backtesting temporal
- **Analise exploratoria**: MCA, ANACOR, K-Means, redes de coautoria, sentiment analysis

## Dados de Entrada

| Arquivo | Descricao |
|---|---|
| `pl_limpo.xlsx` | Base principal de PLs com ementa, autor, partido, UF, situacao |
| `politicos-de-farda.xlsx` | Dados sociologicos de parlamentares ligados a corporacoes de seguranca |

## Estrutura do Pipeline

O script executa **47 secoes** organizadas em blocos logicos:

### Bloco 1 — Pre-processamento (secoes 1-4)

| Secao | Descricao |
|---|---|
| 1 | Leitura, limpeza e padronizacao da base |
| 2 | Construcao da base de texto (ementas) |
| 3 | Frequencias de palavras, bigramas, trigramas |
| 4 | Extracao de leis e artigos citados nas ementas |

### Bloco 2 — Analise Descritiva (secoes 5-9)

| Secao | Descricao |
|---|---|
| 5 | Contagens por UF, partido, autor, situacao |
| 6 | Qui-quadrado e taxas de sucesso por partido/UF |
| 7 | Graficos descritivos |
| 8 | Produtividade por autor e coautoria |
| 9 | Analise temporal (PLs por ano) |

### Bloco 3 — Topic Modeling / LDA (secoes 10-14)

| Secao | Descricao |
|---|---|
| 10 | LDA com 5 topicos |
| 11 | Cruzamento partido x topico |
| 12 | Evolucao dos topicos por ano |
| 13 | Sucesso legislativo por topico |
| 14 | Coautoria e partido x agenda |

### Bloco 4 — Modelos Inferenciais (secoes 15-22B)

| Secao | Descricao |
|---|---|
| 15-16 | Bases de regressao e legislatura x sucesso |
| 17 | Logit principal (topico + partido + ano) com erros clusterizados |
| 17B | Modelo multinivel (intercepto aleatorio por autor) |
| 19-20 | Modelo preditivo complementar e efeito isolado dos topicos |
| 21-22 | Analise de corporacao (logit inferencial + preditivo) |
| 22A | Modelos de contagem (Poisson, NB, ZINB) |
| 22B | ANACOR, efeitos marginais (AME), rede de coautoria |
| 22B-EXT | Extensoes: interacao partido x topico, tempo nao linear, Firth |
| 22B-SOC | Modelo sociologico com variaveis de carreira |

### Bloco 5 — Previsao e Cenarios (secoes 24-30B)

| Secao | Descricao |
|---|---|
| 24-25 | Media de probabilidade por ano, cenarios otimista/pessimista |
| 26-28 | Simulacao por partido, heatmap partido x ano, rankings |
| 29 | ML temporal (RF, XGBoost, Hist-GBT) com backtesting |
| 30-30B | Predicoes finais e previsao futura com ML |

### Bloco 6 — Robustez e Extensoes (secoes 31-47)

| Secao | Descricao |
|---|---|
| 31 | Tabelas-resumo e quadro de robustez |
| 32 | Exportacao final (Excel com 80+ abas) |
| 33 | Correlacao de Pearson (produtividade x sucesso) |
| 34 | K-Means clustering de parlamentares |
| 35 | Sentiment analysis das ementas |
| 36 | Analise espacial (taxa por UF + choropleth) |
| 37 | Probit como robustez do logit |
| 38 | Cross-validation estratificado (5-fold) |
| 39 | Serie temporal anual (decomposicao + Holt-Winters) |
| 40 | MCA (analise de correspondencia multipla) |
| 41 | Integracao sociologica (rede + capital + tipologia) |
| 42 | Interacao topico x partido x corporacao |
| 42B | Legislatura reintegrada + tabela unificada de importancia |
| 43 | Tempo nao linear (spline + blocos historicos) |
| 45 | Modelo de sobrevivencia (Kaplan-Meier + Cox) |
| 46 | Calibration plot do modelo preditivo |
| 47 | Coherence score do LDA (validacao k=4,5,6) |

## Saidas

### Excel

Arquivo `resultados_pln_pl.xlsx` com **80+ abas**, incluindo:

- Bases limpas e de regressao
- Frequencias, cruzamentos, taxas de sucesso
- Coeficientes de todos os modelos (logit, probit, Poisson, Cox)
- Odds ratios, AMEs, metricas de modelo
- Predicoes por ano, partido, corporacao
- Backtesting ML e cenarios futuros

### Graficos (42 PNGs)

Todos salvos na mesma pasta do script:

- Distribuicoes descritivas (`pls_por_ano.png`, `lda_topicos_pl.png`)
- Heatmaps de cruzamento (`heatmap_partido_topico_pl.png`)
- Curvas de modelo (`calibration_plot.png`, `cv_estratificado.png`)
- Analise espacial (`mapa_choropleth_uf.png`)
- Redes (`rede_coautoria.png`)
- Sobrevivencia (`kaplan_meier_topico.png`, `cox_topico.png`)
- Clustering e MCA (`kmeans_perfil_parlamentar.png`, `mca_perfil_parlamentar.png`)

## Instalacao

```bash
git clone https://github.com/miaguchi/pln-pl-consolidado.git
cd pln-pl-consolidado
pip install -r requirements.txt
```

### Dependencias opcionais

```bash
pip install xgboost lifelines gensim prince networkx
```

Estas bibliotecas habilitam funcionalidades extras (XGBoost, Kaplan-Meier/Cox, coherence score do LDA, MCA, rede de coautoria). O script funciona sem elas — as secoes correspondentes sao puladas com aviso.

## Uso

1. Coloque `pl_limpo.xlsx` e `politicos-de-farda.xlsx` na mesma pasta do script
2. Execute:

```bash
python pln_pl_consolidado.py
```

3. O script gera `resultados_pln_pl.xlsx` e 42 graficos `.png` na mesma pasta

## Requisitos do Sistema

- Python >= 3.9
- ~4 GB RAM (para LDA + ML com backtesting)
- Tempo de execucao estimado: 5-15 min dependendo do hardware

## Metodologia

- **LDA**: 5 topicos, `max_iter=30`, `random_state=42`
- **Logit inferencial**: erros clusterizados por autor (Huber-White)
- **Firth (rare events)**: regularizacao L1 como proxy para evento raro (~1.3% de aprovacao)
- **ML**: backtesting temporal com treino em legislaturas anteriores e teste na seguinte
- **Validacao**: probit, cross-validation 5-fold, calibration plot, coherence score

## Referencial Teorico

O pipeline integra metodos de:

- **Ciencia politica**: sucesso legislativo, poder de agenda, corporativismo
- **NLP**: topic modeling (Blei et al., 2003), frequencias lexicas
- **Estatistica**: logit com evento raro (King & Zeng, 2001), modelos de contagem (Cameron & Trivedi)
- **ML**: ensemble methods com validacao temporal (Hastie, Tibshirani & Friedman)

## Licenca

MIT

## Autor

Thiago ([@miaguchi](https://github.com/miaguchi))
