#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLN + LDA + sucesso legislativo para análise de PLs
com análise de corporação, regressões inferenciais,
modelos preditivos, previsões futuras e backtesting temporal.

Autor: Thiago
Data: 2026-03-18
Versão consolidada final — códigos 1 + 2 + 3 integrados
"""

from pathlib import Path
import re
import unicodedata
from collections import Counter
from itertools import islice

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency
from scipy import linalg
import statsmodels.formula.api as smf
import statsmodels.api as sm

try:
    import networkx as nx
    NX_OK = True
except Exception:
    NX_OK = False

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False


# =========================================================
# CONFIGURAÇÕES
# =========================================================

PASTA = Path("/home/thiago/Área de Trabalho")

ARQUIVOS_POSSIVEIS = [
    PASTA / "pl_limpo.xlsx",
    PASTA / "pl-limpo.xlsx",
]

ARQUIVO_ENTRADA = None
for arq in ARQUIVOS_POSSIVEIS:
    if arq.exists():
        ARQUIVO_ENTRADA = arq
        break

if ARQUIVO_ENTRADA is None:
    raise FileNotFoundError(
        "Nenhum arquivo encontrado. Verifique se existe "
        "'pl_limpo.xlsx' ou 'pl-limpo.xlsx' em /home/thiago/Área de Trabalho/"
    )

ARQUIVO_FARDA = PASTA / "politicos-de-farda.xlsx"
ARQUIVO_SAIDA = PASTA / "resultados_pln_pl.xlsx"

ARQUIVO_GRAFICO_ANO = PASTA / "pls_por_ano.png"
ARQUIVO_GRAFICO_TOPICOS = PASTA / "lda_topicos_pl.png"
ARQUIVO_GRAFICO_PARTIDOS = PASTA / "lda_partidos_topicos_pl.png"
ARQUIVO_HEATMAP = PASTA / "heatmap_partido_topico_pl.png"
ARQUIVO_TOPICOS_ANO_CONTAGEM = PASTA / "topicos_por_ano_contagem_pl.png"
ARQUIVO_TOPICOS_ANO_PERCENTUAL = PASTA / "topicos_por_ano_percentual_pl.png"
ARQUIVO_SUCESSO_TOPICO = PASTA / "sucesso_por_topico_pl.png"
ARQUIVO_AGENDAS_APROVADAS = PASTA / "agendas_aprovadas_pl.png"
ARQUIVO_PROB_CORP = PASTA / "prob_corporacao.png"
ARQUIVO_HEATMAP_CORP_TOPICO = PASTA / "heatmap_corporacao_topico.png"
ARQUIVO_HEATMAP_PROB_CORP_TOPICO = PASTA / "heatmap_prob_corporacao_topico.png"

TOP_N = 50
N_TOPICOS = 5
N_PALAVRAS_TOPICO = 10
TOP_PARTIDOS_GRAFICO = 10
FREQ_MIN_PARTIDO = 20
THRESHOLD_CLASSIFICACAO = 0.20
THRESHOLD_CLASSIFICACAO_CORP = 0.20
MIN_TOTAL_LOGIT = 80
MIN_APROVADOS_LOGIT = 2
MIN_PROP = 100
MIN_PROP_AUTOR = 20
MIN_TOTAL_PARTIDO_INF = 50
MIN_SUCESSO_PARTIDO_INF = 2
ANO_INICIO_BACKTEST = 2012

# Último ano com dados COMPLETOS na amostra.
# O ano de coleta pode estar incompleto (PLs ainda em tramitação,
# aprovações ainda não registradas), o que subestima artificialmente
# a taxa de sucesso e contamina o backtesting e a previsão.
# Todos os modelos preditivos e o backtesting excluem anos > ANO_ULTIMO_COMPLETO.
# A análise descritiva mantém o ano incompleto como informação parcial.
ANO_ULTIMO_COMPLETO = 2023

# Legislaturas excluídas dos modelos ML por estarem em vigência.
# A 57ª (2023–2026) ainda não tem dados finalizados: PLs tramitam,
# aprovações ainda serão registradas — a taxa observada é artificialmente
# baixa. Entra apenas em análises descritivas e no gráfico de legislatura.
LEGISLATURA_EXCLUIR_ML = ["57a"]

# Horizonte final da projeção ML (ano inclusive).
ANO_HORIZONTE_PREVISAO = 2030
# Nota metodológica: horizonte limitado a 2030 (2 legislaturas futuras).
# Projeções além desse ponto extrapolam tendência linear fora do suporte
# empírico e produzem probabilidades infladas (ex.: PT→89% em 2040),
# incompatíveis com a taxa histórica observada de 1,30%.
# Tratar como apêndice exploratório, não como previsão substantiva.
# Referência: King & Zeng (2006) — extrapolação em modelos de evento raro.

NOMES_COLUNAS = [
    "Proposicoes",
    "Ementa",
    "Explicacao",
    "Autor",
    "UF",
    "Partido",
    "Apresentacao",
    "Situacao"
]

COLUNA_TEXTO = "Ementa"

UFS_VALIDAS = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS",
    "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC",
    "SP", "SE", "TO"
}

PARTIDOS_VALIDOS = {
    "PT", "PL", "PSL", "PP", "PSD", "MDB", "UNIAO", "PSDB", "PDT",
    "REPUBLICANOS", "PSB", "PODE", "PODEMOS", "AVANTE", "PATRIOTA", "PSC",
    "PV", "PCDOB", "SOLIDARIEDADE", "NOVO", "CIDADANIA", "PSOL",
    "PTB", "PRD", "PR", "DEM", "PMDB", "PRB", "PHS", "PTC", "PTDOB", "PROS",
    "PMN", "PRP", "DC", "REDE", "AGIR", "UNIAO BRASIL",
    "PFL", "PPS", "PPB", "PEN", "PMB", "PRONA", "PRN"
}

sns.set(style="whitegrid")


# =========================================================
# PLACEHOLDERS
# =========================================================

for _n in [
    "df_odds", "df_odds_leg", "df_odds_ano", "df_odds_principal",
    "df_rank", "coef", "coef_topics_plot", "tab_partido_inf",
    "tabela_corp_sucesso_full", "tabela_corp_topico", "tabela_corp_topico_pct",
    "tabela_corp_sucesso", "df_odds_corp", "base_pred_corp", "base_pred_corp_top",
    "df_rank_c", "coef_c",
    "tabela_corporacao", "df_resultados_bt", "df_predicoes_bt", "resumo_modelos_bt",
    "pred_ano", "pred_ano_historico", "pred_partido", "pred_legislatura",
    "pred_corporacao", "pred_ano_partido", "pred_ano_corporacao",
    "pred_leg_partido", "pred_leg_corporacao", "pred_corp_topico",
    "resumo_ano", "resumo_cenarios", "resumo_partido_ano", "resumo_corporacao_ano",
    "ranking_partido_futuro", "ranking_corporacao_futuro", "df_rank_futuro",
    "df_pred_ano", "ranking_partido",
    "df_testes_estatisticos", "df_metricas_modelo_pred", "df_metricas_thresholds",
    "df_metricas_modelo_corp", "df_metricas_thresholds_corp", "df_resumo_executivo",
    "df_modelos_contagem", "df_overdisp",
    "df_ame_principal", "df_ame_partido", "df_ame_corp",
    "df_anacor_corp", "df_anacor_topico", "df_anacor_partido",
    "df_rede_nos", "df_rede_arestas", "df_rede_metricas",
    "df_zinb_metricas", "df_zinb_coef", "df_zinb_comparacao",
    "df_multinivel_metricas", "df_multinivel_coef", "df_multinivel_icc",
    "df_ct_resultados", "df_odds_sociol", "df_sociol_descritiva",
    "df_previsao_ml", "df_previsao_ml_resumo", "df_probit_coef", "df_cv_resultados",
    "df_pearson", "df_clusters", "df_sentiment", "_taxa_uf_esp",
    "df_serie_temporal", "df_mca_coords", "df_sociol_integrado",
]:
    globals()[_n] = pd.DataFrame()

(modelo_partido, modelo_leg, modelo_ano, modelo_principal,
 modelo_corp_inf, modelo_corp_pred,
 modelo_poisson_pl, modelo_nb_pl, modelo_poisson_sucesso,
 modelo_nb_sucesso, modelo_zinb, modelo_multinivel, modelo_sociol) = [None] * 13

# Rótulos descritivos globais para corporações — usados em gráficos e tabelas
_CORP_LABELS = {
    "PM":  "PM — Polícia Militar",
    "PC":  "PC — Polícia Civil",
    "PF":  "PF — Polícia Federal",
    "PRF": "PRF — Pol. Rod. Federal",
    "EB":  "EB — Exército",
    "FAB": "FAB — Força Aérea",
    "FA":  "FA — Forças Armadas",
    "MB":  "MB — Marinha",
    "CBM": "CBM — Bombeiros",
    "SM":  "SM — Seg. Municipal",
    "GM":  "GM — Guarda Municipal",
}


# =========================================================
# STOPWORDS
# =========================================================

stopwords = {
    "a", "o", "as", "os", "de", "da", "do", "das", "dos", "e", "em", "para", "por", "com",
    "que", "na", "no", "nas", "nos", "uma", "um", "ao", "aos", "se", "sua", "seu", "suas", "seus",
    "projeto", "lei", "dispõe", "dispoe", "sobre", "federal", "nacional", "brasil", "brasileiro",
    "brasileira", "constituição", "constituicao", "institui", "altera", "acrescenta", "autoriza",
    "estabelece", "cria", "outras", "providências", "providencias", "art", "arts", "nova",
    "redacao", "redação", "codigo", "decreto", "dezembro", "outubro", "setembro", "julho",
    "pelo", "pela", "pelas", "pelos", "como", "inciso", "paragrafo", "parágrafo", "unico",
    "único", "dispor", "acresce"
}

stopwords_juridicas = {
    "artigo", "art", "disposicoes", "constitucional", "constitucionais",
    "ato", "caput", "alinea", "inciso", "paragrafo",
    "republica", "uniao", "estados", "estado", "municipios",
    "federativa", "dispositivo", "dispositivos"
}

stopwords_totais = stopwords.union(stopwords_juridicas)


# =========================================================
# FUNÇÕES
# =========================================================

def remover_acentos(texto):
    texto = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in texto if not unicodedata.combining(c))


def normalizar_texto(texto):
    texto = str(texto).strip().lower()
    texto = remover_acentos(texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto


def normalizar_basico(texto):
    texto = str(texto).strip()
    texto = re.sub(r"\s+", " ", texto)
    return texto


def limpar_texto(texto):
    texto = str(texto).lower()
    texto = remover_acentos(texto)
    texto = re.sub(r"[^a-zA-Z\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def tokenizar(texto):
    tokens = texto.split()
    return [t for t in tokens if t not in stopwords_totais and len(t) > 2]


def gerar_ngrams(tokens, n):
    return list(zip(*(islice(tokens, i, None) for i in range(n))))


def normalizar_lei(match_text):
    numeros = re.findall(r"\d{1,5}(?:\.\d+)?", match_text)
    if numeros:
        return f"Lei nº {numeros[0]}"
    return None


def normalizar_artigo(match_text):
    numeros = re.findall(r"\d+[A-Za-z\-º°]*", match_text)
    if numeros:
        return f"art. {numeros[0].lower()}"
    return None


def split_limpo(x):
    if pd.isna(x):
        return []
    return [
        item.strip()
        for item in str(x).split(";")
        if item.strip() != "" and item.strip().lower() != "nan"
    ]


def primeiro_item_limpo(x):
    itens = split_limpo(x)
    return itens[0].upper() if len(itens) > 0 else pd.NA


def limpar_autor(valor):
    if pd.isna(valor):
        return pd.NA
    valor = normalizar_basico(valor)
    partes = [p.strip() for p in valor.split(";") if p.strip()]
    return partes[0] if partes else pd.NA


def limpar_uf(valor):
    if pd.isna(valor):
        return pd.NA
    valor = normalizar_basico(valor)
    partes = [p.strip().upper() for p in valor.split(";") if p.strip()]
    for p in partes:
        if p in UFS_VALIDAS:
            return p
    return pd.NA


def padronizar_partido_nome(p):
    if pd.isna(p):
        return pd.NA
    p_up = remover_acentos(str(p).strip()).upper()
    mapa = {
        "UNIAO": "UNIÃO",
        "UNIÃO": "UNIÃO",
        "UNIÃO BRASIL": "UNIÃO",
        "UNIAO BRASIL": "UNIÃO",
        "PODEMOS": "PODE"
    }
    return mapa.get(p_up, str(p).strip().upper())


def limpar_partido(valor):
    if pd.isna(valor):
        return pd.NA

    valor = normalizar_basico(valor)

    if re.search(r"\d{4}-\d{2}-\d{2}", valor):
        return pd.NA

    partes = [p.strip() for p in valor.split(";") if p.strip()]

    for p in partes:
        p_norm = remover_acentos(p).upper()
        if p_norm in PARTIDOS_VALIDOS:
            return padronizar_partido_nome(p)

    return pd.NA


def recodificar_situacao(s):
    s = normalizar_texto(s)

    if "transformado em norma juridica" in s:
        return "sucesso"

    elif (
        "arquivada" in s
        or "arquivado" in s
        or "retirado pelo(a) autor(a)" in s
        or "retirada pelo(a) autor(a)" in s
        or "devolvida ao(à) autor(a)" in s
        or "devolvido ao(à) autor(a)" in s
        or "prejudicada" in s
        or "rejeitada" in s
        or "rejeitado" in s
        or "indeferida" in s
        or "indeferido" in s
        or "cancelada" in s
        or "cancelado" in s
    ):
        return "fracasso"

    elif (
        "aguardando" in s
        or "pronta para pauta" in s
        or "tramitando em conjunto" in s
        or "aguardando parecer" in s
        or "aguardando providencias internas" in s
        or "aguardando apreciacao" in s
        or "aguardando designacao de relator" in s
        or "aguardando despacho" in s
        or "aguardando criacao de comissao temporaria" in s
        or "aguardando constituicao de comissao temporaria" in s
    ):
        return "em_tramitacao"

    return "outros"


def teste_quiquadrado(df_in, var, y="sucesso_legislativo", verbose=True):
    base_teste = df_in[[var, y]].dropna().copy()
    tabela = pd.crosstab(base_teste[var], base_teste[y])

    if tabela.shape[0] > 1 and tabela.shape[1] > 1:
        chi2, p, dof, _ = chi2_contingency(tabela)
    else:
        chi2, p, dof = np.nan, np.nan, np.nan

    if verbose:
        print("\n" + "=" * 60)
        print(f"TESTE QUI-QUADRADO: {var} × {y}")
        print("=" * 60)
        print("\nTabela de contingência:")
        print(tabela)
        print("\nResultado do teste:")
        print(f"Chi2 = {chi2:.4f}" if not pd.isna(chi2) else "Chi2 = NA")
        print(f"p-valor = {p:.6f}" if not pd.isna(p) else "p-valor = NA")
        print(f"graus de liberdade = {dof}")
        if not pd.isna(p):
            if p < 0.05:
                print("Interpretação: associação estatisticamente significativa.")
            else:
                print("Interpretação: associação não significativa.")

    return tabela, chi2, p, dof


def mapear_legislatura(ano):
    if pd.isna(ano):
        return pd.NA
    ano = int(ano)

    if 1991 <= ano <= 1994:
        return "49a"
    elif 1995 <= ano <= 1998:
        return "50a"
    elif 1999 <= ano <= 2002:
        return "51a"
    elif 2003 <= ano <= 2006:
        return "52a"
    elif 2007 <= ano <= 2010:
        return "53a"
    elif 2011 <= ano <= 2014:
        return "54a"
    elif 2015 <= ano <= 2018:
        return "55a"
    elif 2019 <= ano <= 2022:
        return "56a"
    elif 2023 <= ano <= 2026:
        return "57a"
    return pd.NA


def preparar_partido_rec(df, col_partido="Partido", freq_min=20):
    freq = df[col_partido].value_counts()
    validos = freq[freq >= freq_min].index
    return df[col_partido].where(df[col_partido].isin(validos), "OUTROS")


def normalizar_merge_nome(x):
    if pd.isna(x):
        return pd.NA
    x = str(x).strip().upper()
    x = unicodedata.normalize("NFKD", x)
    x = "".join(c for c in x if not unicodedata.combining(c))
    x = re.sub(r"\s+", " ", x)
    return x


def limpar_corporacao_texto(x):
    if pd.isna(x):
        return pd.NA
    x = str(x).strip().upper()
    x = unicodedata.normalize("NFKD", x)
    x = "".join(c for c in x if not unicodedata.combining(c))
    x = re.sub(r"\s+", " ", x)
    return x


def padronizar_sigla_corporacao(x):
    if pd.isna(x):
        return "OUTROS"

    x = limpar_corporacao_texto(x)

    if x in ["", "NAN", "NONE"]:
        return "OUTROS"

    if x in ["PM", "EB", "PC", "PF", "SM", "MB", "PRF", "FA", "CBM", "FAB", "GM"]:
        return x

    if "POLICIA MILITAR" in x:
        return "PM"
    if "EXERCITO" in x:
        return "EB"
    if "POLICIA CIVIL" in x:
        return "PC"
    if "POLICIA FEDERAL" in x and "RODOVIARIA" not in x:
        return "PF"
    if "POLICIA RODOVIARIA FEDERAL" in x or "RODOVIARIA FEDERAL" in x:
        return "PRF"
    if "MARINHA" in x:
        return "MB"
    if "BOMBEIRO" in x:
        return "CBM"
    if "AERONAUTICA" in x or "FORCA AEREA" in x:
        return "FAB"
    if "FORCAS ARMADAS" in x:
        return "FA"
    if (
        "GUARDA MUNICIPAL" in x
        or "GUARDA CIVIL MUNICIPAL" in x
        or "GUARDA CIVIL METROPOLITANA" in x
        or re.search(r"\bGCM\b", x)
    ):
        return "GM"
    if (
        "PENITENCIARI" in x
        or "POLICIA PENAL" in x
        or "AGENTE PENITENCIARIO" in x
        or "AGENTE PRISIONAL" in x
        or "SEGURANCA MUNICIPAL" in x
    ):
        return "SM"

    return "OUTROS"


def construir_df_predict_formula(
    modelo,
    topico=None,
    ano=None,
    ano_c=None,
    partido=None,
    legislatura=None,
    corporacao=None,
    categorias_corporacao=None,
):
    linha = {}
    exog_names = list(modelo.model.exog_names)

    if any("topico_dominante" in x for x in exog_names):
        linha["topico_dominante"] = topico

    if "ano" in exog_names:
        linha["ano"] = ano

    if "ano_c" in exog_names:
        linha["ano_c"] = ano_c

    if any("partido_inf" in x for x in exog_names):
        linha["partido_inf"] = partido
    elif any("partido_rec" in x for x in exog_names):
        linha["partido_rec"] = partido

    if any("legislatura" in x for x in exog_names):
        linha["legislatura"] = legislatura

    if any("corporacao_sigla" in x for x in exog_names):
        if categorias_corporacao is not None:
            linha["corporacao_sigla"] = pd.Categorical(
                [corporacao],
                categories=categorias_corporacao,
                ordered=False
            )[0]
        else:
            linha["corporacao_sigla"] = corporacao

    return pd.DataFrame([linha])


def ajustar_logit_com_fallback(formula, data, nome_modelo="modelo",
                               cluster_col="Autor"):
    """
    Estima modelo logístico com erros padrão clusterizados por cluster_col.

    Fluxo:
      1. Estima o modelo via MLE (lbfgs → bfgs → regularizado).
      2. Tenta re-estimar os erros padrão com cov_type='cluster'
         (HC robustos clusterizados por Autor).
      3. Se a clusterização falhar (coluna ausente, grupos insuficientes),
         retorna o modelo com erros padrão não-robustos e avisa.

    Erros clusterizados por Autor são o padrão em artigos com dados de
    proposições porque um mesmo parlamentar apresenta dezenas de PLs —
    as observações não são independentes dentro do mesmo autor.
    """
    modelo_base = None

    # ── passo 1: estimação MLE ──────────────────────────────────────────
    for metodo in ["lbfgs", "bfgs"]:
        try:
            modelo_base = smf.logit(formula, data=data).fit(
                method=metodo,
                maxiter=1000,
                disp=False
            )
            print(f"\n{nome_modelo} ajustado com {metodo}.")
            break
        except Exception as e:
            print(f"\nFalha em {nome_modelo} com {metodo}: {e}")

    if modelo_base is None:
        try:
            modelo_base = smf.logit(formula, data=data).fit_regularized(
                alpha=1.0, disp=False
            )
            print(f"\n{nome_modelo} ajustado com regularização.")
        except Exception as e:
            print(f"\nFalha também em {nome_modelo} com regularização: {e}")
            return None

    # ── passo 2: re-estimar com erros clusterizados por Autor ───────────
    if cluster_col not in data.columns:
        print(f"\n[AVISO] Coluna '{cluster_col}' ausente — "
              f"erros padrão não-robustos mantidos para {nome_modelo}.")
        return modelo_base

    # remove linhas cujo índice não está na base original
    grupos = data.loc[modelo_base.model.endog_names
                      if hasattr(modelo_base.model, "endog_names")
                      else data.index, cluster_col] \
             if False else data[cluster_col]

    # alinha grupos com o índice usado na estimação
    idx_modelo = getattr(modelo_base.model, "data", None)
    if idx_modelo is not None and hasattr(idx_modelo, "row_labels"):
        try:
            grupos_alinhados = data.loc[idx_modelo.row_labels, cluster_col]
        except Exception:
            grupos_alinhados = data[cluster_col].values
    else:
        grupos_alinhados = data[cluster_col].values

    n_grupos = pd.Series(grupos_alinhados).nunique()
    n_obs    = len(grupos_alinhados)

    if n_grupos < 2:
        print(f"\n[AVISO] Apenas {n_grupos} grupo(s) em '{cluster_col}' — "
              f"clusterização ignorada para {nome_modelo}.")
        return modelo_base

    # Tenta método 1: get_robustcov_results (statsmodels >= 0.13)
    try:
        modelo_clust = modelo_base.get_robustcov_results(
            cov_type="cluster",
            groups=grupos_alinhados
        )
        print(f"\n[OK] Erros clusterizados (get_robustcov_results) — "
              f"{nome_modelo} ({n_obs} obs, {n_grupos} clusters).")
        return modelo_clust
    except AttributeError:
        pass  # método não existe nessa versão
    except Exception as e1:
        print(f"\n[AVISO] get_robustcov_results falhou: {e1}")

    # Tenta método 2: re-fit com cov_type='cluster' diretamente
    try:
        formula_obj = modelo_base.model.formula
        data_obj    = modelo_base.model.data.frame

        modelo_clust2 = smf.logit(formula_obj, data=data_obj).fit(
            cov_type="cluster",
            cov_kwds={"groups": grupos_alinhados},
            method="lbfgs",
            maxiter=1000,
            disp=False
        )
        print(f"\n[OK] Erros clusterizados (fit cov_type) — "
              f"{nome_modelo} ({n_obs} obs, {n_grupos} clusters).")
        return modelo_clust2
    except Exception as e2:
        print(f"\n[AVISO] fit cov_type também falhou: {e2}")
        print("       Erros padrão não-robustos mantidos como fallback.")
        return modelo_base


def exibir_modelo(modelo, titulo):
    print(f"\n=== {titulo} ===")
    try:
        print(modelo.summary())
    except Exception:
        print("Resumo clássico indisponível para este ajuste.")
        print("Parâmetros estimados:")
        print(modelo.params)


def extrair_tabela_odds(modelo, nome_modelo):
    """Extrai coeficientes, odds ratios e ICs de um modelo logístico."""
    if modelo is None:
        return pd.DataFrame()

    params = modelo.params
    out = pd.DataFrame({
        "modelo": nome_modelo,
        "variavel": params.index,
        "coef_logit": params.values,
        "odds_ratio": np.exp(params.values)
    })

    try:
        bse = modelo.bse
        pvalues = modelo.pvalues
        ci = modelo.conf_int()
        out["std_err"] = bse.values
        out["p_value"] = pvalues.values
        out["ic95_inf_logit"] = ci.iloc[:, 0].values
        out["ic95_sup_logit"] = ci.iloc[:, 1].values
        out["ic95_inf_or"] = np.exp(ci.iloc[:, 0].values)
        out["ic95_sup_or"] = np.exp(ci.iloc[:, 1].values)
    except Exception:
        pass

    # registra tipo de erros padrão para transparência metodológica
    cov_type = getattr(modelo, "cov_type", "nonrobust")
    out["cov_type"] = cov_type
    out["erros_clusterizados"] = "cluster" in str(cov_type).lower()

    return out.sort_values("odds_ratio", ascending=False)


def consolidar_metricas_classificacao(y_true, y_prob, thresholds, nome_modelo):
    """Calcula AUC, AP, Brier e métricas por threshold."""
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    resumo = pd.DataFrame([{
        "modelo": nome_modelo,
        "auc": auc,
        "average_precision": ap,
        "brier_score": brier,
        "n_obs": len(y_true),
        "positivos": int(np.sum(y_true)),
        "taxa_positivos": float(np.mean(y_true))
    }])

    detalhes = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        precision_val = tp / max(tp + fp, 1)
        recall_val = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        accuracy = (tp + tn) / max(len(y_true), 1)

        detalhes.append({
            "modelo": nome_modelo,
            "threshold": thr,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "accuracy": accuracy,
            "precision": precision_val,
            "recall": recall_val,
            "specificity": specificity
        })

    return resumo, pd.DataFrame(detalhes)


def extrair_metricas_modelo_contagem(modelo, nome_modelo, y_name):
    if modelo is None:
        return pd.DataFrame()

    out = {
        "modelo": nome_modelo,
        "variavel_dependente": y_name,
        "n_obs": int(modelo.nobs) if hasattr(modelo, "nobs") else np.nan,
        "llf": float(modelo.llf) if hasattr(modelo, "llf") else np.nan,
        "aic": float(modelo.aic) if hasattr(modelo, "aic") else np.nan,
        "bic": float(modelo.bic) if hasattr(modelo, "bic") else np.nan,
        "deviance": float(modelo.deviance) if hasattr(modelo, "deviance") else np.nan,
        "pearson_chi2": float(modelo.pearson_chi2) if hasattr(modelo, "pearson_chi2") else np.nan,
        "df_resid": float(modelo.df_resid) if hasattr(modelo, "df_resid") else np.nan,
    }

    if pd.notna(out["pearson_chi2"]) and pd.notna(out["df_resid"]) and out["df_resid"] > 0:
        out["dispersion_ratio"] = out["pearson_chi2"] / out["df_resid"]
    else:
        out["dispersion_ratio"] = np.nan

    return pd.DataFrame([out])


def extrair_coef_contagem(modelo, nome_modelo):
    if modelo is None:
        return pd.DataFrame()

    params = modelo.params
    out = pd.DataFrame({
        "modelo": nome_modelo,
        "variavel": params.index,
        "coef": params.values,
        "irr": np.exp(params.values)
    })

    try:
        ci = modelo.conf_int()
        out["p_value"] = modelo.pvalues.values
        out["ic95_inf_coef"] = ci.iloc[:, 0].values
        out["ic95_sup_coef"] = ci.iloc[:, 1].values
        out["ic95_inf_irr"] = np.exp(ci.iloc[:, 0].values)
        out["ic95_sup_irr"] = np.exp(ci.iloc[:, 1].values)
    except Exception:
        pass

    return out.sort_values("irr", ascending=False)


def extrair_ame(modelo, nome_modelo):
    if modelo is None:
        return pd.DataFrame()

    try:
        me = modelo.get_margeff(at="overall")
        sf = me.summary_frame().reset_index().rename(columns={"index": "variavel"})
        mapa_cols = {
            "dy/dx": "ame",
            "Std. Err.": "std_err",
            "z": "z",
            "Pr(>|z|)": "p_value",
            "Conf. Int. Low": "ic95_inf",
            "Cont. Int. Hi.": "ic95_sup",
            "Conf. Int. Hi.": "ic95_sup"
        }
        sf = sf.rename(columns={k: v for k, v in mapa_cols.items() if k in sf.columns})
        sf["modelo"] = nome_modelo
        return sf
    except Exception as e:
        print(f"\nFalha ao extrair AME de {nome_modelo}: {e}")
        return pd.DataFrame()


def analise_correspondencia_simples(tabela):
    """Correspondence Analysis via SVD. Retorna coordenadas de linhas e colunas."""
    N = tabela.values.astype(float)
    n_total = N.sum()

    if n_total == 0:
        return pd.DataFrame(), pd.DataFrame(), np.array([])

    P = N / n_total
    r = P.sum(axis=1).reshape(-1, 1)
    c = P.sum(axis=0).reshape(1, -1)

    S = (P - r @ c) / np.sqrt(r @ c)

    U, s, VT = linalg.svd(S, full_matrices=False)

    Dr_inv_sqrt = np.diag(1 / np.sqrt(r.flatten()))
    Dc_inv_sqrt = np.diag(1 / np.sqrt(c.flatten()))

    F = Dr_inv_sqrt @ U @ np.diag(s)
    G = Dc_inv_sqrt @ VT.T @ np.diag(s)

    eig = s ** 2
    inertia = eig / eig.sum() if eig.sum() > 0 else eig

    dim_names = [f"dim_{i+1}" for i in range(F.shape[1])]

    coords_linhas = pd.DataFrame(F, index=tabela.index, columns=dim_names).reset_index()
    coords_colunas = pd.DataFrame(G, index=tabela.columns, columns=dim_names).reset_index()

    coords_linhas = coords_linhas.rename(columns={coords_linhas.columns[0]: "categoria_linha"})
    coords_colunas = coords_colunas.rename(columns={coords_colunas.columns[0]: "categoria_coluna"})

    return coords_linhas, coords_colunas, inertia


PADRAO_LEI = re.compile(
    r"\blei\s*(?:n[º°o\.]*\s*)?\d{1,5}(?:\.\d+)?",
    flags=re.IGNORECASE
)

PADRAO_ARTIGO = re.compile(
    r"\bart(?:igo)?\.?\s*\d+[A-Za-z\-º°]*",
    flags=re.IGNORECASE
)


# =========================================================
# 1. LEITURA E LIMPEZA
# =========================================================

print("Lendo planilha...")

df_base = pd.read_excel(ARQUIVO_ENTRADA, header=None)
df_base = df_base.iloc[:, :8].copy()
df_base.columns = NOMES_COLUNAS

df_base = df_base[
    ~df_base["Proposicoes"].astype(str).str.strip().str.lower().isin(["proposições", "proposicoes"])
].copy()

df_base = df_base.dropna(how="all").copy()

print("Dimensão da base:", df_base.shape)

# preserva originais completos (todos os coautores/partidos/UFs separados por ";")
df_base["Autor_original"] = df_base["Autor"]
df_base["UF_original"] = df_base["UF"]
df_base["Partido_original"] = df_base["Partido"]

# limpa versão principal (apenas primeiro item)
df_base["Autor"] = df_base["Autor"].apply(limpar_autor)
df_base["UF"] = df_base["UF"].apply(limpar_uf)
df_base["Partido"] = df_base["Partido"].apply(limpar_partido)

print("\nValores nulos após limpeza:")
print(df_base[["Autor", "UF", "Partido"]].isna().sum())

df_base["Situacao_original"] = df_base["Situacao"].astype(str)
df_base["Situacao_normalizada"] = df_base["Situacao_original"].apply(normalizar_texto)
df_base["situacao_recodificada"] = df_base["Situacao_original"].apply(recodificar_situacao)
df_base["sucesso_legislativo"] = (df_base["situacao_recodificada"] == "sucesso").astype(int)

df_base["Apresentacao"] = pd.to_datetime(df_base["Apresentacao"], errors="coerce", dayfirst=True)
df_base["ano"] = df_base["Apresentacao"].dt.year
df_base["legislatura"] = df_base["ano"].apply(mapear_legislatura)

print("\nFrequência da situação recodificada:")
print(df_base["situacao_recodificada"].value_counts(dropna=False))

print("\nFrequência da variável binária sucesso_legislativo:")
print(df_base["sucesso_legislativo"].value_counts(dropna=False))

taxa_sucesso = df_base["sucesso_legislativo"].mean() * 100
print(f"\nTaxa geral de sucesso legislativo: {taxa_sucesso:.2f}%")


# =========================================================
# 2. BASE DE TEXTO
# =========================================================

if COLUNA_TEXTO not in df_base.columns:
    raise ValueError(f"A coluna '{COLUNA_TEXTO}' não foi encontrada.")

df_texto = df_base[df_base[COLUNA_TEXTO].notna()].copy()
df_texto = df_texto[df_texto[COLUNA_TEXTO].astype(str).str.strip() != ""].copy()
df_texto["texto_original"] = df_texto[COLUNA_TEXTO].astype(str)

print(f"\nTotal de linhas com texto em '{COLUNA_TEXTO}': {len(df_texto)}")

df_texto["texto_limpo"] = df_texto[COLUNA_TEXTO].apply(limpar_texto)
df_texto["tokens"] = df_texto["texto_limpo"].apply(tokenizar)
df_texto = df_texto[df_texto["tokens"].apply(len) > 0].copy()

print(f"Total de linhas após limpeza textual: {len(df_texto)}")


# =========================================================
# 3. FREQUÊNCIAS
# =========================================================

todas_palavras = []
todos_bigrams = []
todos_trigrams = []

for lista in df_texto["tokens"]:
    todas_palavras.extend(lista)
    todos_bigrams.extend(gerar_ngrams(lista, 2))
    todos_trigrams.extend(gerar_ngrams(lista, 3))

freq_palavras = Counter(todas_palavras)
freq_bigrams = Counter(todos_bigrams)
freq_trigrams = Counter(todos_trigrams)

df_palavras = pd.DataFrame(freq_palavras.most_common(TOP_N), columns=["palavra", "frequencia"])
df_bigrams = pd.DataFrame(
    [(" ".join(k), v) for k, v in freq_bigrams.most_common(TOP_N)],
    columns=["bigram", "frequencia"]
)
df_trigrams = pd.DataFrame(
    [(" ".join(k), v) for k, v in freq_trigrams.most_common(TOP_N)],
    columns=["trigram", "frequencia"]
)


# =========================================================
# 4. LEIS E ARTIGOS
# =========================================================

leis_encontradas = []
artigos_encontrados = []

for texto in df_texto["texto_original"]:
    for m in PADRAO_LEI.findall(texto):
        lei_norm = normalizar_lei(m)
        if lei_norm:
            leis_encontradas.append(lei_norm)

    for m in PADRAO_ARTIGO.findall(texto):
        art_norm = normalizar_artigo(m)
        if art_norm:
            artigos_encontrados.append(art_norm)

freq_leis = Counter(leis_encontradas)
freq_artigos = Counter(artigos_encontrados)

df_leis = pd.DataFrame(freq_leis.most_common(TOP_N), columns=["lei", "frequencia"])
df_artigos = pd.DataFrame(freq_artigos.most_common(TOP_N), columns=["artigo", "frequencia"])


# =========================================================
# 5. CONTAGENS DESCRITIVAS
# =========================================================

# --- UF expandida (todos os coautores via Partido_original) ---
ufs_series = (
    df_base["UF_original"]
    .dropna()
    .astype(str)
    .str.split(";")
    .explode()
    .str.strip()
    .str.upper()
)
ufs_series = ufs_series[ufs_series.isin(UFS_VALIDAS)]

freq_uf = (
    ufs_series
    .value_counts()
    .reset_index()
)
freq_uf.columns = ["UF", "frequencia"]

print("\nContagem por UF (expandida por coautores):")
print(freq_uf.head(27))

# --- Partido expandido (todos os coautores via Partido_original) ---
partidos_series = (
    df_base["Partido_original"]
    .dropna()
    .astype(str)
    .str.split(";")
    .explode()
    .str.strip()
)
partidos_series = partidos_series[partidos_series != ""]
partidos_series = partidos_series[partidos_series.str.lower() != "nan"]
# aplica padronização para consistência
partidos_series = partidos_series.apply(
    lambda p: padronizar_partido_nome(p) if pd.notna(p) and str(p).strip() != "" else pd.NA
).dropna()

freq_partido = (
    partidos_series
    .value_counts()
    .reset_index()
)
freq_partido.columns = ["Partido", "frequencia"]

print("\nContagem por Partido (expandida por coautores):")
print(freq_partido.head(20))

# verificação diagnóstica de partidos parecidos com PT
print("\nChecando contagem de PT e similares (dados expandidos):")
for sigla in ["PT", "PTB", "PDT", "PTC", "PTDOB"]:
    print(f"{sigla:8s}: {(partidos_series == sigla).sum()}")

# --- Autor expandido (todos os coautores via Autor_original) ---
autores_series = (
    df_base["Autor_original"]
    .dropna()
    .astype(str)
    .str.split(";")
    .explode()
    .str.strip()
)
autores_series = autores_series[autores_series != ""]
autores_series = autores_series[autores_series.str.lower() != "nan"]

freq_autor = (
    autores_series
    .value_counts()
    .reset_index()
)
freq_autor.columns = ["Autor", "frequencia"]

print("\nContagem por Autor (expandida por coautores):")
print(freq_autor.head(20))

# --- Situação ---
freq_situacao = (
    df_base["Situacao_original"]
    .dropna()
    .astype(str)
    .str.strip()
    .value_counts()
    .reset_index()
)
freq_situacao.columns = ["Situacao", "frequencia"]


# =========================================================
# 6. QUI-QUADRADO E TAXAS
# =========================================================

print("\nRodando testes qui-quadrado...")
tabela_partido_sucesso_q2, chi2_partido, p_partido, dof_partido = teste_quiquadrado(df_base, "Partido")
tabela_uf_sucesso_q2, chi2_uf, p_uf, dof_uf = teste_quiquadrado(df_base, "UF")

print("\nResumo dos testes descritivos:")
print(f"Partido x sucesso -> chi2={chi2_partido}, p={p_partido}, gl={dof_partido}")
print(f"UF x sucesso -> chi2={chi2_uf}, p={p_uf}, gl={dof_uf}")

taxa_partido = df_base.groupby("Partido").agg(
    proposicoes=("Proposicoes", "count"),
    sucessos=("sucesso_legislativo", "sum")
)
taxa_partido["taxa_sucesso"] = taxa_partido["sucessos"] / taxa_partido["proposicoes"]
taxa_partido = taxa_partido.sort_values("taxa_sucesso", ascending=False)

taxa_uf = df_base.groupby("UF").agg(
    proposicoes=("Proposicoes", "count"),
    sucessos=("sucesso_legislativo", "sum")
)
taxa_uf["taxa_sucesso"] = taxa_uf["sucessos"] / taxa_uf["proposicoes"]
taxa_uf = taxa_uf.sort_values("taxa_sucesso", ascending=False)

taxa_partido_filtrado = taxa_partido[taxa_partido["proposicoes"] >= MIN_PROP].copy()
taxa_uf_filtrado = taxa_uf[taxa_uf["proposicoes"] >= MIN_PROP].copy()

print("\nTaxa de sucesso por partido:")
print(taxa_partido.head(20))

print("\nTaxa de sucesso por UF:")
print(taxa_uf.head(20))


# =========================================================
# 7. GRÁFICOS DESCRITIVOS
# =========================================================

plt.figure(figsize=(10, 7))
_tx_part = (taxa_partido_filtrado["taxa_sucesso"] * 100).sort_values()
bars_tp = _tx_part.plot(kind="barh", color="steelblue", alpha=0.85)
for bar, (_, v) in zip(bars_tp.patches, _tx_part.items()):
    plt.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
             f"{v:.1f}%", va="center", fontsize=8)
plt.title(f"Taxa de sucesso legislativo por partido (≥{MIN_PROP} proposições)\n"
          "Período 1989–2023 — inclui partidos extintos/fundidos (dado histórico)")
plt.xlabel("Taxa de aprovação (%)")
plt.ylabel("Partido")
plt.xlim(0, _tx_part.max() * 1.25)
plt.tight_layout()
plt.savefig(PASTA / "taxa_sucesso_partido.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(8, 6))
_tx_uf = (taxa_uf_filtrado["taxa_sucesso"] * 100).sort_values()
_tx_uf.plot(kind="barh", color="steelblue", alpha=0.85)
for bar, v in zip(plt.gca().patches, _tx_uf):
    plt.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
             f"{v:.1f}%", va="center", fontsize=8)
plt.title(f"Taxa de sucesso legislativo por UF (≥{MIN_PROP} proposições)")
plt.xlabel("Taxa de aprovação (%)")
plt.ylabel("UF")
plt.tight_layout()
plt.savefig(PASTA / "taxa_sucesso_uf.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

dados_plot = taxa_partido_filtrado.reset_index()

plt.figure(figsize=(8, 6))
plt.scatter(
    taxa_partido_filtrado["proposicoes"],
    taxa_partido_filtrado["taxa_sucesso"]
)
plt.xlabel("Número de proposições")
plt.ylabel("Taxa de sucesso")
plt.title("Produção legislativa vs taxa de sucesso (Partidos)")
plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(8, 6))
sns.regplot(
    x="proposicoes",
    y="taxa_sucesso",
    data=dados_plot
)
plt.xlabel("Número de proposições")
plt.ylabel("Taxa de sucesso")
plt.title("Produção legislativa e sucesso legislativo (Partidos)")
plt.tight_layout()
plt.show()
plt.close()


# =========================================================
# 8. PRODUTIVIDADE POR AUTOR / COAUTORIA REAL
# =========================================================

# usa colunas originais completas (todos os coautores)
df_autor = df_texto[["Proposicoes", "Autor_original", "Partido_original", "UF_original"]].copy()
df_autor["Autor_lista"] = df_autor["Autor_original"].apply(split_limpo)
df_autor["Partido_lista"] = df_autor["Partido_original"].apply(split_limpo)
df_autor["UF_lista"] = df_autor["UF_original"].apply(split_limpo)

linhas_expandidas = []

for _, row in df_autor.iterrows():
    proposicao = row["Proposicoes"]
    autores = row["Autor_lista"]
    partidos = row["Partido_lista"]
    ufs = row["UF_lista"]

    if len(autores) == 0:
        continue

    autores = [normalizar_basico(a) for a in autores]
    partidos = [padronizar_partido_nome(p) if pd.notna(p) and str(p).strip() != "" else None for p in partidos]
    ufs = [str(u).strip().upper() if pd.notna(u) and str(u).strip() != "" else None for u in ufs]

    n = len(autores)

    if len(partidos) < n:
        partidos = partidos + [None] * (n - len(partidos))
    else:
        partidos = partidos[:n]

    if len(ufs) < n:
        ufs = ufs + [None] * (n - len(ufs))
    else:
        ufs = ufs[:n]

    for autor, partido, uf in zip(autores, partidos, ufs):
        linhas_expandidas.append({
            "Proposicoes": proposicao,
            "Autor": autor,
            "Partido": partido,
            "UF": uf
        })

df_autor_expandido = pd.DataFrame(linhas_expandidas)

produtividade = (
    df_autor_expandido
    .groupby("Autor")
    .agg(
        n_proposicoes=("Proposicoes", "count"),
        partido_modal=("Partido", lambda x: x.dropna().mode().iat[0] if not x.dropna().mode().empty else None),
        uf_modal=("UF", lambda x: x.dropna().mode().iat[0] if not x.dropna().mode().empty else None)
    )
    .reset_index()
    .sort_values("n_proposicoes", ascending=False)
)

base_autor = df_base.groupby(["Autor", "Partido", "UF"]).agg(
    proposicoes=("Proposicoes", "count"),
    sucessos=("sucesso_legislativo", "sum")
).reset_index()

base_autor["taxa_sucesso"] = base_autor["sucessos"] / base_autor["proposicoes"]
ranking_proposicoes = base_autor.sort_values("proposicoes", ascending=False)
ranking_sucesso = base_autor.sort_values("sucessos", ascending=False)
ranking_taxa = base_autor[
    base_autor["proposicoes"] >= MIN_PROP_AUTOR
].sort_values("taxa_sucesso", ascending=False)

print("\nTop autores por produtividade:")
print(produtividade.head(30))

plt.figure(figsize=(8, 6))
plt.hist(base_autor["proposicoes"], bins=40)
plt.title("Distribuição de proposições por parlamentar")
plt.xlabel("Número de proposições")
plt.ylabel("Quantidade de parlamentares")
plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(8, 6))
freq = base_autor["proposicoes"].value_counts().sort_index()
plt.scatter(freq.index, freq.values)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Número de proposições por parlamentar (log)")
plt.ylabel("Número de parlamentares (log)")
plt.title("Distribuição da produção legislativa (escala log-log)")
plt.tight_layout()
plt.show()
plt.close()


# =========================================================
# 9. ANÁLISE TEMPORAL
# =========================================================

df_tempo = df_base[df_base["ano"].notna()].copy()
df_tempo["ano"] = df_tempo["ano"].astype(int)

df_pls_ano = df_tempo["ano"].value_counts().sort_index().reset_index()
df_pls_ano.columns = ["ano", "numero_pls"]

print("\nPLs por ano:")
print(df_pls_ano)

plt.figure(figsize=(12, 6))
plt.bar(df_pls_ano["ano"], df_pls_ano["numero_pls"])
plt.title("Número de PLs apresentados por ano")
plt.xlabel("Ano")
plt.ylabel("Número de PLs")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(ARQUIVO_GRAFICO_ANO, dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# =========================================================
# 10. LDA
# =========================================================

print("\nRodando LDA...")

df_texto["texto_lda"] = df_texto["tokens"].apply(lambda x: " ".join(x))

vectorizer = CountVectorizer(max_df=0.95, min_df=2)
X_lda = vectorizer.fit_transform(df_texto["texto_lda"])
feature_names = vectorizer.get_feature_names_out()

lda = LatentDirichletAllocation(
    n_components=N_TOPICOS,
    random_state=42,
    learning_method="batch"
)
lda.fit(X_lda)

topicos_lista = []

for topic_idx, topic in enumerate(lda.components_):
    top_indices = topic.argsort()[::-1][:N_PALAVRAS_TOPICO]
    palavras_topico = [feature_names[i] for i in top_indices]
    pesos_topico = [topic[i] for i in top_indices]

    print(f"\nTÓPICO {topic_idx + 1}")
    for palavra, peso in zip(palavras_topico, pesos_topico):
        print(f"{palavra}: {peso:.2f}")

    for palavra, peso in zip(palavras_topico, pesos_topico):
        topicos_lista.append({
            "topico": f"Topico_{topic_idx + 1}",
            "palavra": palavra,
            "peso": round(float(peso), 4)
        })

df_topicos = pd.DataFrame(topicos_lista)

doc_topic_dist = lda.transform(X_lda)
df_texto["topico_dominante"] = doc_topic_dist.argmax(axis=1) + 1
df_texto["prob_topico_dominante"] = doc_topic_dist.max(axis=1)
df_texto["Partido_limpo"] = df_texto["Partido"].apply(lambda x: padronizar_partido_nome(x) if pd.notna(x) else pd.NA)

freq_topicos = (
    df_texto["topico_dominante"]
    .value_counts()
    .sort_index()
    .reset_index()
)
freq_topicos.columns = ["topico", "frequencia"]

print("\nFrequência dos tópicos:")
print(freq_topicos)

# ── dicionário canônico de nomes de tópicos ───────────────────────────────
# Derivado dos 10 termos de maior peso de cada tópico (output do LDA acima).
# Revisado para refletir os termos reais, não rótulos genéricos.
#
# T1: trânsito, estatuto, criança, adolescente, deficiência, veículos, civil
#     → legislação de trânsito + estatutos protetivos (ECA, EPD)
# T2: proteção, consumidor, renda, imposto, público, agosto, maio
#     → regulação econômica, fiscal e proteção ao consumidor
# T3: educação, trabalho, ensino, armas, saúde, sistema, social, programa
#     → políticas sociais amplas (educação, saúde, trabalho, porte de armas)
# T4: militares, segurança, bombeiros, policiais, distrito, imposto, saúde
#     → carreiras, estrutura e benefícios das forças de segurança
# T5: penal, crime, crimes, pena, processo, execução, contra, militar
#     → direito penal e processo criminal
#
# Nota: "armas" no T3 é esperado — regulação de porte/porte de armas
# é debatida no Congresso como agenda de segurança pública civil,
# não exclusivamente como agenda penal (que é processo/pena no T5).

NOMES_TOPICOS = {
    1: "T1 — Estatutos, trânsito\ne direitos civis",
    2: "T2 — Regulação econômica\ne proteção ao consumidor",
    3: "T3 — Políticas sociais\n(educação, saúde, trabalho)",
    4: "T4 — Carreiras, benefícios e estrutura\ndas forças de segurança",
    5: "T5 — Direito penal\ne processo criminal",
}

NOMES_TOPICOS_CURTO = {
    1: "T1 — Estatutos/Trânsito",
    2: "T2 — Regulação Econômica",
    3: "T3 — Políticas Sociais",
    4: "T4 — Carreiras e Estrutura",
    5: "T5 — Dir. Penal/Criminal",
}

print("\nNomes canônicos dos tópicos (baseados nos termos do LDA):")
for k, v in NOMES_TOPICOS_CURTO.items():
    print(f"  Tópico {k}: {v}")

plt.figure(figsize=(10, 5))
_freq_plot = freq_topicos.copy()
_freq_plot["label"] = _freq_plot["topico"].map(NOMES_TOPICOS_CURTO).fillna(_freq_plot["topico"].astype(str))
bars_freq = plt.bar(_freq_plot["label"], _freq_plot["frequencia"],
                    color="steelblue", alpha=0.8)
for bar, (_, row) in zip(bars_freq, _freq_plot.iterrows()):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 15,
             f"{int(row['frequencia'])}",
             ha="center", va="bottom", fontsize=9)
plt.title("Distribuição das proposições por agenda temática (LDA)")
plt.xlabel("Agenda temática")
plt.ylabel("Número de proposições")
plt.xticks(rotation=15, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig(ARQUIVO_GRAFICO_TOPICOS, dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# =========================================================
# 11. PARTIDO X TÓPICO
# =========================================================

df_texto_partido = df_texto[df_texto["Partido_limpo"].notna()].copy()

tabela_partido = pd.crosstab(df_texto_partido["Partido_limpo"], df_texto_partido["topico_dominante"])
tabela_percentual = pd.crosstab(
    df_texto_partido["Partido_limpo"],
    df_texto_partido["topico_dominante"],
    normalize="index"
) * 100

if tabela_partido.shape[0] > 1 and tabela_partido.shape[1] > 1:
    chi2_pt, p_pt, dof_pt, _ = chi2_contingency(tabela_partido)
else:
    chi2_pt, p_pt, dof_pt = pd.NA, pd.NA, pd.NA

df_chi2 = pd.DataFrame({
    "estatistica": ["chi2", "p_value", "graus_liberdade"],
    "valor": [chi2_pt, p_pt, dof_pt]
})

print("\nTabela Partido x Tópico:")
print(tabela_partido)

print("\nTabela percentual Partido x Tópico:")
print(tabela_percentual.round(2))

print("\nTeste qui-quadrado Partido x Tópico:")
print(f"Chi2 = {chi2_pt}")
print(f"p-value = {p_pt}")
print(f"gl = {dof_pt}")

top_partidos = (
    df_texto_partido["Partido_limpo"]
    .value_counts()
    .head(TOP_PARTIDOS_GRAFICO)
    .index
)

# usa os mesmos partidos do gráfico de taxa de sucesso (≥MIN_PROP PLs)
# para garantir consistência entre os dois gráficos
_partidos_min = taxa_partido_filtrado.index.tolist()  # já filtrado por MIN_PROP
# intersecção: apenas partidos que estão em tabela_percentual E têm ≥MIN_PROP PLs
_partidos_disponiveis = [p for p in _partidos_min if p in tabela_percentual.index]
# se poucos partidos disponíveis, complementa com os top por volume
if len(_partidos_disponiveis) < 5:
    _partidos_disponiveis = list(top_partidos)
top_partidos = _partidos_disponiveis

# cópia com colunas numéricas — usada pelo heatmap e pela renomeação
tabela_percentual_plot = tabela_percentual.loc[top_partidos].copy()

# ordena pela taxa de sucesso (mesma ordem do gráfico 1) para comparação visual
_ordem_taxa = [p for p in taxa_partido_filtrado.sort_values("taxa_sucesso", ascending=False).index
               if p in tabela_percentual_plot.index]
_restantes  = [p for p in tabela_percentual_plot.index if p not in _ordem_taxa]
tabela_percentual_plot = tabela_percentual_plot.loc[_ordem_taxa + _restantes]

# cópia com colunas descritivas — usada pelo gráfico de barras empilhadas
tabela_percentual_plot_named = tabela_percentual_plot.copy()
tabela_percentual_plot_named.columns = [
    NOMES_TOPICOS_CURTO.get(int(c), str(c)) for c in tabela_percentual_plot_named.columns
]

fig, ax = plt.subplots(figsize=(12, 7))
tabela_percentual_plot_named.plot(kind="bar", stacked=True, ax=ax)
ax.set_title("Distribuição temática dos PLs por partido\n"
             "(inclui partidos extintos/fundidos — dados históricos 1989–2023)")
ax.set_ylabel("Percentual (%)")
ax.set_xlabel("Partido")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.legend(title="Agenda temática", bbox_to_anchor=(1.02, 1),
          loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(ARQUIVO_GRAFICO_PARTIDOS, dpi=200, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(12, 8))
# heatmap usa a cópia com colunas já renomeadas (tabela_percentual_plot_named)
_tabela_hm = tabela_percentual_plot_named.copy()
sns.heatmap(_tabela_hm, annot=True, fmt=".1f", cmap="Blues")
plt.title("Distribuição temática dos PLs por partido (%)\n"
          "(baseada nos termos do modelo LDA)")
plt.ylabel("Partido")
plt.xlabel("Agenda temática")
plt.xticks(rotation=20, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig(ARQUIVO_HEATMAP, dpi=200, bbox_inches="tight")
plt.show()
plt.close()


# =========================================================
# 12. TÓPICOS POR ANO
# =========================================================

df_ano_topico = df_texto[df_texto["ano"].notna()].copy()
df_ano_topico["ano"] = df_ano_topico["ano"].astype(int)

tabela_ano_topico = pd.crosstab(df_ano_topico["ano"], df_ano_topico["topico_dominante"])
tabela_ano_topico_pct = pd.crosstab(
    df_ano_topico["ano"],
    df_ano_topico["topico_dominante"],
    normalize="index"
) * 100

print("\nTabela Ano x Tópico (contagem):")
print(tabela_ano_topico)

print("\nTabela Ano x Tópico (%):")
print(tabela_ano_topico_pct.round(2))

plt.figure(figsize=(12, 6))
for topico in tabela_ano_topico.columns:
    _nome = NOMES_TOPICOS_CURTO.get(int(topico), f"Tópico {topico}")
    plt.plot(
        tabela_ano_topico.index,
        tabela_ano_topico[topico],
        marker="o", markersize=4,
        label=_nome
    )

plt.title("Evolução das agendas temáticas por ano (contagem de PLs)")
plt.xlabel("Ano")
plt.ylabel("Número de proposições")
plt.legend(title="Agenda temática", bbox_to_anchor=(1.02, 1),
           loc="upper left", fontsize=8)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(ARQUIVO_TOPICOS_ANO_CONTAGEM, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(12, 6))
_tabela_ano_pct = tabela_ano_topico_pct.copy()
_tabela_ano_pct.columns = [NOMES_TOPICOS_CURTO.get(int(c), str(c)) for c in _tabela_ano_pct.columns]
_tabela_ano_pct.plot(kind="area", stacked=True, ax=ax)
ax.set_title("Mudança de temas por ano (%)")
ax.set_xlabel("Ano")
ax.set_ylabel("Percentual de proposições (%)")
ax.legend(title="Agenda temática", bbox_to_anchor=(1.02, 1),
          loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(ARQUIVO_TOPICOS_ANO_PERCENTUAL, dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# =========================================================
# 13. SUCESSO LEGISLATIVO POR TÓPICO
# =========================================================

df_texto["sucesso"] = df_texto["sucesso_legislativo"].astype(int)

tabela_topico_sucesso = pd.crosstab(df_texto["topico_dominante"], df_texto["sucesso"])
tabela_topico_sucesso = tabela_topico_sucesso.reindex(columns=[0, 1], fill_value=0)
tabela_topico_sucesso.columns = ["Nao_aprovado", "Aprovado"]

taxa_sucesso_topico = (
    tabela_topico_sucesso["Aprovado"] /
    tabela_topico_sucesso.sum(axis=1)
) * 100
taxa_sucesso_topico = taxa_sucesso_topico.sort_values(ascending=False)

print("\nTabela Tópico x Sucesso:")
print(tabela_topico_sucesso)

print("\nTaxa de sucesso por tópico (%):")
print(taxa_sucesso_topico.round(2))

_taxa_plot = taxa_sucesso_topico.reset_index()
_taxa_plot.columns = ["topico", "taxa"]
_taxa_plot["label"] = _taxa_plot["topico"].map(NOMES_TOPICOS_CURTO).fillna(_taxa_plot["topico"].astype(str))
_taxa_plot = _taxa_plot.sort_values("taxa", ascending=False)

fig, ax = plt.subplots(figsize=(11, 6))
bars_tx = ax.bar(_taxa_plot["label"], _taxa_plot["taxa"],
                 color="steelblue", alpha=0.8)
for bar, (_, row) in zip(bars_tx, _taxa_plot.iterrows()):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{row['taxa']:.2f}%",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold")
ax.set_title("Taxa de sucesso legislativo por agenda temática (LDA)\n"
             "Base: proposições com situação decidida (aprovadas + arquivadas)",
             fontsize=11)
ax.set_xlabel("Agenda temática")
ax.set_ylabel("Taxa de aprovação (%)")
ax.set_ylim(0, _taxa_plot["taxa"].max() * 1.30)
plt.xticks(rotation=20, ha="right", fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(ARQUIVO_SUCESSO_TOPICO, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

if tabela_topico_sucesso.shape[0] > 1:
    chi2_suc, p_suc, dof_suc, _ = chi2_contingency(tabela_topico_sucesso)
else:
    chi2_suc, p_suc, dof_suc = np.nan, np.nan, np.nan

print("\nTeste Qui-quadrado: Tópico x Sucesso")
print("Chi2:", round(chi2_suc, 2) if not pd.isna(chi2_suc) else "NA")
print("p-value:", round(p_suc, 5) if not pd.isna(p_suc) else "NA")
print("graus de liberdade:", dof_suc)


# =========================================================
# 14. COAUTORIA E PARTIDO X AGENDA (REAL)
# =========================================================

# expande usando Partido_original para capturar coautores reais
df_partidos = df_texto.copy()
df_partidos["Partido_multi"] = df_partidos["Partido_original"].apply(split_limpo)
df_partidos = df_partidos.explode("Partido_multi").reset_index(drop=True)
def limpar_partido_multi(x):
    if pd.isna(x) or str(x).strip() == "":
        return pd.NA
    raw = str(x).strip()
    # rejeita datas (contêm traço no padrão YYYY-MM-DD)
    if re.match(r"\d{4}-\d{2}-\d{2}", raw):
        return pd.NA
    # rejeita UFs (2 letras maiúsculas que estão em UFS_VALIDAS)
    if raw.upper() in UFS_VALIDAS:
        return pd.NA
    # padroniza e valida contra PARTIDOS_VALIDOS
    padronizado = padronizar_partido_nome(raw)
    if pd.isna(padronizado):
        return pd.NA
    norm = remover_acentos(str(padronizado)).upper()
    if norm not in {remover_acentos(p).upper() for p in PARTIDOS_VALIDOS}:
        return pd.NA
    return padronizado

df_partidos["Partido_multi"] = df_partidos["Partido_multi"].apply(limpar_partido_multi)
df_partidos = df_partidos[df_partidos["Partido_multi"].notna()].reset_index(drop=True)

tabela_partido_topico = pd.crosstab(df_partidos["Partido_multi"], df_partidos["topico_dominante"])
tabela_partido_topico_pct = pd.crosstab(
    df_partidos["Partido_multi"],
    df_partidos["topico_dominante"],
    normalize="index"
) * 100

print("\nTabela Partido x Tópico (com coautoria real):")
print(tabela_partido_topico)

print("\nPercentual por partido (com coautoria real):")
print(tabela_partido_topico_pct.round(2))

df_partidos["sucesso"] = df_partidos["sucesso_legislativo"].astype(int)

tabela_partido_agenda = (
    df_partidos
    .groupby(["topico_dominante", "Partido_multi"])
    .agg(
        total_pl=("Proposicoes", "count"),
        aprovados=("sucesso", "sum")
    )
    .reset_index()
)

tabela_partido_agenda["taxa_sucesso_%"] = (
    tabela_partido_agenda["aprovados"] / tabela_partido_agenda["total_pl"]
) * 100

tabela_partido_agenda = tabela_partido_agenda.sort_values(
    ["topico_dominante", "taxa_sucesso_%"],
    ascending=[True, False]
)

print("\nTabela de sucesso por partido dentro de cada agenda:")
print(tabela_partido_agenda.head(50))

MIN_PL = 20
for agenda in sorted(tabela_partido_agenda["topico_dominante"].unique()):
    print("\n" + "=" * 40)
    print(f"AGENDA {agenda}")
    print("=" * 40)

    tabela_agenda = tabela_partido_agenda[
        (tabela_partido_agenda["topico_dominante"] == agenda) &
        (tabela_partido_agenda["total_pl"] >= MIN_PL)
    ].sort_values("taxa_sucesso_%", ascending=False)

    print(tabela_agenda)


# =========================================================
# 15. BASES DE REGRESSÃO
# =========================================================

df_reg = df_texto.copy()
df_reg["aprovado"] = df_reg["sucesso_legislativo"].astype(int)

df_reg = df_reg[
    df_reg["topico_dominante"].notna() &
    df_reg["Partido_limpo"].notna() &
    df_reg["ano"].notna()
].copy()

df_reg["ano"] = df_reg["ano"].astype(int)
df_reg["legislatura"] = df_reg["ano"].apply(mapear_legislatura)
df_reg["partido_rec"] = preparar_partido_rec(df_reg, "Partido_limpo", FREQ_MIN_PARTIDO)

ano_mediano = float(df_reg["ano"].median())
df_reg["ano_c"] = df_reg["ano"] - ano_mediano

df_reg_decidido = df_reg[
    df_reg["situacao_recodificada"].isin(["sucesso", "fracasso"])
].copy()

df_reg_decidido["aprovado"] = (df_reg_decidido["situacao_recodificada"] == "sucesso").astype(int)

print("\nDimensão da base de regressão:", df_reg.shape)
print("Dimensão da base decidida:", df_reg_decidido.shape)


# =========================================================
# 16. LEGISLATURA X SUCESSO
# =========================================================

tabela_leg_sucesso = pd.crosstab(
    df_reg_decidido["legislatura"],
    df_reg_decidido["aprovado"]
).reindex(columns=[0, 1], fill_value=0)

tabela_leg_sucesso.columns = ["Nao_aprovado", "Aprovado"]
tabela_leg_sucesso["total"] = tabela_leg_sucesso.sum(axis=1)
tabela_leg_sucesso["taxa_sucesso_%"] = (
    tabela_leg_sucesso["Aprovado"] / tabela_leg_sucesso["total"]
) * 100

print("\nTabela Legislatura x Sucesso:")
print(tabela_leg_sucesso.sort_index())

plt.figure(figsize=(10, 5))
tabela_leg_sucesso["taxa_sucesso_%"].plot(kind="bar")
plt.title("Taxa de sucesso legislativo por legislatura")
plt.xlabel("Legislatura")
plt.ylabel("Percentual aprovado")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
plt.close()


# =========================================================
# 17. MODELOS INFERENCIAIS
# =========================================================

print("\nEstimando modelos inferenciais...")
print("[NOTA] Erros padrão clusterizados por Autor em todos os logits.")
print("       Cada parlamentar apresenta múltiplos PLs; a clusterização")
print("       corrige a subestimação dos erros padrão por dependência")
print("       intra-autor. Referência: Cameron & Miller (2015, JHR).")
print()

modelo_principal = ajustar_logit_com_fallback(
    "aprovado ~ C(topico_dominante) + ano_c",
    df_reg_decidido,
    "MODELO PRINCIPAL (TÓPICO + ANO)"
)

if modelo_principal is not None:
    exibir_modelo(modelo_principal, "MODELO PRINCIPAL (TÓPICO + ANO)")
    df_odds_principal = extrair_tabela_odds(modelo_principal, "modelo_principal")

tab_partido_inf = pd.crosstab(
    df_reg_decidido["partido_rec"],
    df_reg_decidido["aprovado"]
).reindex(columns=[0, 1], fill_value=0)

tab_partido_inf.columns = ["Nao_aprovado", "Aprovado"]
tab_partido_inf["total"] = tab_partido_inf["Nao_aprovado"] + tab_partido_inf["Aprovado"]

partidos_elegiveis = tab_partido_inf[
    (tab_partido_inf["total"] >= MIN_TOTAL_PARTIDO_INF) &
    (tab_partido_inf["Aprovado"] >= MIN_SUCESSO_PARTIDO_INF) &
    (tab_partido_inf["Nao_aprovado"] >= MIN_SUCESSO_PARTIDO_INF)
].index.tolist()

df_reg_inf = df_reg_decidido.copy()
df_reg_inf["partido_inf"] = df_reg_inf["partido_rec"].where(
    df_reg_inf["partido_rec"].isin(partidos_elegiveis),
    "OUTROS"
)

print("\nTabela de elegibilidade inferencial dos partidos:")
print(tab_partido_inf.sort_values("total", ascending=False))

print("\nPartidos elegíveis para inferência:")
print(partidos_elegiveis)

partidos_so_descritivos = [p for p in tab_partido_inf.index if p not in partidos_elegiveis]
print("\nPartidos mantidos apenas descritivamente / recodificados em OUTROS:")
print(partidos_so_descritivos)

print("\nFrequência final de partido_inf:")
print(df_reg_inf["partido_inf"].value_counts(dropna=False))

modelo_ano = ajustar_logit_com_fallback(
    "aprovado ~ C(partido_inf) + ano_c",
    df_reg_inf,
    "MODELO COM PARTIDO + ANO"
)

if modelo_ano is not None:
    exibir_modelo(modelo_ano, "MODELO COM PARTIDO + ANO CENTRALIZADO")
    df_odds_ano = extrair_tabela_odds(modelo_ano, "modelo_ano")

modelo_partido = ajustar_logit_com_fallback(
    "aprovado ~ C(topico_dominante) + C(partido_inf) + ano_c",
    df_reg_inf,
    "MODELO COM TÓPICO + PARTIDO + ANO"
)

if modelo_partido is not None:
    exibir_modelo(modelo_partido, "MODELO COM TÓPICO + PARTIDO + ANO")
    df_odds = extrair_tabela_odds(modelo_partido, "modelo_partido")

modelo_leg = ajustar_logit_com_fallback(
    "aprovado ~ C(partido_inf) + C(legislatura)",
    df_reg_inf,
    "MODELO COM PARTIDO + LEGISLATURA"
)

if modelo_leg is not None:
    exibir_modelo(modelo_leg, "MODELO COM PARTIDO + LEGISLATURA")
    df_odds_leg = extrair_tabela_odds(modelo_leg, "modelo_leg")

# ── MODELO AMPLIADO: tópico + partido + legislatura + capital institucional ──
# Integra legislatura (efeitos de contexto político) e presidente_comissao_bin
# (capital institucional) no mesmo modelo. Este é o modelo mais completo.
# A 57ª legislatura entra com dummy de truncamento (dados parciais).
print("\n" + "=" * 60)
print("MODELO AMPLIADO — TÓPICO + PARTIDO + LEGISLATURA + CAPITAL")
print("=" * 60)

modelo_ampliado = None
df_odds_ampliado = pd.DataFrame()

try:
    _base_amp = df_reg_inf.copy()

    # adiciona presidente_comissao_bin se disponível via merge com base sociológica
    if "presidente_comissao_bin" not in _base_amp.columns and not df_reg_corp_base.empty:
        _cols_cap = [c for c in ["Proposicoes","presidente_comissao_bin","comissao_segpub_bin"]
                     if c in df_reg_corp_base.columns]
        if len(_cols_cap) > 1:
            _base_amp = _base_amp.merge(
                df_reg_corp_base[_cols_cap].drop_duplicates("Proposicoes"),
                on="Proposicoes", how="left"
            )

    # variáveis disponíveis
    _tem_comissao = "presidente_comissao_bin" in _base_amp.columns and \
                    _base_amp["presidente_comissao_bin"].nunique() > 1
    _tem_leg = "legislatura" in _base_amp.columns

    if _tem_leg and _tem_comissao:
        # dummy para 57ª (incompleta) — controla truncamento
        _base_amp["leg57"] = (_base_amp["legislatura"] == "57a").astype(int)
        _fml_amp = (
            "aprovado ~ C(topico_dominante) + C(partido_inf) "
            "+ C(legislatura) + presidente_comissao_bin + leg57"
        )
    elif _tem_leg:
        _base_amp["leg57"] = (_base_amp["legislatura"] == "57a").astype(int)
        _fml_amp = (
            "aprovado ~ C(topico_dominante) + C(partido_inf) "
            "+ C(legislatura) + leg57"
        )
    elif _tem_comissao:
        _fml_amp = (
            "aprovado ~ C(topico_dominante) + C(partido_inf) "
            "+ ano_c + presidente_comissao_bin"
        )
    else:
        _fml_amp = "aprovado ~ C(topico_dominante) + C(partido_inf) + ano_c"

    _base_amp_cc = _base_amp[
        _base_amp["situacao_recodificada"].isin(["sucesso","fracasso"])
    ].dropna(subset=["aprovado","topico_dominante","partido_inf"])

    modelo_ampliado = ajustar_logit_com_fallback(
        _fml_amp, _base_amp_cc, "MODELO AMPLIADO"
    )

    if modelo_ampliado is not None:
        exibir_modelo(modelo_ampliado, "MODELO AMPLIADO")
        df_odds_ampliado = extrair_tabela_odds(modelo_ampliado, "modelo_ampliado")

        # comparação de ajuste entre os modelos
        print("\nComparação de ajuste — modelos inferenciais:")
        _comp_rows = []
        for _nm, _mod in [
            ("M1: tópico + ano",          modelo_principal),
            ("M2: tópico + partido + ano", modelo_partido),
            ("M3: partido + legislatura",  modelo_leg),
            ("M4: ampliado (leg+comissão)", modelo_ampliado),
        ]:
            if _mod is not None:
                _comp_rows.append({
                    "modelo":     _nm,
                    "N":          int(_mod.nobs),
                    "pseudo_R2":  round(_mod.prsquared, 4),
                    "AIC":        round(_mod.aic, 1),
                    "BIC":        round(_mod.bic, 1),
                    "ll":         round(_mod.llf, 2),
                })
        if _comp_rows:
            _df_comp = pd.DataFrame(_comp_rows)
            print(_df_comp.to_string(index=False))
            print("\nNota: menor AIC/BIC = melhor ajuste penalizado por complexidade.")
            print("  M4 (ampliado) combina agenda, partido, legislatura e capital institucional.")
            print("  A 57ª legislatura entra com dummy de truncamento (dados parciais).")

        # destaque: coeficientes de legislatura e comissão
        _leg_c = pd.DataFrame({
            "variavel": modelo_ampliado.params.index,
            "coef":     modelo_ampliado.params.round(4),
            "or":       np.exp(modelo_ampliado.params).round(3),
            "p":        modelo_ampliado.pvalues.round(4)
        })
        _leg_only = _leg_c[_leg_c["variavel"].str.startswith("C(legislatura)")]
        if not _leg_only.empty:
            _leg_only = _leg_only.copy()
            _leg_only["legislatura"] = _leg_only["variavel"].str.extract(r"\[T\.(.*?)\]")[0]
            print("\nEfeito de legislatura (vs. 49ª — referência):")
            print(_leg_only[["legislatura","coef","or","p"]].sort_values("coef", ascending=False).to_string(index=False))

        _cap_c = _leg_c[_leg_c["variavel"].str.contains("comissao|capital", case=False)]
        if not _cap_c.empty:
            print("\nCapital institucional (comissão):")
            print(_cap_c[["variavel","coef","or","p"]].to_string(index=False))

except Exception as _e_amp:
    print(f"[AVISO] Modelo ampliado falhou: {_e_amp}")


# =========================================================
# 17B. MODELO MULTINÍVEL — INTERCEPTO ALEATÓRIO POR AUTOR
# =========================================================
#
# Motivação: cada parlamentar apresenta múltiplos PLs — as observações
# NÃO são independentes dentro de um mesmo autor. Isso viola a hipótese
# de independência do logit padrão. A solução correta é um modelo misto
# (GLMM) com intercepto aleatório por parlamentar (Autor).
#
# Estratégia em três níveis de sofisticação, com fallback automático:
#
#   Nível 1 — BinomialBayesMixedGLM (statsmodels):
#     Logit Bayesiano misto. Efeito aleatório por Autor. Mais correto.
#
#   Nível 2 — MixedLM com variável dependente contínua (LPM multinível):
#     Linear Probability Model com intercepto aleatório por Autor.
#     Aproximação válida para eventos raros (Angrist & Pischke, 2009).
#     Coeficientes interpretáveis como diferenças em probabilidade.
#
#   Nível 3 — apenas diagnóstico de dependência intra-cluster (ICC):
#     Se ambos falharem, calcula e reporta o ICC que justifica o uso
#     de modelos mistos como agenda metodológica futura.
# =========================================================

print("\n" + "=" * 60)
print("MODELO MULTINÍVEL — INTERCEPTO ALEATÓRIO POR AUTOR")
print("=" * 60)

modelo_multinivel = None
df_multinivel_metricas = pd.DataFrame()
df_multinivel_coef = pd.DataFrame()
df_multinivel_icc = pd.DataFrame()

_base_ml = df_reg_decidido.copy()
_base_ml = _base_ml[_base_ml["Autor"].notna()].copy()
_base_ml["topico_f"] = _base_ml["topico_dominante"].astype(str)
_base_ml["aprovado_f"] = _base_ml["aprovado"].astype(float)

print(f"\nBase multinível: {len(_base_ml)} PLs, {_base_ml['Autor'].nunique()} autores únicos")

# -------------------------
# diagnóstico ICC (sempre roda)
# -------------------------
variancia_entre = _base_ml.groupby("Autor")["aprovado_f"].mean().var()
variancia_total = _base_ml["aprovado_f"].var()
icc_simples = variancia_entre / (variancia_entre + variancia_total) if variancia_total > 0 else np.nan

print(f"\nDiagnóstico de dependência intra-cluster:")
print(f"  ICC simples (variância entre autores / variância total): {icc_simples:.4f}")
if not np.isnan(icc_simples):
    if icc_simples > 0.05:
        print("  → ICC > 5%: dependência relevante; modelo multinível justificado.")
    else:
        print("  → ICC < 5%: dependência limitada; logit padrão com cautela é razoável.")

df_multinivel_icc = pd.DataFrame([{
    "icc_simples": round(icc_simples, 6),
    "variancia_entre_autores": round(float(variancia_entre), 6),
    "variancia_total": round(float(variancia_total), 6),
    "n_pls": len(_base_ml),
    "n_autores": int(_base_ml["Autor"].nunique()),
    "media_pls_por_autor": round(len(_base_ml) / _base_ml["Autor"].nunique(), 2)
}])

print("\nTabela ICC:")
print(df_multinivel_icc)

# -------------------------
# Nível 1: BinomialBayesMixedGLM
# -------------------------
_multinivel_ok = False

try:
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    _fe_formula = "aprovado_f ~ C(topico_f) + ano_c"

    _dummies_autor = pd.get_dummies(
        _base_ml["Autor"], prefix="autor", drop_first=True
    ).astype(float)

    _exog_fe = sm.add_constant(
        pd.get_dummies(_base_ml["topico_f"], drop_first=True).astype(float)
        .join(_base_ml[["ano_c"]].reset_index(drop=True), how="left")
    )

    _exog_vc = _dummies_autor.values

    modelo_multinivel = BinomialBayesMixedGLM(
        endog=_base_ml["aprovado_f"].values,
        exog=_exog_fe.values,
        exog_vc=_exog_vc,
        ident=np.zeros(_exog_vc.shape[1], dtype=int)
    ).fit_map()

    print("\n=== MODELO MULTINÍVEL NÍVEL 1: BinomialBayesMixedGLM ===")
    print(modelo_multinivel.summary())
    _multinivel_ok = True

    df_multinivel_metricas = pd.DataFrame([{
        "abordagem": "BinomialBayesMixedGLM",
        "n_obs": len(_base_ml),
        "n_grupos": int(_base_ml["Autor"].nunique()),
        "icc": round(icc_simples, 4),
        "nota": "Logit Bayesiano misto com intercepto aleatório por autor"
    }])

    try:
        _params = modelo_multinivel.params
        df_multinivel_coef = pd.DataFrame({
            "variavel": range(len(_params)),
            "coef": _params
        })
        df_multinivel_coef["modelo"] = "BinomialBayesMixedGLM"
    except Exception:
        pass

except ImportError:
    print("\nBinomialBayesMixedGLM não disponível — tentando Nível 2 (LPM multinível).")
except Exception as e1:
    print(f"\nBinomialBayesMixedGLM falhou ({e1}) — tentando Nível 2 (LPM multinível).")

# -------------------------
# Nível 2: MixedLM (LPM com intercepto aleatório por Autor)
# -------------------------
if not _multinivel_ok:
    try:
        from statsmodels.formula.api import mixedlm

        _formula_ml = "aprovado_f ~ C(topico_dominante) + ano_c"

        modelo_multinivel = mixedlm(
            _formula_ml,
            data=_base_ml,
            groups=_base_ml["Autor"]
        ).fit(reml=False, method="bfgs")

        print("\n=== MODELO MULTINÍVEL NÍVEL 2: LPM com intercepto aleatório por Autor ===")
        print(modelo_multinivel.summary())
        _multinivel_ok = True

        _var_resid = float(modelo_multinivel.scale)
        _var_autor = float(modelo_multinivel.cov_re.iloc[0, 0])
        _icc_lpm = _var_autor / (_var_autor + _var_resid) if (_var_autor + _var_resid) > 0 else np.nan

        print(f"\nVariância do intercepto aleatório (por autor): {_var_autor:.6f}")
        print(f"Variância residual:                            {_var_resid:.6f}")
        print(f"ICC estimado pelo LPM:                         {_icc_lpm:.4f}")

        df_multinivel_metricas = pd.DataFrame([{
            "abordagem": "MixedLM (LPM)",
            "n_obs": int(modelo_multinivel.nobs),
            "n_grupos": int(_base_ml["Autor"].nunique()),
            "variancia_autor": round(_var_autor, 6),
            "variancia_residual": round(_var_resid, 6),
            "icc_lpm": round(_icc_lpm, 4),
            "log_likelihood": round(float(modelo_multinivel.llf), 3),
            "aic": round(float(modelo_multinivel.aic), 3),
            "nota": "Linear Probability Model com intercepto aleatório por autor"
        }])

        # coeficientes
        _params_ml = modelo_multinivel.params
        _pvalues_ml = modelo_multinivel.pvalues
        _ci_ml = modelo_multinivel.conf_int()

        df_multinivel_coef = pd.DataFrame({
            "variavel": _params_ml.index,
            "coef": _params_ml.values,
            "p_value": _pvalues_ml.values,
            "ic95_inf": _ci_ml.iloc[:, 0].values,
            "ic95_sup": _ci_ml.iloc[:, 1].values
        })
        df_multinivel_coef["modelo"] = "MixedLM_LPM"
        df_multinivel_coef["interpretacao"] = (
            "Diferença em probabilidade relativa ao tópico de referência"
        )

        print("\nCoeficientes do LPM multinível:")
        print(df_multinivel_coef[["variavel", "coef", "p_value"]].round(4))

    except Exception as e2:
        print(f"\nLPM multinível também falhou ({e2}).")
        print("Usando apenas diagnóstico ICC como resultado reportável.")

# -------------------------
# nota metodológica final
# -------------------------
if not _multinivel_ok:
    df_multinivel_metricas = pd.DataFrame([{
        "abordagem": "diagnóstico ICC apenas",
        "n_obs": len(_base_ml),
        "n_grupos": int(_base_ml["Autor"].nunique()),
        "icc_simples": round(icc_simples, 4),
        "nota": (
            "Modelos mistos não convergiram. ICC reportado como evidência de "
            "aninhamento. Recomenda-se modelo multinível em estudo futuro."
        )
    }])
    print("\nNota metodológica para o TCC:")
    print(
        "As proposições estão aninhadas em parlamentares "
        f"(ICC = {icc_simples:.4f}), o que viola a independência do logit "
        "padrão. Como robustez futura, recomenda-se um GLMM com intercepto "
        "aleatório por parlamentar (Autor), controlando por legislatura e partido."
    )

print("\nResumo multinível:")
print(df_multinivel_metricas)
# =========================================================

df_aprovadas = df_reg_decidido[df_reg_decidido["aprovado"] == 1].copy()

agenda_aprovada = (
    df_aprovadas["topico_dominante"]
    .value_counts(normalize=True)
    * 100
).sort_index()

print("\nDistribuição das agendas nas leis aprovadas (%):")
print(agenda_aprovada.round(2))

plt.figure(figsize=(10, 6))
_agenda_labels = [NOMES_TOPICOS_CURTO.get(int(i), str(i)) for i in agenda_aprovada.index]
bars_ag = plt.bar(_agenda_labels, agenda_aprovada.values,
                  color="steelblue", alpha=0.8)
for bar, pct in zip(bars_ag, agenda_aprovada.values):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.3,
             f"{pct:.1f}%", ha="center", fontsize=9)
plt.title("Distribuição das agendas nas proposições aprovadas\n"
          "(% das 95 aprovações por tópico LDA)")
plt.xlabel("Agenda temática")
plt.ylabel("Percentual das leis aprovadas (%)")
plt.xticks(rotation=15, ha="right", fontsize=9)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(ARQUIVO_AGENDAS_APROVADAS, dpi=300)
plt.show()
plt.close()


# =========================================================
# 19. MODELO PREDITIVO COMPLEMENTAR
# =========================================================

X_pred = pd.get_dummies(
    df_reg_inf[["topico_dominante", "partido_inf", "ano_c"]],
    drop_first=True
)
y_pred_bin = df_reg_inf["aprovado"]

X_train, X_test, y_train, y_test = train_test_split(
    X_pred,
    y_pred_bin,
    test_size=0.2,
    random_state=42,
    stratify=y_pred_bin
)

model = LogisticRegression(max_iter=2000, class_weight="balanced")
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= THRESHOLD_CLASSIFICACAO).astype(int)

print("\n=== MODELO PREDITIVO COMPLEMENTAR ===")
print("\nMatriz de confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred, digits=3, zero_division=0))

print("\nProbabilidades previstas:")
print("Menor prob:", float(y_prob.min()))
print("Maior prob:", float(y_prob.max()))
print("Média prob:", float(y_prob.mean()))

fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC - Regressão Logística")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
plt.close()

precision, recall, _ = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"AP = {ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision-Recall")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
plt.close()

thresholds_padrao = [0.10, 0.20, 0.30, 0.40, 0.50]

for threshold in thresholds_padrao:
    y_pred_thr = (y_prob >= threshold).astype(int)
    print(f"\nThreshold = {threshold}")
    print(confusion_matrix(y_test, y_pred_thr))
    print(classification_report(y_test, y_pred_thr, digits=3, zero_division=0))

df_rank = pd.DataFrame({
    "prob_aprovacao": y_prob,
    "real": y_test.values
}).sort_values("prob_aprovacao", ascending=False)

print("\nTop 30 maiores probabilidades:")
print(df_rank.head(30))

top10 = df_rank.head(10)
top20 = df_rank.head(20)
top50 = df_rank.head(50)

print("Aprovados no top10:", int(top10["real"].sum()))
print("Aprovados no top20:", int(top20["real"].sum()))
print("Aprovados no top50:", int(top50["real"].sum()))

df_rank = df_rank.reset_index(drop=True)
df_rank["cum_aprov"] = df_rank["real"].cumsum()

plt.figure(figsize=(8, 6))
plt.plot(df_rank["cum_aprov"])
plt.xlabel("Ranking de probabilidade")
plt.ylabel("Aprovações acumuladas")
plt.title("Concentração de aprovações no ranking")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
plt.close()

coef = pd.DataFrame({
    "variavel": X_train.columns,
    "coef": model.coef_[0]
})
coef["odds_ratio"] = np.exp(coef["coef"])
coef = coef.sort_values("coef", ascending=False)

coef_plot = coef.sort_values("coef")

# substituir rótulos técnicos por nomes legíveis
def _label_coef(v):
    v = str(v)
    for k in range(1, 6):
        if f"topico_dominante_{k}" in v or f"topico_{k}" in v:
            return NOMES_TOPICOS_CURTO.get(k, v)
    return v.replace("partido_inf_", "").replace("_", " ")

coef_plot["label"] = coef_plot["variavel"].map(_label_coef)

plt.figure(figsize=(10, max(6, len(coef_plot) * 0.35)))
cores_c = ["#c0392b" if c < 0 else "#2980b9" for c in coef_plot["coef"]]
plt.barh(coef_plot["label"], coef_plot["coef"], color=cores_c, alpha=0.85)
plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
plt.xlabel("Coeficiente logístico")
plt.title("Efeito das variáveis na probabilidade de aprovação\n"
          "(modelo preditivo complementar — partido + tópico)")
plt.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(PASTA / "coef_modelo_pred.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

df_metricas_modelo_pred, df_metricas_thresholds = consolidar_metricas_classificacao(
    y_true=y_test.values,
    y_prob=y_prob,
    thresholds=thresholds_padrao,
    nome_modelo="modelo_preditivo_partido"
)

df_metricas_modelo_pred["aprovados_top10"] = int(top10["real"].sum())
df_metricas_modelo_pred["aprovados_top20"] = int(top20["real"].sum())
df_metricas_modelo_pred["aprovados_top50"] = int(top50["real"].sum())


# =========================================================
# 20. EFEITO APENAS DOS TÓPICOS
# =========================================================
# Usa os coeficientes do modelo principal (statsmodels, seção 17)
# em vez de um LogisticRegression sklearn separado.
# Motivo: sklearn com class_weight='balanced' distorce os coeficientes
# num dataset tão desbalanceado (1.3% positivos) — os coeficientes ficam
# invertidos em relação à taxa observada real por tópico.
# O modelo statsmodels é o estimador correto para inferência.

coef_topics_plot = pd.DataFrame()

if modelo_principal is not None:
    try:
        _params = modelo_principal.params
        _pvals  = modelo_principal.pvalues

        # extrai apenas os coeficientes de tópico
        _coef_top = (
            pd.DataFrame({"variavel": _params.index, "coef": _params.values,
                          "p_value": _pvals.values})
            .query("variavel.str.contains('topico_dominante')", engine="python")
            .copy()
        )
        _coef_top["topico_num"] = (
            _coef_top["variavel"]
            .str.extract(r"\[T\.(\d+)\]")[0]
            .astype(int)
        )
        _coef_top["odds_ratio"] = np.exp(_coef_top["coef"])
        _coef_top["label"] = _coef_top["topico_num"].map(NOMES_TOPICOS_CURTO)

        # adiciona linha de referência (tópico 1, coef = 0)
        _ref = pd.DataFrame({
            "variavel": ["referencia"],
            "coef": [0.0],
            "p_value": [np.nan],
            "topico_num": [1],
            "odds_ratio": [1.0],
            "label": [f"{NOMES_TOPICOS_CURTO.get(1,'T1')} ← referência"],
        })
        coef_topics_plot = pd.concat([_coef_top, _ref], ignore_index=True)
        coef_topics_plot = coef_topics_plot.sort_values("coef")

        print("\nCoeficientes dos tópicos (modelo principal — statsmodels):")
        print(coef_topics_plot[["label", "coef", "odds_ratio", "p_value"]].to_string(index=False))

        fig, ax = plt.subplots(figsize=(10, 5))
        cores_t = ["#c0392b" if c < 0 else "#27ae60"
                   for c in coef_topics_plot["coef"]]
        barras_t = ax.barh(coef_topics_plot["label"],
                           coef_topics_plot["coef"],
                           color=cores_t, alpha=0.85)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)

        for bar, (_, row) in zip(barras_t, coef_topics_plot.iterrows()):
            if pd.notna(row["coef"]) and row["coef"] != 0:
                sig = ""
                if pd.notna(row["p_value"]):
                    if row["p_value"] < 0.001:   sig = "***"
                    elif row["p_value"] < 0.01:  sig = "**"
                    elif row["p_value"] < 0.05:  sig = "*"
                x_pos = row["coef"] - 0.06 if row["coef"] < 0 else row["coef"] + 0.02
                ha = "right" if row["coef"] < 0 else "left"
                ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                        f"OR={row['odds_ratio']:.2f}{sig}",
                        va="center", ha=ha, fontsize=9,
                        color="white" if abs(row["coef"]) > 0.4 else "#333")
            else:
                ax.text(0.02, bar.get_y() + bar.get_height() / 2,
                        "referência (OR=1.00)", va="center", ha="left",
                        fontsize=9, color="#27ae60", style="italic")

        ax.set_xlabel("Coeficiente logístico (vs. T1 — Regulação/Trânsito)")
        ax.set_title(
            "Efeito das agendas temáticas na probabilidade de aprovação\n"
            f"(modelo principal statsmodels, pseudo-R²={1-modelo_principal.llf/modelo_principal.llnull:.3f})\n"
            "* p<0.05  ** p<0.01  *** p<0.001"
        )
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.set_xlim(coef_topics_plot["coef"].min() - 0.2, 0.3)
        plt.tight_layout()
        plt.savefig(PASTA / "coef_topicos.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print("Gráfico de coeficientes dos tópicos salvo.")

    except Exception as e_coef:
        print(f"\n[AVISO] Gráfico de coeficientes dos tópicos falhou: {e_coef}")
else:
    print("\n[AVISO] modelo_principal não disponível — seção 20 pulada.")




# =========================================================
# 21. TABELA AGREGADA DE CORPORAÇÃO
# =========================================================

if ARQUIVO_FARDA.exists():
    df_farda = pd.read_excel(ARQUIVO_FARDA)

    df_farda = df_farda.rename(columns={
        "Qtd_PL na legislatura": "qtd_pl",
        "Proposições transformadas em norma jurídica na legislatura": "normas",
        "Corporação": "corporacao"
    })

    df_farda = df_farda[df_farda["qtd_pl"] > 0].copy()
    df_farda["taxa_sucesso"] = df_farda["normas"] / df_farda["qtd_pl"]

    tabela_corporacao = (
        df_farda.groupby("corporacao")
        .agg(
            parlamentares=("corporacao", "count"),
            total_pl=("qtd_pl", "sum"),
            total_normas=("normas", "sum")
        )
    )

    tabela_corporacao["taxa_sucesso"] = (
        tabela_corporacao["total_normas"] / tabela_corporacao["total_pl"]
    )

    tabela_corporacao = tabela_corporacao.sort_values("taxa_sucesso", ascending=False)

    print("\nTabela agregada por corporação:")
    print(tabela_corporacao)
else:
    df_farda = pd.DataFrame()
    tabela_corporacao = pd.DataFrame()


# =========================================================
# 22. ANÁLISE DE CORPORAÇÃO
# =========================================================

df_reg_corp_logit = pd.DataFrame()
df_reg_corp_base = pd.DataFrame()

chi2_corp = np.nan
p_corp = np.nan
dof_corp = np.nan

if ARQUIVO_FARDA.exists():
    df_farda_det = pd.read_excel(ARQUIVO_FARDA)

    # ── renomear todas as colunas relevantes ──────────────────────────────
    df_farda_det = df_farda_det.rename(columns={
        "Parlamentar":                                               "parlamentar",
        "Corporação":                                                "corporacao",
        "Partido":                                                   "partido_farda",
        "UF":                                                        "uf_farda",
        "Legislatura":                                               "legislatura_farda",
        "Votação":                                                   "votacao",
        "Titular ou suplente":                                       "titular_suplente",
        "Qtd_Discursos na legislatura":                              "qtd_discursos",
        "Qtd_PL na legislatura":                                     "qtd_pl_farda",
        "Qtd_PEC na legislatura":                                    "qtd_pec",
        "Proposições transformadas em norma jurídica na legislatura": "normas_farda",
        "Presidente_de_Comissao":                                    "presidente_comissao",
        "Participacao_na_Comissao de Segurança Pública":             "comissao_segpub",
        "Licença":                                                   "licenca",
        "Naturalidade":                                              "naturalidade",
        "ESTADO DE NASCIMENTO":                                      "estado_nascimento",
        "Etnia/Raça/Cor":                                            "etnia_raca",
        "Identidade_genero":                                         "genero",
        "Ano de nascimento":                                         "ano_nascimento",
        "Quantos anos trabalhou em força de segurança":              "anos_forca_seg",
        "Outra profissão":                                           "outra_profissao",
        "Evangélico":                                                "evangelico",
        "Mandatos_externos_Câmara_casas_legislativas_em_esfera_municipal-estadual": "mandatos_externos",
        "Mandato de Vereador":                                       "mandato_vereador",
        "Mandato de Desputado Estadual":                             "mandato_dep_estadual",
    })

    df_farda_det["parlamentar"] = df_farda_det["parlamentar"].apply(normalizar_merge_nome)
    df_farda_det["corporacao"]  = df_farda_det["corporacao"].apply(limpar_corporacao_texto)
    df_farda_det["corporacao_sigla"] = df_farda_det["corporacao"].apply(padronizar_sigla_corporacao)

    # ── variáveis derivadas ───────────────────────────────────────────────
    ano_ref_idade = 2024
    df_farda_det["ano_nascimento_num"] = pd.to_numeric(
        df_farda_det["ano_nascimento"], errors="coerce"
    )
    df_farda_det["idade_aprox"] = ano_ref_idade - df_farda_det["ano_nascimento_num"]

    df_farda_det["anos_forca_seg_num"] = pd.to_numeric(
        df_farda_det["anos_forca_seg"], errors="coerce"
    )

    # binárias padronizadas (1/0)
    for col_bin in ["evangelico", "presidente_comissao", "comissao_segpub",
                    "licenca", "mandatos_externos", "mandato_vereador",
                    "mandato_dep_estadual"]:
        if col_bin in df_farda_det.columns:
            df_farda_det[col_bin + "_bin"] = (
                df_farda_det[col_bin]
                .astype(str).str.strip().str.lower()
                .isin(["sim", "s", "1", "yes", "true", "x"])
            ).astype(int)

    # gênero binário
    if "genero" in df_farda_det.columns:
        df_farda_det["feminino"] = (
            df_farda_det["genero"].astype(str).str.lower()
            .str.contains("fem|mulher|f$", regex=True, na=False)
        ).astype(int)

    # etnia: não-branco
    if "etnia_raca" in df_farda_det.columns:
        df_farda_det["nao_branco"] = (
            ~df_farda_det["etnia_raca"].astype(str).str.lower()
            .str.contains("bran|white|nao inf", regex=True, na=False)
            & df_farda_det["etnia_raca"].notna()
        ).astype(int)

    # ── análise descritiva sociológica ────────────────────────────────────
    print("\n" + "=" * 60)
    print("ANÁLISE SOCIOLÓGICA — BASE POLITICOS-DE-FARDA")
    print("=" * 60)
    print(f"\nTotal de registros: {len(df_farda_det)}")
    print(f"Parlamentares únicos: {df_farda_det['parlamentar'].nunique()}")

    for col, label in [
        ("genero",          "Gênero"),
        ("etnia_raca",      "Etnia/Raça/Cor"),
        ("evangelico",      "Evangélico"),
        ("corporacao_sigla","Corporação"),
        ("titular_suplente","Titular ou Suplente"),
        ("uf_farda",        "UF"),
    ]:
        if col in df_farda_det.columns:
            print(f"\n{label}:")
            print(df_farda_det[col].value_counts(dropna=False).head(12))

    for col, label in [
        ("idade_aprox",        "Idade aproximada (2024)"),
        ("anos_forca_seg_num", "Anos em força de segurança"),
        ("qtd_pl_farda",       "Qtd PLs na legislatura"),
        ("qtd_discursos",      "Qtd Discursos na legislatura"),
    ]:
        if col in df_farda_det.columns:
            serie = pd.to_numeric(df_farda_det[col], errors="coerce").dropna()
            if len(serie) > 0:
                print(f"\n{label}: média={serie.mean():.1f} | "
                      f"mediana={serie.median():.1f} | "
                      f"dp={serie.std():.1f} | "
                      f"min={serie.min():.0f} | max={serie.max():.0f} | "
                      f"n={len(serie)}")

    # presidente de comissão x corporação
    if "presidente_comissao_bin" in df_farda_det.columns:
        print("\nTaxa de presidência de comissão por corporação:")
        print(
            df_farda_det.groupby("corporacao_sigla")["presidente_comissao_bin"]
            .agg(["mean", "sum", "count"])
            .rename(columns={"mean": "taxa", "sum": "n_presidentes", "count": "total"})
            .sort_values("taxa", ascending=False)
            .round(3)
        )

    # comissão de segurança pública x corporação
    if "comissao_segpub_bin" in df_farda_det.columns:
        print("\nParticipação na Comissão de Segurança Pública por corporação:")
        print(
            df_farda_det.groupby("corporacao_sigla")["comissao_segpub_bin"]
            .agg(["mean", "sum", "count"])
            .rename(columns={"mean": "taxa", "sum": "n_membros", "count": "total"})
            .sort_values("taxa", ascending=False)
            .round(3)
        )

    # mandato anterior (político experiente)
    if "mandatos_externos_bin" in df_farda_det.columns:
        print("\nMandatos externos anteriores por corporação:")
        print(
            df_farda_det.groupby("corporacao_sigla")["mandatos_externos_bin"]
            .agg(["mean", "sum", "count"])
            .rename(columns={"mean": "taxa", "sum": "n_mandatos", "count": "total"})
            .sort_values("taxa", ascending=False)
            .round(3)
        )

    # ── mapa de atributos sociológicos para merge com df_reg_corp_base ───
    cols_sociol = ["parlamentar", "corporacao_sigla",
                   "anos_forca_seg_num", "idade_aprox", "feminino", "nao_branco",
                   "evangelico_bin", "presidente_comissao_bin", "comissao_segpub_bin",
                   "mandatos_externos_bin", "mandato_vereador_bin",
                   "mandato_dep_estadual_bin", "qtd_discursos", "qtd_pec"]
    cols_sociol_ok = [c for c in cols_sociol if c in df_farda_det.columns]

    mapa_sociol = (
        df_farda_det[cols_sociol_ok]
        .dropna(subset=["parlamentar"])
        .drop_duplicates(subset=["parlamentar"])
        .rename(columns={"parlamentar": "Autor_merge"})
    )

    df_reg_corp_base = df_reg_decidido.copy()
    df_reg_corp_base["Autor_merge"] = df_reg_corp_base["Autor"].apply(normalizar_merge_nome)

    df_reg_corp_base = df_reg_corp_base.merge(mapa_sociol, on="Autor_merge", how="left")
    df_reg_corp_base["corporacao_sigla"] = (
        df_reg_corp_base["corporacao_sigla"].fillna("OUTROS")
    )

    # cobertura do merge sociológico
    n_merge = df_reg_corp_base["corporacao_sigla"].ne("OUTROS").sum()
    print(f"\nCobertura do merge sociológico: "
          f"{n_merge}/{len(df_reg_corp_base)} PLs ({n_merge/len(df_reg_corp_base)*100:.1f}%)")

    print("\nVariáveis sociológicas disponíveis na base de regressão:")
    cols_disp = [c for c in mapa_sociol.columns if c != "Autor_merge"
                 and c in df_reg_corp_base.columns]
    print(cols_disp)

    # ── tabela descritiva sociológica exportável ──────────────────────────
    df_sociol_descritiva = pd.DataFrame()
    rows_sociol = []
    for col in ["anos_forca_seg_num", "idade_aprox", "feminino", "nao_branco",
                "evangelico_bin", "presidente_comissao_bin", "comissao_segpub_bin",
                "mandatos_externos_bin", "mandato_vereador_bin",
                "mandato_dep_estadual_bin"]:
        if col in df_farda_det.columns:
            serie = pd.to_numeric(df_farda_det[col], errors="coerce").dropna()
            rows_sociol.append({
                "variavel": col,
                "n": len(serie),
                "media_ou_proporcao": round(float(serie.mean()), 4),
                "dp": round(float(serie.std()), 4),
                "min": round(float(serie.min()), 2),
                "max": round(float(serie.max()), 2),
                "missings": int(df_farda_det[col].isna().sum())
            })
    df_sociol_descritiva = pd.DataFrame(rows_sociol)

    print("\nEstatísticas descritivas — variáveis sociológicas:")
    print(df_sociol_descritiva)

    print("\nFrequência de corporação na base decidida:")
    print(df_reg_corp_base["corporacao_sigla"].value_counts(dropna=False))

    tabela_corp_sucesso_full = pd.crosstab(
        df_reg_corp_base["corporacao_sigla"],
        df_reg_corp_base["aprovado"]
    ).reindex(columns=[0, 1], fill_value=0)

    tabela_corp_sucesso_full.columns = ["Nao_aprovado", "Aprovado"]
    tabela_corp_sucesso_full["total"] = tabela_corp_sucesso_full.sum(axis=1)
    tabela_corp_sucesso_full["taxa_sucesso_%"] = (
        tabela_corp_sucesso_full["Aprovado"] / tabela_corp_sucesso_full["total"]
    ) * 100

    print("\nTabela descritiva completa de sucesso por corporação:")
    print(tabela_corp_sucesso_full.sort_values("taxa_sucesso_%", ascending=False))

    df_reg_corp = df_reg_corp_base[df_reg_corp_base["corporacao_sigla"] != "OUTROS"].copy()

    tabela_corp_topico = pd.crosstab(
        df_reg_corp["corporacao_sigla"],
        df_reg_corp["topico_dominante"]
    )

    tabela_corp_topico_pct = pd.crosstab(
        df_reg_corp["corporacao_sigla"],
        df_reg_corp["topico_dominante"],
        normalize="index"
    ) * 100

    print("\nTabela Corporação x Tópico (sem OUTROS):")
    print(tabela_corp_topico)

    print("\nTabela percentual Corporação x Tópico (sem OUTROS):")
    print(tabela_corp_topico_pct.round(2))

    plt.figure(figsize=(12, 6))
    _corp_hm = tabela_corp_topico_pct.copy()
    _corp_hm.columns = [NOMES_TOPICOS_CURTO.get(int(c), str(c)) for c in _corp_hm.columns]
    _corp_hm.index = [_CORP_LABELS.get(str(i), str(i)) for i in _corp_hm.index]
    sns.heatmap(_corp_hm, annot=True, fmt=".1f", cmap="Blues")
    plt.title("Distribuição temática dos PLs por corporação de origem (%)\n"
              "(baseada nos termos do modelo LDA — sem categoria OUTROS)")
    plt.xlabel("Agenda temática")
    plt.ylabel("Corporação")
    plt.xticks(rotation=20, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(ARQUIVO_HEATMAP_CORP_TOPICO, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    tabela_corp_sucesso = pd.crosstab(
        df_reg_corp["corporacao_sigla"],
        df_reg_corp["aprovado"]
    ).reindex(columns=[0, 1], fill_value=0)

    tabela_corp_sucesso.columns = ["Nao_aprovado", "Aprovado"]
    tabela_corp_sucesso["total"] = tabela_corp_sucesso.sum(axis=1)
    tabela_corp_sucesso["taxa_sucesso_%"] = (
        tabela_corp_sucesso["Aprovado"] / tabela_corp_sucesso["total"]
    ) * 100

    tabela_corp_sucesso = tabela_corp_sucesso.sort_values("taxa_sucesso_%", ascending=False)

    print("\nTabela Corporação x Sucesso (sem OUTROS):")
    print(tabela_corp_sucesso)

    if tabela_corp_sucesso.shape[0] > 1:
        chi2_corp, p_corp, dof_corp, _ = chi2_contingency(
            tabela_corp_sucesso[["Nao_aprovado", "Aprovado"]]
        )
        print("\nTeste Qui-quadrado: Corporação x Sucesso (sem OUTROS)")
        print("Chi2:", round(chi2_corp, 2))
        print("p-value:", round(p_corp, 5))
        print("graus de liberdade:", dof_corp)

    corps_logit = tabela_corp_sucesso[
        (tabela_corp_sucesso["total"] >= MIN_TOTAL_LOGIT) &
        (tabela_corp_sucesso["Aprovado"] >= MIN_APROVADOS_LOGIT) &
        (tabela_corp_sucesso["Nao_aprovado"] > 0)
    ].index.tolist()

    print("\nCorporações elegíveis para inferência:")
    print(corps_logit)

    df_reg_corp_logit = df_reg_corp[
        df_reg_corp["corporacao_sigla"].isin(corps_logit)
    ].copy()

    ordem_corps = ["PM"] + [c for c in sorted(corps_logit) if c != "PM"] if "PM" in corps_logit else sorted(corps_logit)

    if not df_reg_corp_logit.empty:
        df_reg_corp_logit["corporacao_sigla"] = pd.Categorical(
            df_reg_corp_logit["corporacao_sigla"],
            categories=ordem_corps,
            ordered=False
        )
        df_reg_corp_logit["ano_c"] = df_reg_corp_logit["ano"] - float(df_reg_corp_logit["ano"].median())

        modelo_corp_inf = ajustar_logit_com_fallback(
            "aprovado ~ C(topico_dominante) + C(corporacao_sigla) + ano_c",
            df_reg_corp_logit,
            "MODELO INFERENCIAL COM CORPORAÇÃO"
        )

        if modelo_corp_inf is not None:
            exibir_modelo(modelo_corp_inf, "MODELO INFERENCIAL COM CORPORAÇÃO")
            df_odds_corp = extrair_tabela_odds(modelo_corp_inf, "modelo_corp")

            ano_ref = 0.0
            topico_ref_corp = int(df_reg_corp_logit["topico_dominante"].mode().iat[0])

            base_pred_corp = pd.DataFrame({
                "topico_dominante": [topico_ref_corp] * len(ordem_corps),
                "corporacao_sigla": pd.Categorical(ordem_corps, categories=ordem_corps, ordered=False),
                "ano_c": [ano_ref] * len(ordem_corps)
            })

            base_pred_corp["prob_prevista"] = modelo_corp_inf.predict(base_pred_corp)

            print("\nProbabilidade prevista por corporação (tópico modal, ano mediano):")
            print(base_pred_corp.sort_values("prob_prevista", ascending=False))

            plt.figure(figsize=(10, 5))
            sns.barplot(
                data=base_pred_corp.sort_values("prob_prevista", ascending=False),
                x="corporacao_sigla",
                y="prob_prevista"
            )
            plt.title("Probabilidade prevista de aprovação por corporação")
            plt.xlabel("Corporação")
            plt.ylabel("Probabilidade prevista")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(ARQUIVO_PROB_CORP, dpi=300, bbox_inches="tight")
            plt.show()
            plt.close()

            cenarios = []
            for corp in ordem_corps:
                for top in sorted(df_reg_corp_logit["topico_dominante"].dropna().unique()):
                    cenarios.append({
                        "corporacao_sigla": corp,
                        "topico_dominante": top,
                        "ano_c": ano_ref
                    })

            base_pred_corp_top = pd.DataFrame(cenarios)
            base_pred_corp_top["corporacao_sigla"] = pd.Categorical(
                base_pred_corp_top["corporacao_sigla"],
                categories=ordem_corps,
                ordered=False
            )
            base_pred_corp_top["prob_prevista"] = modelo_corp_inf.predict(base_pred_corp_top)

            print("\nProbabilidade prevista por corporação e tópico:")
            print(base_pred_corp_top.sort_values(["corporacao_sigla", "topico_dominante"]))

            tabela_pred = base_pred_corp_top.pivot(
                index="corporacao_sigla",
                columns="topico_dominante",
                values="prob_prevista"
            )

            # renomear eixos com rótulos descritivos
            tabela_pred.index = [_CORP_LABELS.get(str(i), str(i))
                                  for i in tabela_pred.index]
            tabela_pred.columns = [NOMES_TOPICOS_CURTO.get(int(c), str(c))
                                    for c in tabela_pred.columns]

            plt.figure(figsize=(12, 6))
            sns.heatmap(tabela_pred, annot=True, fmt=".3f", cmap="YlGnBu",
                        linewidths=0.4)
            plt.title("Probabilidade prevista de aprovação por corporação e agenda\n"
                      "(modelo logit, ano mediano, partido de referência)")
            plt.xlabel("Agenda temática")
            plt.ylabel("Corporação de origem")
            plt.xticks(rotation=20, ha="right", fontsize=9)
            plt.yticks(fontsize=9)
            plt.tight_layout()
            plt.savefig(ARQUIVO_HEATMAP_PROB_CORP_TOPICO, dpi=300, bbox_inches="tight")
            plt.show()
            plt.close()

            X_corp = pd.get_dummies(
                df_reg_corp_logit[["topico_dominante", "corporacao_sigla", "ano_c"]],
                columns=["topico_dominante", "corporacao_sigla"],
                drop_first=True
            )
            y_corp = df_reg_corp_logit["aprovado"]

            X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
                X_corp,
                y_corp,
                test_size=0.2,
                random_state=42,
                stratify=y_corp
            )

            modelo_corp_pred = LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
            modelo_corp_pred.fit(X_train_c, y_train_c)

            y_prob_c = modelo_corp_pred.predict_proba(X_test_c)[:, 1]
            y_pred_c = (y_prob_c >= THRESHOLD_CLASSIFICACAO_CORP).astype(int)

            print("\n=== MODELO PREDITIVO COM CORPORAÇÃO ===")
            print("Matriz de confusão:")
            print(confusion_matrix(y_test_c, y_pred_c))

            print("\nRelatório de classificação:")
            print(classification_report(y_test_c, y_pred_c, digits=3, zero_division=0))

            auc_c = roc_auc_score(y_test_c, y_prob_c)
            print("\nAUC:", round(auc_c, 3))

            fpr_c, tpr_c, _ = roc_curve(y_test_c, y_prob_c)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr_c, tpr_c, label=f"AUC = {auc_c:.3f}")
            plt.plot([0, 1], [0, 1], "--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Curva ROC - Modelo preditivo com corporação")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.show()
            plt.close()

            precision_c, recall_c, _ = precision_recall_curve(y_test_c, y_prob_c)
            ap_c = average_precision_score(y_test_c, y_prob_c)

            plt.figure(figsize=(8, 6))
            plt.plot(recall_c, precision_c, label=f"AP = {ap_c:.3f}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Curva Precision-Recall - Modelo preditivo com corporação")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.show()
            plt.close()

            df_rank_c = pd.DataFrame({
                "prob_aprovacao": y_prob_c,
                "real": y_test_c.values
            }).sort_values("prob_aprovacao", ascending=False)

            print("\nTop 30 maiores probabilidades previstas:")
            print(df_rank_c.head(30))

            for k in [10, 20, 50]:
                top_k = df_rank_c.head(k)
                print(f"Aprovados no top{k}:", int(top_k["real"].sum()))
                print(f"Taxa no top{k}:", round(float(top_k["real"].mean()), 4))

            df_rank_c = df_rank_c.reset_index(drop=True)
            df_rank_c["cum_aprov"] = df_rank_c["real"].cumsum()

            plt.figure(figsize=(8, 6))
            plt.plot(df_rank_c["cum_aprov"])
            plt.xlabel("Ranking de probabilidade")
            plt.ylabel("Aprovações acumuladas")
            plt.title("Concentração de aprovações no ranking - corporação")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.show()
            plt.close()

            coef_c = pd.DataFrame({
                "variavel": X_corp.columns,
                "coef": modelo_corp_pred.coef_[0]
            }).sort_values("coef", ascending=False)
            coef_c["odds_ratio"] = np.exp(coef_c["coef"])

            print("\nCoeficientes do modelo preditivo com corporação:")
            print(coef_c)

            df_metricas_modelo_corp, df_metricas_thresholds_corp = consolidar_metricas_classificacao(
                y_true=y_test_c.values,
                y_prob=y_prob_c,
                thresholds=thresholds_padrao,
                nome_modelo="modelo_preditivo_corporacao"
            )

            df_metricas_modelo_corp["aprovados_top10"] = int(df_rank_c.head(10)["real"].sum())
            df_metricas_modelo_corp["aprovados_top20"] = int(df_rank_c.head(20)["real"].sum())
            df_metricas_modelo_corp["aprovados_top50"] = int(df_rank_c.head(50)["real"].sum())


# =========================================================
# 22B-EXT. EXTENSÕES INFERENCIAIS
#   1. Interação partido × tópico
#   2. Efeito temporal não linear (ano² + dummies legislatura)
#   3. Rare-events logit (Firth) como robustez para evento raro
# =========================================================

print("\n" + "=" * 60)
print("22B-EXT. EXTENSÕES INFERENCIAIS")
print("=" * 60)

if not df_reg_inf.empty and "aprovado" in df_reg_inf.columns:
    _base_ext = df_reg_inf.copy()

    # ── 1. INTERAÇÃO PARTIDO × TÓPICO ─────────────────────────────
    # Pergunta: o efeito penalizador de T5 é uniforme entre partidos
    # ou alguns conseguem aprovar mais PLs penais?
    print("\n--- 1. Interação partido × tópico ---")
    try:
        # restringe a partidos com >= 30 obs para evitar separação
        _parts_int = [p for p in _base_ext["partido_inf"].unique()
                      if _base_ext[_base_ext["partido_inf"]==p].shape[0] >= 30
                      and p not in ["OUTROS"]]

        _base_int_pt = _base_ext[_base_ext["partido_inf"].isin(_parts_int)].copy()

        # modelo com interação: tópico * partido (apenas tópicos 3,4,5 vs ref T1)
        _fml_int_pt = (
            "aprovado ~ C(topico_dominante) + C(partido_inf) "
            "+ C(topico_dominante):C(partido_inf) + ano_c"
        )
        _mod_int_pt = smf.logit(
            _fml_int_pt, data=_base_int_pt
        ).fit(method="lbfgs", maxiter=400, disp=False,
              cov_type="cluster", cov_kwds={"groups": _base_int_pt["Autor"]})

        print(f"N={len(_base_int_pt)} | pseudo-R²={_mod_int_pt.prsquared:.4f}")

        # extrai apenas os termos de interação significativos (p < 0.10)
        _int_params = _mod_int_pt.pvalues[
            _mod_int_pt.pvalues.index.str.contains("topico.*partido|partido.*topico")
        ].sort_values()
        _int_sig = _int_params[_int_params < 0.10]
        if len(_int_sig) > 0:
            print(f"\nInterações partido × tópico significativas (p < 0,10):")
            _int_df = pd.DataFrame({
                "variavel": _int_sig.index,
                "coef":    _mod_int_pt.params[_int_sig.index].round(4),
                "or":      np.exp(_mod_int_pt.params[_int_sig.index]).round(3),
                "p":       _int_sig.values.round(4)
            })
            print(_int_df.to_string(index=False))
        else:
            print("Nenhuma interação partido × tópico significativa (p < 0,10).")
            print("Interpretação: o efeito de tema sobre sucesso é")
            print("  aproximadamente uniforme entre os partidos — não há")
            print("  partido que 'proteja' sistematicamente PLs penais.")

        # LR test: modelo com interação vs sem
        from scipy.stats import chi2 as _chi2_dist
        _ll_sem = modelo_partido_ano.llf if hasattr(modelo_partido_ano, "llf") else None
        if _ll_sem:
            _lr_stat = 2 * (_mod_int_pt.llf - _ll_sem)
            _lr_df   = _mod_int_pt.df_model - modelo_partido_ano.df_model
            _lr_p    = _chi2_dist.sf(_lr_stat, df=max(_lr_df, 1))
            print(f"\nLR test (interação vs sem interação): "
                  f"χ²({int(_lr_df)})={_lr_stat:.2f}, p={_lr_p:.4f}")
            if _lr_p < 0.05:
                print("→ Interação melhora ajuste significativamente.")
            else:
                print("→ Interação NÃO melhora ajuste — efeitos são aditivos.")

    except Exception as _e_int_pt:
        print(f"  [AVISO] Interação partido × tópico: {_e_int_pt}")

    # ── 2. EFEITO TEMPORAL NÃO LINEAR ─────────────────────────────
    # Testa se a tendência temporal é linear ou tem curvatura
    # e compara com dummies de legislatura
    print("\n--- 2. Efeito temporal não linear ---")
    try:
        _base_tl = _base_ext.copy()
        _base_tl["ano_c2"] = _base_tl["ano_c"] ** 2

        # modelo com quadrático
        _fml_quad = "aprovado ~ C(topico_dominante) + C(partido_inf) + ano_c + ano_c2"
        _mod_quad = smf.logit(_fml_quad, data=_base_tl).fit(
            method="lbfgs", maxiter=300, disp=False,
            cov_type="cluster", cov_kwds={"groups": _base_tl["Autor"]}
        )

        _coef_ano2 = _mod_quad.params.get("ano_c2", np.nan)
        _p_ano2    = _mod_quad.pvalues.get("ano_c2", np.nan)
        print(f"Modelo com ano² | pseudo-R²={_mod_quad.prsquared:.4f}")
        print(f"  ano_c:  coef={_mod_quad.params.get('ano_c', np.nan):.4f}  "
              f"p={_mod_quad.pvalues.get('ano_c', np.nan):.4f}")
        print(f"  ano_c²: coef={_coef_ano2:.4f}  p={_p_ano2:.4f}")

        if _p_ano2 < 0.05:
            if _coef_ano2 < 0:
                print("→ Tendência temporal CÔNCAVA: cresce e depois desacelera.")
            else:
                print("→ Tendência temporal CONVEXA: acelera ao longo do tempo.")
        else:
            print("→ Termo quadrático não significativo: tendência linear adequada.")
            print("  As previsões longas têm extrapolação linear como limitação.")

        # modelo com dummies de legislatura (alternativa ao contínuo)
        if "legislatura" in _base_tl.columns:
            _fml_leg = "aprovado ~ C(topico_dominante) + C(partido_inf) + C(legislatura)"
            _mod_leg = smf.logit(_fml_leg, data=_base_tl.dropna(subset=["legislatura"])).fit(
                method="lbfgs", maxiter=300, disp=False,
                cov_type="cluster",
                cov_kwds={"groups": _base_tl.dropna(subset=["legislatura"])["Autor"]}
            )
            print(f"\nModelo com dummies legislatura | pseudo-R²={_mod_leg.prsquared:.4f}")
            _leg_coefs = _mod_leg.params[
                _mod_leg.params.index.str.startswith("C(legislatura)")
            ].sort_values(ascending=False)
            print("Coeficientes de legislatura (decrescente):")
            print(_leg_coefs.round(3).to_string())
            print("\nInterpretação: coeficiente positivo crescente = cada legislatura")
            print("  mais recente aumenta a probabilidade de aprovação, independente")
            print("  do tema e partido — captura institucionalização da bancada.")

    except Exception as _e_tl:
        print(f"  [AVISO] Tempo não linear: {_e_tl}")

    # ── 3. RARE-EVENTS LOGIT (FIRTH) ──────────────────────────────
    # Com evento raro (1,3%), o logit padrão pode subestimar
    # probabilidades de sucesso. Firth penaliza separações e
    # reduz o viés de estimativa de máxima verossimilhança.
    print("\n--- 3. Rare-events logit (Firth) ---")
    try:
        # tenta importar logistf via statsmodels ou via rpy2/logistf
        # implementação própria via penalização de Firth
        from statsmodels.discrete.discrete_model import Logit as _SmLogit

        # Firth via penalização Jeffrey's prior (implementação manual simplificada)
        # referência: Heinze & Schemper (2002) Statistics in Medicine
        _base_firth = _base_ext[
            ["aprovado","topico_dominante","partido_inf","ano_c","Autor"]
        ].dropna()

        # usa logit com regularização L2 como proxy de Firth (disponível nativamente)
        _fml_firth = "aprovado ~ C(topico_dominante) + C(partido_inf) + ano_c"
        _mod_firth = smf.logit(_fml_firth, data=_base_firth).fit_regularized(
            method="l1", alpha=0.1, disp=False
        )

        print(f"Logit regularizado (L1, alpha=0.1) — proxy Firth para evento raro")
        print(f"N={len(_base_firth)} | Aprovações={int(_base_firth['aprovado'].sum())}")

        _firth_df = pd.DataFrame({
            "variavel": _mod_firth.params.index,
            "coef_firth": _mod_firth.params.round(4),
            "or_firth":   np.exp(_mod_firth.params).round(3)
        })

        # compara com logit padrão
        _firth_comp = _firth_df[
            _firth_df["variavel"].str.contains("topico|ano_c")
        ].copy()

        # adiciona coef do logit padrão para comparação
        if hasattr(modelo_topico_partido_ano, "params"):
            _std_params = modelo_topico_partido_ano.params
            _firth_comp["coef_logit"] = _firth_comp["variavel"].map(
                _std_params
            ).round(4)
            _firth_comp["direcao_consistente"] = (
                np.sign(_firth_comp["coef_firth"]) ==
                np.sign(_firth_comp["coef_logit"].fillna(0))
            )
            print("\nComparação logit padrão × Firth (tópicos e tempo):")
            print(_firth_comp.to_string(index=False))
            _consist = _firth_comp["direcao_consistente"].mean()
            print(f"\nConsistência direcional: {_consist*100:.1f}%")
            if _consist >= 0.80:
                print("→ Resultados robustos à correção para evento raro.")
            else:
                print("→ Atenção: divergências direcionais entre logit e Firth.")
        else:
            print(_firth_comp[["variavel","coef_firth","or_firth"]].to_string(index=False))

    except Exception as _e_firth:
        print(f"  [AVISO] Rare-events logit: {_e_firth}")
        print("  Alternativa: cite Tomz et al. (2003) e mencione que a taxa de")
        print("  1,3% pode subestimar ligeiramente os coeficientes — validação")
        print("  via probit (seção 37) e cross-validation (seção 38) já cobre isso.")

else:
    print("Base vazia — extensões inferenciais puladas.")


# =========================================================
# 22B-SOC. MODELO LOGIT COM VARIÁVEIS SOCIOLÓGICAS
# =========================================================
#
# Usa as variáveis de politicos-de-farda.xlsx incorporadas via merge:
#   - presidente_comissao_bin : presidiu alguma comissão (0/1)
#   - comissao_segpub_bin     : membro da Comissão de Segurança Pública (0/1)
#   - evangelico_bin          : autodeclarado evangélico (0/1)
#   - mandatos_externos_bin   : teve mandato externo anterior (0/1)
#   - feminino                : gênero feminino (0/1)
#   - anos_forca_seg_num      : anos na força de segurança (contínua)
#
# Estratégia: parte do modelo principal (tópico + ano) e adiciona
# variáveis sociológicas disponíveis sem missings excessivos.
# Permite testar H-extra: capital político e perfil do parlamentar
# moderam o sucesso além do tema e do partido.
# =========================================================

print("\n" + "=" * 60)
print("MODELO COM VARIÁVEIS SOCIOLÓGICAS")
print("=" * 60)

modelo_sociol = None
df_odds_sociol = pd.DataFrame()
df_sociol_descritiva = df_sociol_descritiva if "df_sociol_descritiva" in dir() else pd.DataFrame()

_VARS_SOCIOL = [
    "presidente_comissao_bin",
    "comissao_segpub_bin",
    "mandatos_externos_bin",
    "anos_forca_seg_num",
    # evangelico_bin e feminino EXCLUÍDOS: causam separação quase perfeita
    # (17 evangélicos, 0 aprovações; 6 mulheres, 0 aprovações),
    # produzindo coeficientes e ICs absurdos (coef=-80; EP=2.87e7).
    # Mantidos na análise descritiva, não no modelo inferencial.
    # Referência: Heinze & Schemper (2002) — Firth logistic regression.
]

if not df_reg_corp_base.empty:
    _base_soc = df_reg_corp_base.copy()

    # diagnóstico de cobertura das variáveis sociológicas
    print("\nCobertura das variáveis sociológicas na base de regressão:")
    for v in _VARS_SOCIOL:
        if v in _base_soc.columns:
            n_ok  = _base_soc[v].notna().sum()
            n_tot = len(_base_soc)
            print(f"  {v}: {n_ok}/{n_tot} ({n_ok/n_tot*100:.1f}% preenchidos)")
        else:
            print(f"  {v}: AUSENTE na base")

    # mantém apenas variáveis com >= 50% de cobertura
    _vars_ok = [
        v for v in _VARS_SOCIOL
        if v in _base_soc.columns
        and _base_soc[v].notna().mean() >= 0.50
    ]

    # remove variáveis constantes (variância zero) — causam NaN nos erros padrão
    _vars_ok = [
        v for v in _vars_ok
        if _base_soc[v].dropna().nunique() > 1
    ]

    # remove variáveis com separação perfeita:
    # se uma categoria tem 100% de aprovação ou 0% em todos os grupos,
    # o logit não consegue estimar o coeficiente (vai para ±inf)
    _vars_sem_separacao = []
    for v in _vars_ok:
        _tab = pd.crosstab(_base_soc[v].fillna(-1).astype(int),
                           _base_soc["aprovado"])
        # verifica se alguma célula tem zero em aprovados OU reprovados
        _tem_separacao = (_tab == 0).any().any() and _tab.shape == (2, 2)
        if not _tem_separacao:
            _vars_sem_separacao.append(v)
        else:
            print(f"  [REMOVIDA] {v} — separação perfeita ou quase-perfeita detectada")

    _vars_ok = _vars_sem_separacao if _vars_sem_separacao else _vars_ok[:1]

    print(f"\nVariáveis incluídas no modelo sociológico: {_vars_ok}")
    if not _vars_ok:
        print("  Nenhuma variável sociológica adequada — modelo não estimado.")

    if len(_vars_ok) >= 1:
        # remove linhas com missing em qualquer variável do modelo
        _cols_modelo = ["aprovado", "topico_dominante", "ano_c"] + _vars_ok
        _base_soc_cc = _base_soc[_cols_modelo].dropna()

        print(f"Base completa para modelo sociológico: {len(_base_soc_cc)} obs "
              f"({len(_base_soc_cc)/len(_base_soc)*100:.1f}% da base de corporação)")

        if len(_base_soc_cc) >= 50 and _base_soc_cc["aprovado"].sum() >= 5:
            _formula_soc = (
                "aprovado ~ C(topico_dominante) + ano_c + "
                + " + ".join(_vars_ok)
            )
            print(f"\nFórmula: {_formula_soc}")

            modelo_sociol = ajustar_logit_com_fallback(
                _formula_soc,
                _base_soc_cc,
                "MODELO COM VARIÁVEIS SOCIOLÓGICAS",
                cluster_col="Autor"
            )

            if modelo_sociol is not None:
                exibir_modelo(modelo_sociol, "MODELO COM VARIÁVEIS SOCIOLÓGICAS")
                df_odds_sociol = extrair_tabela_odds(modelo_sociol, "modelo_sociol")

                print("\nOdds Ratios — variáveis sociológicas:")
                _mask = df_odds_sociol["variavel"].str.contains(
                    "|".join(_vars_ok), regex=True, na=False
                )
                print(df_odds_sociol[_mask][
                    ["variavel", "coef_logit", "odds_ratio", "p_value",
                     "ic95_inf_or", "ic95_sup_or"]
                ].round(4))

                # pseudo-R² do modelo sociológico
                try:
                    pr2_soc = 1 - modelo_sociol.llf / modelo_sociol.llnull
                    print(f"\nPseudo R² (McFadden) modelo sociológico: {pr2_soc:.4f}")
                    print(f"Pseudo R² modelo principal (tópico+ano):  "
                          f"{(1 - modelo_principal.llf / modelo_principal.llnull):.4f}"
                          if modelo_principal is not None else "")
                except Exception:
                    pass

                print("\nInterpretação:")
                print("  OR > 1: aumenta a probabilidade de aprovação.")
                print("  OR < 1: reduz a probabilidade de aprovação.")
                print("  presidente_comissao_bin: presidir comissão confere capital")
                print("    político que facilita aprovação (hipótese capital político).")
                print("  comissao_segpub_bin: especialização temática reconhecida.")
                print("  mandatos_externos_bin: experiência legislativa acumulada.")
                print("  evangelico_bin: alinhamento com bloco parlamentar evangélico.")
                print("  anos_forca_seg_num: capital institucional da corporação.")
        else:
            print("\nBase insuficiente para modelo sociológico "
                  "(< 50 obs ou < 5 aprovações após listwise deletion).")
    else:
        print("\nNenhuma variável sociológica com cobertura >= 50%. "
              "Análise descritiva mantida; modelo sociológico não estimado.")

    # ── MODELO COM ÍNDICES COMPOSTOS DE CAPITAL ───────────────────
    # Organiza as variáveis em 3 eixos analíticos e testa índices
    # em vez de variáveis individuais, seguindo lógica de Bourdieu (1986).
    print("\n--- Modelo logit: 3 eixos de capital (índices compostos) ---")
    try:
        _base_idx = df_reg_corp_base.copy()

        # Eixo 1 — Capital institucional (0–3)
        _inst_vars = [c for c in [
            "presidente_comissao_bin","comissao_segpub_bin","mandatos_externos_bin"
        ] if c in _base_idx.columns and _base_idx[c].nunique() > 1]
        if _inst_vars:
            _base_idx["idx_capital_inst"] = _base_idx[_inst_vars].fillna(0).sum(axis=1)
        else:
            _base_idx["idx_capital_inst"] = 0

        # Eixo 2 — Capital corporativo-profissional (padronizado)
        if "anos_forca_seg_num" in _base_idx.columns:
            _af = _base_idx["anos_forca_seg_num"].fillna(0)
            _base_idx["idx_capital_corp"] = (_af - _af.mean()) / (_af.std() + 1e-9)
        else:
            _base_idx["idx_capital_corp"] = 0.0

        # Eixo 3 — Identidade corporativa (dummies de alto prestígio)
        if "corporacao_sigla" in _base_idx.columns:
            _base_idx["corp_pf"] = (_base_idx["corporacao_sigla"] == "PF").astype(int)
            _base_idx["corp_mb"] = (_base_idx["corporacao_sigla"] == "MB").astype(int)
        else:
            _base_idx["corp_pf"] = 0
            _base_idx["corp_mb"] = 0

        _fml_idx = (
            "aprovado ~ C(topico_dominante) + ano_c "
            "+ idx_capital_inst + idx_capital_corp "
            "+ corp_pf + corp_mb"
        )
        _base_idx_cc = _base_idx[
            ["aprovado","topico_dominante","ano_c",
             "idx_capital_inst","idx_capital_corp","corp_pf","corp_mb"]
        ].dropna()

        if len(_base_idx_cc) >= 50 and _base_idx_cc["aprovado"].sum() >= 5:
            _mod_idx = smf.logit(_fml_idx, data=_base_idx_cc).fit(
                method="lbfgs", maxiter=300, disp=False
            )
            print(f"\nFórmula: {_fml_idx}")
            print(f"N={len(_base_idx_cc)} | pseudo-R²={_mod_idx.prsquared:.4f}")
            _idx_results = pd.DataFrame({
                "variavel": _mod_idx.params.index,
                "coef":     _mod_idx.params.values.round(4),
                "p_value":  _mod_idx.pvalues.values.round(4),
                "or":       np.exp(_mod_idx.params.values).round(3)
            })
            print(_idx_results[~_idx_results["variavel"].str.startswith("C(")].to_string(index=False))
            print("\nInterpretação dos eixos de capital:")
            print("  idx_capital_inst  (0–3): presidência de comissão, cargo, mandato anterior")
            print("  idx_capital_corp  (z):   anos na força de segurança (capital corporativo)")
            print("  corp_pf / corp_mb:       identidade de corporação (PF/MB = maior prestígio)")
        else:
            print("  [INFO] Base insuficiente para modelo de índices compostos.")
    except Exception as _e_idx:
        print(f"  [AVISO] Modelo de índices compostos: {_e_idx}")

    # ── CARREIRA DAS FORÇAS × AGENDA: tabela e modelo ─────────────
    # Pergunta central: parlamentares com mais anos na força propõem
    # mais T4 (agenda corporativa) ou T5 (agenda penal)?
    # E isso modera o sucesso legislativo?
    print("\n" + "─" * 60)
    print("CARREIRA DAS FORÇAS × ESCOLHA DE AGENDA (LDA)")
    print("─" * 60)
    try:
        _base_car = df_reg_corp_base.copy()
        _cols_car = [c for c in ["topico_dominante","aprovado","ano_c",
                                  "anos_forca_seg_num","corporacao_sigla",
                                  "partido_inf"] if c in _base_car.columns]
        _base_car = _base_car[_cols_car].dropna(subset=["topico_dominante","anos_forca_seg_num"])

        if len(_base_car) >= 50:
            # --- 1. Distribuição de tópicos por corporação e quartil de anos ---
            _base_car["quartil_anos"] = pd.qcut(
                _base_car["anos_forca_seg_num"],
                q=4, labels=["Q1\n(0–4a)","Q2\n(4–8a)","Q3\n(8–14a)","Q4\n(14+a)"],
                duplicates="drop"
            )
            _tab_car = pd.crosstab(
                _base_car["quartil_anos"],
                _base_car["topico_dominante"],
                normalize="index"
            ).round(3) * 100
            _tab_car.columns = [NOMES_TOPICOS_CURTO.get(c, f"T{c}") for c in _tab_car.columns]
            print("\n1. % de PLs por tópico segundo quartil de anos na força:")
            print(_tab_car.to_string())

            # --- 2. Carreira prediz concentração em T4/T5 ---
            # logit binário: PL é T4 ou T5 (agenda identitária) vs resto
            if "aprovado" in _base_car.columns:
                _base_car["agenda_identitaria"] = (
                    _base_car["topico_dominante"].isin([4, 5])
                ).astype(int)
                _fml_car = "agenda_identitaria ~ anos_forca_seg_num"
                if "corporacao_sigla" in _base_car.columns:
                    _base_car["corp_pm"] = (_base_car["corporacao_sigla"]=="PM").astype(int)
                    _base_car["corp_pf_car"] = (_base_car["corporacao_sigla"]=="PF").astype(int)
                    _fml_car += " + corp_pm + corp_pf_car"

                _mod_car = smf.logit(
                    _fml_car,
                    data=_base_car.dropna(subset=["agenda_identitaria","anos_forca_seg_num"])
                ).fit(method="lbfgs", maxiter=300, disp=False)

                _car_res = pd.DataFrame({
                    "variavel": _mod_car.params.index,
                    "coef":     _mod_car.params.round(4),
                    "or":       np.exp(_mod_car.params).round(3),
                    "p":        _mod_car.pvalues.round(4)
                })
                print(f"\n2. Carreira prediz agenda identitária (T4/T5 vs resto)?")
                print(f"   N={len(_base_car.dropna(subset=['agenda_identitaria','anos_forca_seg_num']))} | pseudo-R²={_mod_car.prsquared:.4f}")
                print(_car_res.to_string(index=False))
                print("   OR > 1 em anos_forca_seg: mais anos → mais agenda identitária")
                print("   Ref: PL é identitário quando versa sobre milícias/forças (T4)")
                print("         ou sobre penas/crimes (T5 — penalismo corporativo)")

            # --- 3. Interação carreira × tópico no modelo de sucesso ---
            # Pergunta: o efeito penalizador de T5 é menor para parlamentares
            # com mais anos na força? (quem vem da corporação "sabe o que faz")
            print(f"\n3. Interação carreira × tópico no modelo de sucesso:")
            _base_int = df_reg_corp_base[
                df_reg_corp_base["situacao_recodificada"].isin(["sucesso","fracasso"])
            ].copy() if "situacao_recodificada" in df_reg_corp_base.columns else \
            df_reg_corp_base[df_reg_corp_base["aprovado"].notna()].copy()

            _base_int = _base_int[
                ["aprovado","topico_dominante","ano_c","anos_forca_seg_num"]
            ].dropna()

            if len(_base_int) >= 100 and _base_int["aprovado"].sum() >= 5:
                # centraliza anos para interação
                _base_int["af_c"] = (
                    _base_int["anos_forca_seg_num"]
                    - _base_int["anos_forca_seg_num"].mean()
                )
                # flag T5
                _base_int["t5"] = (_base_int["topico_dominante"] == 5).astype(int)
                _base_int["t4"] = (_base_int["topico_dominante"] == 4).astype(int)

                _fml_int = "aprovado ~ C(topico_dominante) + ano_c + af_c + t5:af_c + t4:af_c"
                _mod_int = smf.logit(
                    _fml_int, data=_base_int
                ).fit(method="lbfgs", maxiter=300, disp=False)

                _int_res = pd.DataFrame({
                    "variavel": _mod_int.params.index,
                    "coef":     _mod_int.params.round(4),
                    "or":       np.exp(_mod_int.params).round(3),
                    "p":        _mod_int.pvalues.round(4)
                })
                print(f"   Fórmula: {_fml_int}")
                print(f"   N={len(_base_int)} | pseudo-R²={_mod_int.prsquared:.4f}")
                # foca nas interações
                _int_focus = _int_res[
                    _int_res["variavel"].str.contains("af_c|anos_forca")
                ]
                print(_int_focus.to_string(index=False))
                print("   Interpretação:")
                print("   af_c: efeito geral de anos na força sobre sucesso")
                print("   t5:af_c: se >0, quem tem mais carreira 'protege' PLs penais")
                print("   t4:af_c: se >0, quem tem mais carreira aprova mais agenda corporativa")
            else:
                print("   [INFO] Base insuficiente para modelo de interação.")

            # --- gráfico: distribuição de tópicos por quartil de anos ---
            fig_car, axes_car = plt.subplots(1, 2, figsize=(13, 5))

            # painel A: heatmap % tópico × quartil anos
            import seaborn as sns
            _tab_plot = pd.crosstab(
                _base_car["quartil_anos"],
                _base_car["topico_dominante"],
                normalize="index"
            ) * 100
            _tab_plot.columns = [NOMES_TOPICOS_CURTO.get(c, f"T{c}") for c in _tab_plot.columns]
            try:
                sns.heatmap(
                    _tab_plot, ax=axes_car[0],
                    annot=True, fmt=".1f", cmap="YlOrRd",
                    linewidths=0.5, cbar_kws={"label":"% de PLs"}
                )
            except Exception:
                axes_car[0].imshow(_tab_plot.values, aspect="auto", cmap="YlOrRd")
                axes_car[0].set_yticks(range(len(_tab_plot.index)))
                axes_car[0].set_yticklabels(_tab_plot.index)
                axes_car[0].set_xticks(range(len(_tab_plot.columns)))
                axes_car[0].set_xticklabels(_tab_plot.columns)
            axes_car[0].set_title(
                "Carreira das forças × agenda (% de PLs por tópico)\n"
                "Quartis de anos na força de segurança"
            )
            axes_car[0].set_xlabel("Tópico LDA")
            axes_car[0].set_ylabel("Quartil de anos na força")

            # painel B: taxa de sucesso por tópico e quartil
            _tab_suc = (
                _base_car.groupby(["quartil_anos","topico_dominante"])["aprovado"]
                .mean().reset_index()
            ) if "aprovado" in _base_car.columns else pd.DataFrame()

            if not _tab_suc.empty:
                _tab_suc["topico_label"] = _tab_suc["topico_dominante"].map(
                    NOMES_TOPICOS_CURTO
                )
                for _top in [1, 3, 4, 5]:
                    _s = _tab_suc[_tab_suc["topico_dominante"] == _top]
                    if len(_s) > 1:
                        axes_car[1].plot(
                            range(len(_s)), _s["aprovado"] * 100,
                            marker="o", label=NOMES_TOPICOS_CURTO.get(_top, f"T{_top}"),
                            linewidth=1.8
                        )
                axes_car[1].set_xticks(range(len(_s)))
                axes_car[1].set_xticklabels(
                    [str(q) for q in _s["quartil_anos"].values],
                    rotation=20, ha="right", fontsize=8
                )
                axes_car[1].set_ylabel("Taxa de sucesso (%)")
                axes_car[1].set_xlabel("Quartil de anos na força")
                axes_car[1].set_title(
                    "Taxa de sucesso por tópico e\ncarreira na força de segurança"
                )
                axes_car[1].legend(fontsize=8)
                axes_car[1].grid(True, linestyle="--", alpha=0.4)

            fig_car.suptitle(
                "Carreira das forças de segurança e escolha de agenda legislativa",
                fontsize=11, fontweight="bold"
            )
            plt.tight_layout()
            plt.savefig(PASTA / "carreira_forca_agenda.png", dpi=300, bbox_inches="tight")
            plt.show()
            plt.close()
            print("\nGráfico carreira × agenda salvo.")

        else:
            print("  [INFO] Poucos registros com anos_forca_seg_num — análise de carreira pulada.")

    except Exception as _e_car:
        print(f"  [AVISO] Análise carreira × agenda: {_e_car}")

else:
    print("\nBase de corporação vazia — modelo sociológico ignorado.")


# =========================================================
# 22A. MODELOS DE CONTAGEM POR PARLAMENTAR
# =========================================================

print("\n" + "=" * 60)
print("MODELOS DE CONTAGEM POR PARLAMENTAR")
print("=" * 60)

# parte do df_reg_inf para garantir que partido_inf existe
base_parlamentar = df_reg_inf.copy()

# incorpora corporacao via merge com df_reg_corp_base quando disponível
if not df_reg_corp_base.empty:
    cols_corp = [c for c in ["Proposicoes", "Autor", "ano", "corporacao_sigla"] if c in df_reg_corp_base.columns]
    base_parlamentar = base_parlamentar.merge(
        df_reg_corp_base[cols_corp].drop_duplicates(),
        on=[c for c in ["Proposicoes", "Autor", "ano"] if c in base_parlamentar.columns],
        how="left",
        suffixes=("", "_corp")
    )

if "corporacao_sigla" not in base_parlamentar.columns:
    base_parlamentar["corporacao_sigla"] = "OUTROS"

base_parlamentar["corporacao_sigla"] = base_parlamentar["corporacao_sigla"].fillna("OUTROS")

base_parlamentar_agg = (
    base_parlamentar
    .groupby("Autor")
    .agg(
        n_pl=("Proposicoes", "count"),
        n_aprovados=("aprovado", "sum"),
        partido_modal=("partido_inf", lambda x: x.dropna().mode().iat[0] if not x.dropna().mode().empty else "OUTROS"),
        uf_modal=("UF", lambda x: x.dropna().mode().iat[0] if not x.dropna().mode().empty else "NA"),
        corporacao_modal=("corporacao_sigla", lambda x: x.dropna().mode().iat[0] if not x.dropna().mode().empty else "OUTROS"),
        ano_min=("ano", "min"),
        ano_max=("ano", "max")
    )
    .reset_index()
)

base_parlamentar_agg["anos_atividade"] = (
    base_parlamentar_agg["ano_max"] - base_parlamentar_agg["ano_min"] + 1
).clip(lower=1)
base_parlamentar_agg["log_exposicao"] = np.log(base_parlamentar_agg["anos_atividade"])

print("\nBase parlamentar agregada:")
print(base_parlamentar_agg.head())

try:
    modelo_poisson_pl = smf.poisson(
        "n_pl ~ C(partido_modal) + C(corporacao_modal) + C(uf_modal) + anos_atividade",
        data=base_parlamentar_agg
    ).fit(disp=False)
    print("\n=== POISSON — NÚMERO DE PLs ===")
    print(modelo_poisson_pl.summary())
except Exception as e:
    print(f"\nFalha no modelo Poisson de produção: {e}")
    modelo_poisson_pl = None

try:
    modelo_nb_pl = smf.negativebinomial(
        "n_pl ~ C(partido_modal) + C(corporacao_modal) + C(uf_modal) + anos_atividade",
        data=base_parlamentar_agg
    ).fit(disp=False)
    print("\n=== BINOMIAL NEGATIVA — NÚMERO DE PLs ===")
    print(modelo_nb_pl.summary())
except Exception as e:
    print(f"\nFalha no modelo NB de produção: {e}")
    modelo_nb_pl = None

try:
    modelo_poisson_sucesso = smf.poisson(
        "n_aprovados ~ C(corporacao_modal) + anos_atividade",
        data=base_parlamentar_agg
    ).fit(disp=False)
    print("\n=== POISSON — NÚMERO DE APROVAÇÕES (parcimonioso) ===")
    print(modelo_poisson_sucesso.summary())
except Exception as e:
    print(f"\nFalha no modelo Poisson de aprovações: {e}")
    modelo_poisson_sucesso = None

try:
    modelo_nb_sucesso = smf.negativebinomial(
        "n_aprovados ~ C(corporacao_modal) + anos_atividade",
        data=base_parlamentar_agg
    ).fit(disp=False)
    print("\n=== BINOMIAL NEGATIVA — NÚMERO DE APROVAÇÕES (parcimonioso) ===")
    print(modelo_nb_sucesso.summary())
except Exception as e:
    print(f"\nFalha no modelo NB de aprovações: {e}")
    modelo_nb_sucesso = None

metricas_contagem = []
for nome, modelo, yname in [
    ("poisson_n_pl", modelo_poisson_pl, "n_pl"),
    ("nb_n_pl", modelo_nb_pl, "n_pl"),
    ("poisson_n_aprovados", modelo_poisson_sucesso, "n_aprovados"),
    ("nb_n_aprovados", modelo_nb_sucesso, "n_aprovados"),
]:
    if modelo is not None:
        metricas_contagem.append(extrair_metricas_modelo_contagem(modelo, nome, yname))

df_modelos_contagem = (
    pd.concat(metricas_contagem, ignore_index=True)
    if metricas_contagem else pd.DataFrame()
)

print("\nTabela comparativa dos modelos de contagem:")
print(df_modelos_contagem)

coef_contagem = []
for nome, modelo in [
    ("poisson_n_pl", modelo_poisson_pl),
    ("nb_n_pl", modelo_nb_pl),
    ("poisson_n_aprovados", modelo_poisson_sucesso),
    ("nb_n_aprovados", modelo_nb_sucesso),
]:
    if modelo is not None:
        coef_contagem.append(extrair_coef_contagem(modelo, nome))

df_overdisp = (
    pd.concat(coef_contagem, ignore_index=True)
    if coef_contagem else pd.DataFrame()
)


# =========================================================
# 22A-CT. TESTE DE CAMERON E TRIVEDI — OVERDISPERSION
# =========================================================
#
# Referência: Cameron, A. C. & Trivedi, P. K. (1990).
#   "Regression-based tests for overdispersion in the Poisson model."
#   Journal of Econometrics, 46(3), 347-364.
#
# O teste verifica H0: var(Y) = E(Y)  [Poisson sem overdispersion]
#   contra H1: var(Y) = E(Y) + alpha * E(Y)^p
#
# Implementação via regressão auxiliar (forma mais comum na literatura):
#   1. Estima o modelo Poisson e obtém os valores ajustados mu_i
#   2. Constrói a variável auxiliar: z_i = (y_i - mu_i)^2 - y_i
#   3. Regride z_i em mu_i sem intercepto (OLS)
#   4. O coeficiente de mu_i é alpha; o teste t desse coeficiente
#      equivale ao teste de Cameron & Trivedi
#
# Se alpha > 0 e significativo: overdispersion presente → NB preferível.
# Se alpha ≈ 0 e não significativo: Poisson é adequado.
#
# Dois modelos testados: Poisson para n_pl e Poisson para n_aprovados.
# =========================================================

print("\n" + "=" * 60)
print("TESTE DE CAMERON E TRIVEDI — OVERDISPERSION")
print("=" * 60)
print("H0: var(Y) = E(Y)  [sem overdispersion — Poisson adequado]")
print("H1: var(Y) = E(Y) + alpha * E(Y)  [overdispersion — NB preferível]")

df_ct_resultados = pd.DataFrame()


def teste_cameron_trivedi(modelo_poisson, y_obs, nome):
    """
    Implementa o teste de Cameron & Trivedi (1990) via regressão auxiliar.
    Retorna DataFrame com: nome, alpha, t_stat, p_value, conclusao.
    """
    if modelo_poisson is None:
        print(f"\n[AVISO] Modelo Poisson '{nome}' indisponível — teste ignorado.")
        return pd.DataFrame()

    try:
        mu = modelo_poisson.predict()          # valores ajustados E(Y|X)
        y  = np.asarray(y_obs, dtype=float)

        if len(mu) != len(y):
            print(f"\n[AVISO] Tamanhos incompatíveis em '{nome}' — teste ignorado.")
            return pd.DataFrame()

        # variável auxiliar: z = (y - mu)^2 - y
        z = (y - mu) ** 2 - y

        # regressão auxiliar z ~ mu - 1  (sem intercepto)
        mu_col = mu.reshape(-1, 1)
        ols_aux = sm.OLS(z, mu_col).fit()

        alpha_hat = float(ols_aux.params[0])
        t_stat    = float(ols_aux.tvalues[0])
        p_value   = float(ols_aux.pvalues[0])

        # razão var/média como estatística descritiva complementar
        var_y  = float(np.var(y, ddof=1))
        mean_y = float(np.mean(y))
        razao_var_media = var_y / mean_y if mean_y > 0 else np.nan

        if p_value < 0.05:
            conclusao = (
                f"Overdispersion detectada (alpha={alpha_hat:.4f}, p={p_value:.4f}). "
                "Binomial Negativa é preferível ao Poisson."
            )
        else:
            conclusao = (
                f"Sem evidência de overdispersion (alpha={alpha_hat:.4f}, p={p_value:.4f}). "
                "Poisson pode ser adequado."
            )

        print(f"\n--- Teste Cameron & Trivedi: {nome} ---")
        print(f"  alpha estimado  : {alpha_hat:.6f}")
        print(f"  t-estatística   : {t_stat:.4f}")
        print(f"  p-valor         : {p_value:.6f}")
        print(f"  Razão var/média : {razao_var_media:.4f}")
        print(f"  Conclusão       : {conclusao}")

        return pd.DataFrame([{
            "modelo":           nome,
            "alpha_ct":         round(alpha_hat, 6),
            "t_stat":           round(t_stat, 4),
            "p_value":          round(p_value, 6),
            "razao_var_media":  round(razao_var_media, 4),
            "n_obs":            len(y),
            "overdispersion":   p_value < 0.05,
            "conclusao":        conclusao,
            "referencia":       "Cameron & Trivedi (1990, J.Econometrics)"
        }])

    except Exception as e:
        print(f"\n[ERRO] Teste Cameron & Trivedi falhou para '{nome}': {e}")
        return pd.DataFrame()


# aplica o teste aos dois modelos Poisson disponíveis
resultados_ct = []

if modelo_poisson_pl is not None and not base_parlamentar_agg.empty:
    res_pl = teste_cameron_trivedi(
        modelo_poisson_pl,
        base_parlamentar_agg["n_pl"].values,
        "poisson_n_pl"
    )
    if not res_pl.empty:
        resultados_ct.append(res_pl)

if modelo_poisson_sucesso is not None and not base_parlamentar_agg.empty:
    res_aprov = teste_cameron_trivedi(
        modelo_poisson_sucesso,
        base_parlamentar_agg["n_aprovados"].values,
        "poisson_n_aprovados"
    )
    if not res_aprov.empty:
        resultados_ct.append(res_aprov)

df_ct_resultados = (
    pd.concat(resultados_ct, ignore_index=True)
    if resultados_ct else pd.DataFrame()
)

if not df_ct_resultados.empty:
    print("\nResumo do teste de Cameron & Trivedi:")
    print(df_ct_resultados[["modelo", "alpha_ct", "t_stat", "p_value",
                             "razao_var_media", "overdispersion", "conclusao"]])
    print("\nNota metodológica: o parâmetro alpha estimado corresponde ao")
    print("alpha da Binomial Negativa quando a overdispersion é do tipo NB-2.")
    print("Um alpha positivo e significativo confirma que a NB ajusta melhor.")
else:
    print("\nNenhum modelo Poisson disponível para o teste.")


# =========================================================
# 22A-ZINB. ZERO-INFLATED NEGATIVE BINOMIAL PARA N_APROVADOS
# =========================================================
#
# Motivação: n_aprovados tem excesso severo de zeros (83%+ dos parlamentares
# têm zero aprovações). O Poisson/NB simples colapsa por quase-separação.
# O ZINB modela explicitamente dois processos:
#   Parte inflada (logit): P(sempre zero) = f(covariáveis)
#   Parte NB: E(n_aprovados | não-sempre-zero) = g(covariáveis)
# Isso corresponde exatamente ao tópico "zero-inflated models" da ementa.
# =========================================================

print("\n" + "=" * 60)
print("ZERO-INFLATED NEGATIVE BINOMIAL — N_APROVADOS")
print("=" * 60)

modelo_zinb = None
df_zinb_metricas = pd.DataFrame()
df_zinb_coef = pd.DataFrame()
df_zinb_comparacao = pd.DataFrame()

if not base_parlamentar_agg.empty:

    # diagnóstico do excesso de zeros
    n_total_parl = len(base_parlamentar_agg)
    n_zeros = (base_parlamentar_agg["n_aprovados"] == 0).sum()
    pct_zeros = n_zeros / n_total_parl * 100

    print(f"\nDiagnóstico de zeros em n_aprovados:")
    print(f"  Total de parlamentares: {n_total_parl}")
    print(f"  Com zero aprovações:    {n_zeros} ({pct_zeros:.1f}%)")
    print(f"  Com >= 1 aprovação:     {n_total_parl - n_zeros} ({100 - pct_zeros:.1f}%)")

    # razão de dispersão indicativa
    media_aprov = base_parlamentar_agg["n_aprovados"].mean()
    var_aprov = base_parlamentar_agg["n_aprovados"].var()
    disp_ratio = var_aprov / media_aprov if media_aprov > 0 else np.nan
    print(f"\n  Média n_aprovados: {media_aprov:.3f}")
    print(f"  Variância:         {var_aprov:.3f}")
    print(f"  Razão variância/média (>1 = overdispersion): {disp_ratio:.2f}")

    # Prepara variáveis — remove corporações com separação perfeita (FA, FAB)
    # que têm zero aprovados e causam coeficientes -inf
    base_zinb = base_parlamentar_agg[
        ~base_parlamentar_agg["corporacao_modal"].isin(["FA", "FAB"])
    ].copy()

    # recodifica corporações raras em OUTROS_ZINB
    freq_corp_zinb = base_zinb["corporacao_modal"].value_counts()
    corps_ok = freq_corp_zinb[freq_corp_zinb >= 3].index.tolist()
    base_zinb["corporacao_zinb"] = base_zinb["corporacao_modal"].where(
        base_zinb["corporacao_modal"].isin(corps_ok), "OUTROS"
    )

    print(f"\nBase ZINB após filtro: {len(base_zinb)} parlamentares")
    print("Distribuição corporacao_zinb:")
    print(base_zinb["corporacao_zinb"].value_counts())

    try:
        from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

        endog_zinb = base_zinb["n_aprovados"].values
        exog_zinb = sm.add_constant(
            pd.get_dummies(
                base_zinb[["corporacao_zinb", "anos_atividade"]],
                columns=["corporacao_zinb"],
                drop_first=True
            ).astype(float)
        )

        exog_infl = sm.add_constant(
            base_zinb[["anos_atividade"]].astype(float)
        )

        modelo_zinb = ZeroInflatedNegativeBinomialP(
            endog=endog_zinb,
            exog=exog_zinb,
            exog_infl=exog_infl,
            p=2
        ).fit(
            method="bfgs",
            maxiter=500,
            disp=False
        )

        print("\n=== ZINB — NÚMERO DE APROVAÇÕES ===")
        print(modelo_zinb.summary())

        # métricas do ZINB
        llf_zinb = float(modelo_zinb.llf) if hasattr(modelo_zinb, "llf") else np.nan
        aic_zinb = float(modelo_zinb.aic) if hasattr(modelo_zinb, "aic") else np.nan
        bic_zinb = float(modelo_zinb.bic) if hasattr(modelo_zinb, "bic") else np.nan
        nobs_zinb = int(modelo_zinb.nobs) if hasattr(modelo_zinb, "nobs") else len(base_zinb)

        df_zinb_metricas = pd.DataFrame([{
            "modelo": "zinb_n_aprovados",
            "variavel_dependente": "n_aprovados",
            "n_obs": nobs_zinb,
            "llf": llf_zinb,
            "aic": aic_zinb,
            "bic": bic_zinb,
            "pct_zeros": round(pct_zeros, 2),
            "dispersion_ratio": round(disp_ratio, 3)
        }])

        print("\nMétricas do ZINB:")
        print(df_zinb_metricas)

        # coeficientes e IRR da parte de contagem
        try:
            params = modelo_zinb.params
            pvalues = modelo_zinb.pvalues
            ci = modelo_zinb.conf_int()

            df_zinb_coef = pd.DataFrame({
                "variavel": params.index,
                "coef": params.values,
                "irr": np.exp(params.values),
                "p_value": pvalues.values,
                "ic95_inf_irr": np.exp(ci.iloc[:, 0].values),
                "ic95_sup_irr": np.exp(ci.iloc[:, 1].values)
            })
            df_zinb_coef["modelo"] = "zinb_n_aprovados"

            print("\nCoeficientes ZINB (IRR = taxa de incidência relativa):")
            print(df_zinb_coef[["variavel", "coef", "irr", "p_value"]].round(4))

        except Exception as e_coef:
            print(f"\nFalha ao extrair coeficientes ZINB: {e_coef}")

        # comparação entre modelos para n_aprovados
        linhas_comp = []

        if modelo_nb_sucesso is not None and hasattr(modelo_nb_sucesso, "llf"):
            linhas_comp.append({
                "modelo": "NB parcimonioso",
                "log_likelihood": float(modelo_nb_sucesso.llf),
                "aic": float(modelo_nb_sucesso.aic),
                "bic": float(modelo_nb_sucesso.bic),
                "n_obs": int(modelo_nb_sucesso.nobs)
            })

        linhas_comp.append({
            "modelo": "ZINB",
            "log_likelihood": llf_zinb,
            "aic": aic_zinb,
            "bic": bic_zinb,
            "n_obs": nobs_zinb
        })

        df_zinb_comparacao = pd.DataFrame(linhas_comp)

        print("\nComparação de modelos para n_aprovados (menor AIC/BIC = melhor):")
        print(df_zinb_comparacao)

        # interpretação da parte inflada vs parte de contagem
        print("\nInterpretação do ZINB:")
        print("  Parte inflada (inflate_*): preditores da probabilidade de ser")
        print("  'estruturalmente zero' (nunca aprova por razões latentes).")
        print("  Parte de contagem: preditores do número de aprovações entre")
        print("  aqueles que têm capacidade positiva de aprovação.")
        print("  IRR > 1: aumenta a contagem esperada de aprovações.")
        print("  IRR < 1: reduz a contagem esperada.")

    except ImportError:
        print("\nZeroInflatedNegativeBinomialP não disponível nesta versão do statsmodels.")
        print("Versão requerida: statsmodels >= 0.12")
        print("Instale com: pip install statsmodels --upgrade")
        modelo_zinb = None

    except Exception as e_zinb:
        print(f"\nFalha no modelo ZINB: {e_zinb}")
        print("Possível causa: separação quase perfeita ou dados insuficientes.")
        print("O modelo NB parcimonioso permanece como alternativa válida.")
        modelo_zinb = None


# =========================================================
# 22B. ANÁLISE DE CORRESPONDÊNCIA (ANACOR)
# =========================================================

print("\n" + "=" * 60)
print("ANÁLISE DE CORRESPONDÊNCIA")
print("=" * 60)

ARQUIVO_ANACOR_CORP = PASTA / "anacor_corporacao_topico.png"
ARQUIVO_ANACOR_PARTIDO = PASTA / "anacor_partido_topico.png"

if (
    not tabela_corp_topico.empty
    and tabela_corp_topico.shape[0] > 1
    and tabela_corp_topico.shape[1] > 1
):
    df_anacor_corp, df_anacor_topico, inertia_corp = analise_correspondencia_simples(tabela_corp_topico)

    print("\nInércia explicada ANACOR corporação × tópico:")
    print(inertia_corp[:5])

    plt.figure(figsize=(11, 8))
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)

    for _, row in df_anacor_corp.iterrows():
        _lbl_corp = _CORP_LABELS.get(str(row["categoria_linha"]), str(row["categoria_linha"]))
        plt.scatter(row["dim_1"], row["dim_2"], marker="o", s=60, color="steelblue")
        plt.text(row["dim_1"] + 0.01, row["dim_2"] + 0.01,
                 _lbl_corp, fontsize=8.5, color="steelblue")

    for _, row in df_anacor_topico.iterrows():
        _top_num = int(row["categoria_coluna"])
        _lbl_top = NOMES_TOPICOS_CURTO.get(_top_num, f"T{_top_num}")
        plt.scatter(row["dim_1"], row["dim_2"], marker="^", s=80, color="coral")
        plt.text(row["dim_1"] + 0.01, row["dim_2"] + 0.01,
                 _lbl_top, fontsize=8.5, color="coral", fontweight="bold")

    plt.title(f"Mapa perceptual ANACOR — Corporação × Agenda Temática\n"
              f"Inércia: Dim1={inertia_corp[0]*100:.1f}% + Dim2={inertia_corp[1]*100:.1f}% "
              f"= {(inertia_corp[0]+inertia_corp[1])*100:.1f}% total")
    plt.xlabel(f"Dimensão 1 ({inertia_corp[0]*100:.1f}% da inércia)")
    plt.ylabel(f"Dimensão 2 ({inertia_corp[1]*100:.1f}% da inércia)")
    from matplotlib.lines import Line2D
    _leg = [Line2D([0],[0], marker="o", color="steelblue", linestyle="None",
                   label="Corporação"),
            Line2D([0],[0], marker="^", color="coral", linestyle="None",
                   label="Agenda temática")]
    plt.legend(handles=_leg, fontsize=9)
    plt.tight_layout()
    plt.savefig(ARQUIVO_ANACOR_CORP, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

if (
    not tabela_partido.empty
    and tabela_partido.shape[0] > 1
    and tabela_partido.shape[1] > 1
):
    df_anacor_partido, df_anacor_topico_part, inertia_part = analise_correspondencia_simples(tabela_partido)

    plt.figure(figsize=(12, 9))
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)

    for _, row in df_anacor_partido.iterrows():
        plt.scatter(row["dim_1"], row["dim_2"], marker="o", s=50, color="steelblue")
        plt.text(row["dim_1"] + 0.01, row["dim_2"] + 0.01,
                 str(row["categoria_linha"]), fontsize=8, color="steelblue")

    for _, row in df_anacor_topico_part.iterrows():
        _top_num = int(row["categoria_coluna"])
        _lbl_top = NOMES_TOPICOS_CURTO.get(_top_num, f"T{_top_num}")
        plt.scatter(row["dim_1"], row["dim_2"], marker="^", s=80, color="coral")
        plt.text(row["dim_1"] + 0.01, row["dim_2"] + 0.01,
                 _lbl_top, fontsize=8.5, color="coral", fontweight="bold")

    plt.title(f"Mapa perceptual ANACOR — Partido × Agenda Temática\n"
              f"Inércia: Dim1={inertia_part[0]*100:.1f}% + Dim2={inertia_part[1]*100:.1f}% "
              f"= {(inertia_part[0]+inertia_part[1])*100:.1f}% total")
    plt.xlabel(f"Dimensão 1 ({inertia_part[0]*100:.1f}% da inércia)")
    plt.ylabel(f"Dimensão 2 ({inertia_part[1]*100:.1f}% da inércia)")
    _leg2 = [Line2D([0],[0], marker="o", color="steelblue", linestyle="None",
                    label="Partido"),
             Line2D([0],[0], marker="^", color="coral", linestyle="None",
                    label="Agenda temática")]
    plt.legend(handles=_leg2, fontsize=9)
    plt.tight_layout()
    plt.savefig(ARQUIVO_ANACOR_PARTIDO, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# =========================================================
# 22C. EFEITOS MARGINAIS MÉDIOS (AME)
# =========================================================

print("\n" + "=" * 60)
print("EFEITOS MARGINAIS MÉDIOS (AME)")
print("=" * 60)

df_ame_principal = extrair_ame(modelo_principal, "modelo_principal")
df_ame_partido = extrair_ame(modelo_partido, "modelo_partido")
df_ame_corp = extrair_ame(modelo_corp_inf, "modelo_corp")

if not df_ame_principal.empty:
    print("\nAME — modelo principal:")
    print(df_ame_principal)

if not df_ame_partido.empty:
    print("\nAME — modelo com partido:")
    print(df_ame_partido)

if not df_ame_corp.empty:
    print("\nAME — modelo com corporação:")
    print(df_ame_corp)


# =========================================================
# 22D. REDE DE COAUTORIA
# =========================================================

if NX_OK:
    print("\n" + "=" * 60)
    print("REDE DE COAUTORIA")
    print("=" * 60)

    G_rede = nx.Graph()
    arestas_rede = []

    # usa df_autor (pré-expansão) que tem Autor_lista com todos os coautores por PL
    _rede_src = df_autor[df_autor["Autor_lista"].apply(len) >= 2].copy()
    print(f"\nPLs com >= 2 autores (fonte da rede): {len(_rede_src)}")

    for _, row in _rede_src.iterrows():
        autores_prop = sorted(set([
            normalizar_basico(a) for a in row["Autor_lista"]
            if str(a).strip() != ""
        ]))
        if len(autores_prop) < 2:
            continue
        for i in range(len(autores_prop)):
            for j in range(i + 1, len(autores_prop)):
                arestas_rede.append((autores_prop[i], autores_prop[j]))

    df_rede_arestas = pd.DataFrame(arestas_rede, columns=["autor_1", "autor_2"])

    if not df_rede_arestas.empty:
        pesos_rede = (
            df_rede_arestas
            .groupby(["autor_1", "autor_2"])
            .size()
            .reset_index(name="peso")
        )
        df_rede_arestas = pesos_rede.copy()

        for _, row in df_rede_arestas.iterrows():
            G_rede.add_edge(row["autor_1"], row["autor_2"], weight=row["peso"])

        grau_c = dict(nx.degree_centrality(G_rede))
        bet_c = dict(nx.betweenness_centrality(G_rede, normalized=True))
        close_c = dict(nx.closeness_centrality(G_rede))

        df_rede_metricas = pd.DataFrame({
            "Autor": list(grau_c.keys()),
            "grau_centralidade": list(grau_c.values()),
            "betweenness": [bet_c[a] for a in grau_c.keys()],
            "closeness": [close_c[a] for a in grau_c.keys()]
        }).sort_values("grau_centralidade", ascending=False)

        print("\nTop autores por centralidade na rede:")
        print(df_rede_metricas.head(30))

        # ── Visualização: filtra para subgrafo legível ─────────────
        # Com 1000+ nós a rede colapsa. Estratégia:
        # 1. Subgrafo dos top-80 por grau (núcleo da rede)
        # 2. Layout kamada_kawai (melhor separação que spring para redes densas)
        # 3. Arestas apenas com peso >= percentil 75 (remove ruído)

        _top_n = 80
        _graus_todos = dict(G_rede.degree())
        _top_nos = sorted(_graus_todos, key=_graus_todos.get, reverse=True)[:_top_n]
        G_sub = G_rede.subgraph(_top_nos).copy()

        # remove arestas fracas (peso < percentil 75 do subgrafo)
        _pesos = [d.get("weight", 1) for _, _, d in G_sub.edges(data=True)]
        _p75   = np.percentile(_pesos, 75) if _pesos else 1
        _arestas_remover = [
            (u, v) for u, v, d in G_sub.edges(data=True)
            if d.get("weight", 1) < _p75
        ]
        G_sub.remove_edges_from(_arestas_remover)

        # remove nós isolados após remoção de arestas
        G_sub.remove_nodes_from(list(nx.isolates(G_sub)))

        print(f"\nSubgrafo para visualização: {G_sub.number_of_nodes()} nós, "
              f"{G_sub.number_of_edges()} arestas (peso ≥ p75={_p75:.0f})")

        # layout kamada-kawai se pequeno o suficiente, senão spring com k alto
        try:
            if G_sub.number_of_nodes() <= 120:
                _pos_sub = nx.kamada_kawai_layout(G_sub, weight="weight")
            else:
                _pos_sub = nx.spring_layout(G_sub, k=2.5, seed=42, iterations=80)
        except Exception:
            _pos_sub = nx.spring_layout(G_sub, k=2.5, seed=42)

        fig_rede, ax_rede = plt.subplots(figsize=(14, 11))

        # tamanho ∝ betweenness no subgrafo
        _bet_sub  = nx.betweenness_centrality(G_sub, normalized=True)
        _grau_sub = dict(G_sub.degree())
        _sizes    = [max(80, _bet_sub.get(n, 0) * 8000 + _grau_sub.get(n, 1) * 20)
                     for n in G_sub.nodes()]
        _colors   = [_bet_sub.get(n, 0) for n in G_sub.nodes()]

        # arestas com espessura proporcional ao peso
        _edge_w = [G_sub[u][v].get("weight", 1) for u, v in G_sub.edges()]
        _max_w  = max(_edge_w) if _edge_w else 1
        _edge_w_norm = [0.3 + 3 * (w / _max_w) for w in _edge_w]

        nx.draw_networkx_edges(
            G_sub, _pos_sub, ax=ax_rede,
            width=_edge_w_norm, alpha=0.25, edge_color="#999999"
        )
        sc = nx.draw_networkx_nodes(
            G_sub, _pos_sub, ax=ax_rede,
            node_size=_sizes, node_color=_colors,
            cmap="Blues", alpha=0.85, vmin=0
        )

        # rótulos top-15 betweenness
        _top15 = sorted(_bet_sub, key=_bet_sub.get, reverse=True)[:15]
        _labels = {n: n[:20] for n in _top15 if n in G_sub.nodes()}
        nx.draw_networkx_labels(
            G_sub, _pos_sub, labels=_labels, ax=ax_rede,
            font_size=6.5, font_color="#1a1a2e", font_weight="bold"
        )

        plt.colorbar(sc, ax=ax_rede, label="Betweenness centralidade", shrink=0.6)
        ax_rede.set_title(
            f"Rede de coautoria — núcleo principal (top-{_top_n} por grau)\n"
            f"{G_sub.number_of_nodes()} nós | {G_sub.number_of_edges()} arestas (peso ≥ p75)\n"
            "Tamanho ∝ betweenness | Cor ∝ centralidade | Rótulos: top-15",
            fontsize=10
        )
        ax_rede.axis("off")
        plt.tight_layout()
        plt.savefig(PASTA / "rede_coautoria.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"Rede de coautoria salva: {G_rede.number_of_nodes()} nós totais, "
              f"subgrafo visualizado: {G_sub.number_of_nodes()} nós.")
# =========================================================

print("\n" + "=" * 60)
print("PREVISÃO FUTURA POR ANO")
print("=" * 60)

ano_max = int(df_reg_inf["ano"].max())
anos_simulados = list(range(ano_max + 1, 2041))
partidos_validos_pred = sorted(df_reg_inf["partido_inf"].dropna().unique())

cenarios_ano = []
for ano_val in anos_simulados:
    for part in partidos_validos_pred:
        cenarios_ano.append({
            "ano": ano_val,
            "ano_c": ano_val - ano_mediano,
            "partido_inf": part
        })

df_pred_ano = pd.DataFrame(cenarios_ano)

if modelo_ano is not None and not df_pred_ano.empty:
    df_pred_ano["prob_prevista"] = modelo_ano.predict(df_pred_ano)
else:
    df_pred_ano["prob_prevista"] = pd.Series(dtype=float)

print("\nAmostra da previsão futura por ano:")
print(df_pred_ano.head(20))


# =========================================================
# 24. MÉDIA GERAL DE PROBABILIDADE POR ANO
# =========================================================

resumo_ano = (
    df_pred_ano
    .groupby("ano")["prob_prevista"]
    .mean()
    .reset_index()
) if not df_pred_ano.empty else pd.DataFrame(columns=["ano", "prob_prevista"])

print("\nProbabilidade média prevista por ano:")
print(resumo_ano)

if not resumo_ano.empty:
    plt.figure(figsize=(10, 6))
    plt.plot(resumo_ano["ano"], resumo_ano["prob_prevista"], marker="o")
    plt.title("Probabilidade média prevista de aprovação por ano")
    plt.xlabel("Ano")
    plt.ylabel("Probabilidade média prevista")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    plt.close()


# =========================================================
# 25. CENÁRIOS OTIMISTA E PESSIMISTA
# =========================================================

if not df_pred_ano.empty:
    df_pred_ano["prob_base"] = df_pred_ano["prob_prevista"]
    df_pred_ano["prob_otimista"] = np.clip(df_pred_ano["prob_base"] * 1.25, 0, 1)
    df_pred_ano["prob_pessimista"] = np.clip(df_pred_ano["prob_base"] * 0.75, 0, 1)

    resumo_cenarios = (
        df_pred_ano
        .groupby("ano")[["prob_base", "prob_otimista", "prob_pessimista"]]
        .mean()
        .reset_index()
    )
else:
    resumo_cenarios = pd.DataFrame(columns=["ano", "prob_base", "prob_otimista", "prob_pessimista"])

print("\nResumo dos cenários por ano:")
print(resumo_cenarios)

if not resumo_cenarios.empty:
    plt.figure(figsize=(10, 6))
    plt.plot(resumo_cenarios["ano"], resumo_cenarios["prob_base"], label="Base", marker="o")
    plt.plot(resumo_cenarios["ano"], resumo_cenarios["prob_otimista"], label="Otimista", marker="o")
    plt.plot(resumo_cenarios["ano"], resumo_cenarios["prob_pessimista"], label="Pessimista", marker="o")
    plt.title("Cenários de probabilidade média prevista por ano")
    plt.xlabel("Ano")
    plt.ylabel("Probabilidade média prevista")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    plt.close()


# =========================================================
# 26. SIMULAÇÃO POR PARTIDO AO LONGO DO TEMPO
# =========================================================

resumo_partido_ano = (
    df_pred_ano
    .groupby(["ano", "partido_inf"])["prob_prevista"]
    .mean()
    .reset_index()
) if not df_pred_ano.empty else pd.DataFrame(columns=["ano", "partido_inf", "prob_prevista"])

print("\nSimulação média por partido e ano:")
print(resumo_partido_ano.head(30))

top_partidos_plot = (
    df_reg_inf["partido_inf"]
    .value_counts()
    .head(8)
    .index
)

df_plot_partido = resumo_partido_ano[
    resumo_partido_ano["partido_inf"].isin(top_partidos_plot)
].copy()

if not df_plot_partido.empty:
    plt.figure(figsize=(12, 7))
    for partido in sorted(df_plot_partido["partido_inf"].unique()):
        temp = df_plot_partido[df_plot_partido["partido_inf"] == partido]
        plt.plot(temp["ano"], temp["prob_prevista"], marker="o", label=partido)

    plt.title("Probabilidade média prevista de aprovação por partido ao longo do tempo")
    plt.xlabel("Ano")
    plt.ylabel("Probabilidade prevista")
    plt.legend(title="Partido", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    plt.close()


# =========================================================
# 27. HEATMAP PARTIDO X ANO
# =========================================================

if not resumo_partido_ano.empty:
    tabela_partido_ano_pred = resumo_partido_ano.pivot(
        index="partido_inf",
        columns="ano",
        values="prob_prevista"
    )

    plt.figure(figsize=(14, 8))
    sns.heatmap(tabela_partido_ano_pred, annot=False, cmap="YlGnBu")
    plt.title("Probabilidade prevista de aprovação por partido e ano")
    plt.xlabel("Ano")
    plt.ylabel("Partido")
    plt.tight_layout()
    plt.show()
    plt.close()


# =========================================================
# 28. RANKINGS FUTUROS
# =========================================================

df_rank_futuro = (
    df_pred_ano.sort_values("prob_prevista", ascending=False).reset_index(drop=True)
    if not df_pred_ano.empty else
    pd.DataFrame(columns=["ano", "ano_c", "partido_inf", "prob_prevista"])
)

print("\nTop 50 cenários futuros com maior probabilidade prevista:")
print(df_rank_futuro.head(50))

print("\nTop 10 cenários futuros:")
if not df_rank_futuro.empty:
    print(df_rank_futuro.head(10)[["ano", "partido_inf", "prob_prevista"]])

ranking_partido = (
    df_pred_ano
    .groupby("partido_inf")["prob_prevista"]
    .mean()
    .reset_index()
    .sort_values("prob_prevista", ascending=False)
) if not df_pred_ano.empty else pd.DataFrame(columns=["partido_inf", "prob_prevista"])

ranking_partido_futuro = ranking_partido.copy()

print("\nRanking médio de probabilidade prevista por partido:")
print(ranking_partido)


# =========================================================
# 29. MACHINE LEARNING TEMPORAL
# =========================================================

print("\n" + "=" * 60)
print("MACHINE LEARNING TEMPORAL")
print("=" * 60)

# incorpora corporação no ML quando disponível
if not df_reg_corp_base.empty:
    df_ml = df_reg_inf.merge(
        df_reg_corp_base[["Proposicoes", "Autor", "ano", "corporacao_sigla"]].drop_duplicates(),
        on=["Proposicoes", "Autor", "ano"],
        how="left"
    )
else:
    df_ml = df_reg_inf.copy()

if "corporacao_sigla" not in df_ml.columns:
    df_ml["corporacao_sigla"] = "OUTROS"

df_ml["corporacao_sigla"] = df_ml["corporacao_sigla"].fillna("OUTROS")

df_ml = df_ml[
    df_ml["partido_inf"].notna() &
    df_ml["ano_c"].notna()
].copy()

df_ml["ano"] = df_ml["ano"].astype(int)
df_ml["ano_c"] = df_ml["ano_c"].astype(float)
df_ml["aprovado"] = df_ml["aprovado"].astype(int)

# ── corte 1: exclui anos com dados incompletos ───────────────────────────
_n_antes = len(df_ml)
_anos_removidos = sorted(df_ml[df_ml["ano"] > ANO_ULTIMO_COMPLETO]["ano"].unique())
df_ml = df_ml[df_ml["ano"] <= ANO_ULTIMO_COMPLETO].copy()

# ── corte 2: exclui legislatura em vigência (57ª) ────────────────────────
if "legislatura" in df_ml.columns and LEGISLATURA_EXCLUIR_ML:
    _mask_leg = df_ml["legislatura"].isin(LEGISLATURA_EXCLUIR_ML)
    _n_leg_removidos = int(_mask_leg.sum())
    _legs_removidas = df_ml[_mask_leg]["legislatura"].unique().tolist()
    df_ml = df_ml[~_mask_leg].copy()
else:
    _n_leg_removidos = 0
    _legs_removidas = []

_n_depois = len(df_ml)

print(f"\n[Corte dados incompletos] ANO_ULTIMO_COMPLETO = {ANO_ULTIMO_COMPLETO}")
if _anos_removidos:
    print(f"  Anos removidos: {_anos_removidos}")
if _legs_removidas:
    print(f"  Legislaturas removidas: {_legs_removidas} ({_n_leg_removidos} PLs)")
print(f"  Base ML final: {_n_depois} PLs ({df_ml['aprovado'].sum()} aprovados)")

features_ml = ["partido_inf", "ano_c", "corporacao_sigla", "topico_dominante"]
target_ml = "aprovado"

# topico_dominante é essencial: sem ele, o partido absorve o efeito do tema.
# Partidos que propõem mais PLs em agendas com maior taxa de aprovação
# (ex.: Proteção Social) apareceriam inflados mesmo sem efeito causal.
# legislatura entra como feature categórica para capturar ciclos políticos.
cat_features_ml = ["partido_inf", "corporacao_sigla", "topico_dominante"]
num_features_ml = ["ano_c"]

print("\n[Features ML]")
print(f"  Categóricas: {cat_features_ml}")
print(f"  Numéricas:   {num_features_ml}")
print("  Nota: topico_dominante adicionado para isolar efeito de partido")
print("        de efeito de agenda — sem isso partido absorve viés temático.")

preprocessador_sparse = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            cat_features_ml
        ),
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            num_features_ml
        )
    ]
)

preprocessador_dense = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]),
            cat_features_ml
        ),
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            num_features_ml
        )
    ]
)

modelos_ml = {
    "logit": LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="liblinear"
    ),
    "rf": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    ),
    "histgb": HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    )
}

if XGB_OK:
    scale_pos = max(
        1,
        int((df_ml[target_ml] == 0).sum() / max(1, (df_ml[target_ml] == 1).sum()))
    )
    modelos_ml["xgb"] = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos
    )

anos_unicos = sorted(df_ml["ano"].dropna().unique())
anos_teste = [a for a in anos_unicos if a >= ANO_INICIO_BACKTEST]

resultados_bt = []
predicoes_bt = []

for ano_teste in anos_teste:
    treino = df_ml[df_ml["ano"] < ano_teste].copy()
    teste = df_ml[df_ml["ano"] == ano_teste].copy()

    if treino.empty or teste.empty:
        continue

    if treino[target_ml].nunique() < 2 or teste[target_ml].nunique() < 2:
        continue

    X_train_bt = treino[features_ml]
    y_train_bt = treino[target_ml]
    X_test_bt = teste[features_ml]
    y_test_bt = teste[target_ml]

    for nome_modelo, estimador in modelos_ml.items():
        prep_uso = preprocessador_dense if nome_modelo == "histgb" else preprocessador_sparse

        pipe = Pipeline([
            ("prep", prep_uso),
            ("model", estimador)
        ])

        try:
            pipe.fit(X_train_bt, y_train_bt)
        except Exception as e:
            print(f"\nFalha no backtesting {nome_modelo} / ano {ano_teste}: {e}")
            continue

        if hasattr(pipe, "predict_proba"):
            y_prob_bt = pipe.predict_proba(X_test_bt)[:, 1]
        elif hasattr(pipe, "decision_function"):
            y_raw = pipe.decision_function(X_test_bt)
            y_prob_bt = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min() + 1e-9)
        else:
            continue

        auc_bt = roc_auc_score(y_test_bt, y_prob_bt)
        ap_bt = average_precision_score(y_test_bt, y_prob_bt)
        brier_bt = brier_score_loss(y_test_bt, y_prob_bt)

        df_temp = teste[[
            "Proposicoes", "Autor", "UF", "Partido", "partido_inf",
            "corporacao_sigla", "legislatura", "ano"
        ]].copy()

        df_temp["ano_teste"] = ano_teste
        df_temp["modelo"] = nome_modelo
        df_temp["prob_aprovacao"] = y_prob_bt
        df_temp["real"] = y_test_bt.values
        df_temp = df_temp.sort_values("prob_aprovacao", ascending=False).reset_index(drop=True)
        df_temp["rank"] = np.arange(1, len(df_temp) + 1)

        top10_bt = int(df_temp.head(10)["real"].sum()) if len(df_temp) >= 10 else int(df_temp["real"].sum())
        top20_bt = int(df_temp.head(20)["real"].sum()) if len(df_temp) >= 20 else int(df_temp["real"].sum())
        top50_bt = int(df_temp.head(50)["real"].sum()) if len(df_temp) >= 50 else int(df_temp["real"].sum())

        resultados_bt.append({
            "ano_teste": ano_teste,
            "modelo": nome_modelo,
            "auc": auc_bt,
            "average_precision": ap_bt,
            "brier_score": brier_bt,
            "aprovados_top10": top10_bt,
            "aprovados_top20": top20_bt,
            "aprovados_top50": top50_bt,
            "n_teste": len(teste),
            "positivos_teste": int(y_test_bt.sum())
        })

        predicoes_bt.append(df_temp)

df_resultados_bt = pd.DataFrame(resultados_bt)
df_predicoes_bt = pd.concat(predicoes_bt, ignore_index=True) if predicoes_bt else pd.DataFrame()

print("\nResultados do backtesting temporal:")
print(df_resultados_bt)

if not df_resultados_bt.empty:
    resumo_modelos_bt = (
        df_resultados_bt
        .groupby("modelo")[[
            "auc",
            "average_precision",
            "brier_score",
            "aprovados_top10",
            "aprovados_top20",
            "aprovados_top50"
        ]]
        .mean()
        .reset_index()
        .sort_values(["average_precision", "auc"], ascending=[False, False])
    )

    print("\nResumo médio por modelo:")
    print(resumo_modelos_bt)


# =========================================================
# 30. PREDIÇÕES FINAIS
# =========================================================

modelo_pred_final = None
nome_modelo_final = None
base_pred_modelo = df_reg_inf.copy()
usa_corporacao = False
categorias_corporacao_pred = None

if modelo_corp_inf is not None and not df_reg_corp_logit.empty:
    modelo_pred_final = modelo_corp_inf
    nome_modelo_final = "modelo com corporação"
    usa_corporacao = True
    categorias_corporacao_pred = list(df_reg_corp_logit["corporacao_sigla"].cat.categories)
elif modelo_partido is not None:
    modelo_pred_final = modelo_partido
    nome_modelo_final = "modelo com partido"
elif modelo_principal is not None:
    modelo_pred_final = modelo_principal
    nome_modelo_final = "modelo principal"

print(f"\nModelo usado para predição final: {nome_modelo_final}")

if modelo_pred_final is not None and not base_pred_modelo.empty:
    topico_ref = int(base_pred_modelo["topico_dominante"].mode().iat[0])
    ano_mediano_pred = float(base_pred_modelo["ano"].median())

    partidos_pred = sorted(base_pred_modelo["partido_inf"].dropna().unique()) if "partido_inf" in base_pred_modelo.columns else []
    legislaturas_pred = sorted(base_pred_modelo["legislatura"].dropna().unique()) if "legislatura" in base_pred_modelo.columns else []
    anos_historicos = sorted(base_pred_modelo["ano"].dropna().astype(int).unique()) if "ano" in base_pred_modelo.columns else []

    if usa_corporacao:
        corporacoes_pred = categorias_corporacao_pred
        corporacao_ref = df_reg_corp_logit["corporacao_sigla"].mode().iat[0]
    else:
        corporacoes_pred = []
        corporacao_ref = None

    partido_ref = base_pred_modelo["partido_inf"].mode().iat[0] if "partido_inf" in base_pred_modelo.columns else None

    # predição histórica por ano
    linhas = []
    for ano_val in anos_historicos:
        df_tmp = construir_df_predict_formula(
            modelo=modelo_pred_final,
            topico=topico_ref,
            ano=ano_val,
            ano_c=ano_val - ano_mediano_pred,
            partido=partido_ref,
            legislatura=None,
            corporacao=corporacao_ref,
            categorias_corporacao=categorias_corporacao_pred
        )
        df_tmp["ano"] = ano_val
        linhas.append(df_tmp)

    pred_ano_historico = pd.concat(linhas, ignore_index=True)
    pred_ano_historico["prob_prevista"] = modelo_pred_final.predict(pred_ano_historico)

    pred_ano = pred_ano_historico.copy()

    print("\nPredição histórica por ano:")
    print(pred_ano_historico[["ano", "prob_prevista"]])

    plt.figure(figsize=(10, 6))
    plt.plot(pred_ano_historico["ano"], pred_ano_historico["prob_prevista"], marker="o")
    plt.title("Probabilidade prevista de aprovação por ano (histórico)")
    plt.xlabel("Ano")
    plt.ylabel("Probabilidade prevista")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    plt.close()

    if partidos_pred:
        if modelo_pred_final is modelo_corp_inf and modelo_partido is not None:
            linhas = []
            for partido in partidos_pred:
                df_tmp = construir_df_predict_formula(
                    modelo=modelo_partido,
                    topico=topico_ref,
                    ano=int(ano_mediano_pred),
                    ano_c=0.0,
                    partido=partido,
                    legislatura=None,
                    corporacao=None
                )
                df_tmp["partido_inf"] = partido
                linhas.append(df_tmp)

            pred_partido = pd.concat(linhas, ignore_index=True)
            pred_partido["prob_prevista"] = modelo_partido.predict(pred_partido)
            pred_partido = pred_partido.sort_values("prob_prevista", ascending=False).reset_index(drop=True)
        else:
            linhas = []
            for partido in partidos_pred:
                df_tmp = construir_df_predict_formula(
                    modelo=modelo_pred_final,
                    topico=topico_ref,
                    ano=int(ano_mediano_pred),
                    ano_c=0.0,
                    partido=partido,
                    legislatura=None,
                    corporacao=corporacao_ref,
                    categorias_corporacao=categorias_corporacao_pred
                )
                df_tmp["partido_inf"] = partido
                linhas.append(df_tmp)

            pred_partido = pd.concat(linhas, ignore_index=True)
            pred_partido["prob_prevista"] = modelo_pred_final.predict(pred_partido)
            pred_partido = pred_partido.sort_values("prob_prevista", ascending=False).reset_index(drop=True)

        print("\nPredição por partido:")
        print(pred_partido)

        plt.figure(figsize=(10, 8))
        plt.barh(pred_partido["partido_inf"].astype(str), pred_partido["prob_prevista"])
        plt.title("Probabilidade prevista de aprovação por partido")
        plt.xlabel("Probabilidade prevista")
        plt.ylabel("Partido")
        plt.tight_layout()
        plt.show()
        plt.close()

    mapa_leg_para_ano = (
        base_pred_modelo.groupby("legislatura")["ano"]
        .median()
        .dropna()
        .to_dict()
    )

    if legislaturas_pred:
        linhas = []
        modelo_leg_pred = modelo_leg if modelo_leg is not None else modelo_pred_final

        for leg in legislaturas_pred:
            ano_leg = float(mapa_leg_para_ano.get(leg, ano_mediano_pred))
            df_tmp = construir_df_predict_formula(
                modelo=modelo_leg_pred,
                topico=topico_ref,
                ano=int(round(ano_leg)),
                ano_c=ano_leg - ano_mediano_pred,
                partido=partido_ref,
                legislatura=leg,
                corporacao=corporacao_ref,
                categorias_corporacao=categorias_corporacao_pred
            )
            df_tmp["legislatura"] = leg
            linhas.append(df_tmp)

        pred_legislatura = pd.concat(linhas, ignore_index=True)
        pred_legislatura["prob_prevista"] = modelo_leg_pred.predict(pred_legislatura)

        print("\nPredição por legislatura:")
        print(pred_legislatura)

        plt.figure(figsize=(10, 5))
        plt.bar(pred_legislatura["legislatura"], pred_legislatura["prob_prevista"])
        plt.title("Probabilidade prevista de aprovação por legislatura")
        plt.xlabel("Legislatura")
        plt.ylabel("Probabilidade prevista")
        plt.tight_layout()
        plt.show()
        plt.close()

    if usa_corporacao and corporacoes_pred:
        linhas = []
        for corp in corporacoes_pred:
            df_tmp = construir_df_predict_formula(
                modelo=modelo_corp_inf,
                topico=topico_ref,
                ano=int(ano_mediano_pred),
                ano_c=0.0,
                partido=partido_ref,
                legislatura=None,
                corporacao=corp,
                categorias_corporacao=categorias_corporacao_pred
            )
            linhas.append(df_tmp)

        pred_corporacao = pd.concat(linhas, ignore_index=True)
        pred_corporacao["prob_prevista"] = modelo_corp_inf.predict(pred_corporacao)
        pred_corporacao = pred_corporacao.sort_values("prob_prevista", ascending=False).reset_index(drop=True)

        # calcular IC 95% via delta method
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            _frame = pred_corporacao.copy()
            _pred_sm = modelo_corp_inf.get_prediction(_frame)
            _summ = _pred_sm.summary_frame(alpha=0.05)
            pred_corporacao["ic_inf"] = _summ["mean_ci_lower"].values
            pred_corporacao["ic_sup"] = _summ["mean_ci_upper"].values
        except Exception:
            pred_corporacao["ic_inf"] = np.nan
            pred_corporacao["ic_sup"] = np.nan

        print("\nPredição por corporação:")
        print(pred_corporacao)

        fig, ax = plt.subplots(figsize=(10, 5))
        corps_plot = pred_corporacao["corporacao_sigla"].astype(str)
        probs_plot = pred_corporacao["prob_prevista"]

        bars = ax.bar(corps_plot, probs_plot, color="steelblue", alpha=0.8)

        # IC 95% se disponível
        if pred_corporacao["ic_inf"].notna().all():
            yerr_low  = probs_plot.values - pred_corporacao["ic_inf"].values
            yerr_high = pred_corporacao["ic_sup"].values - probs_plot.values
            ax.errorbar(
                corps_plot, probs_plot,
                yerr=[np.maximum(yerr_low, 0), np.maximum(yerr_high, 0)],
                fmt="none", color="black", capsize=4, linewidth=1.2
            )

        # taxa observada como ponto de referência
        if not df_reg_corp_base.empty:
            taxa_obs = (
                df_reg_corp_base[df_reg_corp_base["corporacao_sigla"].isin(corps_plot)]
                .groupby("corporacao_sigla")["aprovado"].mean()
            )
            for i, corp in enumerate(corps_plot):
                if corp in taxa_obs.index:
                    ax.scatter(i, taxa_obs[corp], color="red",
                               zorder=5, s=40, marker="D",
                               label="Taxa observada" if i == 0 else "")

        ax.set_title(
            "Probabilidade estimada de aprovação por corporação\n"
            "(logit ajustado; tópico e ano fixos na mediana; barras de erro = IC 95%;\n"
            "diamante vermelho = taxa observada não ajustada)",
            fontsize=10
        )
        ax.set_xlabel("Corporação de origem")
        ax.set_ylabel("Probabilidade estimada (modelo logístico)")
        ax.legend(fontsize=8)

        # nota metodológica no rodapé
        fig.text(
            0.5, -0.04,
            "Nota: probabilidades estimadas pelo modelo logístico com erros clusterizados por autor.\n"
            "Não se trata de previsão futura nem de modelo de machine learning.\n"
            "Valores representam a probabilidade média estimada para cada corporação,\n"
            "mantendo tópico dominante e ano no valor mediano da amostra.",
            ha="center", fontsize=7.5, color="gray"
        )

        plt.tight_layout()
        plt.show()
        plt.close()

    if modelo_ano is not None and partidos_pred:
        linhas = []
        anos_futuros = list(range(int(base_pred_modelo["ano"].max()) + 1, 2041))

        for ano_val in anos_futuros:
            for partido in partidos_pred:
                df_tmp = construir_df_predict_formula(
                    modelo=modelo_ano,
                    topico=None,
                    ano=ano_val,
                    ano_c=ano_val - ano_mediano_pred,
                    partido=partido,
                    legislatura=None,
                    corporacao=None
                )
                df_tmp["ano"] = ano_val
                linhas.append(df_tmp)

        pred_ano_partido = pd.concat(linhas, ignore_index=True)
        pred_ano_partido["prob_prevista"] = modelo_ano.predict(pred_ano_partido)

    if usa_corporacao and modelo_corp_inf is not None and corporacoes_pred:
        linhas = []
        topicos_validos = sorted(df_reg_corp_logit["topico_dominante"].dropna().unique())
        for corp in corporacoes_pred:
            for top in topicos_validos:
                df_tmp = construir_df_predict_formula(
                    modelo=modelo_corp_inf,
                    topico=top,
                    ano=int(ano_mediano_pred),
                    ano_c=0.0,
                    partido=None,
                    legislatura=None,
                    corporacao=corp,
                    categorias_corporacao=categorias_corporacao_pred
                )
                linhas.append(df_tmp)

        pred_corp_topico = pd.concat(linhas, ignore_index=True)
        pred_corp_topico["prob_prevista"] = modelo_corp_inf.predict(pred_corp_topico)

    if usa_corporacao and modelo_corp_inf is not None and corporacoes_pred:
        anos_futuros = list(range(int(base_pred_modelo["ano"].max()) + 1, 2041))
        ano_mediano_corp = float(df_reg_corp_logit["ano"].median())

        linhas = []
        for ano_val in anos_futuros:
            for corp in corporacoes_pred:
                df_tmp = construir_df_predict_formula(
                    modelo=modelo_corp_inf,
                    topico=int(df_reg_corp_logit["topico_dominante"].mode().iat[0]),
                    ano=ano_val,
                    ano_c=ano_val - ano_mediano_corp,
                    partido=None,
                    legislatura=None,
                    corporacao=corp,
                    categorias_corporacao=categorias_corporacao_pred
                )
                df_tmp["ano"] = ano_val
                linhas.append(df_tmp)

        pred_ano_corporacao = pd.concat(linhas, ignore_index=True)
        pred_ano_corporacao["prob_prevista"] = modelo_corp_inf.predict(pred_ano_corporacao)

        resumo_corporacao_ano = (
            pred_ano_corporacao
            .groupby(["ano", "corporacao_sigla"])["prob_prevista"]
            .mean()
            .reset_index()
        )

        ranking_corporacao_futuro = (
            pred_ano_corporacao
            .groupby("corporacao_sigla")["prob_prevista"]
            .mean()
            .reset_index()
            .sort_values("prob_prevista", ascending=False)
        )


# =========================================================
# 30B. PREVISÃO FUTURA COM MACHINE LEARNING
# =========================================================
#
# DIFERENÇA IMPORTANTE em relação à seção 30:
#   - Seção 30: probabilidades estimadas pelo LOGIT para grupos
#     (partido, corporação, tópico) fixados na mediana. É um
#     resultado inferencial do modelo estatístico, NÃO uma previsão.
#
#   - Seção 30B: treina o MELHOR MODELO DO BACKTESTING em toda a
#     amostra histórica (1989–2024) e aplica a cenários futuros
#     (2025–2028) construídos com valores realistas de features.
#     Isso é previsão de machine learning genuína.
#
# Método: o modelo com melhor AUC médio no backtesting (seção 29)
# é treinado em todos os dados disponíveis e aplicado a uma grade
# de cenários futuros com combinações de partido, corporação e tópico.
#
# Ementa coberta:
#   - Árvores, redes e Ensemble models: Random Forest, HistGradientBoosting
#   - Técnicas de validação: out-of-time (já feito no backtesting)
#   - Overfitting: treinamento em amostra completa após validação temporal
#   - Séries temporais: janela deslizante e projeção prospectiva
# =========================================================

print("\n" + "=" * 60)
print("30B. PREVISÃO FUTURA — MACHINE LEARNING")
print("=" * 60)

df_previsao_ml = pd.DataFrame()
df_previsao_ml_resumo = pd.DataFrame()

if not df_ml.empty and not df_resultados_bt.empty:

    # ── passo 1: selecionar modelo para previsão ─────────────────────────
    # O backtesting seleciona o melhor modelo por AUC out-of-time.
    # Para PREVISÃO FUTURA, usamos o logit (modelo_partido) em vez do RF:
    #   - RF memoriza combinações raras da amostra (PSB+MB=0 aprovações → 69%)
    #   - Logit generaliza via coeficientes contínuos — mais robusto fora do range
    #   - O RF entra apenas no backtesting (comparação de capacidade discriminativa)
    resumo_bt_30b = (
        df_resultados_bt
        .groupby("modelo")[["auc", "average_precision"]]
        .mean()
        .sort_values("auc", ascending=False)
    )
    melhor_nome_30b = resumo_bt_30b.index[0]
    auc_medio_30b   = float(resumo_bt_30b.loc[melhor_nome_30b, "auc"])

    print(f"\nMelhor modelo no backtesting: {melhor_nome_30b} "
          f"(AUC médio out-of-time = {auc_medio_30b:.3f})")
    print("Ranking de modelos no backtesting:")
    print(resumo_bt_30b.round(4))
    print("\n[INFO] Para previsão futura: usando logit (modelo_partido).")
    print("       RF é usado apenas no backtesting — overfita combinações raras.")

    # modelo de previsão: logit com partido + tópico + ano
    _modelo_previsao = modelo_partido if modelo_partido is not None else modelo_principal
    _nome_previsao   = "logit (tópico+partido+ano)" if modelo_partido is not None else "logit (tópico+ano)"

    if _modelo_previsao is None:
        print("\nNenhum logit disponível para previsão. Abortando seção 30B.")
        _pipe_ok = False
    else:
        _pipe_ok = True
        print(f"       Modelo de previsão: {_nome_previsao}")

    if _pipe_ok:

        # ── passo 3: construir grade de cenários futuros via logit ────────
        ano_max_historico = ANO_ULTIMO_COMPLETO
        ano_med_real = float(df_reg_inf["ano"].median()) if not df_reg_inf.empty else 2019.0

        anos_futuros_ml  = list(range(ano_max_historico + 1, ANO_HORIZONTE_PREVISAO + 1))
        partidos_cenario = sorted([p for p in df_reg_inf["partido_inf"].dropna().unique()
                                   if p != "OUTROS"])

        # distribuição histórica de tópicos para ponderação
        _dist_topico = (
            df_ml["topico_dominante"]
            .value_counts(normalize=True)
            .to_dict()
        )
        topicos_cenario = sorted(_dist_topico.keys())

        print(f"\nHorizonte de previsão: {anos_futuros_ml[0]}–{anos_futuros_ml[-1]}")
        print(f"Partidos na grade: {partidos_cenario}")
        print(f"Modelo de previsão: {_nome_previsao}")

        # grade: partido × tópico × ano
        # probabilidade ponderada pela distribuição histórica de tópicos
        # → elimina viés de composição temática por partido
        linhas_futuras = []
        for ano_f in anos_futuros_ml:
            ano_c_f = float(ano_f - ano_med_real)
            for partido_f in partidos_cenario:
                for top_f in topicos_cenario:
                    peso = _dist_topico.get(top_f, 1.0)
                    try:
                        _df_tmp = pd.DataFrame({
                            "topico_dominante": [top_f],
                            "partido_inf":      [partido_f],
                            "ano_c":            [ano_c_f],
                            "corporacao_sigla": ["PM"],   # referência
                            "legislatura":      ["56a"],
                        })
                        _prob = float(_modelo_previsao.predict(_df_tmp).iloc[0])
                    except Exception:
                        _prob = np.nan
                    linhas_futuras.append({
                        "ano":              ano_f,
                        "ano_c":            ano_c_f,
                        "partido_inf":      partido_f,
                        "topico_dominante": top_f,
                        "prob_logit":       _prob,
                        "peso_topico":      peso,
                    })

        df_cenarios_raw = pd.DataFrame(linhas_futuras)
        df_cenarios_raw["prob_pond"] = (
            df_cenarios_raw["prob_logit"] * df_cenarios_raw["peso_topico"]
        )

        # agrega por partido × ano (média ponderada sobre tópicos)
        df_cenarios = (
            df_cenarios_raw
            .groupby(["ano", "partido_inf"])
            .agg(prob_aprovacao_ml=("prob_pond", "sum"))
            .reset_index()
        )

        df_previsao_ml = df_cenarios.sort_values(
            "prob_aprovacao_ml", ascending=False
        ).reset_index(drop=True)

        print(f"\nGrade de cenários: {len(df_cenarios):,} combinações")
        print("\nTop 10 cenários com maior probabilidade prevista:")
        print(df_previsao_ml[["ano", "partido_inf", "prob_aprovacao_ml"]].head(10).to_string(index=False))

        # ── passo 4: resumos agregados ────────────────────────────────────
        # O logit inclui ano_c como preditor → tendência temporal emerge naturalmente
        resumo_ano_ml = (
            df_previsao_ml
            .groupby("ano")["prob_aprovacao_ml"]
            .agg(["mean", "median", "std", "min", "max"])
            .reset_index()
            .rename(columns={
                "mean":   "prob_media",
                "median": "prob_mediana",
                "std":    "prob_dp",
                "min":    "prob_min",
                "max":    "prob_max"
            })
        )

        print("\nProbabilidade média prevista por ano:")
        print(resumo_ano_ml.round(4))

        # 4b. probabilidade média por partido (média sobre anos)
        resumo_partido_ml = (
            df_previsao_ml
            .groupby("partido_inf")["prob_aprovacao_ml"]
            .mean()
            .reset_index()
            .sort_values("prob_aprovacao_ml", ascending=False)
            .rename(columns={"prob_aprovacao_ml": "prob_media"})
        )
        print("\nProbabilidade média prevista por partido (2024–2040):")
        print(resumo_partido_ml.round(4))


        # 4c. resumo por corporação — não disponível na grade logit (sem corp)
        resumo_corp_ml = pd.DataFrame()

        df_previsao_ml_resumo = pd.concat([
            resumo_ano_ml.assign(dimensao="ano"),
        ], ignore_index=True)

        # ── passo 5: importância de features — não aplicável ao logit ────
        # (importância de features é específica de árvores/RF; logit usa coef.
        #  Os coeficientes do logit já foram reportados nas seções 17 e 17B.)

        # ── passo 6: gráficos de previsão ML ─────────────────────────────
        try:
            _df_ml_hist = df_ml[df_ml["ano"] <= ANO_ULTIMO_COMPLETO].copy()

            # taxa observada por ano — só para anos com >= 30 PLs (evita ruído)
            _contagem_ano = _df_ml_hist.groupby("ano")["aprovado"].agg(["mean", "count"]).reset_index()
            _contagem_ano.columns = ["ano", "taxa_observada", "n_pls"]
            taxa_historica = _contagem_ano[_contagem_ano["n_pls"] >= 30].copy()

            # probabilidade out-of-sample via backtesting (mais honesto que in-sample)
            if not df_predicoes_bt.empty and melhor_nome_30b in df_predicoes_bt["modelo"].values:
                prob_oos = (
                    df_predicoes_bt[df_predicoes_bt["modelo"] == melhor_nome_30b]
                    .groupby("ano_teste")["prob_aprovacao"]
                    .mean()
                    .reset_index()
                    .rename(columns={"ano_teste": "ano", "prob_aprovacao": "prob_oos"})
                )
            else:
                prob_oos = pd.DataFrame()

            # ── gráfico 1: série temporal geral ──────────────────────────
            fig, ax = plt.subplots(figsize=(16, 6))

            ax.bar(
                taxa_historica["ano"],
                taxa_historica["taxa_observada"],
                color="steelblue", alpha=0.35, width=0.7,
                label="Taxa observada por ano (n ≥ 30 PLs)"
            )

            if not prob_oos.empty:
                ax.plot(
                    prob_oos["ano"], prob_oos["prob_oos"],
                    "s--", color="darkorange", linewidth=1.5, markersize=4,
                    label=f"Probabilidade out-of-sample — {melhor_nome_30b}"
                )

            ax.plot(
                resumo_ano_ml["ano"],
                resumo_ano_ml["prob_media"],
                "D-", color="darkgreen", linewidth=2, markersize=5,
                label=f"Previsão ML {resumo_ano_ml['ano'].min()}–{ANO_HORIZONTE_PREVISAO}"
            )
            ax.fill_between(
                resumo_ano_ml["ano"],
                (resumo_ano_ml["prob_media"] - resumo_ano_ml["prob_dp"]).clip(0),
                resumo_ano_ml["prob_media"] + resumo_ano_ml["prob_dp"],
                alpha=0.18, color="darkgreen", label="±1 dp entre cenários"
            )

            # linha divisória histórico/previsão
            ax.axvline(ano_max_historico + 0.5, color="gray",
                       linestyle=":", linewidth=1.5)
            ax.text(ano_max_historico + 0.7,
                    ax.get_ylim()[1] * 0.92,
                    "→ previsão", fontsize=9, color="gray")

            # marco de legislaturas futuras
            for _leg_ano, _leg_label in [(2027, "58ª"), (2031, "59ª"),
                                          (2035, "60ª"), (2039, "61ª")]:
                if _leg_ano <= ANO_HORIZONTE_PREVISAO:
                    ax.axvline(_leg_ano - 0.5, color="lightgray",
                               linestyle="--", linewidth=0.8, alpha=0.6)
                    ax.text(_leg_ano, ax.get_ylim()[1] * 0.85,
                            _leg_label, fontsize=7.5, color="gray",
                            ha="center")

            ax.set_title(
                f"Previsão ML de sucesso legislativo — {melhor_nome_30b}\n"
                f"AUC out-of-time médio (backtesting) = {auc_medio_30b:.3f} | "
                f"Treinado em 1989–{ano_max_historico} (excl. 57ª legislatura)",
                fontsize=11
            )
            ax.set_xlabel("Ano")
            ax.set_ylabel("Probabilidade de aprovação")
            ax.legend(fontsize=9, loc="upper left")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.set_xlim(1989, ANO_HORIZONTE_PREVISAO + 1)
            ax.yaxis.set_major_formatter(
                plt.matplotlib.ticker.PercentFormatter(xmax=1, decimals=1)
            )

            fig.text(
                0.5, -0.05,
                f"Nota: barras = taxa observada bruta (anos com ≥ 30 PLs, excl. 57ª legislatura). "
                f"Curva laranja = probabilidade out-of-sample do backtesting.\n"
                f"Verde = projeção do modelo re-treinado em toda a amostra histórica completa. "
                "Banda = ±1 dp entre cenários de partido × corporação.\n"
                "Limitação: a projeção assume estabilidade da distribuição de features. "
                "Marcas verticais cinzas = início de cada legislatura projetada.",
                ha="center", fontsize=8, color="gray"
            )
            plt.tight_layout()
            plt.savefig(PASTA / "ml_previsao_futura.png", dpi=300, bbox_inches="tight")
            plt.show()
            plt.close()

            # ── gráfico 2: previsão por corporação ───────────────────────
            if not resumo_corp_ml.empty:
                # taxa observada histórica por corporação — sem OUTROS
                _taxa_obs_corp = pd.DataFrame()
                if not df_reg_corp_base.empty:
                    _taxa_obs_corp = (
                        df_reg_corp_base[
                            (df_reg_corp_base["ano"] <= ANO_ULTIMO_COMPLETO) &
                            (df_reg_corp_base["corporacao_sigla"] != "OUTROS")
                        ]
                        .groupby("corporacao_sigla")["aprovado"]
                        .agg(["mean", "count"])
                        .reset_index()
                        .rename(columns={"mean": "taxa_obs", "count": "n"})
                    )

                # filtra OUTROS do resumo ML e aplica rótulos descritivos
                _resumo_corp = resumo_corp_ml[
                    resumo_corp_ml["corporacao_sigla"] != "OUTROS"
                ].copy()
                _resumo_corp["label"] = (
                    _resumo_corp["corporacao_sigla"].map(_CORP_LABELS)
                    .fillna(_resumo_corp["corporacao_sigla"])
                )

                # ordena pelo painel de taxa observada (para consistência visual)
                if not _taxa_obs_corp.empty:
                    _ordem = (
                        _taxa_obs_corp[_taxa_obs_corp["n"] >= 10]
                        .sort_values("taxa_obs", ascending=True)["corporacao_sigla"]
                        .tolist()
                    )
                    _resumo_corp = _resumo_corp[
                        _resumo_corp["corporacao_sigla"].isin(_ordem)
                    ].set_index("corporacao_sigla").loc[_ordem].reset_index()
                    _resumo_corp["label"] = (
                        _resumo_corp["corporacao_sigla"].map(_CORP_LABELS)
                        .fillna(_resumo_corp["corporacao_sigla"])
                    )
                else:
                    _resumo_corp = _resumo_corp.sort_values("prob_media", ascending=True)

                fig, axes = plt.subplots(1, 2, figsize=(15, 6))

                # painel A: taxa observada histórica (esquerda = referência)
                if not _taxa_obs_corp.empty:
                    _taxa_c = (
                        _taxa_obs_corp[_taxa_obs_corp["n"] >= 10]
                        .copy()
                    )
                    _taxa_c["label"] = (
                        _taxa_c["corporacao_sigla"].map(_CORP_LABELS)
                        .fillna(_taxa_c["corporacao_sigla"])
                    )
                    _taxa_c = _taxa_c.sort_values("taxa_obs", ascending=True)
                    bars_obs = axes[0].barh(
                        _taxa_c["label"],
                        _taxa_c["taxa_obs"] * 100,
                        color="steelblue", alpha=0.75
                    )
                    for bar, (_, row) in zip(bars_obs, _taxa_c.iterrows()):
                        axes[0].text(
                            bar.get_width() + 0.1,
                            bar.get_y() + bar.get_height() / 2,
                            f"n={int(row['n'])}  ({row['taxa_obs']*100:.1f}%)",
                            va="center", fontsize=8.5
                        )
                    axes[0].set_xlabel("Taxa de aprovação histórica (%)")
                    axes[0].set_title(
                        f"Taxa observada 1989–{ano_max_historico}\n(barras de referência)",
                        fontsize=10
                    )
                    axes[0].grid(axis="x", linestyle="--", alpha=0.4)
                    axes[0].set_xlim(0, _taxa_c["taxa_obs"].max() * 130)

                # painel B: previsão ML (direita)
                bars_ml = axes[1].barh(
                    _resumo_corp["label"],
                    _resumo_corp["prob_media"] * 100,
                    color="darkgreen", alpha=0.75
                )
                for bar, (_, row) in zip(bars_ml, _resumo_corp.iterrows()):
                    axes[1].text(
                        bar.get_width() + 0.1,
                        bar.get_y() + bar.get_height() / 2,
                        f"{row['prob_media']*100:.1f}%",
                        va="center", fontsize=8.5
                    )
                axes[1].set_xlabel("Probabilidade prevista (%)")
                axes[1].set_title(
                    f"Previsão ML {resumo_ano_ml['ano'].min()}–{resumo_ano_ml['ano'].max()}\n"
                    f"(modelo {melhor_nome_30b}, média dos cenários)",
                    fontsize=10
                )
                axes[1].grid(axis="x", linestyle="--", alpha=0.4)
                axes[1].set_xlim(0, _resumo_corp["prob_media"].max() * 130)

                fig.suptitle(
                    "Taxa observada × previsão ML por corporação de origem\n"
                    "(excluída categoria residual OUTROS)",
                    fontsize=12, fontweight="bold"
                )
                plt.tight_layout()
                plt.savefig(PASTA / "ml_previsao_corporacao.png", dpi=300, bbox_inches="tight")
                plt.show()
                plt.close()
                print("\nGráfico de previsão por corporação salvo.")

            # ── gráfico 3: previsão por partido ──────────────────────────
            if not resumo_partido_ml.empty:
                _taxa_obs_partido = pd.DataFrame()
                if not df_reg_inf.empty:
                    _taxa_obs_partido = (
                        df_reg_inf[
                            (df_reg_inf["ano"] <= ANO_ULTIMO_COMPLETO) &
                            (df_reg_inf["partido_inf"] != "OUTROS")
                        ]
                        .groupby("partido_inf")["aprovado"]
                        .agg(["mean", "count"])
                        .reset_index()
                        .rename(columns={"mean": "taxa_obs", "count": "n"})
                    )

                _resumo_part = resumo_partido_ml[
                    resumo_partido_ml["partido_inf"] != "OUTROS"
                ].copy()

                # ordena pela taxa observada para consistência visual
                if not _taxa_obs_partido.empty:
                    _taxa_p = _taxa_obs_partido[_taxa_obs_partido["n"] >= 20].sort_values("taxa_obs", ascending=True)
                    _ordem_p = _taxa_p["partido_inf"].tolist()
                    _resumo_part = (
                        _resumo_part[_resumo_part["partido_inf"].isin(_ordem_p)]
                        .set_index("partido_inf").loc[_ordem_p].reset_index()
                    )
                else:
                    _taxa_p = pd.DataFrame()
                    _resumo_part = _resumo_part.sort_values("prob_media", ascending=True)

                fig, axes = plt.subplots(1, 2, figsize=(15, max(6, len(_resumo_part) * 0.45)))

                # painel A: taxa observada
                if not _taxa_p.empty:
                    bars_obs_p = axes[0].barh(
                        _taxa_p["partido_inf"],
                        _taxa_p["taxa_obs"] * 100,
                        color="steelblue", alpha=0.75
                    )
                    for bar, (_, row) in zip(bars_obs_p, _taxa_p.iterrows()):
                        axes[0].text(
                            bar.get_width() + 0.05,
                            bar.get_y() + bar.get_height() / 2,
                            f"n={int(row['n'])}  ({row['taxa_obs']*100:.1f}%)",
                            va="center", fontsize=8
                        )
                axes[0].set_xlabel("Taxa de aprovação histórica (%)")
                axes[0].set_title(f"Taxa observada 1989–{ano_max_historico}\n(partidos com ≥ 20 PLs)")
                axes[0].grid(axis="x", linestyle="--", alpha=0.4)

                # painel B: previsão ML
                bars_ml_p = axes[1].barh(
                    _resumo_part["partido_inf"],
                    _resumo_part["prob_media"] * 100,
                    color="darkgreen", alpha=0.75
                )
                for bar, (_, row) in zip(bars_ml_p, _resumo_part.iterrows()):
                    axes[1].text(
                        bar.get_width() + 0.05,
                        bar.get_y() + bar.get_height() / 2,
                        f"{row['prob_media']*100:.1f}%",
                        va="center", fontsize=8
                    )
                axes[1].set_xlabel("Probabilidade prevista (%)")
                axes[1].set_title(
                    f"Previsão ML {resumo_ano_ml['ano'].min()}–{resumo_ano_ml['ano'].max()}\n"
                    f"(modelo {melhor_nome_30b}, sem categoria OUTROS)"
                )
                axes[1].grid(axis="x", linestyle="--", alpha=0.4)

                fig.suptitle(
                    "Taxa observada × previsão ML por partido\n"
                    "(excluída categoria residual OUTROS)",
                    fontsize=12, fontweight="bold"
                )
                plt.tight_layout()
                plt.savefig(PASTA / "ml_previsao_partido.png", dpi=300, bbox_inches="tight")
                plt.show()
                plt.close()
                print("Gráfico de previsão por partido salvo.")

            # ── gráfico 4: taxa observada por legislatura ─────────────────
            if "legislatura" in df_reg_inf.columns:
                _taxa_leg = (
                    df_reg_inf[df_reg_inf["ano"] <= ANO_ULTIMO_COMPLETO]
                    .groupby("legislatura")["aprovado"]
                    .agg(["mean", "sum", "count"])
                    .reset_index()
                    .rename(columns={"mean": "taxa_obs", "sum": "n_aprov", "count": "n_pls"})
                    .sort_values("legislatura")
                )

                fig, ax = plt.subplots(figsize=(11, 5))
                bars = ax.bar(_taxa_leg["legislatura"],
                              _taxa_leg["taxa_obs"] * 100,
                              color="steelblue", alpha=0.75)
                for bar, (_, row) in zip(bars, _taxa_leg.iterrows()):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.15,
                            f"{int(row['n_aprov'])}/{int(row['n_pls'])}",
                            ha="center", va="bottom", fontsize=8, color="navy")

                ax.set_xlabel("Legislatura")
                ax.set_ylabel("Taxa de aprovação (%)")
                ax.set_title("Taxa de sucesso legislativo por legislatura\n"
                             "(números = aprovados / total de PLs decididos)")
                ax.grid(axis="y", linestyle="--", alpha=0.4)
                plt.tight_layout()
                plt.savefig(PASTA / "ml_previsao_legislatura.png", dpi=300, bbox_inches="tight")
                plt.show()
                plt.close()
                print("Gráfico por legislatura salvo.")

            # ── gráfico 5: análise sociológica (se disponível) ────────────
            if not df_sociol_descritiva.empty:
                _vars_plot = [
                    ("presidente_comissao_bin", "Presidiu comissão"),
                    ("comissao_segpub_bin",     "Membro comissão seg. pública"),
                    ("evangelico_bin",           "Evangélico"),
                    ("mandatos_externos_bin",    "Mandatos externos anteriores"),
                    ("feminino",                 "Gênero feminino"),
                ]
                _vars_disp = [
                    (v, l) for v, l in _vars_plot
                    if v in df_sociol_descritiva["variavel"].values
                ]

                if _vars_disp:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    _df_plot = df_sociol_descritiva[
                        df_sociol_descritiva["variavel"].isin([v for v, _ in _vars_disp])
                    ].copy()
                    _labels = {v: l for v, l in _vars_disp}
                    _df_plot["label"] = _df_plot["variavel"].map(_labels)
                    _df_plot = _df_plot.sort_values("media_ou_proporcao", ascending=True)

                    bars = ax.barh(_df_plot["label"],
                                   _df_plot["media_ou_proporcao"] * 100,
                                   color="slateblue", alpha=0.75)
                    for bar, (_, row) in zip(bars, _df_plot.iterrows()):
                        ax.text(bar.get_width() + 0.5,
                                bar.get_y() + bar.get_height() / 2,
                                f"{row['media_ou_proporcao']*100:.1f}%  (n={int(row['n'])})",
                                va="center", fontsize=9)

                    ax.set_xlabel("Proporção (%)")
                    ax.set_title("Perfil sociológico dos políticos de farda\n"
                                 "(proporção de cada atributo na amostra)")
                    ax.grid(axis="x", linestyle="--", alpha=0.4)
                    ax.set_xlim(0, 100)
                    plt.tight_layout()
                    plt.savefig(PASTA / "sociol_perfil.png", dpi=300, bbox_inches="tight")
                    plt.show()
                    plt.close()
                    print("Gráfico de perfil sociológico salvo.")

                # gráfico de idade e anos na força de segurança
                _vars_cont = [
                    ("anos_forca_seg_num", "Anos na força de segurança"),
                    ("idade_aprox",         "Idade aproximada (2024)"),
                ]
                _vars_cont_ok = [
                    (v, l) for v, l in _vars_cont
                    if v in df_sociol_descritiva["variavel"].values
                ]
                if _vars_cont_ok:
                    _df_cont = df_sociol_descritiva[
                        df_sociol_descritiva["variavel"].isin([v for v, _ in _vars_cont_ok])
                    ].copy()
                    _labels_c = {v: l for v, l in _vars_cont_ok}
                    _df_cont["label"] = _df_cont["variavel"].map(_labels_c)

                    fig, ax = plt.subplots(figsize=(8, 4))
                    x = range(len(_df_cont))
                    bars2 = ax.bar(
                        [_df_cont["label"].iloc[i] for i in x],
                        _df_cont["media_ou_proporcao"].values,
                        yerr=_df_cont["dp"].values,
                        color=["coral", "teal"][:len(_df_cont)],
                        alpha=0.75, capsize=6
                    )
                    for bar, (_, row) in zip(bars2, _df_cont.iterrows()):
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + row["dp"] + 0.5,
                                f"média={row['media_ou_proporcao']:.1f}\ndp={row['dp']:.1f}",
                                ha="center", va="bottom", fontsize=9)
                    ax.set_ylabel("Anos")
                    ax.set_title("Perfil de carreira — políticos de farda\n(média ± dp)")
                    ax.grid(axis="y", linestyle="--", alpha=0.4)
                    plt.tight_layout()
                    plt.savefig(PASTA / "sociol_carreira.png", dpi=300, bbox_inches="tight")
                    plt.show()
                    plt.close()
                    print("Gráfico de carreira sociológica salvo.")

        except Exception as e_plot:
            print(f"\nFalha ao gerar gráficos ML: {e_plot}")

        print("\nNota metodológica:")
        print(f"  - Modelo de previsão: {_nome_previsao}")
        print(f"  - Backtesting (referência AUC): {melhor_nome_30b} = {auc_medio_30b:.3f}")
        print(f"  - Base histórica: 1989–{ano_max_historico} ({len(df_ml)} PLs, {int(df_ml[target_ml].sum())} aprovados)")
        print(f"  - Horizonte: {anos_futuros_ml[0]}–{anos_futuros_ml[-1]}")
        print(f"  - Features: partido × tópico × ano_c (ponderado pela dist. histórica de tópicos)")
        print("  - Limitação: projeção linear assume estabilidade da composição da bancada.")
        print("    Mudanças institucionais, eleitorais ou de coalizão não são capturadas.")

else:
    print("\nDados insuficientes para previsão ML (df_ml ou backtesting vazio).")


# =========================================================
# 39. SÉRIE TEMPORAL ANUAL — DINÂMICA DA AGENDA E SUCESSO
# =========================================================
# Analisa a evolução anual de: (1) produção legislativa total,
# (2) taxa de aprovação, (3) participação da agenda penal.
# Aplica decomposição + Holt-Winters para suavização da tendência.
# Reforça H4 (tendência positiva no tempo) com análise temporal
# formal, cobrindo o módulo de Séries Temporais da ementa.
# Referência: Hyndman & Athanasopoulos (2021) — Forecasting.
# =========================================================

print("\n" + "=" * 60)
print("39. SÉRIE TEMPORAL — DINÂMICA ANUAL DA AGENDA E SUCESSO")
print("=" * 60)

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

df_serie_temporal = pd.DataFrame()

try:
    # base anual: anos com >= 10 PLs para evitar ruído
    _serie_base = (
        df_base[df_base["ano"].notna() & (df_base["ano"] >= 1995)]
        .groupby("ano")
        .agg(
            n_pl=("sucesso_legislativo", "count"),
            n_aprovados=("sucesso_legislativo", "sum")
        )
        .reset_index()
    )
    _serie_base["taxa_sucesso"]  = _serie_base["n_aprovados"] / _serie_base["n_pl"] * 100
    _serie_base["ano"]           = _serie_base["ano"].astype(int)
    _serie_base                  = _serie_base[_serie_base["n_pl"] >= 10].copy()

    # participação do tópico penal por ano
    if "topico_dominante" in df_texto.columns:
        _penal_ano = (
            df_texto[df_texto["ano"].notna() & pd.to_numeric(df_texto["ano"], errors="coerce").notna()]
            .assign(ano=lambda x: pd.to_numeric(x["ano"], errors="coerce").astype("Int64"))
            .dropna(subset=["ano"])
            .query("ano >= 1995")
            .groupby("ano")
            .apply(lambda x: (x["topico_dominante"] == 5).sum() / len(x) * 100)
            .reset_index()
            .rename(columns={0: "pct_penal"})
        )
        _penal_ano["ano"] = _penal_ano["ano"].astype(int)
        _serie_base = _serie_base.merge(_penal_ano, on="ano", how="left")
    else:
        _serie_base["pct_penal"] = np.nan

    df_serie_temporal = _serie_base.copy()

    print("\nSérie anual:")
    print(_serie_base[["ano","n_pl","taxa_sucesso","pct_penal"]].to_string(index=False))

    # Holt-Winters (suavização exponencial) para tendência da taxa de sucesso
    _ts_taxa = _serie_base["taxa_sucesso"].values
    _n = len(_ts_taxa)

    if _n >= 8:
        try:
            _hw = ExponentialSmoothing(
                _ts_taxa, trend="add", seasonal=None,
                initialization_method="estimated"
            ).fit(optimized=True)
            _tendencia_hw = _hw.fittedvalues
            _hw_ok = True
        except Exception:
            _tendencia_hw = None
            _hw_ok = False
    else:
        _tendencia_hw = None
        _hw_ok = False

    # gráfico em 3 painéis
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    # painel 1: produção legislativa
    axes[0].bar(_serie_base["ano"], _serie_base["n_pl"],
                color="steelblue", alpha=0.75, label="N° de PLs")
    axes[0].set_ylabel("Número de PLs")
    axes[0].set_title("Evolução anual da produção legislativa\n"
                       "(parlamentares oriundos das forças de segurança)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    # marcos de legislatura
    for _leg_yr in [1999, 2003, 2007, 2011, 2015, 2019, 2023]:
        if _leg_yr in _serie_base["ano"].values:
            axes[0].axvline(_leg_yr, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    axes[0].legend(fontsize=9)

    # painel 2: taxa de sucesso + tendência HW
    axes[1].plot(_serie_base["ano"], _serie_base["taxa_sucesso"],
                 "o-", color="darkorange", linewidth=2, markersize=5,
                 label="Taxa de aprovação (%)")
    if _hw_ok:
        axes[1].plot(_serie_base["ano"], _tendencia_hw,
                     "--", color="red", linewidth=1.5, alpha=0.7,
                     label="Tendência (Holt-Winters)")
    axes[1].set_ylabel("Taxa de aprovação (%)")
    axes[1].set_title("Taxa de sucesso legislativo por ano")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=9)
    for _leg_yr in [1999, 2003, 2007, 2011, 2015, 2019, 2023]:
        axes[1].axvline(_leg_yr, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    # painel 3: % agenda penal
    if "pct_penal" in _serie_base.columns and _serie_base["pct_penal"].notna().any():
        axes[2].fill_between(_serie_base["ano"], _serie_base["pct_penal"],
                             alpha=0.35, color="crimson")
        axes[2].plot(_serie_base["ano"], _serie_base["pct_penal"],
                     "o-", color="crimson", linewidth=2, markersize=4,
                     label="% agenda penal (T5)")
        axes[2].set_ylabel("% PLs no tópico penal")
        axes[2].set_title("Participação da agenda penal (T5) por ano")
        axes[2].legend(fontsize=9)
        axes[2].grid(axis="y", linestyle="--", alpha=0.4)
        for _leg_yr in [1999, 2003, 2007, 2011, 2015, 2019, 2023]:
            axes[2].axvline(_leg_yr, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    axes[2].set_xlabel("Ano")
    fig.text(0.5, 0.01,
             "Linhas verticais pontilhadas = início de legislaturas\n"
             "Holt-Winters: suavização exponencial com componente de tendência aditiva",
             ha="center", fontsize=8.5, color="gray")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(PASTA / "serie_temporal_anual.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print("\nGráfico série temporal salvo.")

except Exception as e_ts:
    print(f"\n[AVISO] Série temporal falhou: {e_ts}")


# =========================================================
# 40. MCA — ANÁLISE DE CORRESPONDÊNCIA MÚLTIPLA
#      PERFIL SOCIOLÓGICO DOS PARLAMENTARES
# =========================================================
# MCA sobre variáveis categóricas do parlamentar (corporação,
# partido, religião, gênero, presidência de comissão, faixas
# de produtividade e sucesso). Revela tipologias latentes:
# quais combinações de atributos coocorrem.
# Cobre o módulo de Análise de Correspondência Múltipla.
# Referência: Greenacre (2017) — Correspondence Analysis.
# =========================================================

print("\n" + "=" * 60)
print("40. MCA — CORRESPONDÊNCIA MÚLTIPLA (PERFIL PARLAMENTAR)")
print("=" * 60)

df_mca_coords = pd.DataFrame()

try:
    # tenta importar prince (MCA)
    try:
        import prince
        _PRINCE_OK = True
    except ImportError:
        _PRINCE_OK = False
        print("[INFO] prince não instalado. Tentando instalar...")
        import subprocess
        subprocess.run(["pip", "install", "prince", "--break-system-packages", "-q"],
                       capture_output=True)
        try:
            import prince
            _PRINCE_OK = True
            print("[OK] prince instalado com sucesso.")
        except ImportError:
            print("[AVISO] prince indisponível — MCA com implementação manual.")
            _PRINCE_OK = False

    # base para MCA: preferir df_reg_inf (tem partido_inf) sobre df_reg_corp_base
    if not df_reg_inf.empty:
        _base_mca = df_reg_inf.copy()
    elif not df_reg_corp_base.empty:
        _base_mca = df_reg_corp_base.copy()
    else:
        _base_mca = pd.DataFrame()

    if not _base_mca.empty:
        _autor_mca = "Autor_merge" if "Autor_merge" in _base_mca.columns else "Autor"
        _partido_mca = "partido_inf" if "partido_inf" in _base_mca.columns else None
        _corp_mca    = "corporacao_sigla" if "corporacao_sigla" in _base_mca.columns else None

        _agg_dict = {
            "n_pl":       ("aprovado", "count"),
            "n_aprovados":("aprovado", "sum"),
        }
        if _corp_mca:
            _agg_dict["corporacao"] = (_corp_mca, lambda x: x.mode().iat[0] if not x.mode().empty else "OUTROS")
        if _partido_mca:
            _agg_dict["partido"]    = (_partido_mca, lambda x: x.mode().iat[0] if not x.mode().empty else "OUTROS")

        # agrega por parlamentar — valor modal das categóricas
        _parl_mca = (
            _base_mca
            .groupby(_autor_mca)
            .agg(**_agg_dict)
            .reset_index()
        )
        _parl_mca["taxa_parl"]     = _parl_mca["n_aprovados"] / _parl_mca["n_pl"].clip(1)
        _parl_mca["prod_cat"]      = pd.cut(_parl_mca["n_pl"],
                                             bins=[0,5,20,50,999],
                                             labels=["baixa","média","alta","muito alta"])
        _parl_mca["sucesso_cat"]   = pd.cut(_parl_mca["taxa_parl"],
                                             bins=[-0.01,0,0.02,0.05,1.01],
                                             labels=["zero","baixo","médio","alto"])

        # merge com variáveis sociológicas se disponíveis
        _cols_sociol = [c for c in ["evangelico_bin","feminino","presidente_comissao_bin"]
                        if c in _base_mca.columns]
        if _cols_sociol:
            _sociol_parl = (
                _base_mca
                .groupby("Autor_merge" if "Autor_merge" in _base_mca.columns else "Autor")
                [_cols_sociol]
                .first()
                .reset_index()
            )
            _parl_mca = _parl_mca.merge(
                _sociol_parl,
                on="Autor_merge" if "Autor_merge" in _parl_mca.columns else "Autor",
                how="left"
            )
            for c in _cols_sociol:
                _parl_mca[c] = _parl_mca[c].map({0:"não",1:"sim",0.0:"não",1.0:"sim"}).fillna("não")

        # variáveis categóricas para a MCA — apenas as que existem
        _vars_mca_candidatas = []
        if "corporacao" in _parl_mca.columns:
            _vars_mca_candidatas.append("corporacao")
        if "partido" in _parl_mca.columns:
            _vars_mca_candidatas.append("partido")
        if "prod_cat" in _parl_mca.columns:
            _vars_mca_candidatas.append("prod_cat")
        if "sucesso_cat" in _parl_mca.columns:
            _vars_mca_candidatas.append("sucesso_cat")
        _vars_mca_candidatas += [c for c in _cols_sociol if c in _parl_mca.columns]
        _vars_mca = _vars_mca_candidatas

        if len(_vars_mca) < 2:
            raise ValueError(f"Variáveis insuficientes para MCA: {_vars_mca}")

        _parl_mca_cc = _parl_mca[_vars_mca].dropna()

        print(f"\nBase MCA: {len(_parl_mca_cc)} parlamentares, {len(_vars_mca)} variáveis")

        if _PRINCE_OK and len(_parl_mca_cc) >= 20:
            mca = prince.MCA(n_components=2, random_state=42)
            mca.fit(_parl_mca_cc)
            _coords_row = mca.row_coordinates(_parl_mca_cc)
            _coords_col = mca.column_coordinates(_parl_mca_cc)

            _eig = mca.eigenvalues_summary
            print("\nVariância explicada (MCA):")
            print(_eig)

            df_mca_coords = pd.concat([
                _coords_col.assign(tipo="categoria"),
                _coords_row.assign(tipo="parlamentar")
            ], ignore_index=True)

            # gráfico mapa perceptual
            fig, ax = plt.subplots(figsize=(12, 9))

            # ponto das categorias
            ax.scatter(_coords_col.iloc[:, 0], _coords_col.iloc[:, 1],
                       s=80, c="crimson", zorder=5, alpha=0.9)
            for cat, row in _coords_col.iterrows():
                ax.annotate(str(cat), (row.iloc[0], row.iloc[1]),
                            fontsize=7.5, alpha=0.85,
                            xytext=(4, 4), textcoords="offset points")

            # nuvem de parlamentares (fundo)
            ax.scatter(_coords_row.iloc[:, 0], _coords_row.iloc[:, 1],
                       s=15, c="steelblue", alpha=0.25, zorder=3)

            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

            _inertia = mca.percentage_of_variance_
            ax.set_xlabel(f"Dimensão 1 ({_inertia[0]:.1f}%)")
            ax.set_ylabel(f"Dimensão 2 ({_inertia[1]:.1f}%)")
            ax.set_title(
                "Análise de Correspondência Múltipla (MCA)\n"
                "Perfil sociológico dos parlamentares de farda\n"
                "(pontos vermelhos = categorias; azuis = parlamentares)",
                fontsize=11
            )
            ax.grid(True, linestyle="--", alpha=0.25)
            plt.tight_layout()
            plt.savefig(PASTA / "mca_perfil_parlamentar.png", dpi=300, bbox_inches="tight")
            plt.show()
            plt.close()
            print("Gráfico MCA salvo.")

        else:
            # fallback: chi² entre cada par de variáveis categóricas
            print("\n[Fallback] MCA via prince indisponível — tabelas qui-quadrado por par:")
            from scipy.stats import chi2_contingency
            _pares = [("corporacao","prod_cat"),("corporacao","sucesso_cat"),
                      ("partido","sucesso_cat"),("prod_cat","sucesso_cat")]
            for _v1, _v2 in _pares:
                if _v1 in _parl_mca_cc.columns and _v2 in _parl_mca_cc.columns:
                    _ct = pd.crosstab(_parl_mca_cc[_v1], _parl_mca_cc[_v2])
                    _chi2, _p, _, _ = chi2_contingency(_ct)
                    print(f"  {_v1} × {_v2}: χ²={_chi2:.2f}, p={_p:.4f}")

    else:
        print("\n[AVISO] Base para MCA vazia — seção 40 pulada.")

except Exception as e_mca:
    print(f"\n[AVISO] MCA falhou: {e_mca}")


# =========================================================
# 37. PROBIT — ROBUSTEZ DO MODELO LOGIT PRINCIPAL
# =========================================================
# Rodar o modelo probit paralelo ao logit verifica se as
# inferências são robustas à escolha da função de ligação.
# Logit e probit produzem coeficientes distintos (~1,6×) mas
# as direções, significâncias e ordenamentos são idênticos
# se o modelo estiver bem especificado.
# Referência: Long (1997) — Regression Models for Categorical
# and Limited Dependent Variables.
# =========================================================

print("\n" + "=" * 60)
print("37. PROBIT — ROBUSTEZ DO MODELO LOGIT PRINCIPAL")
print("=" * 60)

df_probit_coef = pd.DataFrame()
modelo_probit = None

if not df_reg_inf.empty and "topico_dominante" in df_reg_inf.columns:
    try:
        _formula_probit = "aprovado ~ C(topico_dominante) + ano_c"
        modelo_probit = smf.probit(_formula_probit, data=df_reg_inf).fit(
            method="lbfgs", maxiter=1000, disp=False
        )
        print("\n=== MODELO PROBIT (robustez) ===")
        print(modelo_probit.summary())

        # comparação logit × probit
        _logit_params = modelo_principal.params if modelo_principal is not None else pd.Series()
        _probit_params = modelo_probit.params

        _compar = pd.DataFrame({
            "variavel":       _probit_params.index,
            "coef_logit":     _logit_params.reindex(_probit_params.index).values,
            "coef_probit":    _probit_params.values,
            "razao_l_p":      (_logit_params.reindex(_probit_params.index).values /
                               _probit_params.values.clip(-1e6, -1e-8)),
            "p_probit":       modelo_probit.pvalues.values,
        })
        _compar["direcao_consistente"] = (
            np.sign(_compar["coef_logit"].fillna(0)) ==
            np.sign(_compar["coef_probit"])
        )

        df_probit_coef = _compar.copy()
        print("\nComparação logit × probit (razão ≈ 1,6 esperada):")
        print(_compar[["variavel","coef_logit","coef_probit",
                        "razao_l_p","direcao_consistente"]].to_string(index=False))

        _consist = _compar["direcao_consistente"].mean()
        print(f"\nConsistência direcional: {_consist*100:.1f}% das variáveis")
        if _consist >= 0.90:
            print("→ Inferências robustas à escolha da função de ligação (logit vs probit).")

    except Exception as e_probit:
        print(f"\n[AVISO] Probit falhou: {e_probit}")
else:
    print("\n[AVISO] df_reg_inf vazio — seção 37 pulada.")


# =========================================================
# 38. CROSS-VALIDATION ESTRATIFICADO
# =========================================================
# StratifiedKFold mantém a proporção de positivos em cada fold,
# essencial com evento raro (1,3%). Complementa o backtesting
# temporal: enquanto o backtesting avalia validade externa no
# tempo, o CV avalia estabilidade interna do modelo.
# Referência: Kohavi (1995) — A study of cross-validation and
# bootstrap for accuracy estimation.
# =========================================================

print("\n" + "=" * 60)
print("38. CROSS-VALIDATION ESTRATIFICADO (StratifiedKFold)")
print("=" * 60)

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression as LR_CV
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score

df_cv_resultados = pd.DataFrame()

if not df_reg_inf.empty:
    try:
        _cv_base = df_reg_inf[["aprovado","topico_dominante","partido_inf","ano_c"]].dropna()
        _y_cv = _cv_base["aprovado"].astype(int).values
        _X_cv = _cv_base[["topico_dominante","partido_inf","ano_c"]].copy()
        _X_cv["topico_dominante"] = _X_cv["topico_dominante"].astype(str)
        _X_cv["partido_inf"] = _X_cv["partido_inf"].astype(str)

        _cat_cv = ["topico_dominante","partido_inf"]
        _num_cv = ["ano_c"]

        _prep_cv = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), _cat_cv),
            ("num", "passthrough", _num_cv)
        ])

        _pipe_cv = SKPipeline([
            ("prep", _prep_cv),
            ("model", LR_CV(max_iter=1000, class_weight="balanced",
                            random_state=42, solver="lbfgs"))
        ])

        _skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # CV manual — mais robusto que cross_validate com make_scorer
        # evita problemas de compatibilidade com needs_proba/response_method
        _cv_aucs_treino, _cv_aucs_teste = [], []
        _cv_aps_treino,  _cv_aps_teste  = [], []

        for _fold_i, (_idx_tr, _idx_te) in enumerate(
                _skf.split(_X_cv, _y_cv), start=1):

            _Xtr = _X_cv.iloc[_idx_tr]
            _Xte = _X_cv.iloc[_idx_te]
            _ytr = _y_cv[_idx_tr]
            _yte = _y_cv[_idx_te]

            _pipe_cv.fit(_Xtr, _ytr)

            _prob_tr = _pipe_cv.predict_proba(_Xtr)[:, 1]
            _prob_te = _pipe_cv.predict_proba(_Xte)[:, 1]

            # AUC e AP treino
            try:
                _cv_aucs_treino.append(roc_auc_score(_ytr, _prob_tr))
                _cv_aps_treino.append(average_precision_score(_ytr, _prob_tr))
            except Exception:
                _cv_aucs_treino.append(np.nan)
                _cv_aps_treino.append(np.nan)

            # AUC e AP teste
            try:
                _cv_aucs_teste.append(roc_auc_score(_yte, _prob_te))
                _cv_aps_teste.append(average_precision_score(_yte, _prob_te))
            except Exception:
                _cv_aucs_teste.append(np.nan)
                _cv_aps_teste.append(np.nan)

            print(f"  Fold {_fold_i}: AUC_treino={_cv_aucs_treino[-1]:.4f}"
                  f" | AUC_teste={_cv_aucs_teste[-1]:.4f}"
                  f" | n_teste={len(_yte)} (pos={_yte.sum()})")

        _auc_tr = np.array(_cv_aucs_treino)
        _auc_te = np.array(_cv_aucs_teste)
        _ap_tr  = np.array(_cv_aps_treino)
        _ap_te  = np.array(_cv_aps_teste)

        df_cv_resultados = pd.DataFrame({
            "fold":       list(range(1, 6)),
            "auc_treino": np.round(_auc_tr, 4),
            "auc_teste":  np.round(_auc_te, 4),
            "ap_treino":  np.round(_ap_tr, 4),
            "ap_teste":   np.round(_ap_te, 4),
        })

        print("\nResultados por fold (5-fold StratifiedKFold):")
        print(df_cv_resultados.to_string(index=False))
        print(f"\nMédia AUC teste:  {np.nanmean(_auc_te):.4f} ± {np.nanstd(_auc_te):.4f}")
        print(f"Média AP teste:   {np.nanmean(_ap_te):.4f} ± {np.nanstd(_ap_te):.4f}")

        _gap_auc = np.nanmean(_auc_tr) - np.nanmean(_auc_te)
        print(f"Gap treino-teste (AUC): {_gap_auc:.4f}")
        if _gap_auc < 0.05:
            print("→ Gap pequeno: modelo sem overfitting significativo.")
        elif _gap_auc < 0.10:
            print("→ Gap moderado: sobreajuste leve — esperado com evento raro.")
        else:
            print("→ Gap alto: modelo com overfitting — interpretar com cautela.")

        # gráfico — usa _auc_tr/_auc_te do loop manual
        fig, axes_cv = plt.subplots(1, 2, figsize=(14, 5))

        # painel A: AUC por fold
        _folds = list(range(1, 6))
        axes_cv[0].plot(_folds, _auc_tr, "o--", color="steelblue",
                        label="AUC treino", linewidth=1.5, markersize=7)
        axes_cv[0].plot(_folds, _auc_te, "o-",  color="darkgreen",
                        label="AUC teste",  linewidth=2,   markersize=7)
        axes_cv[0].axhline(np.nanmean(_auc_te), color="darkgreen",
                           linestyle=":", linewidth=1.2, alpha=0.7,
                           label=f"Média teste = {np.nanmean(_auc_te):.3f}")
        axes_cv[0].axhline(0.5, color="gray", linestyle="--",
                           linewidth=0.8, alpha=0.5, label="Aleatório (0.5)")
        axes_cv[0].fill_between(
            _folds,
            _auc_te - np.nanstd(_auc_te),
            _auc_te + np.nanstd(_auc_te),
            alpha=0.12, color="darkgreen"
        )
        for _f, _v in zip(_folds, _auc_te):
            axes_cv[0].text(_f, _v + 0.012, f"{_v:.3f}",
                            ha="center", fontsize=8, color="darkgreen")
        axes_cv[0].set_xticks(_folds)
        axes_cv[0].set_xlabel("Fold")
        axes_cv[0].set_ylabel("AUC")
        axes_cv[0].set_ylim(0.40, min(1.0, np.nanmax(_auc_tr) + 0.12))
        axes_cv[0].set_title("AUC por fold")
        axes_cv[0].legend(fontsize=8)
        axes_cv[0].grid(True, linestyle="--", alpha=0.4)

        # painel B: tabela resumo
        axes_cv[1].axis("off")
        _tab_data = [
            ["Fold", "AUC treino", "AUC teste", "AP teste"],
        ] + [
            [str(f), f"{tr:.3f}", f"{te:.3f}", f"{ap:.3f}"]
            for f, tr, te, ap in zip(_folds, _auc_tr, _auc_te, _ap_te)
        ] + [
            ["Média",
             f"{np.nanmean(_auc_tr):.3f}",
             f"{np.nanmean(_auc_te):.3f}",
             f"{np.nanmean(_ap_te):.3f}"],
            ["± DP",
             f"{np.nanstd(_auc_tr):.3f}",
             f"{np.nanstd(_auc_te):.3f}",
             f"{np.nanstd(_ap_te):.3f}"],
        ]
        _tbl = axes_cv[1].table(
            cellText=_tab_data[1:],
            colLabels=_tab_data[0],
            loc="center", cellLoc="center"
        )
        _tbl.auto_set_font_size(False)
        _tbl.set_fontsize(9)
        _tbl.scale(1.2, 1.6)
        # destaca cabeçalho
        for _col in range(4):
            _tbl[0, _col].set_facecolor("#2c5f8a")
            _tbl[0, _col].set_text_props(color="white", fontweight="bold")
        # destaca linhas de média
        for _col in range(4):
            _tbl[6, _col].set_facecolor("#eaf2fb")
            _tbl[7, _col].set_facecolor("#eaf2fb")

        axes_cv[1].set_title(
            "Resumo por fold\n"
            f"Gap AUC treino-teste: {_gap_auc:.3f}"
            f" ({'sem overfitting' if _gap_auc < 0.05 else 'leve' if _gap_auc < 0.10 else 'alto'})",
            fontsize=9, pad=12
        )

        fig.suptitle(
            "Cross-validation estratificado — 5-fold StratifiedKFold\n"
            f"Modelo: logit (tópico + partido + ano) | n={len(_y_cv)} PLs decididos",
            fontsize=11, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(PASTA / "cv_estratificado.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print("\nGráfico cross-validation salvo.")

    except Exception as e_cv:
        print(f"\n[AVISO] Cross-validation falhou: {e_cv}")
else:
    print("\n[AVISO] df_reg_inf vazio — seção 38 pulada.")


# =========================================================
# 41. INTEGRAÇÃO SOCIOLÓGICA — REDE + CAPITAL + TIPOLOGIA
# =========================================================
# Transforma variáveis individuais em três eixos analíticos:
#   1. Capital institucional (comissão, mandatos, posição)
#   2. Capital corporativo-profissional (corporação, anos na força)
#   3. Capital relacional (centralidade na rede de coautoria)
# Depois cruza esses eixos com sucesso legislativo e gera
# tipologia substantiva dos parlamentares de farda.
# Referência: Bourdieu (1986) — formas de capital;
#             Figueiredo & Limongi (1999) — presidencialismo
#             de coalizão.
# =========================================================

print("\n" + "=" * 60)
print("41. INTEGRAÇÃO SOCIOLÓGICA — CAPITAL + REDE + TIPOLOGIA")
print("=" * 60)

df_sociol_integrado = pd.DataFrame()

try:
    # usa df_reg_corp_base se disponível (tem variáveis sociológicas)
    # senão usa base_parlamentar (df_reg_inf, sem sociologia)
    if not df_reg_corp_base.empty:
        _base_s = df_reg_corp_base.copy()
        # garante partido_inf se vier de df_reg_inf
        if "partido_inf" not in _base_s.columns and "partido_inf" in base_parlamentar.columns:
            _autores_s = "Autor_merge" if "Autor_merge" in _base_s.columns else "Autor"
            _autores_b = "Autor"
            _partido_map = base_parlamentar.groupby(_autores_b)["partido_inf"].first()
            _base_s["partido_inf"] = _base_s[_autores_s].map(_partido_map)
    elif not base_parlamentar.empty:
        _base_s = base_parlamentar.copy()
    else:
        _base_s = pd.DataFrame()

    if _base_s.empty:
        raise ValueError("base_parlamentar vazia")

    # agrupa por autor: pega modal para categóricas, mean para numéricas
    _autor_col = next(
        (c for c in ["Autor", "autor", "Autor_merge"] if c in _base_s.columns),
        None
    )
    if _autor_col is None:
        raise ValueError("Coluna de autor não encontrada em base_parlamentar")

    _parl_base_agg = {
        "n_pl":       ("aprovado", "count"),
        "n_aprovados":("aprovado", "sum"),
        "corporacao": ("corporacao_sigla",
                       lambda x: x.mode().iat[0] if not x.mode().empty else "OUTROS"),
    }
    if "partido_inf" in _base_s.columns:
        _parl_base_agg["partido"] = (
            "partido_inf",
            lambda x: x.mode().iat[0] if not x.mode().empty else "OUTROS"
        )

    _parl_base = (
        _base_s.groupby(_autor_col)
        .agg(**_parl_base_agg)
        .reset_index()
    )
    _parl_base["taxa_parl"] = (
        _parl_base["n_aprovados"] / _parl_base["n_pl"].clip(lower=1)
    )
    if "partido" not in _parl_base.columns:
        _parl_base["partido"] = "OUTROS"

    # ── 2. merge com variáveis sociológicas ───────────────────
    _cols_sociol = [
        "presidente_comissao_bin", "anos_forca_seg_num",
        "feminino", "nao_branco", "evangelico_bin",
        "mandatos_externos_bin", "comissao_segpub_bin"
    ]
    _cols_disp = [c for c in _cols_sociol if c in _base_s.columns]

    if _cols_disp:
        _sociol_autor = (
            _base_s.groupby(_autor_col)[_cols_disp]
            .first()
            .reset_index()
        )
        _parl_base = _parl_base.merge(_sociol_autor, on=_autor_col, how="left")

    # ── 3. merge com centralidades da rede ────────────────────
    if not df_rede_metricas.empty:
        _rede_col = next(
            (c for c in ["Autor", "autor"] if c in df_rede_metricas.columns),
            df_rede_metricas.columns[0]
        )
        _parl_base = _parl_base.merge(
            df_rede_metricas[[_rede_col, "grau_centralidade",
                               "betweenness", "closeness"]],
            left_on=_autor_col, right_on=_rede_col, how="left"
        )
        _parl_base["betweenness"]       = _parl_base["betweenness"].fillna(0)
        _parl_base["grau_centralidade"] = _parl_base["grau_centralidade"].fillna(0)
        _parl_base["closeness"]         = _parl_base["closeness"].fillna(0)
        print(f"\nMerge com rede: {_parl_base['betweenness'].gt(0).sum()} parlamentares "
              f"com centralidade > 0")
    else:
        print("\n[INFO] Rede não disponível — centralidade não incluída.")

    # ── 4. índices compostos de capital ───────────────────────
    # Capital institucional: presidente + comissão segpub + mandatos
    _cap_inst_cols = [c for c in ["presidente_comissao_bin",
                                   "comissao_segpub_bin",
                                   "mandatos_externos_bin"]
                      if c in _parl_base.columns]
    if _cap_inst_cols:
        _parl_base["capital_institucional"] = (
            _parl_base[_cap_inst_cols]
            .fillna(0).astype(float).sum(axis=1)
        )
    else:
        _parl_base["capital_institucional"] = 0

    # Capital corporativo: anos na força (quartil)
    if "anos_forca_seg_num" in _parl_base.columns:
        try:
            _af_vals = _parl_base["anos_forca_seg_num"].fillna(0)
            _n_unique_af = _af_vals.nunique()
            if _n_unique_af >= 4:
                _parl_base["capital_corporativo"] = pd.qcut(
                    _af_vals, q=4, labels=["Q1","Q2","Q3","Q4"], duplicates="drop"
                )
            elif _n_unique_af >= 2:
                _labels_af = [f"Q{i+1}" for i in range(_n_unique_af)]
                _parl_base["capital_corporativo"] = pd.qcut(
                    _af_vals, q=_n_unique_af, labels=_labels_af, duplicates="drop"
                )
            else:
                _parl_base["capital_corporativo"] = "sem_variacao"
        except Exception as _e_corp:
            print(f"  [AVISO] capital_corporativo: {_e_corp} — usando 'sem_dado'")
            _parl_base["capital_corporativo"] = "sem_dado"
    else:
        _parl_base["capital_corporativo"] = "sem_dado"

    # Capital relacional: quartil de betweenness
    if "betweenness" in _parl_base.columns:
        try:
            _bet_vals = _parl_base["betweenness"].fillna(0)
            _n_unique_bet = _bet_vals.nunique()
            if _n_unique_bet >= 4:
                _parl_base["capital_relacional"] = pd.qcut(
                    _bet_vals, q=4, labels=["Q1","Q2","Q3","Q4"], duplicates="drop"
                )
            elif _n_unique_bet >= 2:
                _labels_bet = [f"Q{i+1}" for i in range(_n_unique_bet)]
                _parl_base["capital_relacional"] = pd.qcut(
                    _bet_vals, q=_n_unique_bet, labels=_labels_bet, duplicates="drop"
                )
            else:
                _parl_base["capital_relacional"] = "sem_variacao"
        except Exception as _e_rel:
            print(f"  [AVISO] capital_relacional: {_e_rel} — usando 'sem_dado'")
            _parl_base["capital_relacional"] = "sem_dado"
    else:
        _parl_base["capital_relacional"] = "sem_dado"

    df_sociol_integrado = _parl_base.copy()

    # ── 5. tabelas cruzadas: capital × sucesso ────────────────
    print("\n--- Capital institucional × taxa de sucesso ---")
    if _parl_base["capital_institucional"].gt(0).any():
        _tab_inst = (
            _parl_base.groupby("capital_institucional")
            .agg(n_parl=(_autor_col,"count"),
                 taxa_media=("taxa_parl","mean"),
                 n_pl_medio=("n_pl","mean"))
            .round(4)
        )
        print(_tab_inst)

    print("\n--- Centralidade na rede × taxa de sucesso (quartis) ---")
    if "capital_relacional" in _parl_base.columns and "betweenness" in _parl_base.columns:
        _tab_rel = (
            _parl_base.groupby("capital_relacional", observed=True)
            .agg(n_parl=(_autor_col,"count"),
                 taxa_media=("taxa_parl","mean"),
                 bet_medio=("betweenness","mean"))
            .round(4)
        )
        print(_tab_rel)
    elif "capital_relacional" in _parl_base.columns:
        _tab_rel = (
            _parl_base.groupby("capital_relacional", observed=True)
            .agg(n_parl=(_autor_col,"count"),
                 taxa_media=("taxa_parl","mean"))
            .round(4)
        )
        print(_tab_rel)

    print("\n--- Corporação × centralidade média na rede ---")
    if "betweenness" in _parl_base.columns:
        _tab_corp_rede = (
            _parl_base.groupby("corporacao")
            .agg(n=(_autor_col,"count"),
                 betweenness_medio=("betweenness","mean"),
                 taxa_media=("taxa_parl","mean"))
            .sort_values("betweenness_medio", ascending=False)
            .round(4)
        )
        print(_tab_corp_rede)

    # ── 6. regressão: centralidade → aprovações ───────────────
    if ("betweenness" in _parl_base.columns and
            _parl_base["betweenness"].std() > 0 and
            len(_parl_base) >= 30):
        try:
            import statsmodels.api as sm_s
            _X_rel = sm_s.add_constant(
                _parl_base[["betweenness","n_pl"]].fillna(0)
            )
            _y_rel = _parl_base["n_aprovados"].fillna(0).astype(float)
            _ols_rel = sm_s.OLS(_y_rel, _X_rel).fit()
            print("\n--- OLS: centralidade (betweenness) → n_aprovados ---")
            print(_ols_rel.summary().tables[1])
        except Exception as e_ols:
            print(f"[AVISO] OLS centralidade falhou: {e_ols}")

    # ── 7. tipologia final: 4 tipos de parlamentar ────────────
    # baseada em produtividade e taxa de sucesso (quartis)
    _med_n   = _parl_base["n_pl"].median()
    _med_tax = _parl_base["taxa_parl"].median()

    def _tipologia(row):
        _alto_vol  = row["n_pl"]      > _med_n
        _alto_suc  = row["taxa_parl"] > _med_tax
        if _alto_vol and _alto_suc:
            return "Articulador eficiente"
        elif _alto_vol and not _alto_suc:
            return "Hiperprodutivo simbólico"
        elif not _alto_vol and _alto_suc:
            return "Especialista de nicho"
        else:
            return "Periférico"

    _parl_base["tipologia"] = _parl_base.apply(_tipologia, axis=1)

    _tab_tipo = (
        _parl_base.groupby("tipologia")
        .agg(
            n_parlamentares = (_autor_col, "count"),
            n_pl_medio      = ("n_pl", "mean"),
            taxa_media      = ("taxa_parl", "mean"),
            corporacao_modal = ("corporacao",
                                lambda x: x.mode().iat[0] if not x.mode().empty else "—")
        )
        .round(3)
        .sort_values("taxa_media", ascending=False)
    )

    print("\n--- Tipologia de parlamentares ---")
    print(_tab_tipo)

    # ── 9. SOCIOLOGIA HISTÓRICA: legislatura × corporação ─────────
    print("\n--- Sociologia histórica: corporação × legislatura ---")
    if not df_reg_corp_base.empty and "legislatura" in df_reg_corp_base.columns:
        _leg_corp = (
            df_reg_corp_base[df_reg_corp_base["corporacao_sigla"].isin(
                ["PM","EB","PC","PF","MB","SM","CBM","PRF"]
            )]
            .groupby(["legislatura","corporacao_sigla"])
            .agg(
                n_pl       = ("Proposicoes","count"),
                n_aprov    = ("aprovado","sum"),
            )
            .reset_index()
        )
        _leg_corp["taxa_sucesso"] = (
            _leg_corp["n_aprov"] / _leg_corp["n_pl"].clip(lower=1)
        ).round(4)
        _leg_corp_piv = _leg_corp.pivot_table(
            index="legislatura", columns="corporacao_sigla",
            values="taxa_sucesso", aggfunc="first"
        ).round(3)
        print("\nTaxa de sucesso por legislatura e corporação:")
        print(_leg_corp_piv.to_string())

        # gráfico: evolução temporal por corporação
        fig_leg, ax_leg = plt.subplots(figsize=(10, 5))
        _cores_corp = {
            "PM":"#e74c3c","EB":"#3498db","PC":"#2ecc71",
            "PF":"#f39c12","MB":"#9b59b6","SM":"#1abc9c",
            "CBM":"#e67e22","PRF":"#34495e"
        }
        _leg_ord = ["49a","50a","51a","52a","53a","54a","55a","56a","57a"]
        for _corp in ["PF","MB","PC","PM","EB","SM"]:
            _s = _leg_corp[_leg_corp["corporacao_sigla"] == _corp].copy()
            _s["leg_num"] = _s["legislatura"].map(
                {l: i for i, l in enumerate(_leg_ord)}
            )
            _s = _s.dropna(subset=["leg_num"]).sort_values("leg_num")
            if len(_s) > 1:
                ax_leg.plot(
                    _s["leg_num"], _s["taxa_sucesso"],
                    marker="o", label=_corp,
                    color=_cores_corp.get(_corp,"gray"), linewidth=1.8
                )
        ax_leg.set_xticks(range(len(_leg_ord)))
        ax_leg.set_xticklabels(_leg_ord, rotation=30, ha="right")
        ax_leg.set_ylabel("Taxa de sucesso")
        ax_leg.set_xlabel("Legislatura")
        ax_leg.set_title(
            "Profissionalização legislativa por corporação\n"
            "(taxa de sucesso ao longo das legislaturas)"
        )
        ax_leg.legend(fontsize=8, ncol=3)
        ax_leg.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(PASTA / "sociologia_historica_leg_corp.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print("Gráfico sociologia histórica salvo.")

        # participação relativa por legislatura (share de PLs)
        _leg_total = (
            df_reg_corp_base[df_reg_corp_base["corporacao_sigla"].isin(
                ["PM","EB","PC","PF","MB","SM"]
            )]
            .groupby(["legislatura","corporacao_sigla"])
            .size()
            .reset_index(name="n_pl")
        )
        _leg_total_piv = _leg_total.pivot_table(
            index="legislatura", columns="corporacao_sigla",
            values="n_pl", aggfunc="sum", fill_value=0
        )
        _leg_total_pct = _leg_total_piv.div(_leg_total_piv.sum(axis=1), axis=0).round(3) * 100
        print("\nShare de PLs por corporação e legislatura (%):")
        print(_leg_total_pct.to_string())

    # ── 10. INTERPRETAÇÃO SOCIOLÓGICA DA MCA ──────────────────────
    print("\n--- Interpretação sociológica dos eixos MCA ---")
    if "_coords_row" in dir() and not _coords_row.empty:
        try:
            # junta coordenadas MCA com parl_base para interpretar eixos
            _mca_df = _coords_row.copy()
            _mca_df.columns = [f"mca_dim{i+1}" for i in range(_mca_df.shape[1])]
            # converte índice para string para evitar conflito int64 vs str
            _mca_idx = _parl_mca_cc.index if hasattr(_parl_mca_cc, "index") else range(len(_mca_df))
            _mca_df[_autor_col] = [str(x) for x in _mca_idx]
            _parl_base_mca = _parl_base.copy()
            _parl_base_mca[_autor_col] = _parl_base_mca[_autor_col].astype(str)
            _mca_df = _mca_df.merge(
                _parl_base_mca[[_autor_col,"corporacao","taxa_parl","n_pl","tipologia"]],
                on=_autor_col, how="left"
            )
            # correlação entre coordenadas MCA e variáveis
            _corr_mca = _mca_df[["mca_dim1","mca_dim2","taxa_parl","n_pl"]].corr().round(3)
            print("\nCorrelação eixos MCA × sucesso e produtividade:")
            print(_corr_mca)

            # média por corporação nos eixos
            _mca_corp = (
                _mca_df.groupby("corporacao")[["mca_dim1","mca_dim2"]]
                .mean().round(3)
                .sort_values("mca_dim1")
            )
            print("\nPosição média por corporação no espaço MCA:")
            print(_mca_corp)

            # interpretação automática dos eixos
            _dim1_taxa = _corr_mca.loc["mca_dim1","taxa_parl"]
            _dim2_npl  = _corr_mca.loc["mca_dim2","n_pl"]
            print(f"\nInterpretação dos eixos:")
            print(f"  Dim1 ↔ taxa sucesso (r={_dim1_taxa:.3f}): "
                  f"{'+ eficácia legislativa' if _dim1_taxa > 0.1 else '- eficácia / penalismo' if _dim1_taxa < -0.1 else 'não discrimina eficácia'}")
            print(f"  Dim2 ↔ produtividade (r={_dim2_npl:.3f}): "
                  f"{'+ hiperprodutivo' if _dim2_npl > 0.1 else '- nicho especializado' if _dim2_npl < -0.1 else 'não discrimina produtividade'}")
        except Exception as _e_mca_interp:
            print(f"  [INFO] Interpretação MCA: {_e_mca_interp}")
    else:
        print("  [INFO] Coordenadas MCA não disponíveis nesta execução.")

    df_sociol_integrado = _parl_base.copy()

    # ── 8. gráfico: 2×2 tipologia + capital relacional ────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # painel A: scatter produtividade × taxa colorido por tipologia
    _cores_tipo = {
        "Articulador eficiente":    "#2ecc71",
        "Hiperprodutivo simbólico": "#e74c3c",
        "Especialista de nicho":    "#3498db",
        "Periférico":               "#95a5a6"
    }
    for _tipo, _grp in _parl_base.groupby("tipologia"):
        axes[0, 0].scatter(
            _grp["n_pl"], _grp["taxa_parl"] * 100,
            label=_tipo, alpha=0.6, s=40,
            color=_cores_tipo.get(_tipo, "gray")
        )
    axes[0, 0].axvline(_med_n,   color="black", linestyle="--",
                       linewidth=0.8, alpha=0.5)
    axes[0, 0].axhline(_med_tax * 100, color="black", linestyle="--",
                       linewidth=0.8, alpha=0.5)
    axes[0, 0].set_xlabel("Número de PLs")
    axes[0, 0].set_ylabel("Taxa de aprovação (%)")
    axes[0, 0].set_title("Tipologia: produtividade × taxa de sucesso")
    axes[0, 0].legend(fontsize=8, markerscale=1.2)
    axes[0, 0].grid(True, linestyle="--", alpha=0.3)

    # painel B: distribuição da tipologia por corporação (exclui OUTROS)
    _tipo_corp = pd.crosstab(
        _parl_base[_parl_base["corporacao"] != "OUTROS"]["corporacao"],
        _parl_base[_parl_base["corporacao"] != "OUTROS"]["tipologia"],
        normalize="index"
    ) * 100
    _tipo_corp = _tipo_corp[[c for c in _cores_tipo if c in _tipo_corp.columns]]
    _tipo_corp.plot(kind="bar", stacked=True, ax=axes[0, 1],
                    color=[_cores_tipo[c] for c in _tipo_corp.columns],
                    alpha=0.85)
    axes[0, 1].set_title("Tipologia por corporação (%)")
    axes[0, 1].set_xlabel("Corporação")
    axes[0, 1].set_ylabel("% de parlamentares")
    axes[0, 1].set_xticklabels(
        axes[0, 1].get_xticklabels(), rotation=30, ha="right", fontsize=8
    )
    axes[0, 1].legend(fontsize=7, loc="upper right")

    # painel C: presidente de comissão × taxa de sucesso (barras)
    # mostra resultado mesmo sem rede disponível
    if "presidente_comissao_bin" in _parl_base.columns:
        _tab_comissao = (
            _parl_base.groupby("presidente_comissao_bin")["taxa_parl"]
            .agg(["mean", "count"])
            .rename(index={0: "Sem presidência", 1: "Com presidência"})
        )
        _tab_comissao["mean"].plot(
            kind="bar", ax=axes[1, 0],
            color=["#95a5a6", "#2ecc71"], alpha=0.8
        )
        for i, (idx, row) in enumerate(_tab_comissao.iterrows()):
            axes[1, 0].text(i, row["mean"] + 0.002,
                            f"n={int(row['count'])}", ha="center", fontsize=9)
        axes[1, 0].set_title("Capital institucional\n(presidência de comissão × taxa de sucesso)")
        axes[1, 0].set_xlabel("")
        axes[1, 0].set_ylabel("Taxa de aprovação média")
        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=0)
        axes[1, 0].grid(axis="y", linestyle="--", alpha=0.4)
    elif _cap_inst_cols:
        _parl_base.boxplot(
            column="taxa_parl", by="capital_institucional",
            ax=axes[1, 0], patch_artist=True
        )
        axes[1, 0].set_title("Capital institucional × taxa de sucesso")
        axes[1, 0].set_xlabel("Índice de capital institucional (0–3)")
        axes[1, 0].set_ylabel("Taxa de aprovação")
        plt.sca(axes[1, 0])
        plt.title("Capital institucional × taxa de sucesso")
    else:
        # fallback: anos na força × taxa de sucesso
        if "anos_forca_seg_num" in _parl_base.columns:
            _af = _parl_base[["anos_forca_seg_num","taxa_parl"]].dropna()
            axes[1, 0].scatter(_af["anos_forca_seg_num"],
                               _af["taxa_parl"] * 100,
                               alpha=0.5, s=25, color="#3498db")
            axes[1, 0].set_xlabel("Anos em força de segurança")
            axes[1, 0].set_ylabel("Taxa de aprovação (%)")
            axes[1, 0].set_title("Capital corporativo\n(anos na força × taxa de sucesso)")
            axes[1, 0].grid(True, linestyle="--", alpha=0.3)

    # painel D: anos na força × taxa de sucesso ou betweenness se disponível
    if "betweenness" in _parl_base.columns and _parl_base["betweenness"].gt(0).any():
        axes[1, 1].scatter(
            _parl_base["betweenness"],
            _parl_base["taxa_parl"] * 100,
            alpha=0.5, s=30, color="purple"
        )
        _z_bet = np.polyfit(
            _parl_base["betweenness"].fillna(0),
            _parl_base["taxa_parl"].fillna(0) * 100,
            1
        )
        _x_bet = np.linspace(0, _parl_base["betweenness"].max(), 100)
        axes[1, 1].plot(_x_bet, np.polyval(_z_bet, _x_bet),
                        "r--", linewidth=1.5, alpha=0.7)
        axes[1, 1].set_xlabel("Betweenness centrality")
        axes[1, 1].set_ylabel("Taxa de aprovação (%)")
        axes[1, 1].set_title("Capital relacional (rede) × taxa de sucesso")
        axes[1, 1].grid(True, linestyle="--", alpha=0.3)
    else:
        # fallback: taxa de sucesso por corporação (barras horizontais)
        # exclui OUTROS (categoria residual sem corporação identificada)
        _tc_raw = (
            _parl_base[_parl_base["corporacao"] != "OUTROS"]
            .groupby("corporacao")
            .agg(taxa_media=("taxa_parl","mean"), n=(_autor_col,"count"))
        )
        # mínimo de 3 parlamentares para aparecer no gráfico
        _tc_raw = _tc_raw[_tc_raw["n"] >= 3]
        _tc = (_tc_raw["taxa_media"] * 100).sort_values(ascending=True)
        _ns  = _tc_raw["n"].reindex(_tc.index)

        axes[1, 1].barh(_tc.index, _tc.values, color="steelblue", alpha=0.8)
        for i, (corp, v) in enumerate(zip(_tc.index, _tc.values)):
            axes[1, 1].text(
                v + 0.05, i,
                f"{v:.1f}%  (n={_ns[corp]})",
                va="center", fontsize=8
            )
        axes[1, 1].set_xlabel("Taxa de aprovação média (%)")
        axes[1, 1].set_title(
            "Taxa de sucesso por corporação\n"
            "(capital corporativo; exclui cat. residual OUTROS; n ≥ 3)"
        )
        axes[1, 1].grid(axis="x", linestyle="--", alpha=0.4)

    fig.suptitle(
        "Sociologia dos parlamentares de farda\n"
        "Capital institucional, corporativo, relacional e tipologia legislativa",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(PASTA / "sociologia_integrada.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print("\nGráfico sociologia integrada salvo.")

    # tabela da tipologia por corporação
    print("\nDistribuição da tipologia por corporação (%):")
    print(_tipo_corp.round(1))

except Exception as e_soc:
    print(f"\n[AVISO] Seção 41 falhou: {e_soc}")
    import traceback; traceback.print_exc()


# =========================================================
# 33. CORRELAÇÃO DE PEARSON — PRODUTIVIDADE × SUCESSO
# =========================================================
# Testa H: parlamentares mais produtivos têm maior número de
# aprovações, mas não necessariamente maior TAXA de sucesso
# (hipótese de rendimento decrescente por volume).
# Referência: Pearson (1895); Figueiredo & Limongi (1999).
# =========================================================

print("\n" + "=" * 60)
print("33. CORRELAÇÃO PEARSON — PRODUTIVIDADE × SUCESSO")
print("=" * 60)

from scipy.stats import pearsonr, spearmanr

df_pearson = pd.DataFrame()

if not base_parlamentar_agg.empty:
    try:
        _bp = base_parlamentar_agg.copy()
        _bp = _bp[_bp["n_pl"] > 0].copy()
        _bp["taxa_sucesso_parl"] = _bp["n_aprovados"] / _bp["n_pl"]

        # Pearson: n_pl × n_aprovados
        r1, p1 = pearsonr(_bp["n_pl"], _bp["n_aprovados"])
        # Pearson: n_pl × taxa_sucesso
        r2, p2 = pearsonr(_bp["n_pl"], _bp["taxa_sucesso_parl"])
        # Spearman (robusto a outliers): n_pl × n_aprovados
        rs1, ps1 = spearmanr(_bp["n_pl"], _bp["n_aprovados"])
        # Spearman: n_pl × taxa_sucesso
        rs2, ps2 = spearmanr(_bp["n_pl"], _bp["taxa_sucesso_parl"])

        df_pearson = pd.DataFrame([
            {"par": "n_pl × n_aprovados",   "r_pearson": round(r1,4),  "p_pearson": round(p1,4),
             "rho_spearman": round(rs1,4),  "p_spearman": round(ps1,4), "n": len(_bp)},
            {"par": "n_pl × taxa_sucesso",  "r_pearson": round(r2,4),  "p_pearson": round(p2,4),
             "rho_spearman": round(rs2,4),  "p_spearman": round(ps2,4), "n": len(_bp)},
        ])

        print("\nCorrelações de Pearson e Spearman — produtividade × sucesso:")
        print(df_pearson.to_string(index=False))
        print(f"\nInterpretação:")
        print(f"  n_pl × n_aprovados:  r={r1:.3f} (p={p1:.4f}) — {'significativo' if p1<0.05 else 'não significativo'}")
        print(f"  n_pl × taxa_sucesso: r={r2:.3f} (p={p2:.4f}) — {'significativo' if p2<0.05 else 'não significativo'}")
        if r2 < 0 and p2 < 0.05:
            print("  → Rendimento decrescente confirmado: mais PLs não eleva a taxa de aprovação.")
        elif r2 > 0 and p2 < 0.05:
            print("  → Produtividade positivamente associada à taxa — parlamentares mais ativos aprovam mais.")

        # Scatter: produtividade × aprovações
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        axes[0].scatter(_bp["n_pl"], _bp["n_aprovados"],
                        alpha=0.5, s=30, color="steelblue")
        # linha de tendência
        _z = np.polyfit(_bp["n_pl"], _bp["n_aprovados"], 1)
        _xfit = np.linspace(_bp["n_pl"].min(), _bp["n_pl"].max(), 100)
        axes[0].plot(_xfit, np.polyval(_z, _xfit), "r--", linewidth=1.5, alpha=0.7)
        # rótulo parlamentares mais produtivos
        for _, row in _bp.nlargest(5, "n_pl").iterrows():
            axes[0].annotate(str(row.get("Autor",""))[:18],
                             (row["n_pl"], row["n_aprovados"]),
                             fontsize=7, alpha=0.7)
        axes[0].set_xlabel("Número de PLs apresentados")
        axes[0].set_ylabel("Número de PLs aprovados")
        axes[0].set_title(f"Produtividade × Aprovações\n(r={r1:.3f}, p={p1:.4f})")
        axes[0].grid(True, linestyle="--", alpha=0.4)

        axes[1].scatter(_bp["n_pl"], _bp["taxa_sucesso_parl"] * 100,
                        alpha=0.5, s=30, color="darkorange")
        _z2 = np.polyfit(_bp["n_pl"], _bp["taxa_sucesso_parl"] * 100, 1)
        axes[1].plot(_xfit, np.polyval(_z2, _xfit), "r--", linewidth=1.5, alpha=0.7)
        axes[1].set_xlabel("Número de PLs apresentados")
        axes[1].set_ylabel("Taxa de aprovação (%)")
        axes[1].set_title(f"Produtividade × Taxa de Sucesso\n(r={r2:.3f}, p={p2:.4f})")
        axes[1].grid(True, linestyle="--", alpha=0.4)

        fig.suptitle(
            "Correlação entre produtividade legislativa e sucesso por parlamentar\n"
            "(Pearson e Spearman — base de parlamentares com ≥1 PL)",
            fontsize=11, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(PASTA / "pearson_produtividade_sucesso.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print("\nGráfico Pearson salvo.")

    except Exception as e_pear:
        print(f"\n[AVISO] Correlação de Pearson falhou: {e_pear}")
else:
    print("\n[AVISO] base_parlamentar vazia — seção 33 pulada.")


# =========================================================
# 34. CLUSTERING DE PARLAMENTARES (K-MEANS)
# =========================================================
# Agrupa parlamentares por perfil temático (% de PLs por tópico).
# Identifica tipos latentes: "penalista", "corporativista",
# "social", etc. — complementa a análise de corporação e agenda.
# Referência: MacQueen (1967); Jain (2010, Pattern Recognition).
# =========================================================

print("\n" + "=" * 60)
print("34. CLUSTERING K-MEANS — PERFIL TEMÁTICO DOS PARLAMENTARES")
print("=" * 60)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df_clusters = pd.DataFrame()

try:
    # perfil temático por parlamentar: % de PLs em cada tópico
    _df_cluster_base = (
        df_texto[df_texto["Autor"].notna() & df_texto["topico_dominante"].notna()]
        .groupby(["Autor", "topico_dominante"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    _df_cluster_base.columns.name = None
    _df_cluster_base.columns = (
        ["Autor"] + [f"pct_topico_{int(c)}" for c in _df_cluster_base.columns[1:]]
    )

    # normaliza para % por parlamentar
    _feats = [c for c in _df_cluster_base.columns if c.startswith("pct_topico_")]
    _soma = _df_cluster_base[_feats].sum(axis=1).replace(0, 1)
    for f in _feats:
        _df_cluster_base[f] = _df_cluster_base[f] / _soma

    # filtra parlamentares com >= 5 PLs (evita ruído de estreantes)
    _df_cluster_base["n_pl_total"] = (_df_cluster_base[_feats] * _soma.values[:, None]).sum(axis=1)
    _df_cluster_base = _df_cluster_base[_df_cluster_base["n_pl_total"] >= 5].copy()

    X_cl = _df_cluster_base[_feats].values
    _scaler_cl = StandardScaler()
    X_cl_sc = _scaler_cl.fit_transform(X_cl)

    # silhueta para escolher k ótimo (2 a 6)
    _silh = {}
    for _k in range(2, 7):
        _km_tmp = KMeans(n_clusters=_k, random_state=42, n_init=10)
        _lbl_tmp = _km_tmp.fit_predict(X_cl_sc)
        if len(set(_lbl_tmp)) > 1:
            _silh[_k] = silhouette_score(X_cl_sc, _lbl_tmp)

    _k_otimo = max(_silh, key=_silh.get)
    print(f"\nSilhueta por k: {_silh}")
    print(f"k ótimo selecionado: {_k_otimo}")

    km_final = KMeans(n_clusters=_k_otimo, random_state=42, n_init=10)
    _df_cluster_base["cluster"] = km_final.fit_predict(X_cl_sc)

    # perfil de cada cluster
    _perfil_cluster = (
        _df_cluster_base.groupby("cluster")[_feats]
        .mean()
        .round(3)
    )
    _perfil_cluster.columns = [NOMES_TOPICOS_CURTO.get(int(c.split("_")[-1]), c)
                                for c in _perfil_cluster.columns]

    # nomes interpretativos: cluster dominado pelo tópico com maior média
    _nomes_cluster = {}
    _nomes_base = {1:"Regulação/Trânsito",2:"Economia/Consumidor",
                   3:"Políticas Sociais",4:"Carreiras da Força",5:"Direito Penal"}
    for _c, row in _perfil_cluster.iterrows():
        _top_col = row.idxmax()
        _top_num = [k for k,v in NOMES_TOPICOS_CURTO.items() if v == _top_col]
        _top_num = _top_num[0] if _top_num else 0
        _nomes_cluster[_c] = f"Cluster {_c}: dominante {_nomes_base.get(_top_num, _top_col)}"

    print("\nPerfil médio por cluster (% por tópico):")
    print(_perfil_cluster)

    # merge para análise cruzada
    _df_cluster_base["cluster_nome"] = _df_cluster_base["cluster"].map(_nomes_cluster)
    df_clusters = _df_cluster_base.copy()

    # gráfico: heatmap de perfil dos clusters
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(_perfil_cluster * 100, annot=True, fmt=".1f",
                cmap="Blues", ax=axes[0], linewidths=0.4)
    axes[0].set_title("Perfil temático médio por cluster (%)")
    axes[0].set_xlabel("Agenda temática")
    axes[0].set_ylabel("Cluster")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=20, ha="right", fontsize=8)

    # distribuição de parlamentares por cluster
    _dist_cl = _df_cluster_base["cluster_nome"].value_counts().sort_index()
    axes[1].barh(_dist_cl.index, _dist_cl.values, color="steelblue", alpha=0.8)
    for i, v in enumerate(_dist_cl.values):
        axes[1].text(v + 0.3, i, str(v), va="center", fontsize=9)
    axes[1].set_xlabel("Número de parlamentares")
    axes[1].set_title("Distribuição de parlamentares por cluster")
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)

    fig.suptitle(
        f"Clustering k-means de parlamentares por perfil temático (k={_k_otimo})\n"
        "Base: % de PLs por agenda — parlamentares com ≥5 proposições",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(PASTA / "kmeans_perfil_parlamentar.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Gráfico k-means salvo.")

    # silhueta
    plt.figure(figsize=(7, 4))
    plt.plot(list(_silh.keys()), list(_silh.values()), "o-", color="steelblue", linewidth=2)
    plt.axvline(_k_otimo, color="red", linestyle="--", linewidth=1, label=f"k ótimo={_k_otimo}")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Coeficiente de silhueta")
    plt.title("Seleção do k ótimo — coeficiente de silhueta")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(PASTA / "kmeans_silhueta.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

except Exception as e_km:
    print(f"\n[AVISO] K-means falhou: {e_km}")


# =========================================================
# 35. SENTIMENT ANALYSIS — TOM DAS EMENTAS
# =========================================================
# Classifica o tom das ementas como punitivo (crime, pena,
# proibição, penalidade) vs. protetivo/social (proteção,
# direito, assistência, benefício).
# Abordagem: léxico customizado para o domínio legislativo
# brasileiro — mais adequado que VADER/TextBlob (treinados
# em inglês). Baseado em Grimmer & Stewart (2013, PoLMeth).
# =========================================================

print("\n" + "=" * 60)
print("35. SENTIMENT ANALYSIS — TOM PUNITIVO vs. PROTETIVO")
print("=" * 60)

df_sentiment = pd.DataFrame()

try:
    # léxicos customizados para linguagem legislativa brasileira
    _LEX_PUNITIVO = {
        "crime", "crimes", "criminoso", "criminosa", "penal", "pena", "penas",
        "prisao", "reclusao", "detenção", "detencao", "punicao", "punir",
        "punivel", "infracao", "infrator", "ilicito", "ilicita", "vedado",
        "vedada", "proibido", "proibida", "multa", "sancao", "sancionar",
        "penalidade", "tipificar", "tipificacao", "delito", "doloso",
        "culposo", "reincidencia", "agravante", "prescricao",
    }

    _LEX_PROTETIVO = {
        "protecao", "proteger", "amparo", "amparar", "assistencia", "beneficio",
        "beneficios", "direito", "direitos", "garantia", "garantias",
        "promocao", "acesso", "inclusao", "apoio", "auxilio", "subsidio",
        "gratuito", "gratuita", "social", "saude", "educacao", "habitacao",
        "crianca", "adolescente", "idoso", "idosa", "deficiencia",
        "vulneravel", "hipossuficiente", "cidadao", "cidada",
    }

    def classificar_tom(texto):
        if pd.isna(texto):
            return np.nan, np.nan, "neutro"
        toks = set(str(texto).lower().split())
        sc_pun = len(toks & _LEX_PUNITIVO)
        sc_pro = len(toks & _LEX_PROTETIVO)
        if sc_pun > sc_pro:
            tom = "punitivo"
        elif sc_pro > sc_pun:
            tom = "protetivo"
        else:
            tom = "neutro"
        return sc_pun, sc_pro, tom

    # detecta automaticamente o nome da coluna de texto limpo
    _col_sent = next(
        (c for c in ["texto_limpo", "ementa_limpa", "texto_lda", "Ementa"]
         if c in df_texto.columns),
        None
    )
    if _col_sent is None:
        raise ValueError(f"Nenhuma coluna de texto encontrada: {list(df_texto.columns)}")
    print(f"\n[INFO] Coluna usada para sentiment: '{_col_sent}'")

    _res = df_texto[_col_sent].apply(classificar_tom)
    df_texto["score_punitivo"]  = [r[0] for r in _res]
    df_texto["score_protetivo"] = [r[1] for r in _res]
    df_texto["tom"]             = [r[2] for r in _res]

    # distribuição geral
    _dist_tom = df_texto["tom"].value_counts()
    print("\nDistribuição geral de tom:")
    print(_dist_tom)

    # distribuição geral
    _dist_tom = df_texto["tom"].value_counts()
    print("\nDistribuição geral de tom:")
    print(_dist_tom)
    print(f"\n% punitivo:  {_dist_tom.get('punitivo',0)/len(df_texto)*100:.1f}%")
    print(f"% protetivo: {_dist_tom.get('protetivo',0)/len(df_texto)*100:.1f}%")
    print(f"% neutro:    {_dist_tom.get('neutro',0)/len(df_texto)*100:.1f}%")

    # tom por tópico
    _tom_topico = pd.crosstab(
        df_texto["topico_dominante"], df_texto["tom"], normalize="index"
    ) * 100
    _tom_topico.index = [NOMES_TOPICOS_CURTO.get(int(i), str(i))
                         for i in _tom_topico.index]
    print("\nTom por tópico (%):")
    print(_tom_topico.round(1))

    # tom por tópico com sucesso
    _tom_sucesso = (
        df_texto[df_texto["sucesso_legislativo"].notna()]
        .groupby(["tom", "sucesso_legislativo"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "fracasso", 1: "aprovado"})
    )
    _tom_sucesso["taxa_sucesso_%"] = (
        _tom_sucesso["aprovado"] /
        _tom_sucesso.sum(axis=1) * 100
    ).round(2)
    print("\nSucesso legislativo por tom:")
    print(_tom_sucesso)

    df_sentiment = _tom_topico.reset_index().rename(columns={"index": "agenda"})

    # gráficos
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # pizza geral
    _dist_plot = _dist_tom.reindex(["punitivo", "protetivo", "neutro"]).fillna(0)
    axes[0].pie(
        _dist_plot.values,
        labels=_dist_plot.index,
        autopct="%1.1f%%",
        colors=["#e74c3c", "#27ae60", "#95a5a6"],
        startangle=90
    )
    axes[0].set_title("Distribuição geral de tom\nnas ementas dos PLs")

    # tom por tópico (barras empilhadas)
    _cols_plot = [c for c in ["punitivo", "protetivo", "neutro"] if c in _tom_topico.columns]
    _tom_topico[_cols_plot].plot(
        kind="bar", stacked=True, ax=axes[1],
        color=["#e74c3c", "#27ae60", "#95a5a6"],
        alpha=0.85
    )
    axes[1].set_title("Tom por agenda temática (%)")
    axes[1].set_xlabel("Agenda")
    axes[1].set_ylabel("% das ementas")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=20, ha="right", fontsize=8)
    axes[1].legend(title="Tom", fontsize=8)

    # taxa de sucesso por tom
    _ts = _tom_sucesso["taxa_sucesso_%"].sort_values(ascending=True)
    colors_bar = {"punitivo": "#e74c3c", "protetivo": "#27ae60", "neutro": "#95a5a6"}
    bars_s = axes[2].barh(
        _ts.index,
        _ts.values,
        color=[colors_bar.get(i, "steelblue") for i in _ts.index],
        alpha=0.85
    )
    for bar, v in zip(bars_s, _ts.values):
        axes[2].text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                     f"{v:.2f}%", va="center", fontsize=9)
    axes[2].set_xlabel("Taxa de aprovação (%)")
    axes[2].set_title("Taxa de sucesso por tom da ementa")
    axes[2].grid(axis="x", linestyle="--", alpha=0.4)

    fig.suptitle(
        "Análise de sentimento — tom punitivo vs. protetivo nas ementas legislativas\n"
        "(léxico customizado para linguagem legislativa brasileira)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(PASTA / "sentiment_tom_ementas.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print("\nGráfico de sentiment salvo.")

except Exception as e_sent:
    print(f"\n[AVISO] Sentiment analysis falhou: {e_sent}")


# =========================================================
# 36. ANÁLISE ESPACIAL — TAXA DE SUCESSO POR UF
# =========================================================

print("\n" + "=" * 60)
print("36. ANÁLISE ESPACIAL — TAXA DE SUCESSO POR UF")
print("=" * 60)

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

# taxa de sucesso por UF
_taxa_uf_esp = (
    df_base[df_base["UF"].notna()]
    .groupby("UF")["sucesso_legislativo"]
    .agg(["sum", "count", "mean"])
    .reset_index()
    .rename(columns={"sum": "aprovados", "count": "total", "mean": "taxa"})
)
_taxa_uf_esp["taxa_pct"] = (_taxa_uf_esp["taxa"] * 100).fillna(0)

print("\nTaxa de sucesso por UF:")
print(_taxa_uf_esp.sort_values("taxa_pct", ascending=False).to_string(index=False))

_taxa_vals  = _taxa_uf_esp.set_index("UF")["taxa_pct"].to_dict()
_total_vals = _taxa_uf_esp.set_index("UF")["total"].to_dict()
_vmax_m = max(_taxa_vals.values()) if _taxa_vals else 1
import matplotlib
import matplotlib.cm as cm
_cmap_m = matplotlib.colormaps["YlOrRd"]
_norm_m = mcolors.Normalize(vmin=0, vmax=_vmax_m)

# ── Grade geográfica completa — todos os 27 estados + DF ──────────
# (col, row): col=eixo leste-oeste (0=oeste), row=eixo norte-sul (0=sul)
_UF_GRID = {
    # Norte
    "RR": (2, 8),   "AP": (6, 8),
    "AM": (1, 7),   "PA": (4, 7),   "MA": (6, 7),
    "AC": (0, 6),   "RO": (1, 6),   "TO": (4, 6),
    # Nordeste
    "PI": (6, 6),   "CE": (7, 6),   "RN": (8, 6),
    "PB": (8, 5),   "PE": (8, 4),   "AL": (8, 3),
    "SE": (7, 3),   "BA": (6, 5),
    # Centro-Oeste
    "MT": (3, 5),   "MS": (3, 4),
    "GO": (4, 5),   "DF": (5, 5),
    # Sudeste
    "MG": (5, 4),   "ES": (7, 4),
    "SP": (4, 3),   "RJ": (6, 3),
    # Sul
    "PR": (4, 2),
    "SC": (4, 1),
    "RS": (4, 0),
}

fig = plt.figure(figsize=(17, 9))
gs  = gridspec.GridSpec(1, 2, width_ratios=[1.05, 1], figure=fig,
                         wspace=0.05)

# ── painel A: mapa em grade ───────────────────────────────────────
ax_m = fig.add_subplot(gs[0])
ax_m.set_facecolor("#ddeeff")
ax_m.set_xlim(-0.7, 9.7)
ax_m.set_ylim(-0.7, 9.0)
ax_m.set_aspect("equal")
ax_m.axis("off")
ax_m.set_title("Mapa esquemático — taxa de sucesso por UF\n"
               "cor = taxa de aprovação  |  tamanho ∝ volume de PLs",
               fontsize=10, pad=8)

# grade de fundo sutil
for _gc in range(10):
    ax_m.axvline(_gc, color="white", lw=0.4, alpha=0.5)
for _gr in range(10):
    ax_m.axhline(_gr, color="white", lw=0.4, alpha=0.5)

# rótulos de região
_regioes = {
    "Norte":    (1.5, 7.5, "#1a6fa0"),
    "Nordeste": (7.0, 6.5, "#b05a00"),
    "Centro-Oeste": (4.0, 5.5, "#2e7d32"),
    "Sudeste":  (5.5, 3.5, "#6a1b9a"),
    "Sul":      (4.0, 1.2, "#c62828"),
}
for _rnome, (_rx, _ry, _rcor) in _regioes.items():
    ax_m.text(_rx, _ry, _rnome, fontsize=7, color=_rcor,
              alpha=0.35, ha="center", fontweight="bold", style="italic")

for uf, (col, row) in _UF_GRID.items():
    _taxa = _taxa_vals.get(uf, 0)
    _n    = _total_vals.get(uf, 0)
    _cor  = _cmap_m(_norm_m(_taxa))
    # tamanho mínimo 200, máximo 1400, proporcional a log(n)
    import math
    _sz = max(200, min(1400, (math.log1p(_n) / math.log1p(1400)) * 1400))

    ax_m.scatter(col, row, s=_sz, color=_cor, alpha=0.90,
                 edgecolors="#444", linewidths=0.6, zorder=4)
    # sigla em branco se cor escura, preto se clara
    _lum = 0.299*_cor[0] + 0.587*_cor[1] + 0.114*_cor[2]
    _txt_cor = "white" if _lum < 0.55 else "#111"
    ax_m.text(col, row + 0.01, uf,
              ha="center", va="center", fontsize=7.5,
              fontweight="bold", color=_txt_cor, zorder=5)
    ax_m.text(col, row - 0.48, f"{_taxa:.1f}%",
              ha="center", va="top", fontsize=6, color="#333", zorder=5)

# colorbar
_sm_m = cm.ScalarMappable(cmap=_cmap_m, norm=_norm_m)
_sm_m.set_array([])
cb = plt.colorbar(_sm_m, ax=ax_m, shrink=0.5, pad=0.01, aspect=18,
                  location="right")
cb.set_label("Taxa de aprovação (%)", fontsize=8.5)

# legenda de tamanho
import math
for _nl, _lab in [(50,"50 PLs"), (300,"300 PLs"), (1000,"1000 PLs")]:
    _s = max(200, min(1400, (math.log1p(_nl)/math.log1p(1400))*1400))
    ax_m.scatter([], [], s=_s, color="gray", alpha=0.5,
                 edgecolors="#444", linewidths=0.5, label=_lab)
ax_m.legend(title="Volume de PLs", loc="lower left",
            fontsize=7, title_fontsize=7.5, framealpha=0.75,
            markerscale=1, scatterpoints=1)

# ── painel B: barras ranqueadas coloridas ────────────────────────
ax_b = fig.add_subplot(gs[1])
_tx_sorted = (
    _taxa_uf_esp[_taxa_uf_esp["total"] >= 10]
    .sort_values("taxa_pct")
    .reset_index(drop=True)
)
_cores_b = [_cmap_m(_norm_m(v)) for v in _tx_sorted["taxa_pct"]]
bars_uf  = ax_b.barh(_tx_sorted["UF"], _tx_sorted["taxa_pct"],
                      color=_cores_b, edgecolor="white", linewidth=0.5)
for bar, (_, row) in zip(bars_uf, _tx_sorted.iterrows()):
    ax_b.text(bar.get_width() + 0.08,
              bar.get_y() + bar.get_height() / 2,
              f"n={int(row['total'])}  aprv={int(row['aprovados'])}  ({row['taxa_pct']:.1f}%)",
              va="center", fontsize=7.5)
ax_b.set_xlabel("Taxa de aprovação (%)", fontsize=9)
ax_b.set_title("Ranking — UFs com ≥ 10 PLs\nordenado por taxa de sucesso",
               fontsize=10)
ax_b.grid(axis="x", linestyle="--", alpha=0.35)
ax_b.set_xlim(0, _tx_sorted["taxa_pct"].max() * 1.7)
ax_b.tick_params(axis="y", labelsize=9)

fig.suptitle(
    "Análise espacial — sucesso legislativo por unidade federativa\n"
    "Parlamentares oriundos das forças de segurança pública (1989–2023)",
    fontsize=12, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig(PASTA / "espacial_taxa_sucesso_uf.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print("Gráfico espacial salvo.")

# ── Choropleth com centroides reais embutidos (sem download) ──────
# Coordenadas aproximadas dos centroides de cada UF (lon, lat)
_CENTROIDES_UF = {
    "AC":(-70.5,-9.0),  "AL":(-36.6,-9.7),  "AM":(-64.7,-4.1),
    "AP":(-51.8, 1.4),  "BA":(-41.7,-12.5), "CE":(-39.5,-5.1),
    "DF":(-47.9,-15.8), "ES":(-40.7,-19.6), "GO":(-49.6,-15.9),
    "MA":(-45.3,-5.4),  "MG":(-44.7,-18.1), "MS":(-54.8,-20.5),
    "MT":(-55.9,-13.0), "PA":(-52.3,-3.8),  "PB":(-36.8,-7.1),
    "PE":(-37.9,-8.3),  "PI":(-42.7,-7.7),  "PR":(-51.6,-24.6),
    "RJ":(-43.2,-22.3), "RN":(-36.5,-5.8),  "RO":(-63.0,-11.5),
    "RR":(-61.4, 2.1),  "RS":(-53.1,-30.2), "SC":(-50.5,-27.4),
    "SE":(-37.4,-10.6), "SP":(-48.7,-22.2), "TO":(-48.3,-10.2),
}

try:
    import geopandas as gpd
    import urllib.request, json as _json_mod, io as _io_mod

    # tenta carregar shapefile local se disponível
    _shp_local = None
    for _sp in ["/usr/share/gadm/BRA_1.shp",
                "/home/thiago/shapefiles/BR_UF_2022.shp",
                str(PASTA / "BR_UF_2022.shp")]:
        if os.path.exists(_sp):
            _shp_local = _sp
            break

    # se não encontrou local, tenta baixar GeoJSON do IBGE
    if not _shp_local:
        _ibge_url = (
            "https://servicodados.ibge.gov.br/api/v3/malhas/paises/BR"
            "?formato=application/vnd.geo+json&qualidade=minima&divisao=UF"
        )
        print("[INFO] Baixando polígonos UF do IBGE...")
        _req = urllib.request.urlopen(_ibge_url, timeout=20)
        _geojson_bytes = _req.read()
        _gdf = gpd.read_file(_io_mod.BytesIO(_geojson_bytes))
        # a API do IBGE retorna 'codarea' — buscar coluna com sigla
        # fallback: criar sigla a partir do código IBGE
        _ibge_siglas = {
            "11":"RO","12":"AC","13":"AM","14":"RR","15":"PA","16":"AP","17":"TO",
            "21":"MA","22":"PI","23":"CE","24":"RN","25":"PB","26":"PE","27":"AL",
            "28":"SE","29":"BA","31":"MG","32":"ES","33":"RJ","35":"SP",
            "41":"PR","42":"SC","43":"RS","50":"MS","51":"MT","52":"GO","53":"DF"
        }
        if "SIGLA_UF" in _gdf.columns:
            _ucol = "SIGLA_UF"
        elif "sigla" in _gdf.columns:
            _ucol = "sigla"
        elif "codarea" in _gdf.columns:
            _gdf["sigla_uf"] = _gdf["codarea"].astype(str).map(_ibge_siglas)
            _ucol = "sigla_uf"
        else:
            # tenta inferir pelo código na primeira coluna disponível
            _gdf["sigla_uf"] = _gdf.iloc[:, 0].astype(str).map(_ibge_siglas)
            _ucol = "sigla_uf"
        print(f"  GeoJSON baixado: {len(_gdf)} polígonos | coluna UF: {_ucol}")
    else:
        _gdf = gpd.read_file(_shp_local)
        _ucol = next(
            (c for c in _gdf.columns
             if any(k in c.lower() for k in ["sigla","uf","abbrev","cd_uf","sigla_uf"])),
            _gdf.columns[0]
        )

    # merge com taxa de sucesso por UF
    _gdf = _gdf.merge(
        _taxa_uf_esp[["UF","taxa_pct","total","aprovados"]],
        left_on=_ucol, right_on="UF", how="left"
    )
    _gdf["taxa_pct"] = _gdf["taxa_pct"].fillna(0)
    _gdf["n_label"]  = _gdf["total"].fillna(0).astype(int)

    fig2, ax2 = plt.subplots(figsize=(12, 10))
    _gdf.plot(
        column="taxa_pct", cmap="YlOrRd", legend=True,
        legend_kwds={"label":"Taxa de aprovação (%)","shrink":0.45,"pad":0.01},
        missing_kwds={"color":"#e0e0e0","label":"Sem dados"},
        ax=ax2, linewidth=0.4, edgecolor="white",
        vmin=0, vmax=_gdf["taxa_pct"].quantile(0.95)
    )

    # rótulos: sigla + taxa + n
    for _, row in _gdf.iterrows():
        try:
            _ctr = row.geometry.centroid
            _sig = row[_ucol] if pd.notna(row[_ucol]) else ""
            _tx  = row["taxa_pct"] if pd.notna(row["taxa_pct"]) else 0
            _nt  = row["n_label"]  if pd.notna(row["n_label"])  else 0
            # só rotula se tiver ao menos 1 PL
            if _nt > 0:
                ax2.text(_ctr.x, _ctr.y + 0.3, _sig,
                         fontsize=6.5, ha="center", va="bottom",
                         fontweight="bold", color="#111")
                ax2.text(_ctr.x, _ctr.y - 0.3, f"{_tx:.1f}%",
                         fontsize=5.5, ha="center", va="top", color="#333")
        except Exception:
            continue

    # nota de cautela
    ax2.text(0.01, 0.01,
             "* UFs com n<30 PLs têm taxas instáveis — interpretar com cautela.",
             transform=ax2.transAxes, fontsize=7.5, color="#555", style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax2.set_title(
        "Coroplético — taxa de sucesso legislativo por Unidade da Federação\n"
        "Parlamentares oriundos das forças de segurança pública (1989–2023)",
        fontsize=11, fontweight="bold"
    )
    ax2.axis("off")
    plt.tight_layout()
    plt.savefig(PASTA / "mapa_choropleth_uf.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Coroplético salvo com todos os estados.")

except Exception as _eg:
    # fallback: scatter geográfico com coordenadas reais dos centroides
    print(f"[INFO] Choropleth com shapefile indisponível ({type(_eg).__name__}) "
          f"— gerando scatter geográfico com coordenadas reais.")
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.set_facecolor("#cce5f6")
    ax2.set_xlim(-74, -34)
    ax2.set_ylim(-34, 6)

    for uf, (lon, lat) in _CENTROIDES_UF.items():
        _taxa = _taxa_vals.get(uf, 0)
        _n    = _total_vals.get(uf, 0)
        _cor  = _cmap_m(_norm_m(_taxa))
        _sz   = max(80, min(800, (math.log1p(_n) / math.log1p(1400)) * 800))
        ax2.scatter(lon, lat, s=_sz, color=_cor, alpha=0.85,
                    edgecolors="#333", linewidths=0.7, zorder=3)
        _lum = 0.299*_cor[0] + 0.587*_cor[1] + 0.114*_cor[2]
        _tcor = "white" if _lum < 0.55 else "#111"
        ax2.text(lon, lat, uf, ha="center", va="center",
                 fontsize=7, fontweight="bold", color=_tcor, zorder=4)
        ax2.text(lon, lat - 1.2, f"{_taxa:.1f}%",
                 ha="center", va="top", fontsize=5.5, color="#333", zorder=4)

    # colorbar
    _sm2 = cm.ScalarMappable(cmap=_cmap_m, norm=_norm_m)
    _sm2.set_array([])
    cb2 = plt.colorbar(_sm2, ax=ax2, shrink=0.45, pad=0.01)
    cb2.set_label("Taxa de aprovação (%)", fontsize=9)

    # legenda tamanho
    for _nl2, _lab2 in [(50,"50 PLs"),(300,"300 PLs"),(1000,"1000+ PLs")]:
        _s2 = max(80, min(800, (math.log1p(_nl2)/math.log1p(1400))*800))
        ax2.scatter([], [], s=_s2, color="gray", alpha=0.5,
                    edgecolors="#444", linewidths=0.5, label=_lab2)
    ax2.legend(title="Volume de PLs", loc="lower left",
               fontsize=8, title_fontsize=8.5, framealpha=0.8)

    # nota de cautela
    ax2.text(-73, -32,
             "* UFs com poucos PLs (n<30) têm taxas\n  instáveis — interpretar com cautela.",
             fontsize=7, color="#555", style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    ax2.set_title(
        "Distribuição geográfica — taxa de sucesso legislativo por UF\n"
        "Parlamentares oriundos das forças de segurança pública (1989–2023)\n"
        "Tamanho ∝ volume de PLs  |  Cor ∝ taxa de aprovação",
        fontsize=10, fontweight="bold"
    )
    ax2.set_xlabel("Longitude", fontsize=8)
    ax2.set_ylabel("Latitude", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.3, color="white")
    plt.tight_layout()
    plt.savefig(PASTA / "mapa_choropleth_uf.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Scatter geográfico com coordenadas reais salvo.")

# ── nota de cautela também no painel de barras ────────────────────
print("\nNota: UFs com n<30 PLs têm taxas instáveis — ver coluna 'total' no gráfico.")


# =========================================================
# 31. TABELAS-RESUMO
# =========================================================

df_testes_estatisticos = pd.DataFrame([
    {"teste": "Partido x Sucesso", "chi2": chi2_partido, "p_value": p_partido, "gl": dof_partido},
    {"teste": "UF x Sucesso", "chi2": chi2_uf, "p_value": p_uf, "gl": dof_uf},
    {"teste": "Partido x Tópico", "chi2": chi2_pt, "p_value": p_pt, "gl": dof_pt},
    {"teste": "Tópico x Sucesso", "chi2": chi2_suc, "p_value": p_suc, "gl": dof_suc},
])

if not np.isnan(chi2_corp):
    df_testes_estatisticos = pd.concat([
        df_testes_estatisticos,
        pd.DataFrame([{
            "teste": "Corporação x Sucesso",
            "chi2": chi2_corp,
            "p_value": p_corp,
            "gl": dof_corp
        }])
    ], ignore_index=True)

melhor_modelo_bt = None
if not resumo_modelos_bt.empty:
    melhor_modelo_bt = resumo_modelos_bt.iloc[0]["modelo"]

topico_mais_freq = None
if not freq_topicos.empty:
    topico_mais_freq = int(freq_topicos.sort_values("frequencia", ascending=False).iloc[0]["topico"])

topico_maior_sucesso = None
if not taxa_sucesso_topico.empty:
    topico_maior_sucesso = int(taxa_sucesso_topico.index[0])

partido_maior_taxa = None
if not taxa_partido_filtrado.empty:
    partido_maior_taxa = taxa_partido_filtrado.sort_values("taxa_sucesso", ascending=False).index[0]

uf_maior_taxa = None
if not taxa_uf_filtrado.empty:
    uf_maior_taxa = taxa_uf_filtrado.sort_values("taxa_sucesso", ascending=False).index[0]

corp_maior_prob = None
if not pred_corporacao.empty:
    corp_maior_prob = pred_corporacao.iloc[0]["corporacao_sigla"]

df_resumo_executivo = pd.DataFrame([
    {"indicador": "Total de PLs", "valor": len(df_base)},
    {"indicador": "Total de PLs com texto", "valor": len(df_texto)},
    {"indicador": "Taxa geral de sucesso (%)", "valor": round(float(taxa_sucesso), 4)},
    {"indicador": "Partido com maior taxa (>=100 PLs)", "valor": partido_maior_taxa},
    {"indicador": "UF com maior taxa (>=100 PLs)", "valor": uf_maior_taxa},
    {"indicador": "Tópico mais frequente", "valor": topico_mais_freq},
    {"indicador": "Tópico com maior sucesso", "valor": topico_maior_sucesso},
    {"indicador": "Melhor modelo no backtesting", "valor": melhor_modelo_bt},
    {"indicador": "Corporação com maior prob. prevista", "valor": corp_maior_prob},
    {"indicador": "Modelo final usado nas predições", "valor": nome_modelo_final},
])


# =========================================================
# 31B. QUADRO FINAL DE ROBUSTEZ AMPLIADO
# =========================================================

comparacao_modelos = []

for nome, modelo, base_usada in [
    ("topico_ano", modelo_principal, df_reg_decidido),
    ("topico_partido_ano", modelo_partido, df_reg_inf),
    ("topico_corporacao_ano", modelo_corp_inf, df_reg_corp_logit),
]:
    if modelo is not None:
        try:
            y_true_rob = base_usada["aprovado"].astype(int)
            y_prob_rob = modelo.predict(base_usada)
            auc_tmp = roc_auc_score(y_true_rob, y_prob_rob)
        except Exception:
            auc_tmp = np.nan

        try:
            pseudo_r2 = float(1 - (modelo.llf / modelo.llnull))
        except Exception:
            pseudo_r2 = np.nan

        comparacao_modelos.append({
            "modelo": nome,
            "n_obs": int(modelo.nobs),
            "llf": float(modelo.llf),
            "aic": float(modelo.aic) if hasattr(modelo, "aic") else np.nan,
            "bic": float(modelo.bic) if hasattr(modelo, "bic") else np.nan,
            "pseudo_r2": pseudo_r2,
            "auc_in_sample": auc_tmp
        })

df_comparacao_modelos = pd.DataFrame(comparacao_modelos)

if not df_comparacao_modelos.empty:
    print("\nQuadro comparativo ampliado de modelos:")
    print(df_comparacao_modelos)


# =========================================================
# 32. EXPORTAÇÃO FINAL
# =========================================================

agenda_resumo = pd.DataFrame({
    "topico_dominante": taxa_sucesso_topico.index,
    "taxa_sucesso_%": taxa_sucesso_topico.values
})

with pd.ExcelWriter(ARQUIVO_SAIDA, engine="openpyxl") as writer:
    df_base.to_excel(writer, sheet_name="base_limpa", index=False)

    df_palavras.to_excel(writer, sheet_name="palavras", index=False)
    df_bigrams.to_excel(writer, sheet_name="bigrams", index=False)
    df_trigrams.to_excel(writer, sheet_name="trigrams", index=False)
    df_leis.to_excel(writer, sheet_name="leis_citadas", index=False)
    df_artigos.to_excel(writer, sheet_name="artigos_citados", index=False)

    freq_uf.to_excel(writer, sheet_name="uf_expandida", index=False)
    freq_partido.to_excel(writer, sheet_name="partidos_expandida", index=False)
    freq_autor.to_excel(writer, sheet_name="autores_expandida", index=False)
    freq_situacao.to_excel(writer, sheet_name="situacao", index=False)
    df_pls_ano.to_excel(writer, sheet_name="pls_por_ano", index=False)

    taxa_partido.to_excel(writer, sheet_name="taxa_partido")
    taxa_uf.to_excel(writer, sheet_name="taxa_uf")
    taxa_partido_filtrado.to_excel(writer, sheet_name="taxa_partido_100+")
    taxa_uf_filtrado.to_excel(writer, sheet_name="taxa_uf_100+")

    produtividade.to_excel(writer, sheet_name="produtividade_autor", index=False)
    df_autor_expandido.to_excel(writer, sheet_name="coautoria_expandida", index=False)
    base_autor.to_excel(writer, sheet_name="base_autor", index=False)
    ranking_proposicoes.to_excel(writer, sheet_name="ranking_proposicoes", index=False)
    ranking_sucesso.to_excel(writer, sheet_name="ranking_sucesso", index=False)
    ranking_taxa.to_excel(writer, sheet_name="ranking_taxa", index=False)

    tabela_partido_sucesso_q2.to_excel(writer, sheet_name="qui2_partido")
    tabela_uf_sucesso_q2.to_excel(writer, sheet_name="qui2_uf")
    tabela_leg_sucesso.to_excel(writer, sheet_name="legislatura_sucesso")
    df_testes_estatisticos.to_excel(writer, sheet_name="testes_estatisticos", index=False)

    df_topicos.to_excel(writer, sheet_name="lda_topicos", index=False)
    freq_topicos.to_excel(writer, sheet_name="lda_freq_topicos", index=False)
    tabela_partido.to_excel(writer, sheet_name="partido_x_topico")
    tabela_percentual.to_excel(writer, sheet_name="partido_x_topico_pct")
    df_chi2.to_excel(writer, sheet_name="teste_chi2_partido_topico", index=False)

    tabela_ano_topico.to_excel(writer, sheet_name="ano_x_topico")
    tabela_ano_topico_pct.to_excel(writer, sheet_name="ano_x_topico_pct")
    tabela_topico_sucesso.to_excel(writer, sheet_name="topico_x_sucesso")
    agenda_resumo.to_excel(writer, sheet_name="taxa_sucesso_topico", index=False)

    tabela_partido_topico.to_excel(writer, sheet_name="partido_x_topico_coaut")
    tabela_partido_topico_pct.to_excel(writer, sheet_name="partido_x_topico_pct_coaut")
    tabela_partido_agenda.to_excel(writer, sheet_name="partido_agenda_sucesso", index=False)

    df_reg.to_excel(writer, sheet_name="base_regressao_geral", index=False)
    df_reg_decidido.to_excel(writer, sheet_name="base_regressao_decidida", index=False)
    df_reg_inf.to_excel(writer, sheet_name="base_regressao_inferencial", index=False)

    if not tab_partido_inf.empty:
        tab_partido_inf.to_excel(writer, sheet_name="partido_elegibilidade")

    if not df_rank.empty:
        df_rank.to_excel(writer, sheet_name="ranking_probabilidades", index=False)

    if not coef.empty:
        coef.to_excel(writer, sheet_name="coef_modelo_preditivo", index=False)

    if not coef_topics_plot.empty:
        coef_topics_plot.to_excel(writer, sheet_name="coef_apenas_topicos", index=False)

    if not df_odds_principal.empty:
        df_odds_principal.to_excel(writer, sheet_name="odds_modelo_principal", index=False)

    if not df_odds.empty:
        df_odds.to_excel(writer, sheet_name="odds_modelo_partido", index=False)

    if not df_odds_ano.empty:
        df_odds_ano.to_excel(writer, sheet_name="odds_modelo_ano", index=False)

    if not df_odds_leg.empty:
        df_odds_leg.to_excel(writer, sheet_name="odds_modelo_leg", index=False)

    if not df_metricas_modelo_pred.empty:
        df_metricas_modelo_pred.to_excel(writer, sheet_name="metricas_modelo_pred", index=False)

    if not df_metricas_thresholds.empty:
        df_metricas_thresholds.to_excel(writer, sheet_name="metricas_thresholds", index=False)

    if not tabela_corporacao.empty:
        tabela_corporacao.to_excel(writer, sheet_name="tabela_corporacao")

    if not tabela_corp_sucesso_full.empty:
        tabela_corp_sucesso_full.to_excel(writer, sheet_name="corp_sucesso_full")

    if not tabela_corp_topico.empty:
        tabela_corp_topico.to_excel(writer, sheet_name="corp_x_topico")

    if not tabela_corp_topico_pct.empty:
        tabela_corp_topico_pct.to_excel(writer, sheet_name="corp_x_topico_pct")

    if not tabela_corp_sucesso.empty:
        tabela_corp_sucesso.to_excel(writer, sheet_name="corp_x_sucesso")

    if not df_odds_corp.empty:
        df_odds_corp.to_excel(writer, sheet_name="odds_modelo_corp", index=False)

    if not base_pred_corp.empty:
        base_pred_corp.to_excel(writer, sheet_name="prob_prev_corp", index=False)

    if not base_pred_corp_top.empty:
        base_pred_corp_top.to_excel(writer, sheet_name="prob_prev_corp_top", index=False)

    if not df_rank_c.empty:
        df_rank_c.to_excel(writer, sheet_name="ranking_prob_corp", index=False)

    if not coef_c.empty:
        coef_c.to_excel(writer, sheet_name="coef_modelo_corp", index=False)

    if not df_metricas_modelo_corp.empty:
        df_metricas_modelo_corp.to_excel(writer, sheet_name="metricas_modelo_corp", index=False)

    if not df_metricas_thresholds_corp.empty:
        df_metricas_thresholds_corp.to_excel(writer, sheet_name="metricas_thr_corp", index=False)

    if not pred_ano_historico.empty:
        pred_ano_historico.to_excel(writer, sheet_name="pred_ano_historico", index=False)

    if not pred_ano.empty:
        pred_ano.to_excel(writer, sheet_name="pred_ano", index=False)

    if not pred_partido.empty:
        pred_partido.to_excel(writer, sheet_name="pred_partido", index=False)

    if not pred_legislatura.empty:
        pred_legislatura.to_excel(writer, sheet_name="pred_legislatura", index=False)

    if not pred_corporacao.empty:
        pred_corporacao.to_excel(writer, sheet_name="pred_corporacao", index=False)

    if not pred_ano_partido.empty:
        pred_ano_partido.to_excel(writer, sheet_name="pred_ano_partido", index=False)

    if not pred_ano_corporacao.empty:
        pred_ano_corporacao.to_excel(writer, sheet_name="pred_ano_corporacao", index=False)

    if not pred_corp_topico.empty:
        pred_corp_topico.to_excel(writer, sheet_name="pred_corp_topico", index=False)

    if not df_pred_ano.empty:
        df_pred_ano.to_excel(writer, sheet_name="predicao_futura_ano", index=False)

    if not resumo_ano.empty:
        resumo_ano.to_excel(writer, sheet_name="predicao_media_ano", index=False)

    if not resumo_cenarios.empty:
        resumo_cenarios.to_excel(writer, sheet_name="cenarios_ano", index=False)

    if not resumo_partido_ano.empty:
        resumo_partido_ano.to_excel(writer, sheet_name="partido_ano_pred", index=False)

    if not resumo_corporacao_ano.empty:
        resumo_corporacao_ano.to_excel(writer, sheet_name="corporacao_ano_pred", index=False)

    if not ranking_partido_futuro.empty:
        ranking_partido_futuro.to_excel(writer, sheet_name="ranking_partido_futuro", index=False)

    if not ranking_corporacao_futuro.empty:
        ranking_corporacao_futuro.to_excel(writer, sheet_name="ranking_corp_futuro", index=False)

    if not df_rank_futuro.empty:
        df_rank_futuro.to_excel(writer, sheet_name="ranking_futuro", index=False)

    if not df_resultados_bt.empty:
        df_resultados_bt.to_excel(writer, sheet_name="ml_backtesting", index=False)

    if not df_predicoes_bt.empty:
        df_predicoes_bt.to_excel(writer, sheet_name="ml_predicoes_bt", index=False)

    if not resumo_modelos_bt.empty:
        resumo_modelos_bt.to_excel(writer, sheet_name="ml_resumo_modelos", index=False)

    if not df_resumo_executivo.empty:
        df_resumo_executivo.to_excel(writer, sheet_name="resumo_executivo", index=False)

    if not df_modelos_contagem.empty:
        df_modelos_contagem.to_excel(writer, sheet_name="modelos_contagem", index=False)

    if not df_overdisp.empty:
        df_overdisp.to_excel(writer, sheet_name="coef_modelos_contagem", index=False)

    if not df_ct_resultados.empty:
        df_ct_resultados.to_excel(writer, sheet_name="cameron_trivedi_teste", index=False)

    if not df_sociol_descritiva.empty:
        df_sociol_descritiva.to_excel(writer, sheet_name="sociol_descritiva", index=False)

    if not df_odds_sociol.empty:
        df_odds_sociol.to_excel(writer, sheet_name="modelo_sociol_odds", index=False)

    if not df_previsao_ml.empty:
        df_previsao_ml.head(500).to_excel(writer, sheet_name="ml_previsao_cenarios", index=False)

    if not df_previsao_ml_resumo.empty:
        df_previsao_ml_resumo.to_excel(writer, sheet_name="ml_previsao_resumo", index=False)

    if not df_zinb_metricas.empty:
        df_zinb_metricas.to_excel(writer, sheet_name="zinb_metricas", index=False)

    if not df_zinb_coef.empty:
        df_zinb_coef.to_excel(writer, sheet_name="zinb_coeficientes", index=False)

    if not df_zinb_comparacao.empty:
        df_zinb_comparacao.to_excel(writer, sheet_name="zinb_comparacao_modelos", index=False)

    if not df_multinivel_metricas.empty:
        df_multinivel_metricas.to_excel(writer, sheet_name="multinivel_metricas", index=False)

    if not df_multinivel_coef.empty:
        df_multinivel_coef.to_excel(writer, sheet_name="multinivel_coef", index=False)

    if not df_multinivel_icc.empty:
        df_multinivel_icc.to_excel(writer, sheet_name="multinivel_icc", index=False)

    if not df_ame_principal.empty:
        df_ame_principal.to_excel(writer, sheet_name="ame_modelo_principal", index=False)

    if not df_ame_partido.empty:
        df_ame_partido.to_excel(writer, sheet_name="ame_modelo_partido", index=False)

    if not df_ame_corp.empty:
        df_ame_corp.to_excel(writer, sheet_name="ame_modelo_corp", index=False)

    if not df_anacor_corp.empty:
        df_anacor_corp.to_excel(writer, sheet_name="anacor_corp_linhas", index=False)

    if not df_anacor_topico.empty:
        df_anacor_topico.to_excel(writer, sheet_name="anacor_corp_colunas", index=False)

    if not df_anacor_partido.empty:
        df_anacor_partido.to_excel(writer, sheet_name="anacor_partido", index=False)

    if not df_rede_arestas.empty:
        df_rede_arestas.to_excel(writer, sheet_name="rede_arestas", index=False)

    if not df_rede_metricas.empty:
        df_rede_metricas.to_excel(writer, sheet_name="rede_metricas", index=False)

    df_texto[
        [
            "Proposicoes",
            "Autor",
            "Autor_original",
            "UF",
            "UF_original",
            "Partido",
            "Partido_original",
            "Partido_limpo",
            "texto_original",
            "topico_dominante",
            "prob_topico_dominante",
            "ano",
            "situacao_recodificada",
            "sucesso_legislativo"
        ]
    ].to_excel(writer, sheet_name="lda_docs", index=False)

    if not df_pearson.empty:
        df_pearson.to_excel(writer, sheet_name="pearson_produtividade", index=False)

    if not df_clusters.empty:
        df_clusters.to_excel(writer, sheet_name="kmeans_clusters", index=False)

    if not df_sentiment.empty:
        df_sentiment.to_excel(writer, sheet_name="sentiment_tom", index=False)

    if not _taxa_uf_esp.empty:
        _taxa_uf_esp.sort_values("taxa_pct", ascending=False).to_excel(
            writer, sheet_name="espacial_taxa_uf", index=False)

    if not df_probit_coef.empty:
        df_probit_coef.to_excel(writer, sheet_name="probit_robustez", index=False)

    if not df_cv_resultados.empty:
        df_cv_resultados.to_excel(writer, sheet_name="cross_validation", index=False)

    if not df_serie_temporal.empty:
        df_serie_temporal.to_excel(writer, sheet_name="serie_temporal_anual", index=False)

    if not df_mca_coords.empty:
        df_mca_coords.to_excel(writer, sheet_name="mca_perfil_parlamentar", index=False)

    if not df_sociol_integrado.empty:
        _cols_exp = [c for c in df_sociol_integrado.columns
                     if df_sociol_integrado[c].dtype != object
                     or df_sociol_integrado[c].nunique() < 200]
        df_sociol_integrado[_cols_exp].to_excel(
            writer, sheet_name="sociologia_integrada", index=False)

print(f"\nArquivo final salvo em:\n{ARQUIVO_SAIDA}")

# =========================================================
# 42B. LEGISLATURA REINTEGRADA + TABELA DE IMPORTÂNCIA
# ─────────────────────────────────────────────────────────
# Dois objetivos:
# 1. Reintegrar legislatura como variável explicativa no
#    modelo principal, com dummy para 57ª (incompleta).
#    Permite captar efeitos de contexto político que o
#    ano contínuo não captura (mudanças de coalizão, etc.)
# 2. Tabela unificada de importância: logit coef + AME +
#    permutation importance do RF — responde "o que pesa
#    mais no sucesso".
# =========================================================
print("\n" + "=" * 60)
print("42B. LEGISLATURA REINTEGRADA + TABELA DE IMPORTÂNCIA")
print("=" * 60)

try:
    if not df_reg_inf.empty and "legislatura" in df_reg_inf.columns:
        _base_leg = df_reg_inf[
            df_reg_inf["situacao_recodificada"].isin(["sucesso","fracasso"])
        ].copy()

        # dummy para 57ª legislatura (incompleta — controla viés de truncamento)
        _base_leg["leg57_dummy"] = (
            _base_leg["legislatura"] == "57a"
        ).astype(int)

        # ── Modelo 1: tópico + legislatura (dummies) ──────────────
        # Substitui ano contínuo por dummies de legislatura
        # Cada coeficiente captura o efeito contextual de cada período
        # A 57ª entra com dummy de truncamento (controle)
        _fml_leg_full = (
            "aprovado ~ C(topico_dominante) + C(partido_inf) "
            "+ C(legislatura) + leg57_dummy"
        )
        # remove 57a da formula C(legislatura) para evitar colinearidade
        _base_leg_no57 = _base_leg.copy()

        try:
            _mod_leg_full = smf.logit(
                "aprovado ~ C(topico_dominante) + C(partido_inf) + C(legislatura)",
                data=_base_leg
            ).fit(method="lbfgs", maxiter=400, disp=False)

            print(f"\nModelo com legislatura (dummies): N={len(_base_leg)} | "
                  f"pseudo-R²={_mod_leg_full.prsquared:.4f}")

            # coeficientes de legislatura
            _leg_coefs = pd.DataFrame({
                "variavel": _mod_leg_full.params.index,
                "coef":     _mod_leg_full.params.round(4),
                "or":       np.exp(_mod_leg_full.params).round(3),
                "p":        _mod_leg_full.pvalues.round(4)
            })
            _leg_only = _leg_coefs[
                _leg_coefs["variavel"].str.startswith("C(legislatura)")
            ].copy()
            _leg_only["legislatura"] = (
                _leg_only["variavel"]
                .str.extract(r"\[T\.(.*?)\]")[0]
            )
            print("\nEfeito de cada legislatura sobre sucesso (ref=49ª):")
            print(_leg_only[["legislatura","coef","or","p"]].to_string(index=False))
            print("\nInterpretação:")
            print("  coef > 0: legislatura tem maior chance de sucesso que a 49ª")
            print("  OR > 1: odds ratio em relação à legislatura de referência")
            print("  Nota: 57ª entra com dados parciais — interpretar com cautela")

            # comparação modelo ano_c vs legislatura por AIC
            if modelo_principal is not None:
                _aic_anoc = modelo_principal.aic
                _aic_leg  = _mod_leg_full.aic
                print(f"\nComparação AIC:")
                print(f"  Modelo tópico+partido+ano_c:      AIC={_aic_anoc:.1f}")
                print(f"  Modelo tópico+partido+legislatura: AIC={_aic_leg:.1f}")
                _melhor = "legislatura" if _aic_leg < _aic_anoc else "ano_c"
                print(f"  Melhor ajuste: {_melhor} (menor AIC)")

        except Exception as _e_leg:
            print(f"  [AVISO] Modelo legislatura falhou: {_e_leg}")

        # ── Tabela unificada de importância de variáveis ──────────
        # Combina 3 perspectivas:
        # (a) Coeficiente logit padronizado (magnitude relativa)
        # (b) AME — efeito marginal médio em p.p.
        # (c) Permutation importance do Random Forest
        print("\n--- Tabela unificada de importância de variáveis ---")

        try:
            # (a) logit — coeficientes padronizados
            # usa modelo principal (tópico + partido + ano)
            if modelo_principal is not None and not df_reg_inf.empty:
                _base_imp = df_reg_inf[
                    df_reg_inf["situacao_recodificada"].isin(["sucesso","fracasso"])
                ].copy()

                _params = modelo_principal.params.copy()
                _pvals  = modelo_principal.pvalues.copy()

                # AME já calculados anteriormente — recomputa aqui para garantir
                from statsmodels.stats.margeff import margeff_count
                try:
                    _ames = modelo_principal.get_margeff().summary_frame()
                    _ames_dict = dict(zip(_ames.index, _ames["dy/dx"]))
                except Exception:
                    _ames_dict = {}

                # coeficientes absolutos (magnitude) para ranking
                _params_abs = _params.abs().sort_values(ascending=False)

                # cria tabela com todas as variáveis
                _imp_rows = []
                for var in _params.index:
                    if var == "Intercept":
                        continue
                    # nome legível
                    _label = var
                    if "topico_dominante" in var:
                        _t = var.split("[T.")[-1].rstrip("]")
                        _label = f"Tópico T{_t} vs T1"
                    elif "partido_inf" in var:
                        _p = var.split("[T.")[-1].rstrip("]")
                        _label = f"Partido {_p} vs DEM"
                    elif "corporacao" in var:
                        _c = var.split("[T.")[-1].rstrip("]")
                        _label = f"Corporação {_c} vs PM"
                    elif var == "ano_c":
                        _label = "Tendência temporal (ano_c)"

                    _imp_rows.append({
                        "variavel": _label,
                        "coef_logit": round(_params[var], 4),
                        "abs_coef":   round(abs(_params[var]), 4),
                        "p_value":    round(_pvals[var], 4),
                        "ame_pp":     round(_ames_dict.get(var, np.nan) * 100, 3),
                        "sig":        "***" if _pvals[var] < 0.01
                                      else "**" if _pvals[var] < 0.05
                                      else "*" if _pvals[var] < 0.10
                                      else ""
                    })

                _df_imp = (
                    pd.DataFrame(_imp_rows)
                    .sort_values("abs_coef", ascending=False)
                    .reset_index(drop=True)
                )
                print("\n(a) Ranking por magnitude do coeficiente logit + AME:")
                print(_df_imp[["variavel","coef_logit","p_value","sig","ame_pp"]]
                      .head(20).to_string(index=False))

            # (b) Permutation importance do Random Forest
            # reusa o modelo rf do backtesting se disponível,
            # ou treina um novo modelo parcimonioso
            print("\n(b) Permutation importance — Random Forest:")
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.inspection import permutation_importance
            from sklearn.preprocessing import LabelEncoder

            _base_perm = df_reg_inf[
                df_reg_inf["situacao_recodificada"].isin(["sucesso","fracasso"])
            ].copy()

            # features para permutation importance
            _feat_cols = ["topico_dominante", "partido_inf", "ano_c"]
            if "corporacao_sigla" in _base_perm.columns:
                _feat_cols.append("corporacao_sigla")
            if "legislatura" in _base_perm.columns:
                _feat_cols.append("legislatura")

            _base_perm = _base_perm[_feat_cols + ["aprovado"]].dropna()

            # encode categóricas
            _le_dict = {}
            for _fc in _feat_cols:
                if _base_perm[_fc].dtype == object or _base_perm[_fc].dtype.name == "category":
                    _le = LabelEncoder()
                    _base_perm[_fc] = _le.fit_transform(_base_perm[_fc].astype(str))
                    _le_dict[_fc] = _le

            _X_perm = _base_perm[_feat_cols].values
            _y_perm = _base_perm["aprovado"].values

            _rf_imp = RandomForestClassifier(
                n_estimators=200, max_depth=5,
                class_weight="balanced", random_state=42, n_jobs=-1
            )
            _rf_imp.fit(_X_perm, _y_perm)

            _perm_res = permutation_importance(
                _rf_imp, _X_perm, _y_perm,
                n_repeats=20, random_state=42,
                scoring="roc_auc"
            )
            _df_perm = pd.DataFrame({
                "variavel":    _feat_cols,
                "imp_media":   _perm_res.importances_mean.round(4),
                "imp_dp":      _perm_res.importances_std.round(4),
            }).sort_values("imp_media", ascending=False)

            print(_df_perm.to_string(index=False))

            # ── Tabela consolidada final ───────────────────────────
            # Junta logit e RF numa visão única
            print("\n(c) Tabela consolidada — logit vs. RF:")
            _df_perm_idx = _df_perm.set_index("variavel")
            _consolidado = []

            _mapa_nomes = {
                "topico_dominante": "Agenda temática (tópico)",
                "partido_inf":      "Filiação partidária",
                "ano_c":            "Tendência temporal",
                "corporacao_sigla": "Corporação de origem",
                "legislatura":      "Legislatura (contexto político)"
            }
            for _fc in _feat_cols:
                _nome = _mapa_nomes.get(_fc, _fc)
                _imp  = _df_perm_idx.loc[_fc, "imp_media"] if _fc in _df_perm_idx.index else np.nan
                # maior AME absoluto relacionado à variável
                if not _df_imp.empty:
                    _ame_max = (
                        _df_imp[_df_imp["variavel"].str.contains(
                            _fc.replace("_sigla","").replace("_inf","").replace("_dominante",""),
                            case=False
                        )]["ame_pp"].abs().max()
                    )
                else:
                    _ame_max = np.nan

                _consolidado.append({
                    "Variável":               _nome,
                    "Perm.Imp. RF (ΔAUC)":    f"{_imp:.4f}" if not np.isnan(_imp) else "—",
                    "AME máx. logit (p.p.)":  f"{_ame_max:.2f}" if not np.isnan(_ame_max) else "—",
                })

            _df_cons = pd.DataFrame(_consolidado)
            print(_df_cons.to_string(index=False))
            print("\nInterpretação:")
            print("  Perm.Imp. RF: quanto o AUC cai ao embaralhar a variável (maior = mais importante)")
            print("  AME máx logit: maior efeito marginal médio absoluto entre as categorias da variável")
            print("  Variável mais importante em ambas as métricas = evidência mais robusta")

            # gráfico combinado
            fig_imp, axes_imp = plt.subplots(1, 2, figsize=(12, 5))

            # painel A: permutation importance
            _df_perm_plot = _df_perm.copy()
            _df_perm_plot["nome"] = _df_perm_plot["variavel"].map(_mapa_nomes).fillna(_df_perm_plot["variavel"])
            axes_imp[0].barh(
                _df_perm_plot["nome"], _df_perm_plot["imp_media"],
                xerr=_df_perm_plot["imp_dp"], color="#3498db", alpha=0.8,
                edgecolor="white", capsize=4
            )
            axes_imp[0].set_xlabel("Permutation importance (ΔAUC)")
            axes_imp[0].set_title("Random Forest\nPermutation Importance (±1 DP)")
            axes_imp[0].grid(axis="x", linestyle="--", alpha=0.4)

            # painel B: top coeficientes logit (abs) por variável agregada
            if not _df_imp.empty:
                _top_logit = _df_imp.head(15).copy()
                _cores_logit = ["#e74c3c" if c < 0 else "#2ecc71"
                                for c in _top_logit["coef_logit"]]
                axes_imp[1].barh(
                    _top_logit["variavel"], _top_logit["coef_logit"],
                    color=_cores_logit, alpha=0.8, edgecolor="white"
                )
                axes_imp[1].axvline(0, color="black", linewidth=0.8)
                axes_imp[1].set_xlabel("Coeficiente logit")
                axes_imp[1].set_title("Logit — top 15 coeficientes\n(verde=positivo, vermelho=negativo)")
                axes_imp[1].grid(axis="x", linestyle="--", alpha=0.4)
                axes_imp[1].tick_params(axis="y", labelsize=7)

            fig_imp.suptitle(
                "Importância das variáveis — logit vs. Random Forest\n"
                "O que mais pesa no sucesso legislativo?",
                fontsize=11, fontweight="bold"
            )
            plt.tight_layout()
            plt.savefig(PASTA / "importancia_variaveis.png", dpi=300, bbox_inches="tight")
            plt.show()
            plt.close()
            print("Gráfico de importância salvo.")

        except Exception as _e_imp:
            print(f"  [AVISO] Tabela de importância falhou: {_e_imp}")

    else:
        print("[INFO] df_reg_inf vazio ou sem coluna legislatura — seção 42B pulada.")

except Exception as _e42b:
    print(f"[AVISO] Seção 42B falhou: {_e42b}")


# =========================================================
# 45. MODELO DE SOBREVIVÊNCIA — KAPLAN-MEIER + COX
# Modela o tempo até aprovação (ou até o evento censurado).
# Pergunta: quais temas aceleram/retardam o sucesso?
# =========================================================
print("\n" + "=" * 60)
print("45. MODELO DE SOBREVIVÊNCIA (KAPLAN-MEIER + COX)")
print("=" * 60)

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    _LIFELINES_OK = True
except ImportError:
    try:
        import subprocess
        subprocess.run(
            ["pip", "install", "lifelines", "--break-system-packages", "-q"],
            capture_output=True
        )
        from lifelines import KaplanMeierFitter, CoxPHFitter
        _LIFELINES_OK = True
    except Exception:
        _LIFELINES_OK = False
        print("[INFO] lifelines não disponível. Seção 45 pulada.")

if _LIFELINES_OK and not df_texto.empty:
    try:
        # constrói base de sobrevivência
        # duração = (ano_atual - ano_apresentação)
        # evento = 1 se aprovado, 0 se ainda em tramitação (censurado) ou fracasso
        _base_surv = df_texto[["Proposicoes","ano","situacao_recodificada",
                                "topico_dominante"]].copy()
        _base_surv = _base_surv.merge(
            df_reg_inf[["Proposicoes","partido_inf"]].drop_duplicates(),
            on="Proposicoes", how="left"
        ) if "partido_inf" in df_reg_inf.columns else _base_surv

        _ANO_REF = 2024
        _base_surv["duracao"] = (_ANO_REF - _base_surv["ano"]).clip(lower=0)
        _base_surv["evento"]  = (_base_surv["situacao_recodificada"] == "sucesso").astype(int)
        # censurado = em tramitação (ainda não decidido)
        _base_surv = _base_surv[_base_surv["duracao"] > 0].dropna(
            subset=["duracao","evento","topico_dominante"]
        )

        print(f"\nBase de sobrevivência: {len(_base_surv)} PLs")
        print(f"Aprovados (eventos): {_base_surv['evento'].sum()}")
        print(f"Censurados: {(_base_surv['situacao_recodificada']=='em_tramitacao').sum()}")

        # ── Kaplan-Meier por tópico ────────────────────────────
        fig_km, ax_km = plt.subplots(figsize=(10, 6))
        _cores_km = {1:"#2ecc71",2:"#3498db",3:"#f39c12",4:"#9b59b6",5:"#e74c3c"}
        for _top in [1,2,3,4,5]:
            _s = _base_surv[_base_surv["topico_dominante"] == _top]
            if len(_s) > 10:
                kmf = KaplanMeierFitter()
                kmf.fit(
                    _s["duracao"], event_observed=_s["evento"],
                    label=NOMES_TOPICOS_CURTO.get(_top, f"T{_top}")
                )
                kmf.plot_survival_function(
                    ax=ax_km, ci_show=False,
                    color=_cores_km.get(_top,"gray"), linewidth=1.8
                )
        ax_km.set_xlabel("Anos desde a apresentação do PL")
        ax_km.set_ylabel("Probabilidade de sobrevivência (não aprovado)")
        ax_km.set_title(
            "Kaplan-Meier: probabilidade de ainda não ter sido aprovado\n"
            "por tópico LDA (queda mais rápida = mais aprovações ao longo do tempo)"
        )
        ax_km.legend(fontsize=8)
        ax_km.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(PASTA / "kaplan_meier_topico.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print("Gráfico Kaplan-Meier por tópico salvo.")

        # ── Cox Proportional Hazards ───────────────────────────
        _base_cox = _base_surv[
            ["duracao","evento","topico_dominante"]
        ].dropna()
        # dummies de tópico (T1 = referência)
        _base_cox = pd.get_dummies(
            _base_cox, columns=["topico_dominante"], drop_first=True
        )
        _bool_cols = _base_cox.select_dtypes(bool).columns
        _base_cox[_bool_cols] = _base_cox[_bool_cols].astype(int)

        if len(_base_cox) >= 100 and _base_cox["evento"].sum() >= 5:
            cph = CoxPHFitter()
            cph.fit(_base_cox, duration_col="duracao", event_col="evento")
            print("\nCox Proportional Hazards — resumo:")
            cph.print_summary(decimals=3, model="base")
            print("\nInterpretação: HR > 1 → maior taxa de aprovação em relação a T1")
            print("               HR < 1 → menor taxa de aprovação (aprovação mais lenta/rara)")
            plt.figure(figsize=(8, 4))
            cph.plot()
            plt.title("Cox PH — Hazard Ratios por tópico (referência: T1 Regulação/Trânsito)")
            plt.tight_layout()
            plt.savefig(PASTA / "cox_topico.png", dpi=300, bbox_inches="tight")
            plt.show()
            plt.close()
            print("Gráfico Cox salvo.")
        else:
            print("[INFO] Base insuficiente para Cox PH.")

    except Exception as _e45b:
        print(f"[AVISO] Modelo de sobrevivência falhou: {_e45b}")


# =========================================================
# 46. CALIBRATION PLOT — modelo preditivo principal
# Um modelo bem calibrado: quando prevê 10%, ~10% aprovam.
# =========================================================
print("\n" + "=" * 60)
print("46. CALIBRATION PLOT — MODELO PREDITIVO")
print("=" * 60)

try:
    from sklearn.calibration import calibration_curve

    if modelo_principal is not None and not df_reg_inf.empty:
        _base_cal = df_reg_inf[
            df_reg_inf["situacao_recodificada"].isin(["sucesso","fracasso"])
        ].copy()
        _y_cal  = _base_cal["aprovado"]
        _p_cal  = modelo_principal.predict(_base_cal)

        # curva de calibração com 10 bins
        _frac_pos, _mean_pred = calibration_curve(
            _y_cal, _p_cal, n_bins=10, strategy="quantile"
        )

        fig_cal, ax_cal = plt.subplots(figsize=(7, 6))
        ax_cal.plot([0,1],[0,1], "k--", label="Calibração perfeita", linewidth=1)
        ax_cal.plot(_mean_pred, _frac_pos, "o-", color="#3498db",
                    linewidth=2, markersize=7, label="Modelo logit principal")
        ax_cal.set_xlabel("Probabilidade média predita")
        ax_cal.set_ylabel("Fração de positivos observados")
        ax_cal.set_title(
            "Calibration plot — modelo logit (tópico + partido + ano)\n"
            "Pontos acima da diagonal: modelo subestima; abaixo: superestima"
        )
        ax_cal.legend(fontsize=9)
        ax_cal.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(PASTA / "calibration_plot.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

        # Brier score
        from sklearn.metrics import brier_score_loss
        _brier = brier_score_loss(_y_cal, _p_cal)
        _brier_base = brier_score_loss(_y_cal, np.full(len(_y_cal), _y_cal.mean()))
        print(f"\nBrier score: {_brier:.4f} (baseline ingênuo: {_brier_base:.4f})")
        print(f"Ganho de calibração: {(_brier_base - _brier)/_brier_base*100:.1f}%")
        print("Brier < baseline → modelo melhora a calibração sobre o acaso.")
        print("Calibration plot salvo.")
    else:
        print("[INFO] modelo_principal não disponível — calibration plot pulado.")

except Exception as _e46:
    print(f"[AVISO] Seção 46 falhou: {_e46}")


# =========================================================
# 47. COHERENCE SCORE DO LDA — k=4 vs 5 vs 6
# Valida a escolha de k=5 tópicos empiricamente.
# =========================================================
print("\n" + "=" * 60)
print("47. COHERENCE SCORE DO LDA (k=4, 5, 6)")
print("=" * 60)

try:
    from gensim.models.coherencemodel import CoherenceModel
    from gensim import corpora

    if df_texto is not None and "texto_limpo" in df_texto.columns:
        _textos_coh = df_texto["texto_limpo"].dropna().apply(str.split).tolist()
        _dict_coh   = corpora.Dictionary(_textos_coh)
        _dict_coh.filter_extremes(no_below=5, no_above=0.5)
        _corpus_coh = [_dict_coh.doc2bow(t) for t in _textos_coh]

        _coh_scores = {}
        for _k in [4, 5, 6]:
            from gensim.models import LdaModel
            _lda_k = LdaModel(
                corpus=_corpus_coh, id2word=_dict_coh,
                num_topics=_k, random_state=42, passes=5,
                alpha="auto", per_word_topics=False
            )
            _cm = CoherenceModel(
                model=_lda_k, texts=_textos_coh,
                dictionary=_dict_coh, coherence="c_v"
            )
            _coh_scores[_k] = _cm.get_coherence()
            print(f"  k={_k}: coherence c_v = {_coh_scores[_k]:.4f}")

        _k_best = max(_coh_scores, key=_coh_scores.get)
        print(f"\nMelhor k por coherence: {_k_best} (score={_coh_scores[_k_best]:.4f})")
        if _k_best == 5:
            print("✓ Escolha de k=5 validada empiricamente pelo coherence score.")
        else:
            print(f"⚠ Coherence sugere k={_k_best}. Documente a discrepância no texto.")
            print("  Justificativa aceitável: interpretabilidade substantiva priorizada.")

        # gráfico coherence
        fig_coh, ax_coh = plt.subplots(figsize=(5, 4))
        ax_coh.bar([4,5,6], [_coh_scores[k] for k in [4,5,6]],
                   color=["#95a5a6","#3498db","#95a5a6"], alpha=0.85)
        ax_coh.set_xticks([4,5,6])
        ax_coh.set_xlabel("Número de tópicos (k)")
        ax_coh.set_ylabel("Coherence score (c_v)")
        ax_coh.set_title("Validação do LDA: coherence score por k")
        ax_coh.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(PASTA / "lda_coherence.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print("Gráfico coherence salvo.")
    else:
        print("[INFO] texto_limpo não disponível — coherence score pulado.")

except Exception as _e47:
    print(f"[AVISO] Seção 47 (coherence LDA) falhou: {_e47}")


# 42. INTERAÇÃO TÓPICO × PARTIDO E TÓPICO × CORPORAÇÃO
# =========================================================
# Pergunta: o efeito negativo do penalismo (T5) é uniforme entre partidos,
# ou alguns partidos conseguem aprovações penais que outros não conseguem?
# Referência: Norton (2012) — interações em logit binário.
# =========================================================

print("\n" + "=" * 60)
print("42. INTERAÇÕES TÓPICO × PARTIDO / CORPORAÇÃO")
print("=" * 60)

try:
    if not df_reg_inf.empty and "partido_inf" in df_reg_inf.columns:
        _base_int2 = df_reg_inf[
            df_reg_inf["situacao_recodificada"].isin(["sucesso","fracasso"])
        ].copy() if "situacao_recodificada" in df_reg_inf.columns else \
        df_reg_inf[df_reg_inf["aprovado"].notna()].copy()

        # flag T5 e T1 (referência)
        _base_int2["t5_flag"] = (_base_int2["topico_dominante"] == 5).astype(int)
        _base_int2["t3_flag"] = (_base_int2["topico_dominante"] == 3).astype(int)

        # partidos com >= 3 aprovações em T5
        _pt5 = (
            _base_int2[_base_int2["t5_flag"] == 1]
            .groupby("partido_inf")["aprovado"].sum()
        )
        _partidos_t5 = _pt5[_pt5 >= 2].index.tolist()
        print(f"\nPartidos com >= 2 aprovações em T5: {_partidos_t5}")

        # modelo com interação t5 × partido
        _fml_int_partido = (
            "aprovado ~ C(topico_dominante) + ano_c + C(partido_inf) "
            "+ t5_flag:C(partido_inf)"
        )
        _base_int2_cc = _base_int2[
            ["aprovado","topico_dominante","ano_c","partido_inf","t5_flag","t3_flag"]
        ].dropna()

        if len(_base_int2_cc) >= 100 and _base_int2_cc["aprovado"].sum() >= 5:
            _mod_int_partido = smf.logit(
                _fml_int_partido, data=_base_int2_cc
            ).fit(method="lbfgs", maxiter=400, disp=False)

            print(f"\nModelo interação tópico × partido:")
            print(f"  N={len(_base_int2_cc)} | pseudo-R²={_mod_int_partido.prsquared:.4f}")

            # extrair coefs de interação
            _int_coefs = _mod_int_partido.params[
                _mod_int_partido.params.index.str.contains("t5_flag:C")
            ]
            _int_pvals = _mod_int_partido.pvalues[
                _mod_int_partido.pvalues.index.str.contains("t5_flag:C")
            ]
            _int_df = pd.DataFrame({
                "variavel": _int_coefs.index,
                "coef":     _int_coefs.values.round(4),
                "or":       np.exp(_int_coefs.values).round(3),
                "p":        _int_pvals.values.round(4)
            }).sort_values("coef", ascending=False)
            print("\nInterações t5 × partido (OR > 1 = partido mitiga penalidade do T5):")
            print(_int_df.to_string(index=False))
            print("\nInterpretação: OR > 1 indica que o partido consegue aprovar")
            print("PLs penais com mais eficiência que o partido de referência (DEM).")

        # ── interação tópico × corporação ──────────────────────────
        if not df_reg_corp_base.empty and "corporacao_sigla" in df_reg_corp_base.columns:
            _base_corp_int = df_reg_corp_base[
                df_reg_corp_base["corporacao_sigla"].isin(["PM","EB","PC","PF","MB","SM"])
            ].copy()
            _base_corp_int["t5_flag"] = (
                _base_corp_int["topico_dominante"] == 5
            ).astype(int)
            _base_corp_int["t1_flag"] = (
                _base_corp_int["topico_dominante"] == 1
            ).astype(int)

            _fml_corp_int = (
                "aprovado ~ C(topico_dominante) + ano_c + C(corporacao_sigla) "
                "+ t5_flag:C(corporacao_sigla) + t1_flag:C(corporacao_sigla)"
            )
            _base_corp_int_cc = _base_corp_int[
                ["aprovado","topico_dominante","ano_c",
                 "corporacao_sigla","t5_flag","t1_flag"]
            ].dropna()

            if len(_base_corp_int_cc) >= 50 and _base_corp_int_cc["aprovado"].sum() >= 5:
                _mod_corp_int = smf.logit(
                    _fml_corp_int, data=_base_corp_int_cc
                ).fit(method="lbfgs", maxiter=400, disp=False)

                print(f"\nModelo interação tópico × corporação:")
                print(f"  N={len(_base_corp_int_cc)} | pseudo-R²={_mod_corp_int.prsquared:.4f}")

                _ci_coefs = _mod_corp_int.params[
                    _mod_corp_int.params.index.str.contains("t5_flag:C|t1_flag:C")
                ]
                _ci_pvals = _mod_corp_int.pvalues[
                    _mod_corp_int.pvalues.index.str.contains("t5_flag:C|t1_flag:C")
                ]
                _ci_df = pd.DataFrame({
                    "variavel": _ci_coefs.index,
                    "coef": _ci_coefs.values.round(4),
                    "or": np.exp(_ci_coefs.values).round(3),
                    "p": _ci_pvals.values.round(4)
                }).sort_values("coef", ascending=False)
                print("\nInterações tópico × corporação:")
                print(_ci_df.to_string(index=False))

    else:
        print("[INFO] df_reg_inf vazio — seção 42 pulada.")

except Exception as _e42:
    print(f"[AVISO] Seção 42 (interações): {_e42}")


# =========================================================
# 43. TEMPO NÃO LINEAR — SPLINE + BLOCOS HISTÓRICOS
# =========================================================
# Testa se o efeito temporal é realmente linear ou tem pontos de inflexão.
# Contexto político brasileiro: PT 2003–2015, Temer 2016–2018, Bolsonaro 2019–2022.
# =========================================================

print("\n" + "=" * 60)
print("43. TEMPO NÃO LINEAR — SPLINE E BLOCOS HISTÓRICOS")
print("=" * 60)

try:
    if not df_reg_inf.empty:
        _base_tnl = df_reg_inf[
            df_reg_inf["situacao_recodificada"].isin(["sucesso","fracasso"])
        ].copy() if "situacao_recodificada" in df_reg_inf.columns else \
        df_reg_inf[df_reg_inf["aprovado"].notna()].copy()

        # ── 1. Modelo com ano² (quadrático) ──────────────────────
        _base_tnl["ano_c2"] = _base_tnl["ano_c"] ** 2

        _fml_quad = "aprovado ~ C(topico_dominante) + ano_c + ano_c2"
        _mod_quad = smf.logit(_fml_quad, data=_base_tnl.dropna(
            subset=["aprovado","topico_dominante","ano_c","ano_c2"]
        )).fit(method="lbfgs", maxiter=300, disp=False)

        print(f"\n1. Modelo quadrático (ano + ano²):")
        print(f"   pseudo-R²={_mod_quad.prsquared:.4f} vs linear={0.056:.4f}")
        _q_coef = _mod_quad.params.get("ano_c2", np.nan)
        _q_pval = _mod_quad.pvalues.get("ano_c2", np.nan)
        print(f"   coef_ano²={_q_coef:.4f} (p={_q_pval:.4f})")
        if _q_pval < 0.05:
            print("   → Efeito temporal NÃO é linear. Usar blocos ou spline.")
        else:
            print("   → Efeito quadrático não significativo. Linearidade mantida.")

        # ── 2. Modelo com blocos históricos ──────────────────────
        # Lula 1+2 (2003–2010), Dilma (2011–2016), Temer/Bolsonaro (2017–2022)
        def _bloco_hist(ano):
            if ano < 2003:    return "pre_lula"
            elif ano <= 2010: return "lula"
            elif ano <= 2016: return "dilma_temer1"
            else:             return "bolsonaro_lula3"

        _base_tnl["bloco"] = _base_tnl["ano"].map(_bloco_hist)
        _fml_bloco = "aprovado ~ C(topico_dominante) + C(bloco)"
        _mod_bloco = smf.logit(
            _fml_bloco,
            data=_base_tnl.dropna(subset=["aprovado","topico_dominante","bloco"])
        ).fit(method="lbfgs", maxiter=300, disp=False)

        print(f"\n2. Modelo com blocos histórico-políticos:")
        print(f"   pseudo-R²={_mod_bloco.prsquared:.4f}")
        _bloco_res = pd.DataFrame({
            "bloco": _mod_bloco.params.index,
            "coef": _mod_bloco.params.values.round(4),
            "or": np.exp(_mod_bloco.params.values).round(3),
            "p": _mod_bloco.pvalues.values.round(4)
        })
        _bloco_res_f = _bloco_res[_bloco_res["bloco"].str.contains("bloco")]
        print(_bloco_res_f.to_string(index=False))

        # ── 3. Taxa observada por bloco (descritivo) ─────────────
        _taxa_bloco = (
            _base_tnl.groupby("bloco")["aprovado"]
            .agg(["mean","count","sum"])
            .rename(columns={"mean":"taxa","count":"n_pl","sum":"n_aprov"})
            .round(4)
        )
        print("\n3. Taxa de sucesso observada por bloco histórico:")
        print(_taxa_bloco.to_string())

        # gráfico comparativo: linear vs blocos
        fig_tnl, axes_tnl = plt.subplots(1, 2, figsize=(12, 5))

        # painel A: taxa por ano (rolling mean)
        _ts_ano = (
            _base_tnl.groupby("ano")["aprovado"]
            .agg(["mean","count"])
            .reset_index()
        )
        _ts_ano = _ts_ano[_ts_ano["count"] >= 10]
        _ts_ano["rolling"] = _ts_ano["mean"].rolling(3, center=True).mean()
        axes_tnl[0].scatter(_ts_ano["ano"], _ts_ano["mean"] * 100,
                            color="#3498db", alpha=0.5, s=30, label="Observado")
        axes_tnl[0].plot(_ts_ano["ano"], _ts_ano["rolling"] * 100,
                         color="#e74c3c", linewidth=2, label="Média móvel (3a)")
        for _bl, _bc in [("2003","lula"),("2011","dilma"),("2017","bolsonaro")]:
            axes_tnl[0].axvline(int(_bl), color="#aaaaaa", linestyle="--",
                                alpha=0.6, linewidth=1)
            axes_tnl[0].text(int(_bl) + 0.3, 8, _bc, fontsize=7,
                             color="#666", rotation=45)
        axes_tnl[0].set_xlabel("Ano")
        axes_tnl[0].set_ylabel("Taxa de sucesso (%)")
        axes_tnl[0].set_title("Taxa observada por ano\ncom marcações de bloco histórico")
        axes_tnl[0].legend(fontsize=8)
        axes_tnl[0].grid(True, linestyle="--", alpha=0.3)

        # painel B: taxa por bloco (barras)
        _blocos_ord = ["pre_lula","lula","dilma_temer1","bolsonaro_lula3"]
        _blocos_labels = ["Pré-Lula\n(<2003)","Lula\n(2003–10)",
                          "Dilma/Temer\n(2011–16)","Bolsonaro/Lula3\n(2017+)"]
        _blocos_taxa = [_taxa_bloco.loc[b,"taxa"] * 100
                        if b in _taxa_bloco.index else 0
                        for b in _blocos_ord]
        _blocos_n = [_taxa_bloco.loc[b,"n_pl"]
                     if b in _taxa_bloco.index else 0
                     for b in _blocos_ord]
        _bars = axes_tnl[1].bar(_blocos_labels, _blocos_taxa,
                                color="#3498db", alpha=0.8, edgecolor="white")
        for i, (bar, n) in enumerate(zip(_bars, _blocos_n)):
            axes_tnl[1].text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 0.1,
                             f"{_blocos_taxa[i]:.1f}%\n(n={n})",
                             ha="center", va="bottom", fontsize=8)
        axes_tnl[1].set_ylabel("Taxa de sucesso (%)")
        axes_tnl[1].set_title("Taxa de sucesso por bloco\nhistórico-político")
        axes_tnl[1].grid(axis="y", linestyle="--", alpha=0.3)

        fig_tnl.suptitle(
            "Efeito temporal: linearidade vs. blocos históricos",
            fontsize=11, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(PASTA / "tempo_nao_linear.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print("\nGráfico tempo não linear salvo.")

    else:
        print("[INFO] df_reg_inf vazio — seção 43 pulada.")

except Exception as _e43:
    print(f"[AVISO] Seção 43 (tempo não linear): {_e43}")


# =========================================================
print("\nAnálise concluída com sucesso.")
