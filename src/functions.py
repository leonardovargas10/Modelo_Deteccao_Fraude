"""Funções reutilizáveis do projeto de detecção de fraude."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


RANDOM_STATE = 42


def plota_barras(variavel, df, titulo, top_n=None, rotation=0):
    contagem = df[variavel].value_counts(dropna=False).head(top_n)
    ax = contagem.plot.bar(color="#1FB3E5", title=titulo)
    ax.set_ylabel("Quantidade")
    ax.tick_params(axis="x", rotation=rotation)
    for barra, valor in zip(ax.patches, contagem.values):
        ax.annotate(
            f"{valor / contagem.sum():.1%}",
            (barra.get_x() + barra.get_width() / 2, barra.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.show()


def analisa_distribuicao_via_percentis(df, variaveis):
    return df[variaveis].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T


def taxa_por_grupo(df, coluna, target="is_fraud", top_n=20):
    tabela = (
        df.groupby(coluna, observed=True)[target]
        .agg(qtd="size", fraudes="sum", taxa_fraude="mean")
        .sort_values(["taxa_fraude", "qtd"], ascending=False)
    )
    return tabela.head(top_n)


def haversine_km(lat1, lon1, lat2, lon2):
    """Calcula a distância do arco entre dois pontos da Terra."""
    raio_terra = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * raio_terra * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def media_std_historica(df, grupo, valor, prefixo):
    """Calcula estatísticas acumuladas excluindo a observação atual."""
    g = df.groupby(grupo, sort=False, observed=True)[valor]
    n = df.groupby(grupo, sort=False, observed=True).cumcount().astype("float32")
    soma = g.cumsum() - df[valor]
    soma2 = (
        df.assign(_quadrado=df[valor] ** 2)
        .groupby(grupo, sort=False, observed=True)["_quadrado"]
        .cumsum()
        - df[valor] ** 2
    )
    media = soma / n.replace(0, np.nan)
    variancia = (soma2 / n.replace(0, np.nan) - media**2).clip(lower=0)
    df[f"media_{prefixo}"] = media.fillna(0).astype("float32")
    df[f"std_{prefixo}"] = np.sqrt(variancia).fillna(0).astype("float32")
    return df


def cria_features(df):
    d = df.sort_values(["trans_date_trans_time", "trans_num"]).reset_index(drop=True).copy()

    d["log_amt"] = np.log1p(d.amt).astype("float32")
    d["hora"] = d.trans_date_trans_time.dt.hour.astype("int8")
    d["dia_semana"] = d.trans_date_trans_time.dt.dayofweek.astype("int8")
    d["fim_semana"] = (d.dia_semana >= 5).astype("int8")
    d["hora_sin"] = np.sin(2 * np.pi * d.hora / 24).astype("float32")
    d["hora_cos"] = np.cos(2 * np.pi * d.hora / 24).astype("float32")
    d["idade"] = ((d.trans_date_trans_time.dt.normalize() - d.dob).dt.days / 365.25).astype("float32")
    d["distancia_cliente_lojista_km"] = haversine_km(
        d.lat, d.long, d.merch_lat, d.merch_long
    ).astype("float32")

    g_cartao = d.groupby("cc_num", sort=False, observed=True)
    d["qtd_transacoes_cartao_hist"] = g_cartao.cumcount().astype("int32")
    d["segundos_ultima_transacao"] = (
        g_cartao.trans_date_trans_time.diff()
        .dt.total_seconds()
        .fillna(-1)
        .clip(-1, 30 * 86400)
        .astype("float32")
    )
    d = media_std_historica(d, "cc_num", "amt", "amt_cartao_hist")
    d["zscore_amt_cartao"] = (
        ((d.amt - d.media_amt_cartao_hist) / d.std_amt_cartao_hist.replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(-20, 20)
        .astype("float32")
    )
    primeiro_cartao = g_cartao.trans_date_trans_time.transform("min")
    dias_relacionamento = (d.trans_date_trans_time - primeiro_cartao).dt.total_seconds() / 86400
    d["freq_transacoes_cartao_dia"] = (
        d.qtd_transacoes_cartao_hist / np.maximum(dias_relacionamento, 1)
    ).astype("float32")

    g_lojista = d.groupby("merchant", sort=False, observed=True)
    d["qtd_transacoes_lojista_hist"] = g_lojista.cumcount().astype("int32")
    d = media_std_historica(d, "merchant", "amt", "amt_lojista_hist")

    d["qtd_transacoes_par_hist"] = d.groupby(
        ["cc_num", "merchant"], sort=False, observed=True
    ).cumcount().astype("int16")
    primeira_aresta = d.qtd_transacoes_par_hist.eq(0).astype("int8")
    d["grau_cartao_hist"] = (
        primeira_aresta.groupby(d.cc_num, sort=False).cumsum() - primeira_aresta
    ).astype("int16")
    d["grau_lojista_hist"] = (
        primeira_aresta.groupby(d.merchant, sort=False).cumsum() - primeira_aresta
    ).astype("int16")
    d["novo_par_cartao_lojista"] = primeira_aresta
    d["participacao_lojista_cartao"] = (
        d.qtd_transacoes_par_hist / d.qtd_transacoes_cartao_hist.replace(0, np.nan)
    ).fillna(0).astype("float32")
    return d


def metricas_modelo(nome, etapa, y_true, proba, cutoff=None):
    auc = roc_auc_score(y_true, proba)
    resultado = {
        "Modelo": nome,
        "Etapa": etapa,
        "Gini": 2 * auc - 1,
        "PR_AUC": average_precision_score(y_true, proba),
        "Taxa_Fraude": np.mean(y_true),
    }
    if cutoff is not None:
        pred = (np.asarray(proba) >= cutoff).astype(int)
        resultado.update(
            {
                "Cutoff": cutoff,
                "Precisao": precision_score(y_true, pred, zero_division=0),
                "Recall": recall_score(y_true, pred, zero_division=0),
                "F1": f1_score(y_true, pred, zero_division=0),
                "Alert_Rate": pred.mean(),
            }
        )
    return pd.DataFrame([resultado])


def tabela_capacidade(y_true, proba, capacidades=(0.001, 0.0025, 0.005, 0.01, 0.02)):
    aux = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)}).sort_values(
        "p", ascending=False
    )
    total_transacoes, total_fraudes = len(aux), int(aux.y.sum())
    prevalencia = total_fraudes / max(total_transacoes, 1)
    linhas = []
    for cap in capacidades:
        n = max(1, int(np.ceil(total_transacoes * cap)))
        fila = aux.head(n)
        fraudes_capturadas = int(fila.y.sum())
        linhas.append(
            {
                "Capacidade_maxima": f"{cap:.2%}",
                "Total_transacoes": total_transacoes,
                "Alertas_qtd": n,
                "Taxa_alertas": f"{n / total_transacoes:.2%}",
                "Fraudes_capturadas": fraudes_capturadas,
                "Precisao": f"{fraudes_capturadas / n:.2%}",
                "Fraudes_totais": total_fraudes,
                "Recall": f"{fraudes_capturadas / max(total_fraudes, 1):.2%}",
                "Prevalencia_base": f"{prevalencia:.2%}",
                "Cutoff_score": round(float(fila.p.min()), 4),
            }
        )
    return pd.DataFrame(linhas)


def modelo_lightgbm(parametros=None):
    base = dict(
        objective="binary",
        n_estimators=225,
        learning_rate=0.035,
        num_leaves=12,
        max_depth=4,
        min_child_samples=250,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=2,
        reg_lambda=5,
        subsample_freq=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    if parametros:
        base.update(parametros)
    return LGBMClassifier(**base)


def otimizacao_hyperopt(x_train, y_train, x_valid, y_valid, max_evals=20):
    espaco = {
        "n_estimators": hp.quniform("n_estimators", 100, 350, 25),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.015), np.log(0.08)),
        "num_leaves": hp.quniform("num_leaves", 7, 21, 2),
        "max_depth": hp.quniform("max_depth", 3, 5, 1),
        "min_child_samples": hp.quniform("min_child_samples", 150, 600, 50),
        "subsample": hp.uniform("subsample", 0.65, 1),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(0.1), np.log(20)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(0.5), np.log(30)),
    }

    inteiros = ["n_estimators", "num_leaves", "max_depth", "min_child_samples"]

    def converte(parametros):
        parametros = parametros.copy()
        for coluna in inteiros:
            parametros[coluna] = int(parametros[coluna])
        return parametros

    def objetivo(parametros):
        parametros = converte(parametros)
        modelo = modelo_lightgbm(parametros).fit(x_train, y_train, categorical_feature="auto")
        proba_train = modelo.predict_proba(x_train)[:, 1]
        proba_valid = modelo.predict_proba(x_valid)[:, 1]
        ap_valid = average_precision_score(y_valid, proba_valid)
        ap_train = average_precision_score(y_train, proba_train)
        gini_train = 2 * roc_auc_score(y_train, proba_train) - 1
        gini_valid = 2 * roc_auc_score(y_valid, proba_valid) - 1
        gap_excessivo = max(0, gini_train - gini_valid - 0.05)
        gap_ap_excessivo = max(0, ap_train - ap_valid - 0.08)
        perda = -ap_valid + 0.25 * gap_excessivo + 0.75 * gap_ap_excessivo
        return {
            "loss": perda,
            "status": STATUS_OK,
            "ap_valid": ap_valid,
            "ap_train": ap_train,
            "gap_ap": ap_train - ap_valid,
            "gini_train": gini_train,
            "gini_valid": gini_valid,
        }

    trials = Trials()
    best = fmin(
        objetivo,
        espaco,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(RANDOM_STATE),
        show_progressbar=True,
    )
    best = converte(best)
    modelo = modelo_lightgbm(best).fit(x_train, y_train, categorical_feature="auto")
    return modelo, best, trials


def cria_rating(score, cortes):
    return pd.cut(
        score,
        bins=[-np.inf] + list(cortes) + [np.inf],
        labels=["A", "B", "C", "D", "E"],
        include_lowest=True,
    )


def ajusta_calibrador_score(score, y, metodo):
    if metodo == "isotonic":
        modelo = IsotonicRegression(out_of_bounds="clip").fit(score, y)
    else:
        modelo = LogisticRegression(C=1, random_state=RANDOM_STATE).fit(
            np.asarray(score).reshape(-1, 1), y
        )
    return modelo


def prediz_calibrador(modelo, score, metodo):
    if metodo == "isotonic":
        return modelo.predict(score)
    return modelo.predict_proba(np.asarray(score).reshape(-1, 1))[:, 1]


def retorno_financeiro_incremental(
    df, pred, taxa_recuperacao=0.75, custo_revisao=2, custo_atrito_fp=5
):
    pred = np.asarray(pred).astype(bool)
    fraude = df.is_fraud.to_numpy().astype(bool)
    valores = df.amt.to_numpy()
    tp, fp = pred & fraude, pred & ~fraude
    perda_sem_modelo = valores[fraude].sum()
    perda_residual = valores[fraude & ~pred].sum() + (1 - taxa_recuperacao) * valores[tp].sum()
    custo_operacional = pred.sum() * custo_revisao + fp.sum() * custo_atrito_fp
    retorno_incremental = perda_sem_modelo - perda_residual - custo_operacional
    return {
        "Perda_sem_modelo": perda_sem_modelo,
        "Fraude_evitada": taxa_recuperacao * valores[tp].sum(),
        "Custo_revisao_atrito": custo_operacional,
        "Retorno_incremental": retorno_incremental,
    }


def escolhe_cutoff_politica(
    df_politica,
    score,
    capacidade_max=0.01,
    taxa_recuperacao=0.75,
    custo_revisao=2,
    custo_atrito_fp=5,
):
    aux = df_politica[["is_fraud", "amt"]].copy()
    aux["score"] = np.asarray(score)
    candidatos = np.unique(np.quantile(score, np.linspace(0.90, 0.9999, 600)))
    linhas = []
    for cutoff in candidatos:
        pred = aux.score >= cutoff
        financeiro = retorno_financeiro_incremental(
            aux, pred, taxa_recuperacao, custo_revisao, custo_atrito_fp
        )
        linhas.append(
            {
                "cutoff": cutoff,
                "alert_rate": pred.mean(),
                "precision": precision_score(aux.is_fraud, pred, zero_division=0),
                "recall": recall_score(aux.is_fraud, pred, zero_division=0),
                "retorno_incremental": financeiro["Retorno_incremental"],
            }
        )
    curva = pd.DataFrame(linhas)
    elegiveis = curva[curva.alert_rate <= capacidade_max]
    melhor = elegiveis.sort_values(
        ["retorno_incremental", "precision"], ascending=False
    ).iloc[0]
    return float(melhor.cutoff), curva


def metricas_mensais(df, score, cutoff):
    aux = df[["trans_date_trans_time", "is_fraud"]].copy()
    aux["score"] = np.asarray(score)
    aux["mes"] = aux.trans_date_trans_time.dt.to_period("M").astype(str)
    linhas = []
    for mes, dados_mes in aux.groupby("mes"):
        pred = dados_mes.score >= cutoff
        auc = roc_auc_score(dados_mes.is_fraud, dados_mes.score)
        linhas.append(
            {
                "Mes": mes,
                "N": len(dados_mes),
                "Taxa_Fraude": dados_mes.is_fraud.mean(),
                "Gini": 2 * auc - 1,
                "PR_AUC": average_precision_score(dados_mes.is_fraud, dados_mes.score),
                "Precisao": precision_score(dados_mes.is_fraud, pred, zero_division=0),
                "Recall": recall_score(dados_mes.is_fraud, pred, zero_division=0),
                "Alert_Rate": pred.mean(),
            }
        )
    return pd.DataFrame(linhas)


def plot_shap(modelo, x_amostra, titulo):
    explainer = shap.TreeExplainer(modelo)
    valores = explainer(x_amostra)
    shap.plots.beeswarm(valores, max_display=20, show=False)
    plt.title(titulo)
    plt.tight_layout()
    plt.show()
    return explainer, valores


def define_amostra(data):
    return np.select(
        [
            data < pd.Timestamp("2020-01-01"),
            data < pd.Timestamp("2020-04-01"),
            data < pd.Timestamp("2020-06-01"),
            data < pd.Timestamp("2020-06-21 12:14:00"),
            data < pd.Timestamp("2020-10-01"),
        ],
        ["Treino", "Validacao", "Calibracao", "Politica", "Teste"],
        default="OOT",
    )
