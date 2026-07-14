import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": text.splitlines(True)
    }


cells = []

cells.append(md("""# Modelo robusto de detecção de fraude

Este notebook revisita o projeto original mantendo suas etapas e seu estilo, mas separa três decisões que antes estavam misturadas:

1. **Target:** o modelo aprende `high_risk`, o rótulo mais próximo de fraude confirmada disponível nesta base sintética. `moderate_risk` não é tratado como fraude por definição.
2. **Ranking:** PR-AUC (*Average Precision*) mede se fraudes ficam no topo da fila, sem depender de um cutoff arbitrário.
3. **Política:** calibração e threshold transformam o ranking em ações, respeitando capacidade operacional, precisão mínima e custos.

> Limitação essencial: `anomaly` e `risk_score` são rótulos/scores sintéticos, não chargebacks confirmados. Em produção, a target ideal deve vir de fraude confirmada após uma janela de maturação. Resultados altos nesta base não equivalem automaticamente a performance real."""))

cells.append(md("""# <font color='orange'>Library e configuração</font>

As funções continuam concentradas no início. Foi removida a dependência de GPU para tornar a execução reproduzível em CPU e todas as sementes foram fixadas."""))

cells.append(code("""from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
                             recall_score, f1_score, fbeta_score, log_loss,
                             brier_score_loss, confusion_matrix, precision_recall_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import shap

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font_scale=1.05)
pd.set_option('display.max_columns', 100)
Path('models').mkdir(exist_ok=True)
"""))

cells.append(code("""def plota_barras(variavel, df, titulo, rotation=0, top_n=None):
    counts = df[variavel].value_counts(dropna=False).head(top_n)
    ax = counts.plot.bar(figsize=(9, 4), color='#1FB3E5', title=titulo)
    ax.set_ylabel('Quantidade')
    ax.tick_params(axis='x', rotation=rotation)
    total = counts.sum()
    for p, v in zip(ax.patches, counts.values):
        ax.annotate(f'{v / total:.1%}', (p.get_x() + p.get_width()/2, p.get_height()),
                    ha='center', va='bottom')
    plt.tight_layout(); plt.show()


def analisa_distribuicao_via_percentis(df, variaveis):
    return df[variaveis].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T


def separa_feature_target(target, vars_drop, dados):
    return dados.drop(columns=[target] + vars_drop), dados[target].astype(int)
"""))

cells.append(code("""def expected_calibration_error(y_true, proba, n_bins=10):
    aux = pd.DataFrame({'y': np.asarray(y_true), 'p': np.asarray(proba)})
    aux['bin'] = pd.cut(aux['p'], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    tab = aux.groupby('bin', observed=False).agg(n=('y','size'), taxa_real=('y','mean'), prob_media=('p','mean')).dropna()
    ece = ((tab['n'] / len(aux)) * (tab['taxa_real'] - tab['prob_media']).abs()).sum()
    return float(ece)


def metricas_classificacao(modelo, y_true, proba, cutoff=0.5, etapa='Validação'):
    pred = (np.asarray(proba) >= cutoff).astype(int)
    prev = np.mean(y_true)
    return pd.DataFrame([{
        'Modelo': modelo, 'Etapa': etapa, 'Cutoff': cutoff,
        'Prevalencia': prev, 'PR-AUC_AP': average_precision_score(y_true, proba),
        'Lift_AP': average_precision_score(y_true, proba) / prev,
        'ROC_AUC': roc_auc_score(y_true, proba),
        'Precisao': precision_score(y_true, pred, zero_division=0),
        'Recall': recall_score(y_true, pred, zero_division=0),
        'F0.5': fbeta_score(y_true, pred, beta=.5, zero_division=0),
        'F1': f1_score(y_true, pred, zero_division=0),
        'Alert_rate': pred.mean(), 'Brier': brier_score_loss(y_true, proba),
        'LogLoss': log_loss(y_true, np.clip(proba, 1e-6, 1-1e-6)),
        'ECE': expected_calibration_error(y_true, proba)
    }])


def tabela_top_k(y_true, proba, percentuais=(.01, .02, .05, .10, .15, .20)):
    aux = pd.DataFrame({'y': np.asarray(y_true), 'p': np.asarray(proba)}).sort_values('p', ascending=False)
    rows = []
    for pct in percentuais:
        n = max(1, int(np.ceil(len(aux) * pct)))
        top = aux.head(n)
        rows.append({'top_pct': pct, 'n_alertas': n, 'precisao_top_k': top.y.mean(),
                     'recall_top_k': top.y.sum()/max(aux.y.sum(), 1),
                     'lift': top.y.mean()/max(aux.y.mean(), 1e-12), 'cutoff': top.p.min()})
    return pd.DataFrame(rows)
"""))

cells.append(code("""def seleciona_cutoff_politica(y_true, proba, capacidade_max=.10, precisao_min=.35,
                                custo_revisao=2.0, custo_fraude=100.0, custo_fp=5.0):
    # Otimização somente na validação. A restrição evita a política 'marcar quase tudo'.
    candidatos = np.unique(np.quantile(proba, np.linspace(0, 1, 501)))
    rows = []
    for cutoff in candidatos:
        pred = (proba >= cutoff).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
        alert_rate = pred.mean()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        custo = fn*custo_fraude + fp*custo_fp + (tp+fp)*custo_revisao
        rows.append((cutoff, alert_rate, precision, recall, custo, tn, fp, fn, tp))
    curva = pd.DataFrame(rows, columns=['cutoff','alert_rate','precision','recall','custo','tn','fp','fn','tp'])
    elegiveis = curva[(curva.alert_rate <= capacidade_max) & (curva.precision >= precisao_min)]
    if elegiveis.empty:
        elegiveis = curva[curva.alert_rate <= capacidade_max]
    melhor = elegiveis.sort_values(['custo','precision'], ascending=[True,False]).iloc[0]
    return float(melhor.cutoff), curva


def aplica_politica(proba, cutoff_revisao, cutoff_bloqueio=None):
    # Bloqueio automático é opcional e deve exigir precisão muito alta.
    cutoff_bloqueio = 1.01 if cutoff_bloqueio is None else cutoff_bloqueio
    return np.select([proba >= cutoff_bloqueio, proba >= cutoff_revisao],
                     ['bloquear', 'revisar'], default='aprovar')
"""))

cells.append(code("""def cria_features(df):
    # Uma única ordenação global garante que todo shift/cumcount use apenas o passado.
    d = df.sort_values(['data_hora_transacao', 'transacao']).copy()
    d['hora_sin'] = np.sin(2*np.pi*d['hora']/24)
    d['hora_cos'] = np.cos(2*np.pi*d['hora']/24)
    d['dia_semana'] = d.data_hora_transacao.dt.dayofweek
    d['fim_de_semana'] = (d.dia_semana >= 5).astype(int)
    d['log_valor_transacao'] = np.log1p(d.valor_transacao)

    for chave, sufixo in [('id_enviador','env'), ('id_recebedor','rec')]:
        g = d.groupby(chave, sort=False)
        d[f'qtd_transacoes_historico_{sufixo}'] = g.cumcount()
        d[f'segundos_desde_ultima_{sufixo}'] = g.data_hora_transacao.diff().dt.total_seconds().fillna(-1)
        d[f'media_valor_historico_{sufixo}'] = g.valor_transacao.transform(lambda s: s.expanding().mean().shift())
        d[f'std_valor_historico_{sufixo}'] = g.valor_transacao.transform(lambda s: s.expanding().std().shift()).fillna(0)
        d[f'media_sessao_historico_{sufixo}'] = g.duracao_sessao_atividade.transform(lambda s: s.expanding().mean().shift())
        ip_anterior = g.prefixo_ip.shift()
        reg_anterior = g.regiao_geografica.shift()
        d[f'mudanca_ip_{sufixo}'] = ((d.prefixo_ip != ip_anterior) & ip_anterior.notna()).astype(int)
        d[f'mudanca_regiao_{sufixo}'] = ((d.regiao_geografica != reg_anterior) & reg_anterior.notna()).astype(int)

    # Features de grafo simples, causais e baratas: graus e recorrência da aresta.
    d['grau_saida_enviador'] = d.groupby('id_enviador').cumcount()
    d['grau_entrada_recebedor'] = d.groupby('id_recebedor').cumcount()
    d['qtd_transacoes_par_historico'] = d.groupby(['id_enviador','id_recebedor']).cumcount()
    d['novo_par'] = (d.qtd_transacoes_par_historico == 0).astype(int)
    d['razao_graus'] = (d.grau_saida_enviador + 1) / (d.grau_entrada_recebedor + 1)

    historicas = [c for c in d if 'historico_' in c or 'media_' in c]
    d[historicas] = d[historicas].replace([np.inf,-np.inf], np.nan).fillna(0)
    return d.sort_values('transacao')
"""))

cells.append(code("""def aplica_feature_selection(target, vars_drop, df, cobertura_importancia=.99, correlacao_max=.95):
    x, y = separa_feature_target(target, vars_drop, df)
    rf = RandomForestClassifier(n_estimators=250, min_samples_leaf=20,
                                class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(x, y)
    imp = pd.DataFrame({'feature': x.columns, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False)
    imp['importance_acumulada'] = imp.importance.cumsum()
    candidatas = imp.loc[imp.importance_acumulada.shift(fill_value=0) < cobertura_importancia, 'feature'].tolist()
    corr = x[candidatas].corr(method='spearman').abs()
    remover = set()
    for i, col in enumerate(corr.columns):
        for anterior in corr.columns[:i]:
            if corr.loc[col, anterior] > correlacao_max:
                remover.add(col); break
    finais = [c for c in candidatas if c not in remover]
    return finais, imp
"""))

cells.append(code("""def Classificador(classificador, x_train, y_train, x_valid, y_valid):
    if classificador == 'Regressão Logística':
        modelo = Pipeline([('imputer', SimpleImputer(strategy='median')),
                           ('scaler', StandardScaler()),
                           ('modelo', LogisticRegression(max_iter=1000, C=1.0,
                                                        class_weight=None, random_state=RANDOM_STATE))])
    elif classificador == 'LightGBM':
        modelo = LGBMClassifier(n_estimators=300, learning_rate=.03, num_leaves=15,
                                max_depth=5, min_child_samples=50, subsample=.8,
                                colsample_bytree=.8, reg_alpha=1, reg_lambda=3,
                                random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    else:
        raise ValueError('Classificador não reconhecido')
    modelo.fit(x_train, y_train)
    return modelo, modelo.predict_proba(x_train)[:,1], modelo.predict_proba(x_valid)[:,1]
"""))

cells.append(code("""def otimizacao_hyperopt(x_train, y_train, x_valid, y_valid, max_evals=40):
    # AP/PR-AUC é threshold-free e adequada ao ranking desbalanceado.
    espaco = {
        'n_estimators': hp.quniform('n_estimators', 150, 600, 25),
        'learning_rate': hp.loguniform('learning_rate', np.log(.01), np.log(.08)),
        'num_leaves': hp.quniform('num_leaves', 7, 45, 2),
        'max_depth': hp.quniform('max_depth', 3, 8, 1),
        'min_child_samples': hp.quniform('min_child_samples', 20, 150, 10),
        'subsample': hp.uniform('subsample', .65, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', .6, 1),
        'reg_alpha': hp.loguniform('reg_alpha', np.log(.01), np.log(20)),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(.1), np.log(30))}

    def normaliza(p):
        p = p.copy()
        for c in ['n_estimators','num_leaves','max_depth','min_child_samples']:
            p[c] = int(p[c])
        return p

    def objective(p):
        p = normaliza(p)
        model = LGBMClassifier(**p, objective='binary', random_state=RANDOM_STATE,
                               n_jobs=-1, verbosity=-1)
        model.fit(x_train, y_train)
        pv = model.predict_proba(x_valid)[:,1]
        pt = model.predict_proba(x_train)[:,1]
        ap_valid = average_precision_score(y_valid, pv)
        gap = max(0, average_precision_score(y_train, pt) - ap_valid - .05)
        return {'loss': -ap_valid + .20*gap, 'status': STATUS_OK, 'ap_valid': ap_valid}

    trials = Trials()
    best = fmin(objective, espaco, algo=tpe.suggest, max_evals=max_evals,
                trials=trials, rstate=np.random.default_rng(RANDOM_STATE), show_progressbar=True)
    best = normaliza(best)
    model = LGBMClassifier(**best, objective='binary', random_state=RANDOM_STATE,
                           n_jobs=-1, verbosity=-1)
    model.fit(x_train, y_train)
    return model, best, trials
"""))

cells.append(code("""def calibracao_probabilidade(modelo, x_cal, y_cal, metodo='isotonic'):
    # O modelo permanece congelado: outubro ajusta somente o calibrador.
    calibrado = CalibratedClassifierCV(
        estimator=FrozenEstimator(modelo), method=metodo, cv=None
    )
    calibrado.fit(x_cal, y_cal)
    return calibrado


def plot_shap(modelo, x_amostra, titulo='SHAP global'):
    explainer = shap.TreeExplainer(modelo)
    valores = explainer.shap_values(x_amostra)
    if isinstance(valores, list): valores = valores[-1]
    shap.summary_plot(valores, x_amostra, show=False)
    plt.title(titulo); plt.tight_layout(); plt.show()
    return explainer
"""))

cells.append(md("""# <font color='orange'>1. Leitura do Dataset</font>

Os campos são renomeados como no projeto original. `risk_score`, `anomaly` e `transaction_type` ficam disponíveis para auditoria, mas não entram como features: são resultados ou descrições posteriores que podem revelar o rótulo."""))

cells.append(code("""df_metaverse = pd.read_csv('./data/metaverse_transactions_dataset.csv')
df_metaverse = df_metaverse.rename(columns={
    'timestamp':'data_hora_transacao', 'hour_of_day':'hora',
    'sending_address':'id_enviador', 'receiving_address':'id_recebedor',
    'amount':'valor_transacao', 'transaction_type':'tipo_transacao',
    'location_region':'regiao_geografica', 'ip_prefix':'prefixo_ip',
    'login_frequency':'frequencia_login', 'session_duration':'duracao_sessao_atividade',
    'purchase_pattern':'padrao_comportamental_de_compras', 'age_group':'recencia_atividade',
    'anomaly':'risk_rank'})
df_metaverse['data_hora_transacao'] = pd.to_datetime(df_metaverse.data_hora_transacao, utc=True)
df_metaverse = df_metaverse.sort_values('data_hora_transacao').reset_index(drop=True)
df_metaverse['transacao'] = np.arange(len(df_metaverse))
df_metaverse['ano'] = df_metaverse.data_hora_transacao.dt.year
df_metaverse['safra'] = df_metaverse.data_hora_transacao.dt.strftime('%m')
df_metaverse['dia'] = df_metaverse.data_hora_transacao.dt.day
df_metaverse.shape, df_metaverse.head(3)
"""))

cells.append(md("""# <font color='orange'>2. Análise da Target</font>

## 2.1 Decisão de target

A target anterior (`moderate_risk` + `high_risk`) respondia “há algum risco?”, não “é fraude?”. Isso elevava a prevalência a ~19% e confundia fila de investigação com fraude confirmada.

Nesta revisão:

- `high_risk = 1` é a **target principal**, por ser o proxy mais conservador disponível;
- `moderate_risk` é classe 0 na modelagem, mas pode receber tratamento de política se sua probabilidade ficar alta;
- `risk_score` não entra no modelo porque é o mecanismo que originou o rótulo;
- em uma operação real, substituiríamos tudo por `fraude_confirmada_apos_maturacao`.

Isso não “cria verdade” onde ela não existe; apenas alinha melhor a pergunta analítica ao objetivo."""))

cells.append(code("""df_metaverse['fraude'] = (df_metaverse.risk_rank == 'high_risk').astype(int)
target = 'fraude'
display(df_metaverse.risk_rank.value_counts().to_frame('n').assign(pct=lambda x: x.n/x.n.sum()))
display(pd.crosstab(df_metaverse.risk_rank, pd.cut(df_metaverse.risk_score, bins=10), normalize='index'))
plota_barras('fraude', df_metaverse, 'Target principal: high_risk', rotation=0)
"""))

cells.append(md("""## 2.2 Separação temporal e governança

- Treino: janeiro a setembro.
- Validação/calibração/política: outubro.
- Teste: novembro, usado uma única vez para comparação final.
- OOT: dezembro, aproxima uma safra futura.

Nenhum hiperparâmetro, calibrador ou cutoff deve ser escolhido olhando teste/OOT. Features históricas são calculadas em ordem temporal e usam `shift`, portanto uma linha só conhece eventos anteriores."""))

cells.append(code("""df_train_raw = df_metaverse.loc[~df_metaverse.safra.isin(['10','11','12'])].copy()
df_valid_raw = df_metaverse.loc[df_metaverse.safra.eq('10')].copy()
df_test_raw  = df_metaverse.loc[df_metaverse.safra.eq('11')].copy()
df_oot_raw   = df_metaverse.loc[df_metaverse.safra.eq('12')].copy()

resumo_split = pd.DataFrame({
    'Treino': [len(df_train_raw), df_train_raw[target].mean()],
    'Validacao': [len(df_valid_raw), df_valid_raw[target].mean()],
    'Teste': [len(df_test_raw), df_test_raw[target].mean()],
    'OOT': [len(df_oot_raw), df_oot_raw[target].mean()]}, index=['n','prevalencia']).T
display(resumo_split)
"""))

cells.append(md("""# <font color='orange'>3. Análise Exploratória</font>

A EDA descritiva usa a base inteira apenas para entender o dataset; toda decisão que aprende parâmetros (seleção, modelo, calibração e cutoff) respeita os conjuntos temporais."""))

cells.append(md("""## 3.1 Tipos de dados e nulos"""))
cells.append(code("""display(df_metaverse.dtypes.to_frame('dtype'))
display(df_metaverse.isna().mean().sort_values(ascending=False).to_frame('pct_nulos').head(15))
"""))

cells.append(md("""## 3.2 Variáveis numéricas

As comparações abaixo são associações, não causalidade. Em particular, frequência de login e duração de sessão parecem fazer parte do processo sintético que gerou `risk_score`; por isso, o notebook mantém um alerta de “proxy leakage”."""))

cells.append(code("""num_cols = ['valor_transacao','prefixo_ip','frequencia_login','duracao_sessao_atividade']
display(analisa_distribuicao_via_percentis(df_train_raw, num_cols))
display(df_train_raw.groupby(target)[num_cols].median().T.rename(columns={0:'legitima',1:'fraude'}))
"""))

cells.append(md("""## 3.3 Variáveis categóricas e leakage

`scam` e `phishing` descrevem o evento depois que ele foi reconhecido; `tipo_transacao` é excluída. `risk_score`, `risk_rank` e categorias derivadas do mesmo mecanismo também não serão features."""))

cells.append(code("""for col in ['tipo_transacao','regiao_geografica','padrao_comportamental_de_compras','recencia_atividade']:
    display(pd.crosstab(df_train_raw[col], df_train_raw[target], normalize='index').sort_values(1, ascending=False))
"""))

cells.append(md("""## 3.4 Variáveis temporais"""))

cells.append(code("""taxa_hora = df_train_raw.groupby('hora')[target].agg(['count','mean'])
taxa_hora['mean'].plot(figsize=(10,4), marker='o', title='Taxa de fraude por hora — treino')
plt.ylabel('taxa'); plt.show()
"""))

cells.append(md("""# <font color='orange'>4. Feature Engineering</font>

Mantemos RFM/comportamento e adicionamos um bloco leve de grafo:

- grau de saída histórico do enviador;
- grau de entrada histórico do recebedor;
- recorrência histórica da aresta enviador→recebedor;
- flag de par novo e razão entre graus.

Não há construção de grafo futuro nem centralidades calculadas com dezembro para explicar janeiro."""))

cells.append(code("""df_model_final = cria_features(df_metaverse)
df_train = df_model_final.loc[~df_model_final.safra.isin(['10','11','12'])].copy()
df_valid = df_model_final.loc[df_model_final.safra.eq('10')].copy()
df_test  = df_model_final.loc[df_model_final.safra.eq('11')].copy()
df_oot   = df_model_final.loc[df_model_final.safra.eq('12')].copy()
df_model_final.filter(regex='grau|par_historico|novo_par').head()
"""))

cells.append(code("""# Auditoria causal: contagens históricas não podem ser negativas e começam em zero.
assert (df_model_final[['grau_saida_enviador','grau_entrada_recebedor','qtd_transacoes_par_historico']] >= 0).all().all()
assert df_model_final.groupby('id_enviador').head(1).grau_saida_enviador.eq(0).all()
assert df_model_final.groupby('id_recebedor').head(1).grau_entrada_recebedor.eq(0).all()
print('Auditoria causal básica aprovada.')
"""))

cells.append(md("""# <font color='orange'>5. Modelagem</font>"""))

cells.append(md("""## 5.1 Métricas

**Primária — PR-AUC/AP:** resume precisão e recall ao longo do ranking e deve ser comparada à prevalência. Reportamos também `Lift_AP = AP / prevalência`.

**Operacionais:** precisão, recall, F0.5 (prioriza precisão), alert rate e desempenho no top-k. Recall alto não é ruim isoladamente; ele vira problema quando exige alertar uma fração inviável da base.

**Probabilísticas:** Brier, LogLoss e ECE avaliam calibração. ROC-AUC permanece como diagnóstico, não como critério principal.

> Não existe threshold universal. O cutoff final pertence à política e é escolhido na validação sob restrições declaradas."""))

cells.append(code("""VARS_DROP = ['ano','safra','data_hora_transacao','transacao','risk_score','risk_rank',
             'tipo_transacao','id_enviador','id_recebedor','regiao_geografica','prefixo_ip',
             'padrao_comportamental_de_compras','recencia_atividade']
features_candidatas = [c for c in df_train.columns if c not in VARS_DROP + [target]]
print(f'{len(features_candidatas)} features candidatas')
"""))

cells.append(md("""## 5.2 Feature selection e pré-processamento

A seleção aprende somente no treino. O bug conceitual anterior — usar `0.9` como importância individual — foi substituído por cobertura acumulada de 99%, seguida de remoção de correlação de Spearman acima de 0,95. A Regressão Logística recebe imputação e padronização dentro do pipeline; LightGBM usa os valores originais."""))

cells.append(code("""features, feature_importances = aplica_feature_selection(target, VARS_DROP, df_train,
                                                         cobertura_importancia=.99,
                                                         correlacao_max=.95)
display(feature_importances.head(30))
print(f'{len(features)} features finais:', features)
"""))

cells.append(code("""x_train, y_train = df_train[features], df_train[target]
x_valid, y_valid = df_valid[features], df_valid[target]
x_test,  y_test  = df_test[features],  df_test[target]
x_oot,   y_oot   = df_oot[features],   df_oot[target]
"""))

cells.append(md("""## 5.3 Benchmarks — Regressão Logística e LightGBM

Os benchmarks não usam `class_weight` artificial. Desbalanceamento é tratado no ranking e na política; inflar pesos alteraria a escala de probabilidade e repetiria o comportamento agressivo anterior."""))

cells.append(code("""reg_logistic, p_train_lr, p_valid_lr = Classificador('Regressão Logística', x_train, y_train, x_valid, y_valid)
lightgbm, p_train_lgb, p_valid_lgb = Classificador('LightGBM', x_train, y_train, x_valid, y_valid)

metricas_benchmark = pd.concat([
    metricas_classificacao('Regressão Logística', y_train, p_train_lr, etapa='Treino'),
    metricas_classificacao('Regressão Logística', y_valid, p_valid_lr, etapa='Validação'),
    metricas_classificacao('LightGBM', y_train, p_train_lgb, etapa='Treino'),
    metricas_classificacao('LightGBM', y_valid, p_valid_lgb, etapa='Validação')])
display(metricas_benchmark)
"""))

cells.append(code("""display(tabela_top_k(y_valid, p_valid_lgb))
"""))

cells.append(md("""## 5.4 Hyperopt e explicabilidade SHAP

O Hyperopt maximiza AP na validação e aplica uma penalidade pequena quando o gap treino–validação excede 5 p.p. A busca não escolhe class weight nem cutoff: capacidade operacional não deve ser escondida em hiperparâmetros do modelo."""))

cells.append(code("""model_otimizado, best_hiperpams, trials = otimizacao_hyperopt(
    x_train, y_train, x_valid, y_valid, max_evals=40)
p_train_opt = model_otimizado.predict_proba(x_train)[:,1]
p_valid_opt = model_otimizado.predict_proba(x_valid)[:,1]
display(pd.DataFrame([best_hiperpams]))
display(pd.concat([
    metricas_classificacao('Hyperopt + LightGBM', y_train, p_train_opt, etapa='Treino'),
    metricas_classificacao('Hyperopt + LightGBM', y_valid, p_valid_opt, etapa='Validação')]))
"""))

cells.append(code("""amostra_shap = x_valid.sample(min(1000, len(x_valid)), random_state=RANDOM_STATE)
explainer = plot_shap(model_otimizado, amostra_shap, 'SHAP global — validação')
"""))

cells.append(code("""# Comparação SHAP entre uma fraude e uma transação legítima de maior score.
idx_fraude = p_valid_opt[y_valid.to_numpy() == 1].argmax()
idx_legitima = p_valid_opt[y_valid.to_numpy() == 0].argmax()
amostras_locais = pd.concat([x_valid[y_valid.eq(1)].iloc[[idx_fraude]],
                             x_valid[y_valid.eq(0)].iloc[[idx_legitima]]])
sv = explainer(amostras_locais)
shap.plots.waterfall(sv[0], show=False); plt.title('Fraude — SHAP local'); plt.show()
shap.plots.waterfall(sv[1], show=False); plt.title('Legítima de alto risco — SHAP local'); plt.show()
"""))

cells.append(md("""## 5.5 Calibração + política de decisão

### Desenho sem vazamento

O modelo foi treinado até setembro. Outubro calibra e define a política. Novembro e dezembro permanecem intocados. Em uma implantação real, usaríamos validação cruzada temporal ou uma janela específica para calibração.

### Política proposta

- **aprovar:** abaixo do cutoff;
- **revisar:** acima do cutoff, limitado inicialmente a 10% das transações;
- **bloquear:** desabilitado por padrão nesta base, pois não há fraude confirmada nem custo validado para justificar automação irreversível.

O cutoff minimiza um custo ilustrativo na validação, sujeito a `alert_rate ≤ 10%` e `precision ≥ 20%`. Esses números são parâmetros de negócio e devem ser substituídos por capacidade e custos reais. O score contínuo ordena a fila; a probabilidade isotônica é usada para estimar risco, pois seus empates podem distorcer um corte operacional."""))

cells.append(code("""modelo_calibrado = calibracao_probabilidade(model_otimizado, x_valid, y_valid, metodo='isotonic')
p_valid_cal = modelo_calibrado.predict_proba(x_valid)[:,1]

# O score contínuo ordena a fila; a probabilidade calibrada estima risco.
# Isso evita que empates da regressão isotônica distorçam a capacidade.
cutoff_revisao, curva_politica = seleciona_cutoff_politica(
    y_valid.to_numpy(), p_valid_opt, capacidade_max=.10, precisao_min=.20,
    custo_revisao=2, custo_fraude=100, custo_fp=5)
print('Cutoff de revisão:', round(cutoff_revisao, 4))
display(curva_politica.loc[curva_politica.cutoff.sub(cutoff_revisao).abs().nsmallest(1).index])
display(metricas_classificacao('LGBM score + política', y_valid, p_valid_opt,
                               cutoff=cutoff_revisao, etapa='Validação'))
"""))

cells.append(code("""fig, ax = plt.subplots(1, 2, figsize=(13,4))
CalibrationDisplay.from_predictions(y_valid, p_valid_opt, n_bins=10, name='Antes', ax=ax[0])
CalibrationDisplay.from_predictions(y_valid, p_valid_cal, n_bins=10, name='Isotônica', ax=ax[0])
ax[0].set_title('Calibração — validação')
curva_plot = curva_politica.sort_values('alert_rate')
ax[1].plot(curva_plot.alert_rate, curva_plot.precision, label='Precisão')
ax[1].plot(curva_plot.alert_rate, curva_plot.recall, label='Recall')
ax[1].axvline(.10, color='red', ls='--', label='Capacidade')
ax[1].set(xlabel='Alert rate', ylabel='Métrica', title='Trade-off operacional'); ax[1].legend()
plt.tight_layout(); plt.show()
"""))

cells.append(md("""## 5.6 Teste e OOT finais

Esta é a primeira aplicação da política congelada em novembro e dezembro. O notebook reporta métricas de ranking, calibração e operação em conjunto; um único número nunca é chamado de “modelo bom” sem seu contexto."""))

cells.append(code("""p_test_raw = model_otimizado.predict_proba(x_test)[:,1]
p_oot_raw = model_otimizado.predict_proba(x_oot)[:,1]
p_test_cal = modelo_calibrado.predict_proba(x_test)[:,1]
p_oot_cal = modelo_calibrado.predict_proba(x_oot)[:,1]

metricas_finais = pd.concat([
    metricas_classificacao('Regressão Logística', y_test, reg_logistic.predict_proba(x_test)[:,1], etapa='Teste'),
    metricas_classificacao('LightGBM benchmark', y_test, lightgbm.predict_proba(x_test)[:,1], etapa='Teste'),
    metricas_classificacao('LGBM score + política', y_test, p_test_raw, cutoff_revisao, 'Teste'),
    metricas_classificacao('LGBM score + política', y_oot, p_oot_raw, cutoff_revisao, 'OOT')])
display(metricas_finais)
"""))

cells.append(code("""display(pd.concat({
    'Validação': tabela_top_k(y_valid, p_valid_cal),
    'Teste': tabela_top_k(y_test, p_test_cal),
    'OOT': tabela_top_k(y_oot, p_oot_cal)}))
"""))

cells.append(code("""df_analise_final = pd.concat([
    df_valid.assign(score_fraude=p_valid_opt, prob_fraude=p_valid_cal, etapa='Validacao'),
    df_test.assign(score_fraude=p_test_raw, prob_fraude=p_test_cal, etapa='Teste'),
    df_oot.assign(score_fraude=p_oot_raw, prob_fraude=p_oot_cal, etapa='OOT')])
df_analise_final['acao'] = aplica_politica(df_analise_final.score_fraude.to_numpy(), cutoff_revisao)
display(pd.crosstab(df_analise_final.etapa, df_analise_final.acao, normalize='index'))
"""))

cells.append(md("""## 5.7 Sensibilidade financeira

Sem custos reais, “retorno de R$ 5,5 milhões” seria falsa precisão. A célula abaixo varia custos de fraude perdida e falso positivo. A decisão só é robusta se continuar razoável em cenários plausíveis."""))

cells.append(code("""def custo_politica(y, proba, cutoff, custo_fn, custo_fp, custo_revisao=2):
    pred = proba >= cutoff
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return fn*custo_fn + fp*custo_fp + pred.sum()*custo_revisao

sensibilidade = []
for custo_fn in [25, 50, 100, 250]:
    for custo_fp in [1, 5, 10, 25]:
        sensibilidade.append({'custo_fn':custo_fn, 'custo_fp':custo_fp,
            'custo_teste':custo_politica(y_test, p_test_raw, cutoff_revisao, custo_fn, custo_fp)})
display(pd.DataFrame(sensibilidade).pivot(index='custo_fn', columns='custo_fp', values='custo_teste'))
"""))

cells.append(md("""# <font color='orange'>6. Conclusões e governança</font>

O resultado deve ser considerado satisfatório somente se:

- AP superar claramente a prevalência em teste e OOT;
- lift/precisão no top-k for útil dentro da capacidade disponível;
- alert rate não explodir fora da validação;
- calibração permanecer aceitável;
- SHAP não revelar dependência de variável indisponível no momento da decisão;
- a conclusão sobreviver à análise de sensibilidade.

### Próximos passos para produção

1. Trocar o proxy `high_risk` por chargeback/fraude confirmada com janela de maturação e prevenção de label delay.
2. Validar quais campos existem **antes** da autorização da transação.
3. Medir custo real de revisão, perda, atrito e capacidade por turno.
4. Monitorar prevalência, AP, precision@k, recall@k, PSI/drift, ECE e custo por safra.
5. Usar shadow mode/A-B test antes de bloquear automaticamente.

Assim, o modelo produz um ranking; a política decide a ação. Essa separação impede que recall alto seja confundido com uma operação saudável."""))

cells.append(code("""# Persistência: modelo, features, target e política via um único bundle versionável.
artefato = {
    'modelo': modelo_calibrado,
    'features': features,
    'target': target,
    'cutoff_score_revisao': cutoff_revisao,
    'cutoff_bloqueio': None,
    'target_definition': "risk_rank == 'high_risk'",
    'periodo_treino': '2022-01 a 2022-09',
    'periodo_calibracao_politica': '2022-10'}
joblib.dump(artefato, './models/modelo_fraude_robusto.pkl')
print('Artefato salvo em models/modelo_fraude_robusto.pkl')
"""))

cells.append(md("""## Checklist de execução

- [ ] Todas as células executam do zero.
- [ ] Teste e OOT não foram usados em tuning/calibração/cutoff.
- [ ] PR-AUC foi comparada à prevalência.
- [ ] Alert rate e precision@k cabem na operação.
- [ ] Custos foram substituídos por valores reais antes de produção.
- [ ] Features foram validadas quanto à disponibilidade online.
- [ ] Bloqueio automático permanece desativado até validação prospectiva."""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}
    },
    "nbformat": 4, "nbformat_minor": 5
}

path = Path('Modelo_Deteccao_Fraude.ipynb')
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(f'Notebook reconstruído: {len(cells)} células')
