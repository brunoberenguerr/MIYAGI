# -*- coding: utf-8 -*-
"""Inferência adversarial para a série final, sem escolher o melhor resultado.

Reporta em conjunto o teste ingênuo, erros Newey--West e bootstrap em blocos.
Os comprimentos de bloco (21, 63 e 252 pregões) representam aproximadamente
um mês, um trimestre e um ano; todos são mostrados, nenhum é selecionado pelo
intervalo que favorece a estratégia.
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from backtest_miyagi import calcular_metricas, calcular_retornos, rodar_backtest
from dados_miyagi import AQUI, carregar_dados_oficiais

SAIDA = AQUI / "resultados"
SEMENTE = 20260813
REPLICACOES = 2_000
TESTES_DECLARADOS_MINIMOS = 4


def erro_hac_media(x: np.ndarray, defasagem: int) -> float:
    """Erro-padrão Newey--West da média, com pesos de Bartlett."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    centrado = x - x.mean()
    lrv = float(centrado @ centrado / n)
    for lag in range(1, min(defasagem, n - 1) + 1):
        gamma = float(centrado[lag:] @ centrado[:-lag] / n)
        lrv += 2.0 * (1.0 - lag / (defasagem + 1.0)) * gamma
    return sqrt(max(lrv, 0.0) / n)


def sharpe_anual(x: np.ndarray, periodos_ano: int = 252) -> float:
    x = np.asarray(x, dtype=float)
    desvio = x.std(ddof=1)
    return float(x.mean() / desvio * sqrt(periodos_ano)) if desvio > 0 else np.nan


def bootstrap_blocos(
    x: np.ndarray, bloco: int, replicacoes: int = REPLICACOES
) -> tuple[float, float, float]:
    """IC percentil do Sharpe por moving-block bootstrap circular."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    rng = np.random.default_rng(SEMENTE + bloco)
    n_blocos = int(np.ceil(n / bloco))
    amostras = np.empty(replicacoes)
    offsets = np.arange(bloco)
    for i in range(replicacoes):
        inicios = rng.integers(0, n, size=n_blocos)
        idx = ((inicios[:, None] + offsets) % n).ravel()[:n]
        amostras[i] = sharpe_anual(x[idx])
    q = np.quantile(amostras, [0.025, 0.5, 0.975])
    return float(q[0]), float(q[1]), float(q[2])


def main() -> None:
    SAIDA.mkdir(exist_ok=True)
    precos, cdi, _ = carregar_dados_oficiais()
    res = rodar_backtest(precos, calcular_retornos(precos), cdi)
    retornos = res["retornos"].dropna()
    excesso = retornos - cdi.reindex(retornos.index).fillna(0.0)
    x = excesso.to_numpy()
    n = len(x)

    media = float(x.mean())
    se_iid = float(x.std(ddof=1) / sqrt(n))
    t_iid = media / se_iid
    p_bilateral = float(2 * stats.t.sf(abs(t_iid), df=n - 1))
    p_unilateral = float(stats.t.sf(t_iid, df=n - 1))

    linhas = [{
        "frequencia": "diaria",
        "estimador": "iid",
        "defasagem": 0,
        "t": t_iid,
        "p_bilateral": p_bilateral,
    }]
    for lag in (5, 21, 63, 252):
        se = erro_hac_media(x, lag)
        t = media / se
        linhas.append({
            "frequencia": "diaria", "estimador": "HAC-Newey-West",
            "defasagem": lag, "t": t,
            "p_bilateral": float(2 * stats.norm.sf(abs(t))),
        })

    mensal_total = (1.0 + retornos).resample("ME").prod() - 1.0
    mensal_cdi = (1.0 + cdi.reindex(retornos.index).fillna(0.0)).resample("ME").prod() - 1.0
    mensal = (mensal_total - mensal_cdi).dropna().to_numpy()
    for lag in (0, 12, 24):
        se = (mensal.std(ddof=1) / sqrt(len(mensal))
              if lag == 0 else erro_hac_media(mensal, lag))
        t = float(mensal.mean() / se)
        linhas.append({
            "frequencia": "mensal",
            "estimador": "iid" if lag == 0 else "HAC-Newey-West",
            "defasagem": lag,
            "t": t,
            "p_bilateral": float(2 * stats.norm.sf(abs(t))),
        })

    testes = pd.DataFrame(linhas)
    testes.to_csv(SAIDA / "auditoria_testes_inferencia.csv", index=False)

    boots = []
    for bloco in (21, 63, 252):
        baixo, mediana, alto = bootstrap_blocos(x, bloco)
        boots.append({"bloco_dias": bloco, "ic_2_5": baixo,
                      "mediana": mediana, "ic_97_5": alto})
    boots = pd.DataFrame(boots)
    boots.to_csv(SAIDA / "auditoria_bootstrap_sharpe.csv", index=False)

    metricas = calcular_metricas(retornos, cdi)
    resumo = pd.Series({
        "observacoes_diarias": n,
        "sharpe_anual": metricas["sharpe"],
        "t_iid": t_iid,
        "p_iid_bilateral": p_bilateral,
        "p_iid_unilateral": p_unilateral,
        "p_bonferroni_4_bilateral": min(1.0, p_bilateral * TESTES_DECLARADOS_MINIMOS),
        "p_bonferroni_4_unilateral": min(1.0, p_unilateral * TESTES_DECLARADOS_MINIMOS),
        "testes_contados_bonferroni": TESTES_DECLARADOS_MINIMOS,
    }, name="valor")
    resumo.to_csv(SAIDA / "auditoria_estatistica_resumo.csv")

    print(resumo.to_string())
    print("\nTestes de média excedente:")
    print(testes.to_string(index=False, float_format=lambda z: f"{z:.4f}"))
    print("\nBootstrap em blocos — IC 95% do Sharpe anual:")
    print(boots.to_string(index=False, float_format=lambda z: f"{z:.3f}"))
    print("\nNota: quatro é apenas o mínimo documentado, não a contagem completa")
    print("do caminho de pesquisa. Por isso Bonferroni(4) é um limite otimista.")


if __name__ == "__main__":
    main()
