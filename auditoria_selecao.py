# -*- coding: utf-8 -*-
"""Teste de universo sem vazamento futuro, com as limitações declaradas.

O funil oficial exige 15 anos de histórico e o pool começa em 2000. Portanto
não existe seleção honesta com esse critério para iniciar o backtest em 2005.
Este script não reduz o requisito para forçar uma série longa. Ele forma o
primeiro universo no fim de 2015 e o utiliza apenas a partir de 2016; depois o
reestima anualmente usando somente dados existentes até cada corte.

É um teste *pseudo* point-in-time: os CSVs do Yahoo e do FRED não são vintages
arquivados na data original. Logo, ele remove o vazamento explícito do corte e
do clustering, mas não prova ausência de revisões ou sobrevivência do provedor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from backtest_miyagi import calcular_metricas, calcular_retornos, rodar_backtest
from dados_miyagi import (AQUI, alinhar_ao_calendario, carregar_cdi,
                          carregar_pool_oficial)
from funil_expandido import (CORTE_CLUSTER, correlacao_pareada, elegiveis,
                             medoide, retornos_mensais)

SAIDA = AQUI / "resultados"


def selecionar_ate(precos: pd.DataFrame, corte: pd.Timestamp) -> list[str]:
    """Executa o funil usando estritamente observações até ``corte``."""
    passado = precos.loc[:corte]
    aptos = elegiveis(passado)
    if len(aptos) < 2:
        return []
    corr = correlacao_pareada(retornos_mensais(passado[aptos]))
    D = 1.0 - corr.to_numpy()
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    labels = fcluster(
        linkage(squareform(D, checks=False), method="average"),
        t=CORTE_CLUSTER,
        criterion="distance",
    )
    clusters: dict[int, list[str]] = {}
    for ativo, label in zip(aptos, labels):
        clusters.setdefault(int(label), []).append(ativo)
    return sorted(medoide(membros, corr, passado) for membros in clusters.values())


def main() -> None:
    SAIDA.mkdir(exist_ok=True)
    pool = carregar_pool_oficial()
    cdi = carregar_cdi()

    # Seleção no último dia disponível de cada ano; começa quando o próprio
    # requisito declarado de 15 anos se torna possível.
    cortes = []
    for ano in range(2015, int(pool.index.max().year)):
        datas = pool.index[pool.index.year == ano]
        if len(datas):
            cortes.append(datas[-1])

    universos: dict[pd.Timestamp, list[str]] = {}
    registros = []
    anterior: set[str] | None = None
    for corte in cortes:
        selecionados = selecionar_ate(pool, corte)
        inicio_vigencia = pd.Timestamp(corte.year + 1, 1, 1)
        universos[inicio_vigencia] = selecionados
        atual = set(selecionados)
        jaccard = (len(atual & anterior) / len(atual | anterior)
                   if anterior is not None and (atual | anterior) else np.nan)
        registros.append({
            "data_corte": corte,
            "inicio_vigencia": inicio_vigencia,
            "n_ativos": len(selecionados),
            "jaccard_vs_ano_anterior": jaccard,
            "ativos": " ".join(selecionados),
        })
        anterior = atual

    tabela = pd.DataFrame(registros)
    tabela.to_csv(SAIDA / "auditoria_universo_point_in_time.csv", index=False)

    todos = sorted(set().union(*map(set, universos.values())))
    precos = alinhar_ao_calendario(pool[todos], cdi.index)
    retornos = calcular_retornos(precos)
    primeiro = min(universos)
    res = rodar_backtest(
        precos, retornos, cdi, universo_por_data=universos, inicio=primeiro
    )
    metricas = calcular_metricas(res["retornos"], cdi)
    resumo = pd.Series({
        "inicio": res["retornos"].index.min(),
        "fim": res["retornos"].index.max(),
        "cagr": metricas["cagr"],
        "vol": metricas["vol"],
        "sharpe": metricas["sharpe"],
        "max_drawdown": metricas["max_drawdown"],
        "rotulo_metodologico": "pseudo-point-in-time; dados sem vintage",
    }, name="valor")
    resumo.to_csv(SAIDA / "auditoria_selecao_resumo.csv")

    print(tabela[["data_corte", "inicio_vigencia", "n_ativos",
                  "jaccard_vs_ano_anterior"]].to_string(index=False))
    print("\nResultado 2016+ com seleção anual defasada:")
    print(resumo.to_string())
    print("\nEste teste não autoriza chamar 2005-2026 de fora da amostra.")


if __name__ == "__main__":
    main()
