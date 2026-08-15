# -*- coding: utf-8 -*-
"""Sensibilidade do carry cambial a uma defasagem de publicação.

Os arquivos públicos não contêm vintages. Portanto este teste não afirma qual
taxa estava efetivamente disponível em cada data. Ele infere o carry que já foi
gravado em ``pool_carrego.csv`` e o desloca por 21 pregões (aproximadamente um
mês), mantendo universo, parâmetros e demais retornos inalterados.

O cenário mede a ordem de grandeza do possível lag; não converte o backtest em
point-in-time e não substitui dados ALFRED ou registros históricos de release.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_miyagi import calcular_metricas, calcular_retornos, rodar_backtest
from dados_miyagi import (
    AQUI,
    alinhar_ao_calendario,
    carregar_cdi,
    carregar_pool_oficial,
    carregar_universo_oficial,
)

SAIDA = AQUI / "resultados"
DEFASAGEM_PREGOES = 21


def reconstruir_indice(retornos: pd.Series, mascara_observada: pd.Series) -> pd.Series:
    """Reconstrói índice sem transformar lacunas em cotações observadas."""
    primeira = retornos.first_valid_index()
    indice = pd.Series(np.nan, index=retornos.index, dtype=float)
    if primeira is None:
        return indice
    trecho = retornos.loc[primeira:]
    indice.loc[primeira:] = (1.0 + trecho.fillna(0.0)).cumprod()
    indice.loc[~mascara_observada] = np.nan
    return indice


def cagr_indice(indice: pd.Series) -> float:
    serie = indice.dropna()
    if len(serie) < 2:
        return np.nan
    anos = (serie.index[-1] - serie.index[0]).days / 365.25
    return float((serie.iloc[-1] / serie.iloc[0]) ** (1.0 / anos) - 1.0)


def main() -> None:
    SAIDA.mkdir(exist_ok=True)
    cdi = carregar_cdi()
    universo = carregar_universo_oficial()
    pool_atual = carregar_pool_oficial()
    pool_spot = pd.read_csv(
        AQUI / "dados" / "pool_expandido.csv", index_col=0, parse_dates=True
    ).sort_index()

    atual = alinhar_ao_calendario(pool_atual[universo], cdi.index)
    spot = alinhar_ao_calendario(pool_spot[universo], cdi.index)
    r_atual = calcular_retornos(atual)
    r_spot = calcular_retornos(spot)

    diferenca_nivel = (pool_atual[universo] - pool_spot[universo]).abs()
    pares = sorted(
        a for a in universo
        if a.endswith("=X") and diferenca_nivel[a].fillna(0.0).max() > 1e-12
    )

    painel_lag = atual.copy()
    linhas = []
    for par in pares:
        carry_modelado = r_atual[par] - r_spot[par]
        carry_defasado = carry_modelado.shift(DEFASAGEM_PREGOES)
        r_defasado = r_spot[par] + carry_defasado
        painel_lag[par] = reconstruir_indice(r_defasado, atual[par].notna())
        linhas.append({
            "par": par,
            "cagr_carry_sem_lag": cagr_indice(atual[par]),
            "cagr_carry_defasado_21_pregoes": cagr_indice(painel_lag[par]),
        })

    r_lag = calcular_retornos(painel_lag)
    base = rodar_backtest(atual, r_atual, cdi)
    lag = rodar_backtest(painel_lag, r_lag, cdi)
    m_base = calcular_metricas(base["retornos"], cdi)
    m_lag = calcular_metricas(lag["retornos"], cdi)

    por_par = pd.DataFrame(linhas)
    por_par["diferenca_cagr"] = (
        por_par["cagr_carry_defasado_21_pregoes"]
        - por_par["cagr_carry_sem_lag"]
    )
    por_par.to_csv(
        SAIDA / "auditoria_lag_carrego_por_par.csv", index=False,
        float_format="%.12g",
    )

    resumo = pd.DataFrame([
        {
            "cenario": "carry_sem_lag_publicacao",
            "cagr": m_base["cagr"], "vol": m_base["vol"],
            "sharpe": m_base["sharpe"], "max_drawdown": m_base["max_drawdown"],
        },
        {
            "cenario": "carry_defasado_21_pregoes",
            "cagr": m_lag["cagr"], "vol": m_lag["vol"],
            "sharpe": m_lag["sharpe"], "max_drawdown": m_lag["max_drawdown"],
        },
    ])
    resumo.to_csv(
        SAIDA / "auditoria_lag_carrego_resumo.csv", index=False,
        float_format="%.12g",
    )

    print("MIYAGI — SENSIBILIDADE AO LAG DO CARRY")
    print("=" * 72)
    print(f"Pares corrigidos: {', '.join(pares)}")
    print(f"Defasagem de stress: {DEFASAGEM_PREGOES} pregões")
    print("\nRetorno anualizado dos índices cambiais:")
    print(por_par.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nBacktest com universo fixo:")
    print(resumo.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nEste teste não cria vintages. Ele apenas mede a sensibilidade do")
    print("resultado a uma defasagem conservadora e reproduzível.")


if __name__ == "__main__":
    main()
