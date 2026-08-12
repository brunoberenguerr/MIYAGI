# -*- coding: utf-8 -*-
"""
Diagnóstico: a escolha do estimador de volatilidade explica a divergência
com o pré-relatório?
=========================================================================

CONTEXTO
--------
Ao reconstruir o backtest do Miyagi, os números não bateram com os do
pré-relatório de julho. Parte da diferença já foi explicada (um bug de
calendário que subestimava o CDI). Mas duas diferenças continuaram de pé:

    drawdown máximo:  -20,4% (reconstruído)  vs  -14,5% (pré-relatório)
    retorno em 2008:   +4,6% (reconstruído)  vs   +9,9% (pré-relatório)

A especificação do projeto permitia DUAS formas de medir volatilidade —
"vol EWMA ou janela de 60 dias". A reconstrução usou a janela simples.
Este script testa a outra opção.

O QUE ESTE SCRIPT NÃO É
-----------------------
Não é uma busca pelo estimador que dá o melhor número. Isso seria
overfitting, e invalidaria o resultado. É um teste de UMA hipótese
específica, declarada antes de rodar: "a divergência vem do estimador".

O resultado é reportado seja ele qual for — inclusive se o EWMA piorar,
ou se não explicar nada. Um diagnóstico só vale se você aceita a resposta
antes de conhecê-la.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_miyagi import (
    calcular_metricas,
    carregar_dados,
    calcular_retornos,
    retorno_por_ano,
    rodar_backtest,
)

# Os números que queremos explicar, extraídos do pré-relatório de julho.
PRE_RELATORIO = {
    "cagr": 0.155,
    "vol": 0.109,
    "sharpe": 0.57,
    "max_drawdown": -0.145,
    "2008": 0.099,
    "2020": 0.080,
}


def main() -> None:
    print("=" * 78)
    print("DIAGNÓSTICO — o estimador de volatilidade explica a divergência?")
    print("=" * 78)

    precos, cdi = carregar_dados()
    retornos = calcular_retornos(precos)

    resultados = {}
    for estimador in ("janela", "ewma"):
        print(f"\nrodando estimador '{estimador}'...", flush=True)
        r = rodar_backtest(precos, retornos, cdi, estimador=estimador)
        serie = r["retornos"]
        m = calcular_metricas(serie, cdi, estimador)
        anos = retorno_por_ano(serie)
        resultados[estimador] = {
            "metricas": m,
            "por_ano": anos,
            "exposicao": r["exposicao"],
            "giro": r["giro"],
            "custos": r["custos"],
        }

    # ---------------------------------------------------------------- tabela
    print("\n" + "=" * 78)
    print(f"{'':<16}{'janela 60d':>14}{'EWMA 0,94':>14}{'pré-relat.':>14}"
          f"{'quem bate?':>18}")
    print("-" * 78)

    linhas = [
        ("CAGR", "cagr", "{:.1%}"),
        ("Volatilidade", "vol", "{:.1%}"),
        ("Sharpe", "sharpe", "{:.2f}"),
        ("Max Drawdown", "max_drawdown", "{:.1%}"),
    ]
    for rotulo, chave, fmt in linhas:
        v_jan = resultados["janela"]["metricas"][chave]
        v_ewma = resultados["ewma"]["metricas"][chave]
        v_pre = PRE_RELATORIO[chave]
        # Qual dos dois chega mais perto do pré-relatório?
        mais_perto = "EWMA" if abs(v_ewma - v_pre) < abs(v_jan - v_pre) else "janela"
        print(f"{rotulo:<16}{fmt.format(v_jan):>14}{fmt.format(v_ewma):>14}"
              f"{fmt.format(v_pre):>14}{mais_perto:>18}")

    for ano in (2008, 2020):
        v_jan = resultados["janela"]["por_ano"].get(ano, np.nan)
        v_ewma = resultados["ewma"]["por_ano"].get(ano, np.nan)
        v_pre = PRE_RELATORIO[str(ano)]
        mais_perto = "EWMA" if abs(v_ewma - v_pre) < abs(v_jan - v_pre) else "janela"
        print(f"{'retorno ' + str(ano):<16}{v_jan:>13.1%}{v_ewma:>14.1%}"
              f"{v_pre:>14.1%}{mais_perto:>18}")
    print("=" * 78)

    # ------------------------------------------------- por que eles diferem
    # A pista mecânica: quanto o robô se alavanca. Se um estimador enxerga o
    # mercado mais calmo, ele autoriza posições maiores — e leva tombos
    # maiores quando erra.
    print("\nALAVANCAGEM (a explicação mecânica da diferença de drawdown)")
    print(f"{'':<16}{'média':>12}{'máxima':>12}{'% no teto 3x':>16}{'giro médio':>14}")
    print("-" * 70)
    for est in ("janela", "ewma"):
        exp = resultados[est]["exposicao"]
        no_teto = float((exp >= 2.99).mean())
        giro = resultados[est]["giro"].mean()
        print(f"{est:<16}{exp.mean():>11.2f}x{exp.max():>11.2f}x"
              f"{no_teto:>15.0%}{giro:>14.2f}")

    # ------------------------------------------------------------ veredito
    print("\n" + "=" * 78)
    m_jan = resultados["janela"]["metricas"]
    m_ewma = resultados["ewma"]["metricas"]

    # Distância total até o pré-relatório, normalizada por métrica para que
    # Sharpe e CAGR pesem de forma comparável.
    def distancia(m, por_ano):
        d = 0.0
        for chave in ("cagr", "vol", "max_drawdown"):
            d += abs(m[chave] - PRE_RELATORIO[chave]) / abs(PRE_RELATORIO[chave])
        d += abs(m["sharpe"] - PRE_RELATORIO["sharpe"]) / PRE_RELATORIO["sharpe"]
        for ano in (2008, 2020):
            d += abs(por_ano.get(ano, np.nan) - PRE_RELATORIO[str(ano)]) \
                / abs(PRE_RELATORIO[str(ano)])
        return d

    d_jan = distancia(m_jan, resultados["janela"]["por_ano"])
    d_ewma = distancia(m_ewma, resultados["ewma"]["por_ano"])

    print(f"Distância total até o pré-relatório:  janela={d_jan:.2f}  EWMA={d_ewma:.2f}")
    print()
    if d_ewma < d_jan * 0.7:
        print("VEREDITO: o EWMA explica boa parte da divergência.")
        print("  O pré-relatório provavelmente usou EWMA, e a reconstrução com")
        print("  janela simples é que estava fora do padrão.")
    elif d_ewma < d_jan:
        print("VEREDITO: o EWMA aproxima, mas NÃO fecha a conta.")
        print("  Parte da divergência vem do estimador; o resto tem outra causa")
        print("  que este teste não identifica.")
    else:
        print("VEREDITO: o estimador NÃO explica a divergência.")
        print("  A hipótese testada foi refutada. A causa está em outro lugar —")
        print("  possivelmente no tratamento de dados ou na construção do sinal.")
    print("=" * 78)


if __name__ == "__main__":
    main()
