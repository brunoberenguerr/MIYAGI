# -*- coding: utf-8 -*-
"""
MIYAGI — parâmetros treinados: duas abordagens, lado a lado
============================================================

A PERGUNTA
----------
O Miyagi original não treina nada: sinal 12-1 fixo, vindo da literatura.
Faz sentido dar a ele parâmetros que APRENDEM com os dados?

E, principalmente: qual é a forma HONESTA de fazer isso?

Este arquivo implementa as duas formas e compara. A diferença entre elas é o
ponto pedagógico do trabalho.

    ABORDAGEM 1 — WALK-FORWARD (causal na escolha do horizonte)
    A cada mês, o robô escolhe o horizonte do sinal olhando APENAS o passado.
    Em março de 2013 ele decide com o que sabia em março de 2013. Não existe
    look-ahead do horizonte por construção. Isso não torna os 21 anos fora da
    amostra: o universo histórico ainda foi selecionado com dados futuros.

    ABORDAGEM 2 — TREINO/TESTE CLÁSSICO
    Escolhe o melhor horizonte em 2007-2016, congela, e aplica em 2017-2026.

POR QUE TESTAR O HORIZONTE, E NÃO OUTRA COISA
---------------------------------------------
Não foi escolha arbitrária. Um diagnóstico histórico anterior estimou queda da
duração média das tendências de 11,5 para 6,9 meses. O script auditado atual
usa pesos derivados e produz outra medição; os valores antigos são preservados
apenas como registro da hipótese que motivou este teste.

O sinal olha 12 meses para trás. Se as tendências duram 7, o robô entra
atrasado por construção. Existe portanto uma RAZÃO ECONÔMICA, medida antes de
qualquer teste, para acreditar que um horizonte adaptativo ajudaria.

Essa ordem importa: hipótese econômica primeiro, teste depois. O contrário
(testar 20 variantes e inventar a explicação da vencedora) é o que produz
backtests bonitos e falsos.

A RESSALVA QUE PRECISA ACOMPANHAR A ABORDAGEM 2
------------------------------------------------
O período 2017-2026 JÁ FOI ANALISADO por nós em detalhe antes deste teste.
Sabemos o Sharpe dele, sabemos quais ativos falharam, sabemos que as
tendências encurtaram.

Um holdout que já foi visto não é mais um holdout. Reportar o resultado da
abordagem 2 como "out-of-sample" seria dar a um número conhecido uma etiqueta
que sugere o contrário. Ele é reportado aqui COM essa ressalva — e a
comparação com a abordagem 1 existe justamente para tornar a diferença
visível.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager

import numpy as np
import pandas as pd

import backtest_miyagi as bt
from backtest_miyagi import (
    DIAS_UTEIS_ANO, calcular_metricas, carregar_dados,
    calcular_retornos, rodar_backtest,
)
from dados_miyagi import selecionar_etfs

# Família de horizontes candidatos. Todos são valores usados na literatura de
# momentum -- não é uma grade fina desenhada para garimpar o melhor número.
HORIZONTES = (3, 6, 9, 12, 18)

JANELA_APRENDIZADO_MESES = 36   # quanto passado o walk-forward olha para decidir
INICIO_HOLDOUT = "2017"
FIM_DESIGN = "2016"


@contextmanager
def parametro(**overrides):
    antigos = {k: getattr(bt, k) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(bt, k, v)
        yield
    finally:
        for k, v in antigos.items():
            setattr(bt, k, v)


def sharpe_de(r: pd.Series, cdi: pd.Series) -> float:
    if len(r) < 60:
        return np.nan
    excesso = r - cdi.reindex(r.index).fillna(0.0)
    if excesso.std() == 0:
        return np.nan
    return float(excesso.mean() / excesso.std() * np.sqrt(DIAS_UTEIS_ANO))


def series_por_horizonte(
    precos, retornos, cdi, ativos_financiados: set[str] | None = None,
) -> dict[int, pd.Series]:
    """Roda o backtest uma vez para cada horizonte candidato.

    Cada série é causal por construção (o motor já garante isso). Ter todas
    permite que o walk-forward, em cada data, olhe o desempenho passado de
    cada horizonte sem nunca tocar o futuro.
    """
    out = {}
    for h in HORIZONTES:
        print(f"    rodando horizonte {h}-1 ...", flush=True)
        with parametro(JANELA_SINAL_MESES=h):
            out[h] = rodar_backtest(
                precos, retornos, cdi,
                ativos_financiados=ativos_financiados,
            )["retornos"]
    return out


def walk_forward(series: dict[int, pd.Series], cdi: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Escolhe o horizonte mês a mês usando SOMENTE o passado.

    Regra: no fim de cada mês, calcula o Sharpe dos últimos 36 meses de cada
    horizonte candidato -- usando apenas retornos ANTERIORES à data -- e adota
    o vencedor para o mês seguinte.

    A linha que garante a causalidade é o corte `.loc[:data]` com `data`
    exclusivo: nenhum retorno do mês que será negociado entra na decisão.
    """
    idx = series[HORIZONTES[0]].index
    meses = pd.Series(idx, index=idx).resample("ME").last().dropna()

    escolhas, retorno_wf = {}, {}
    for i, data in enumerate(meses.values[:-1]):
        data = pd.Timestamp(data)
        inicio_janela = data - pd.DateOffset(months=JANELA_APRENDIZADO_MESES)

        # --- decisão: só passado ---
        pontuacao = {}
        for h, s in series.items():
            passado = s.loc[inicio_janela:data]        # <= data, nunca depois
            sh = sharpe_de(passado, cdi)
            if np.isfinite(sh):
                pontuacao[h] = sh
        if not pontuacao:
            continue
        melhor = max(pontuacao, key=pontuacao.get)
        escolhas[data] = melhor

        # --- aplicação: o mês SEGUINTE, com o horizonte escolhido ---
        proxima = pd.Timestamp(meses.values[i + 1])
        trecho = series[melhor].loc[(series[melhor].index > data)
                                    & (series[melhor].index <= proxima)]
        for d, v in trecho.items():
            retorno_wf[d] = v

    return pd.Series(retorno_wf).sort_index(), pd.Series(escolhas).sort_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--financiamento", choices=["overlay", "etfs"], default="etfs",
        help="etfs = stress financiado atual | overlay = estimando histórico",
    )
    args = ap.parse_args()
    print("=" * 84)
    print("MIYAGI — PARÂMETROS TREINADOS: WALK-FORWARD vs TREINO/TESTE")
    print("=" * 84)

    precos, cdi = carregar_dados()
    retornos = calcular_retornos(precos)
    financiados = (
        selecionar_etfs(list(precos.columns))
        if args.financiamento == "etfs" else set()
    )
    print(f"Convenção de financiamento: {args.financiamento}")
    if financiados:
        print("Stress a CDI sobre ETFs com retornos ainda em moeda de origem;")
        print("não é uma carteira implementável em reais.")

    print("\n[1/3] Rodando cada horizonte candidato...")
    series = series_por_horizonte(
        precos, retornos, cdi, ativos_financiados=financiados
    )

    print("\n[2/3] Walk-forward (decisão mensal, só passado)...")
    r_wf, escolhas = walk_forward(series, cdi)

    print("[3/3] Treino/teste clássico...")
    # TREINO: melhor horizonte em 2007-2016 (design)
    sharpes_design = {h: sharpe_de(s.loc[:FIM_DESIGN], cdi) for h, s in series.items()}
    h_treinado = max(sharpes_design, key=sharpes_design.get)

    print(f"\n  Sharpe de cada horizonte no período de TREINO (2005-{FIM_DESIGN}):")
    for h in HORIZONTES:
        marca = "   <- vencedor, congelado" if h == h_treinado else ""
        print(f"    {h:>2}-1 meses: {sharpes_design[h]:>6.2f}{marca}")

    # ================================================================ tabela
    def bloco(r, rotulo, ini=None, fim=None):
        s = r if ini is None else r.loc[ini:fim]
        m = calcular_metricas(s, cdi, rotulo)
        anos = m["anos"]
        return {"nome": rotulo, "cagr": m["cagr"], "vol": m["vol"],
                "sharpe": m["sharpe"], "dd": m["max_drawdown"],
                "t": m["sharpe"] * np.sqrt(anos) if anos > 0 else np.nan}

    r_base = series[12]
    r_treinado = series[h_treinado]

    print("\n" + "=" * 84)
    print(f"PSEUDO-HOLDOUT JÁ OBSERVADO ({INICIO_HOLDOUT}-2026)")
    print("=" * 84)
    cdi_hold = calcular_metricas(
        cdi.reindex(r_base.index).fillna(0.0).loc[INICIO_HOLDOUT:], cdi)["cagr"]
    print(f"  CDI no período: {cdi_hold:.1%} a.a.\n")
    print(f"  {'abordagem':<34}{'CAGR':>8}{'Vol':>8}{'Sharpe':>9}{'t':>7}{'Max DD':>9}")
    print("  " + "-" * 76)

    linhas = [
        bloco(r_base, "Base (12-1 fixo, sem treino)", INICIO_HOLDOUT, "2026"),
        bloco(r_wf, "1. Walk-forward (horizonte causal)", INICIO_HOLDOUT, "2026"),
        bloco(r_treinado, f"2. Treino/teste ({h_treinado}-1 congelado)",
              INICIO_HOLDOUT, "2026"),
    ]
    for d in linhas:
        alerta = "  <- perde do CDI" if d["cagr"] < cdi_hold else ""
        print(f"  {d['nome']:<34}{d['cagr']:>8.1%}{d['vol']:>8.1%}"
              f"{d['sharpe']:>9.2f}{d['t']:>7.2f}{d['dd']:>9.1%}{alerta}")

    print("\n" + "=" * 84)
    print("PERÍODO INTEIRO (2005-2026)")
    print("=" * 84)
    print(f"  {'abordagem':<34}{'CAGR':>8}{'Vol':>8}{'Sharpe':>9}{'t':>7}{'Max DD':>9}")
    print("  " + "-" * 76)
    for d in [bloco(r_base, "Base (12-1 fixo, sem treino)"),
              bloco(r_wf, "1. Walk-forward (horizonte causal)"),
              bloco(r_treinado, f"2. Treino/teste ({h_treinado}-1 congelado)")]:
        print(f"  {d['nome']:<34}{d['cagr']:>8.1%}{d['vol']:>8.1%}"
              f"{d['sharpe']:>9.2f}{d['t']:>7.2f}{d['dd']:>9.1%}")

    # ============================================================== escolhas
    print("\n" + "=" * 84)
    print("O QUE O WALK-FORWARD ESCOLHEU AO LONGO DO TEMPO")
    print("=" * 84)
    print("  Se a tese das 'tendências encurtando' estiver certa, o robô deveria")
    print("  migrar para horizontes MAIS CURTOS depois de 2016.\n")
    por_periodo = escolhas.groupby(escolhas.index.year // 5 * 5)
    print(f"  {'período':<14}{'horizonte médio':>18}{'mais escolhido':>18}")
    print("  " + "-" * 50)
    for bloco_ano, vals in por_periodo:
        moda = vals.mode()
        print(f"  {int(bloco_ano)}-{int(bloco_ano)+4:<9}{vals.mean():>16.1f}m"
              f"{int(moda.iloc[0]) if len(moda) else 0:>16}-1")
    print(f"\n  Trocas de horizonte: {int((escolhas.diff() != 0).sum())} "
          f"em {len(escolhas)} meses")

    # =============================================================== leitura
    print("\n" + "=" * 84)
    print("LEITURA")
    print("=" * 84)
    s_base = linhas[0]["sharpe"]
    s_wf = linhas[1]["sharpe"]
    s_tt = linhas[2]["sharpe"]

    print(f"""
  No holdout ({INICIO_HOLDOUT}-2026):
      base 12-1 fixo ....... Sharpe {s_base:.2f}
      walk-forward ......... Sharpe {s_wf:.2f}   ({s_wf - s_base:+.2f} vs base)
      treino/teste ......... Sharpe {s_tt:.2f}   ({s_tt - s_base:+.2f} vs base)

  COMO INTERPRETAR CADA UM:

  O walk-forward protege apenas a escolha mensal do horizonte: ela usa o
  passado disponível na data. Ele não é estimativa limpa de desempenho futuro,
  porque o universo foi escolhido com a amostra completa e o período já havia
  sido analisado pela equipe.

  O treino/teste tem uma contaminação que nenhum código conserta: nós já
  tínhamos analisado 2017-2026 antes de rodar este teste. Sabíamos o
  Sharpe, sabíamos que as tendências encurtaram, sabíamos quais ativos
  falharam. Escolher um parâmetro "no treino" sabendo o que aconteceu no
  teste não é out-of-sample, por mais que o código respeite as datas.

  Essa é a diferença entre disciplina de CÓDIGO e disciplina de PROCESSO.
  A primeira o motor garante; a segunda depende de quando o pesquisador
  olhou os dados -- e, no nosso caso, já olhamos.
""")


if __name__ == "__main__":
    main()
