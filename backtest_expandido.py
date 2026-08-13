# -*- coding: utf-8 -*-
"""
MIYAGI — backtest com o universo expandido (40 ativos)
=======================================================

A PREVISÃO A BATER
------------------
Registrada em `funil_expandido.py` e commitada ANTES deste backtest existir:

    universo atual ....  8 ativos |  7,0 apostas efetivas | Sharpe 0,51
    universo expandido  40 ativos | 11,3 apostas efetivas | Sharpe 0,65 (previsto)

    Sharpe_previsto = 0,51 × raiz(11,3 / 7,0) = 0,65

Nada abaixo foi ajustado para chegar nesse número. O sinal, a janela de
volatilidade, o alvo de risco, o teto de alavancagem e os custos são
exatamente os mesmos do modelo atual — a única coisa que mudou foi a lista
de ativos.

UM RISCO TÉCNICO QUE PRECISA SER MEDIDO
---------------------------------------
Com 40 ativos e uma janela de 60 dias, a matriz de covariância é estimada com
q = 40/60 = 0,67 (ativos por observação). Nessa região a covariância amostral
SUBESTIMA a variância da carteira -- e como o robô se alavanca até atingir o
alvo de risco, subestimar risco vira alavancagem excessiva.

Se isso acontecer, a volatilidade REALIZADA vai estourar o alvo de 10%. O
script mede isso explicitamente: é o primeiro lugar onde um resultado bom
demais seria desmascarado.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import backtest_miyagi as bt
from backtest_miyagi import (
    calcular_metricas, calcular_retornos, rodar_backtest, retorno_por_ano,
)
from treino_parametros import HORIZONTES, parametro, walk_forward

AQUI = Path(__file__).resolve().parent

SHARPE_ATUAL = 0.51        # walk-forward, universo de 8
PREVISTO = 0.65            # registrado antes deste script existir
N_EFF_ATUAL, N_EFF_NOVO = 7.0, 11.3


def carregar_expandido():
    """Carrega o pool expandido, restrito aos 40 ativos que o funil selecionou."""
    precos = pd.read_csv(AQUI / "dados" / "pool_expandido.csv",
                         index_col=0, parse_dates=True)
    universo = (AQUI / "dados" / "universo_expandido.txt").read_text(
        encoding="utf-8").split()
    universo = [a for a in universo if a in precos.columns]

    cdi = pd.read_csv(AQUI / "dados" / "cdi.csv", index_col=0, parse_dates=True)
    cdi = cdi.iloc[:, 0].sort_index() / 100.0

    # Mesmo tratamento de calendário do modelo atual: dias úteis do CDI.
    calendario = cdi.index
    precos = precos[universo].ffill(limit=5).reindex(calendario).ffill(limit=5)
    cdi = cdi.reindex(calendario).ffill().fillna(0.0)
    return precos, cdi, universo


def main() -> None:
    print("=" * 82)
    print("MIYAGI — BACKTEST COM UNIVERSO EXPANDIDO")
    print("=" * 82)
    print(f"  Previsão registrada antes deste backtest: Sharpe {PREVISTO:.2f}")
    print(f"  (= {SHARPE_ATUAL:.2f} × raiz({N_EFF_NOVO}/{N_EFF_ATUAL}))")

    precos, cdi, universo = carregar_expandido()
    retornos = calcular_retornos(precos)
    print(f"\n  Universo: {len(universo)} ativos | "
          f"{precos.index.min():%Y-%m} a {precos.index.max():%Y-%m}")

    # --- base 12-1 no universo novo --------------------------------------
    print("\n[1/2] Base (12-1 fixo) no universo expandido...", flush=True)
    res_base = rodar_backtest(precos, retornos, cdi)
    r_base = res_base["retornos"]
    m_base = calcular_metricas(r_base, cdi)

    # --- walk-forward no universo novo -----------------------------------
    print("[2/2] Walk-forward no universo expandido...", flush=True)
    series = {}
    for h in HORIZONTES:
        print(f"      horizonte {h}-1 ...", flush=True)
        with parametro(JANELA_SINAL_MESES=h):
            series[h] = rodar_backtest(precos, retornos, cdi)["retornos"]
    r_wf, escolhas = walk_forward(series, cdi)
    m_wf = calcular_metricas(r_wf, cdi)

    # ================================================================ saída
    print("\n" + "=" * 82)
    print("RESULTADO")
    print("=" * 82)
    print(f"  {'':<34}{'CAGR':>8}{'Vol':>8}{'Sharpe':>9}{'t':>7}{'Max DD':>9}")
    print("  " + "-" * 76)

    def linha(nome, m, r):
        anos = m["anos"]
        t = m["sharpe"] * np.sqrt(anos)
        print(f"  {nome:<34}{m['cagr']:>8.1%}{m['vol']:>8.1%}"
              f"{m['sharpe']:>9.2f}{t:>7.2f}{m['max_drawdown']:>9.1%}")
        return t

    print("  universo de 8 ativos (referência)")
    print(f"  {'  base 12-1':<34}{0.150:>8.1%}{0.110:>8.1%}{0.41:>9.2f}"
          f"{1.88:>7.2f}{-0.204:>9.1%}")
    print(f"  {'  walk-forward':<34}{0.163:>8.1%}{0.110:>8.1%}{0.51:>9.2f}"
          f"{2.36:>7.2f}{-0.201:>9.1%}")
    print("\n  universo de 40 ativos (novo)")
    linha("  base 12-1", m_base, r_base)
    t_wf = linha("  walk-forward", m_wf, r_wf)

    # ------------------------------------------------- controle de risco
    print("\n" + "=" * 82)
    print("CONTROLE DE RISCO — o alvo de 10% foi respeitado?")
    print("=" * 82)
    print("  Se a volatilidade realizada estourar o alvo, a matriz de covariância")
    print("  está subestimando o risco e a alavancagem saiu de controle.\n")
    print(f"  {'':<26}{'vol realizada':>16}{'alvo':>8}{'exposição média':>18}")
    print("  " + "-" * 68)
    print(f"  {'base 12-1':<26}{m_base['vol']:>15.1%}{0.10:>8.0%}"
          f"{res_base['exposicao'].mean():>17.2f}x")
    print(f"  {'walk-forward':<26}{m_wf['vol']:>15.1%}{0.10:>8.0%}"
          f"{res_base['exposicao'].mean():>17.2f}x")

    estourou = m_wf["vol"] > 0.135
    if estourou:
        print("\n  [!] A vol realizada passou de 13,5%. Com 40 ativos e janela de 60")
        print("      dias, a covariância amostral subestima o risco da carteira.")
        print("      O resultado ABAIXO deve ser lido com essa ressalva.")
    else:
        print("\n  [ok] Vol realizada dentro do esperado — o alvo de risco segurou")
        print("       mesmo com 5x mais ativos.")

    # ------------------------------------------------- veredito da previsão
    print("\n" + "=" * 82)
    print("PREVISÃO vs REALIZADO")
    print("=" * 82)
    obtido = m_wf["sharpe"]
    erro = obtido - PREVISTO
    print(f"""
      previsto (antes do backtest) ....... {PREVISTO:.2f}
      realizado (walk-forward, 40 ativos)  {obtido:.2f}
      erro ............................... {erro:+.2f}
""")
    if abs(erro) <= 0.10:
        print("  VEREDITO: previsão CONFIRMADA (erro <= 0,10).")
        print("  O ganho de diversificação se comportou como a teoria previa.")
        print("  Prever antes e acertar vale mais que qualquer numero isolado.")
    elif erro > 0.10:
        print("  VEREDITO: resultado ACIMA do previsto.")
        print("  Isso NAO deve ser comemorado sem investigar. Diversificar não")
        print("  deveria entregar mais do que a teoria permite -- verifique vies")
        print("  de selecao, vazamento de dados ou subestimacao de risco.")
    else:
        print("  VEREDITO: resultado ABAIXO do previsto.")
        print("  A correlacao media superestimou a diversificacao real. Isso e")
        print("  um achado legitimo: em regimes de estresse os ativos se movem")
        print("  juntos, e a correlacao media nao captura isso.")

    # ------------------------------------------------------------ por ano
    print("\n" + "=" * 82)
    print("SUB-PERÍODOS — a fraqueza pós-2016 melhorou?")
    print("=" * 82)
    print(f"  {'período':<16}{'8 ativos (WF)':>18}{'40 ativos (WF)':>18}{'CDI':>9}")
    print("  " + "-" * 62)
    ref_8 = {"2005-2010": 0.32, "2011-2015": 1.40, "2016-2020": -0.04, "2021-2026": 0.03}
    for rot, (ini, fim) in {"2005-2010": ("2005", "2010"),
                            "2011-2015": ("2011", "2015"),
                            "2016-2020": ("2016", "2020"),
                            "2021-2026": ("2021", "2026")}.items():
        sub = r_wf.loc[ini:fim]
        if sub.empty:
            continue
        m = calcular_metricas(sub, cdi)
        c = calcular_metricas(cdi.reindex(sub.index).fillna(0.0).loc[ini:fim], cdi)
        print(f"  {rot:<16}{ref_8[rot]:>18.2f}{m['sharpe']:>18.2f}{c['cagr']:>9.1%}")

    # ------------------------------------------------------------- grava
    saida = AQUI / "resultados"
    saida.mkdir(exist_ok=True)
    pd.DataFrame({"walk_forward_40": r_wf, "base_40": r_base}).to_csv(
        saida / "serie_expandido.csv")
    print(f"\n  Séries gravadas em resultados/serie_expandido.csv")


if __name__ == "__main__":
    main()
