# -*- coding: utf-8 -*-
"""
MIYAGI — funil de seleção sobre o pool expandido
=================================================

O QUE ESTE SCRIPT FAZ
---------------------
Aplica ao pool de 114 candidatos EXATAMENTE o mesmo funil que produziu os 8
ativos originais:

    1. correlação de retornos MENSAIS em log, pareada (mín. 60 meses de overlap)
    2. clustering hierárquico, average linkage, sobre a distância  D = 1 − ρ
    3. corte em 0,35  (junta o que tem correlação média > 0,65)
    4. elegibilidade: histórico ≥ 15 anos
    5. um representante por cluster

O critério NÃO mudou. O que mudou foi o tamanho do pool: 26 → 114 candidatos.
Mais variedade produz mais clusters, e mais clusters produzem mais apostas
independentes — que é a única coisa que a teoria diz que aumenta o Sharpe.

A REGRA DO REPRESENTANTE
------------------------
Escolher qual ativo representa cada cluster é o ponto onde o viés entraria mais
facilmente ("fico com o que rendeu mais"). Por isso a regra é mecânica e
CEGA A RETORNO:

    representante = MEDOIDE do cluster
                    (o ativo mais correlacionado com os próprios colegas,
                     isto é, o que melhor representa o fator comum daquele grupo)
    desempate     = maior histórico disponível

Retorno, Sharpe e desempenho não entram em lugar nenhum desta decisão.

E O QUE ESTE SCRIPT NÃO FAZ
---------------------------
Não roda backtest. De propósito.

A sequência honesta é: selecionar → REGISTRAR A PREVISÃO → só então medir.
Se o backtest rodasse aqui, seria impossível provar (inclusive para nós mesmos)
que a previsão não foi escrita depois de ver o resultado.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

AQUI = Path(__file__).resolve().parent

CORTE_CLUSTER = 0.35        # idêntico ao funil original: junta ρ > 0,65
MIN_OVERLAP_MESES = 60      # idêntico ao original
MIN_ANOS_HISTORICO = 15     # idêntico ao original

# Universo atual, para comparação
ATUAIS = ["^BVSP", "SPY", "IEF", "BRL=X", "EURUSD=X", "JPY=X", "GLD", "DBC"]


def retornos_mensais(precos: pd.DataFrame) -> pd.DataFrame:
    """Retornos mensais em log — a mesma base do funil original."""
    mensal = precos.resample("ME").last()
    return np.log(mensal / mensal.shift(1))


def correlacao_pareada(ret: pd.DataFrame) -> pd.DataFrame:
    """Correlação par a par, exigindo overlap mínimo.

    Pares com pouca sobreposição produzem correlações instáveis que bagunçam o
    clustering. O original exigia 60 meses; mantido.
    """
    corr = ret.corr(min_periods=MIN_OVERLAP_MESES)
    return corr.fillna(0.0).clip(-1, 1)


def elegiveis(precos: pd.DataFrame) -> list[str]:
    """Ativos com pelo menos 15 anos de histórico."""
    fim = precos.index.max()
    out = []
    for c in precos.columns:
        s = precos[c].dropna()
        if len(s) and (fim - s.index.min()).days / 365.25 >= MIN_ANOS_HISTORICO:
            out.append(c)
    return sorted(out)


def apostas_efetivas(corr: pd.DataFrame, ativos: list[str]) -> float:
    """Nº de apostas INDEPENDENTES equivalentes.

        N_eff  =  N / (1 + (N-1) · ρ_médio)

    Traduzindo: 8 ativos que andam sempre juntos (ρ=1) valem 1 aposta;
    8 ativos totalmente independentes (ρ=0) valem 8. O universo atual tem
    ρ médio 0,04 e vale ~6,1.

    É este número — e não a contagem de ativos — que a teoria liga ao Sharpe.
    """
    sub = corr.loc[ativos, ativos].to_numpy()
    n = len(ativos)
    if n < 2:
        return float(n)
    iu = np.triu_indices(n, k=1)
    rho = float(np.nanmean(sub[iu]))
    return n / (1 + (n - 1) * rho), rho


def medoide(cluster: list[str], corr: pd.DataFrame, precos: pd.DataFrame) -> str:
    """O ativo que melhor representa o cluster. Cego a retorno."""
    if len(cluster) == 1:
        return cluster[0]
    sub = corr.loc[cluster, cluster]
    centralidade = sub.sum(axis=1)                    # soma das correlações
    maximo = centralidade.max()
    empatados = [a for a in cluster if centralidade[a] >= maximo - 1e-9]
    if len(empatados) == 1:
        return empatados[0]
    # desempate: maior histórico
    return max(empatados, key=lambda a: len(precos[a].dropna()))


def main() -> None:
    print("=" * 80)
    print("FUNIL DE SELEÇÃO — pool expandido")
    print("=" * 80)

    precos = pd.read_csv(AQUI / "dados" / "pool_expandido.csv",
                         index_col=0, parse_dates=True)
    print(f"Pool: {precos.shape[1]} candidatos | "
          f"{precos.index.min():%Y-%m} a {precos.index.max():%Y-%m}")

    apt = elegiveis(precos)
    print(f"Elegíveis (≥ {MIN_ANOS_HISTORICO} anos de histórico): {len(apt)}")

    ret = retornos_mensais(precos[apt])
    corr = correlacao_pareada(ret)

    # --- clustering, idêntico ao original --------------------------------
    D = 1 - corr.to_numpy()
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(Z, t=CORTE_CLUSTER, criterion="distance")

    clusters: dict[int, list[str]] = {}
    for ativo, lab in zip(apt, labels):
        clusters.setdefault(lab, []).append(ativo)

    print(f"\nClusters formados (corte ρ > {1 - CORTE_CLUSTER:.2f}): {len(clusters)}")

    # --- representantes ---------------------------------------------------
    selecionados = []
    print("\n" + "=" * 80)
    print("CLUSTERS E REPRESENTANTES")
    print("=" * 80)
    for lab, membros in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        rep = medoide(membros, corr, precos)
        selecionados.append(rep)
        outros = [m for m in membros if m != rep]
        marca = "  <- ja no universo atual" if rep in ATUAIS else ""
        print(f"  [{len(membros):>2} ativos]  {rep:<12}{marca}")
        if outros:
            print(f"              representa: {', '.join(outros[:11])}"
                  f"{'...' if len(outros) > 11 else ''}")
    selecionados = sorted(selecionados)

    # --- comparação -------------------------------------------------------
    n_atual, rho_atual = apostas_efetivas(corr, [a for a in ATUAIS if a in corr.index])
    n_novo, rho_novo = apostas_efetivas(corr, selecionados)

    print("\n" + "=" * 80)
    print("UNIVERSO ATUAL vs EXPANDIDO")
    print("=" * 80)
    print(f"  {'':<22}{'ativos':>9}{'ρ médio':>11}{'apostas efetivas':>20}")
    print("  " + "-" * 62)
    print(f"  {'atual':<22}{len([a for a in ATUAIS if a in corr.index]):>9}"
          f"{rho_atual:>11.3f}{n_atual:>20.1f}")
    print(f"  {'expandido':<22}{len(selecionados):>9}{rho_novo:>11.3f}{n_novo:>20.1f}")

    print(f"\n  Novos ativos ({len([s for s in selecionados if s not in ATUAIS])}):")
    novos = [s for s in selecionados if s not in ATUAIS]
    for i in range(0, len(novos), 8):
        print("    " + ", ".join(novos[i:i + 8]))
    saiu = [a for a in ATUAIS if a not in selecionados]
    if saiu:
        print(f"\n  Do universo atual, NÃO foram selecionados: {', '.join(saiu)}")
        print("    (foram absorvidos por clusters cujo medoide é outro ativo)")

    # =================================================================
    # A PREVISÃO — registrada ANTES de qualquer backtest
    # =================================================================
    SHARPE_ATUAL = 0.51        # walk-forward, período inteiro, universo de 8
    fator = np.sqrt(n_novo / n_atual)
    previsto = SHARPE_ATUAL * fator

    print("\n" + "=" * 80)
    print("PREVISÃO REGISTRADA ANTES DO BACKTEST")
    print("=" * 80)
    print(f"""
  A teoria de portfólio liga o Sharpe ao número de apostas INDEPENDENTES:

        Sharpe  ≈  IR × raiz(N_efetivo)

  Se o IR (a qualidade do sinal por aposta) não mudar ao trocar de universo,
  então o Sharpe deve escalar pela raiz da razão entre as apostas efetivas:

        Sharpe_novo  =  {SHARPE_ATUAL:.2f} × raiz({n_novo:.1f} / {n_atual:.1f})
                     =  {SHARPE_ATUAL:.2f} × {fator:.3f}
                     =  {previsto:.2f}

  ESTA É A PREVISÃO. O backtest ainda NÃO foi rodado com o universo novo.

  Como interpretar o resultado quando ele sair:

    perto de {previsto:.2f}      o mecanismo está entendido. É o melhor
                       resultado possível -- prever antes e acertar vale
                       muito mais que um número alto sem explicação.

    muito ACIMA        sinal de alerta. Se diversificar entregasse mais do
                       que a teoria permite, provavelmente há erro no
                       backtest ou viés na seleção.

    muito ABAIXO       a diversificação medida pela correlação não se
                       traduziu em diversificação real. Também é achado:
                       significa que a correlação média subestima a
                       dependência entre os ativos (caudas, regimes).
""")

    # --- grava a seleção --------------------------------------------------
    saida = AQUI / "dados" / "universo_expandido.txt"
    saida.write_text("\n".join(selecionados), encoding="utf-8")
    print(f"  Seleção gravada: dados/universo_expandido.txt ({len(selecionados)} ativos)")
    print(f"  Previsão a bater: Sharpe {previsto:.2f}")


if __name__ == "__main__":
    main()
