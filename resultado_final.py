# -*- coding: utf-8 -*-
"""
MIYAGI — cenários auditados de carry e financiamento
=====================================================

Refaz o funil e o backtest sobre o painel com proxy de carry
(`pool_carrego.csv`), em
que os pares de câmbio passaram a ser índices de RETORNO TOTAL em vez de preço
à vista.

POR QUE O FUNIL TAMBÉM É REFEITO
--------------------------------
O carrego muda o retorno das moedas, e o funil seleciona por correlação de
retornos. Refazê-lo mantém a coerência: a seleção é feita sobre exatamente a
mesma série que o backtest vai negociar.

O critério NÃO muda -- mesmo clustering, mesmo corte, mesmo medoide, mesma
cegueira a desempenho.

FINANCIAMENTO
-------------
A saída preserva o overlay histórico para rastreabilidade e mostra ao lado o
cenário que financia a CDI os ETFs de ``Adjusted Close``. O segundo não corrige
roll, dividendos ausentes, FX, borrow ou margem; ele resolve apenas a dupla
contagem identificável do caixa nos ETFs sob o numerário simplificado do
projeto. Como os retornos continuam nas moedas de origem, nenhum dos dois deve
ser chamado de retorno implementável em reais.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

import backtest_miyagi as bt
from backtest_miyagi import calcular_metricas, calcular_retornos, rodar_backtest
from funil_expandido import (ATUAIS, CORTE_CLUSTER, apostas_efetivas,
                             correlacao_pareada, elegiveis, medoide,
                             retornos_mensais)
from dados_miyagi import (alinhar_ao_calendario, carregar_cdi,
                          carregar_pool_oficial, selecionar_etfs)

AQUI = Path(__file__).resolve().parent


def roda_funil(precos: pd.DataFrame) -> tuple[list[str], float, float]:
    apt = elegiveis(precos)
    corr = correlacao_pareada(retornos_mensais(precos[apt]))
    D = 1 - corr.to_numpy()
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(Z, t=CORTE_CLUSTER, criterion="distance")

    clusters: dict[int, list[str]] = {}
    for ativo, lab in zip(apt, labels):
        clusters.setdefault(lab, []).append(ativo)

    sel = sorted(medoide(m, corr, precos) for m in clusters.values())
    n_eff, rho = apostas_efetivas(corr, sel)
    return sel, n_eff, rho


def prepara(precos: pd.DataFrame, cdi: pd.Series, universo: list[str]):
    p = alinhar_ao_calendario(precos[universo], cdi.index)
    return p, calcular_retornos(p)


def main() -> None:
    print("=" * 80)
    print("MIYAGI — CENÁRIOS AUDITADOS: CARRY E FINANCIAMENTO")
    print("=" * 80)

    cdi = carregar_cdi()

    antes = pd.read_csv(AQUI / "dados" / "pool_expandido.csv", index_col=0, parse_dates=True)
    depois = carregar_pool_oficial()

    print("\n[1/4] Refazendo o funil sobre o painel com proxy de carry...", flush=True)
    sel_novo, neff_novo, rho_novo = roda_funil(depois)
    sel_velho, neff_velho, rho_velho = roda_funil(antes)

    print(f"\n  {'':<26}{'ativos':>9}{'ρ médio':>11}{'apostas efetivas':>19}")
    print("  " + "-" * 65)
    print(f"  {'sem carrego (antes)':<26}{len(sel_velho):>9}{rho_velho:>11.3f}{neff_velho:>19.1f}")
    print(f"  {'com proxy de carry':<26}{len(sel_novo):>9}{rho_novo:>11.3f}{neff_novo:>19.1f}")

    entrou = sorted(set(sel_novo) - set(sel_velho))
    saiu = sorted(set(sel_velho) - set(sel_novo))
    if entrou:
        print(f"\n  Entraram: {', '.join(entrou)}")
    if saiu:
        print(f"  Saíram:   {', '.join(saiu)}")
    if not entrou and not saiu:
        print("\n  A seleção não mudou — o carrego é um drift suave e quase não")
        print("  altera as correlações, que medem co-movimento.")

    print(f"\n  TRY=X ainda está no universo? "
          f"{'SIM' if 'TRY=X' in sel_novo else 'NÃO'}")

    print("\n[2/4] Backtest do overlay histórico...", flush=True)
    p_novo, r_novo = prepara(depois, cdi, sel_novo)
    res_novo = rodar_backtest(p_novo, r_novo, cdi.reindex(p_novo.index).ffill().fillna(0.0))
    m_novo = calcular_metricas(res_novo["retornos"],
                               cdi.reindex(res_novo["retornos"].index).fillna(0.0))

    etfs = selecionar_etfs(sel_novo)
    print("[3/4] Backtest com ETFs financiados a CDI...", flush=True)
    res_financiado = rodar_backtest(
        p_novo,
        r_novo,
        cdi.reindex(p_novo.index).ffill().fillna(0.0),
        ativos_financiados=etfs,
    )
    m_financiado = calcular_metricas(
        res_financiado["retornos"],
        cdi.reindex(res_financiado["retornos"].index).fillna(0.0),
    )

    print("[4/4] Backtest sobre o painel antigo, para comparação...", flush=True)
    p_velho, r_velho = prepara(antes, cdi, sel_velho)
    res_velho = rodar_backtest(p_velho, r_velho, cdi.reindex(p_velho.index).ffill().fillna(0.0))
    m_velho = calcular_metricas(res_velho["retornos"],
                                cdi.reindex(res_velho["retornos"].index).fillna(0.0))

    p8, cdi8 = bt.carregar_dados()
    r8 = calcular_retornos(p8)
    res8_overlay = rodar_backtest(p8, r8, cdi8)
    res8_financiado = rodar_backtest(
        p8, r8, cdi8, ativos_financiados=selecionar_etfs(list(p8.columns))
    )
    m8_overlay = calcular_metricas(res8_overlay["retornos"], cdi8)
    m8_financiado = calcular_metricas(res8_financiado["retornos"], cdi8)

    # ================================================================
    print("\n" + "=" * 80)
    print("RESULTADO")
    print("=" * 80)
    print(f"  {'':<38}{'CAGR':>8}{'Vol':>8}{'Sharpe':>9}{'t':>7}{'Max DD':>9}")
    print("  " + "-" * 78)

    def linha(nome, m):
        t = m["sharpe"] * np.sqrt(m["anos"])
        print(f"  {nome:<38}{m['cagr']:>8.1%}{m['vol']:>8.1%}"
              f"{m['sharpe']:>9.2f}{t:>7.2f}{m['max_drawdown']:>9.1%}")
        return t

    print("  referências (8 ativos, painel oficial atual)")
    linha("  overlay", m8_overlay)
    linha("  ETFs financiados a CDI", m8_financiado)
    print("\n  universo expandido")
    linha("  SEM carrego (o resultado anterior)", m_velho)
    linha("  overlay com proxy de carry", m_novo)
    linha("  ETFs financiados a CDI", m_financiado)

    queda = m_novo["sharpe"] - m_velho["sharpe"]
    impacto_financiamento = m_financiado["sharpe"] - m_novo["sharpe"]
    print(f"\n  Impacto do proxy de carry: {queda:+.2f} de Sharpe")
    print(f"  Impacto do financiamento dos ETFs: "
          f"{impacto_financiamento:+.2f} de Sharpe")
    print("  O overlay é preservado como histórico; o cenário financiado")
    print("  ainda usa retornos em moeda de origem. Nenhum é implementável em reais.")

    # --- contribuição da lira, agora ------------------------------------
    pesos = res_financiado["pesos_diarios"].reindex(
        res_financiado["retornos"].index
    ).fillna(0.0)
    retornos_contrib = r_novo.reindex(
        res_financiado["retornos"].index
    )[sel_novo].fillna(0.0)
    contrib = pesos * retornos_contrib
    cdi_contrib = cdi.reindex(contrib.index).fillna(0.0)
    contrib.loc[:, sorted(etfs)] = contrib[sorted(etfs)].sub(
        pesos[sorted(etfs)].mul(cdi_contrib, axis=0)
    )
    anos = (
        res_financiado["retornos"].index[-1]
        - res_financiado["retornos"].index[0]
    ).days / 365.25
    c_ano = (contrib.sum() / anos).sort_values(ascending=False)

    print("\n" + "=" * 80)
    print("CONTRIBUIÇÃO POR ATIVO — a concentração sumiu?")
    print("=" * 80)
    print("  5 maiores:")
    for a, v in c_ano.head(5).items():
        print(f"    {a:<12}{v:>+8.2%}")
    print("  5 menores:")
    for a, v in c_ano.tail(5).items():
        print(f"    {a:<12}{v:>+8.2%}")
    if "TRY=X" in c_ano.index:
        pos = list(c_ano.index).index("TRY=X") + 1
        print(f"\n  TRY=X: {c_ano['TRY=X']:+.2%} a.a. — {pos}º de {len(c_ano)}")
        print(f"  (antes era +1,92% a.a. e o 1º colocado)")

    exposicao_ausente = res_financiado["exposicao_retorno_ausente"]
    print("\n  QUALIDADE DE DADOS DURANTE POSIÇÕES")
    print(f"  Dias com posição e algum retorno ausente: "
          f"{int((exposicao_ausente > 0).sum())} "
          f"({(exposicao_ausente > 0).mean():.1%} dos dias)")
    print(f"  Maior exposição sem retorno observável: "
          f"{exposicao_ausente.max():.1%}")
    print("  Nesses dias o P&L do ativo é mantido em zero e a exposição é")
    print("  marcada para auditoria; isto não equivale a uma cotação observada.")

    # --- sub-períodos ----------------------------------------------------
    print("\n" + "=" * 80)
    print("SUB-PERÍODOS")
    print("=" * 80)
    r = res_financiado["retornos"]
    print(f"  {'período':<14}{'CAGR':>9}{'Sharpe':>9}{'CDI':>9}{'':>6}")
    print("  " + "-" * 47)
    for ini, fim in [("2005", "2010"), ("2011", "2015"), ("2016", "2020"), ("2021", "2026")]:
        sub = r.loc[ini:fim]
        if sub.empty:
            continue
        m = calcular_metricas(sub, cdi)
        c = calcular_metricas(cdi.reindex(sub.index).fillna(0.0), cdi)
        marca = "ok" if m["cagr"] > c["cagr"] else "--"
        print(f"  {ini}-{fim:<9}{m['cagr']:>9.1%}{m['sharpe']:>9.2f}{c['cagr']:>9.1%}{marca:>6}")

    pd.DataFrame({"retorno": res_novo["retornos"]}).to_csv(
        AQUI / "resultados" / "serie_overlay_historico.csv"
    )
    pd.DataFrame({"retorno": r}).to_csv(AQUI / "resultados" / "serie_final.csv")
    c_ano.to_frame("contrib_anual").to_csv(AQUI / "resultados" / "contribuicao_final.csv")
    (AQUI / "dados" / "universo_final.txt").write_text("\n".join(sel_novo), encoding="utf-8")
    print("\n  Gravados: resultados/serie_overlay_historico.csv,")
    print("           resultados/serie_final.csv (stress de ETFs financiados),")
    print("           dados/universo_final.txt")


if __name__ == "__main__":
    main()
