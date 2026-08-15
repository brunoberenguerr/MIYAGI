# -*- coding: utf-8 -*-
"""
MIYAGI — figuras e tabelas do relatório
========================================

Gera tudo que o relatório precisa e que ainda não existia:

  1. dendrograma       a "árvore das apostas independentes" -- mostra quais
                       candidatos foram agrupados e por quê
  2. matriz            heatmap de correlação dos 40 selecionados
  3. patrimônio        curva do universo de 40 vs. 8 vs. Ibovespa vs. CDI
  4. por ativo         quanto cada um dos 40 somou ao resultado
  5. ordens            todas as ordens de compra e venda, em CSV
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

from backtest_miyagi import (DIAS_UTEIS_ANO, calcular_metricas,
                             calcular_retornos, carregar_dados,
                             rodar_backtest)
from backtest_expandido import carregar_expandido
from funil_expandido import (CORTE_CLUSTER, correlacao_pareada, elegiveis,
                             retornos_mensais)
from dados_miyagi import carregar_pool_oficial, selecionar_etfs

AQUI = Path(__file__).resolve().parent
SAIDA = AQUI / "resultados"
NAVY, CINZA, VERM, VERDE = "#1B4965", "#8D99AE", "#C1121F", "#1A5E39"


def fig_dendrograma():
    """A árvore que transformou 101 candidatos em 40 apostas independentes."""
    precos = carregar_pool_oficial()
    apt = elegiveis(precos)
    corr = correlacao_pareada(retornos_mensais(precos[apt]))

    D = 1 - corr.to_numpy()
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method="average")

    fig, ax = plt.subplots(figsize=(14, 8))
    dendrogram(Z, labels=apt, ax=ax, color_threshold=CORTE_CLUSTER,
               leaf_font_size=7.5, above_threshold_color=CINZA)
    ax.axhline(CORTE_CLUSTER, color=VERM, ls="--", lw=1.4)
    ax.text(0.995, CORTE_CLUSTER + 0.012,
            f"  corte em {CORTE_CLUSTER}  (junta o que tem correlação > "
            f"{1 - CORTE_CLUSTER:.2f})",
            color=VERM, fontsize=9, ha="right", transform=ax.get_yaxis_transform())
    ax.set_ylabel("distância  =  1 − correlação")
    ax.set_title("A árvore das apostas independentes — 101 candidatos elegíveis\n"
                 "Cada cor é um cluster; abaixo do corte, os ativos andam juntos "
                 "demais para valerem como apostas separadas",
                 fontsize=12, fontweight="bold", color=NAVY, loc="left")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    fig.tight_layout()
    fig.savefig(SAIDA / "dendrograma.png", dpi=150)
    plt.close(fig)
    print("  dendrograma.png")
    return corr


def fig_matriz(corr: pd.DataFrame, universo: list[str]):
    """Heatmap dos 40 selecionados, ordenado pelo clustering."""
    sub = corr.loc[universo, universo]
    D = 1 - sub.to_numpy()
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method="average")
    ordem = [universo[i] for i in dendrogram(Z, no_plot=True)["leaves"]]
    Co = sub.loc[ordem, ordem]
    n = len(ordem)

    fig, ax = plt.subplots(figsize=(12.5, 10.5))
    im = ax.imshow(Co.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(ordem, rotation=90, fontsize=7.5)
    ax.set_yticklabels(ordem, fontsize=7.5)
    for i in range(n):
        for j in range(n):
            v = Co.to_numpy()[i, j]
            if abs(v) >= 0.20 and i != j:
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                        fontsize=5.5, color="white" if abs(v) > 0.55 else "black")

    iu = np.triu_indices(n, k=1)
    rho = float(np.nanmean(Co.to_numpy()[iu]))
    n_eff = n / (1 + (n - 1) * rho)
    ax.set_title(f"Correlação entre os 40 ativos selecionados\n"
                 f"ρ médio = {rho:.3f}  →  {n_eff:.1f} apostas efetivas de {n} ativos"
                 f"   (números omitidos quando |ρ| < 0,20)",
                 fontsize=12, fontweight="bold", color=NAVY, loc="left")
    fig.colorbar(im, shrink=0.6, label="correlação")
    fig.tight_layout()
    fig.savefig(SAIDA / "matriz_correlacao_40.png", dpi=150)
    plt.close(fig)
    print("  matriz_correlacao_40.png")


def fig_patrimonio(r40, r8, ibov, cdi):
    """Curva do stress financiado: 40 ativos vs 8 vs referências."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for inicio, fim, rot in [("2008-01-01", "2009-03-31", "crise 2008"),
                             ("2020-02-01", "2020-06-30", "COVID"),
                             ("2021-06-01", "2022-12-31", "choque inflacionário")]:
        ax.axvspan(pd.Timestamp(inicio), pd.Timestamp(fim), color="#000", alpha=0.05)
        ax.text(pd.Timestamp(inicio), 0.02, f" {rot}", transform=ax.get_xaxis_transform(),
                fontsize=7.5, color="#555", va="bottom")

    for serie, cor, lw, rot in [((1 + r40).cumprod(), NAVY, 2.1, "MIYAGI · 40 (stress ETFs)"),
                                ((1 + r8).cumprod(), "#5B8CA8", 1.5, "MIYAGI · 8 (stress ETFs)"),
                                ((1 + ibov).cumprod(), CINZA, 1.3, "Ibovespa"),
                                ((1 + cdi).cumprod(), VERM, 1.3, "CDI")]:
        ax.plot(serie.index, serie, color=cor, lw=lw, label=rot,
                ls="--" if rot == "CDI" else "-")

    ax.set_yscale("log")
    ax.set_ylabel("patrimônio (escala log, base 1)")
    ax.set_title("Patrimônio acumulado — ETFs financiados a CDI\n"
                 "Stress contábil; retornos ainda nas moedas de origem",
                 fontsize=12, fontweight="bold", color=NAVY, loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(alpha=0.22, lw=0.6)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    fig.tight_layout()
    fig.savefig(SAIDA / "patrimonio_40.png", dpi=150)
    plt.close(fig)
    print("  patrimonio_40.png")


def fig_por_ativo(contrib: pd.Series):
    """Quanto cada ativo somou (ou tirou) do resultado, anualizado."""
    c = contrib.sort_values()
    cores = [VERM if v < 0 else NAVY for v in c.values]

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(range(len(c)), c.values * 100, color=cores, height=0.72)
    ax.set_yticks(range(len(c)))
    ax.set_yticklabels(c.index, fontsize=8.5)
    ax.axvline(0, color="#333", lw=0.9)
    ax.set_xlabel("contribuição anualizada para o retorno (%)")
    ax.set_title("Contribuição por ativo líquida do funding dos ETFs\n"
                 f"{(c > 0).sum()} de {len(c)} ativos contribuíram positivamente",
                 fontsize=12, fontweight="bold", color=NAVY, loc="left")
    ax.grid(axis="x", alpha=0.22, lw=0.6)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    fig.tight_layout()
    fig.savefig(SAIDA / "desempenho_por_ativo.png", dpi=150)
    plt.close(fig)
    print("  desempenho_por_ativo.png")


def extrair_ordens(
    pesos_alvo: pd.DataFrame,
    pesos_antes_rebalanceamento: pd.DataFrame,
) -> pd.DataFrame:
    """Toda ordem efetiva entre o peso derivado e o novo alvo.

    Se o peso pré-trade derivado é +0,20 e o novo alvo é +0,35, houve compra de
    0,15. Comparar dois alvos mensais ignoraria a deriva entre eles e poderia
    publicar uma ordem diferente da usada para calcular giro e custo.
    """
    pre_trade = pesos_antes_rebalanceamento.reindex(
        index=pesos_alvo.index, columns=pesos_alvo.columns
    ).fillna(0.0)
    delta = pesos_alvo - pre_trade

    linhas = []
    for data, row in delta.iterrows():
        for ativo, mudanca in row.items():
            if abs(mudanca) < 1e-6:
                continue
            peso_novo = pesos_alvo.loc[data, ativo]
            linhas.append({
                "data": data.date(),
                "ativo": ativo,
                "lado": "COMPRA" if mudanca > 0 else "VENDA",
                "tamanho": round(abs(mudanca), 5),
                "peso_final": round(peso_novo, 5),
                "posicao": ("comprado" if peso_novo > 1e-6
                            else "vendido" if peso_novo < -1e-6 else "zerado"),
            })
    return pd.DataFrame(linhas)


def main() -> None:
    SAIDA.mkdir(exist_ok=True)
    print("Gerando figuras e tabelas do relatório...\n")

    precos, cdi, universo = carregar_expandido()
    retornos = calcular_retornos(precos)

    print("[1/5] árvore de clusters")
    corr = fig_dendrograma()

    print("[2/5] matriz de correlação")
    fig_matriz(corr, [a for a in universo if a in corr.index])

    print("[3/5] rodando o backtest de 40 ativos")
    etfs = selecionar_etfs(universo)
    res = rodar_backtest(
        precos, retornos, cdi, ativos_financiados=etfs
    )
    r40 = res["retornos"]

    # --- referências ---
    ibov = pd.read_csv(AQUI / "dados" / "prices.csv", index_col=0, parse_dates=True)
    ibov = calcular_retornos(ibov)["^BVSP"].reindex(r40.index).fillna(0.0)
    cdi_al = cdi.reindex(r40.index).fillna(0.0)
    p8, cdi8 = carregar_dados()
    etfs8 = selecionar_etfs(list(p8.columns))
    r8 = rodar_backtest(
        p8, calcular_retornos(p8), cdi8, ativos_financiados=etfs8
    )["retornos"]
    r8 = r8.reindex(r40.index).fillna(0.0)

    print("[4/5] figuras de resultado")
    fig_patrimonio(r40, r8, ibov, cdi_al)

    pesos_diarios = res["pesos_diarios"].reindex(r40.index).fillna(0.0)
    anos = (r40.index[-1] - r40.index[0]).days / 365.25
    contribuicoes = pesos_diarios * retornos.reindex(r40.index)[universo].fillna(0.0)
    contribuicoes.loc[:, sorted(etfs)] = contribuicoes[sorted(etfs)].sub(
        pesos_diarios[sorted(etfs)].mul(cdi_al, axis=0)
    )
    contrib = contribuicoes.sum() / anos
    fig_por_ativo(contrib)

    print("[5/5] extraindo ordens")
    giro_reconstruido = (
        res["pesos"] - res["pesos_antes_rebalanceamento"]
    ).abs().sum(axis=1)
    if not np.allclose(
        giro_reconstruido.to_numpy(), res["giro"].to_numpy(),
        rtol=0.0, atol=1e-12,
    ):
        raise RuntimeError(
            "Ordens publicadas não reconciliam com o giro cobrado pelo motor."
        )
    ordens = extrair_ordens(
        res["pesos"], res["pesos_antes_rebalanceamento"]
    )
    ordens.to_csv(SAIDA / "ordens_completas.csv", index=False)

    # --- sumários que cabem num relatório --------------------------------
    print(f"\n{'=' * 70}")
    print(f"ORDENS GERADAS: {len(ordens):,}".replace(",", "."))
    print(f"{'=' * 70}")
    print(f"  compras: {(ordens.lado == 'COMPRA').sum():,}".replace(",", "."))
    print(f"  vendas : {(ordens.lado == 'VENDA').sum():,}".replace(",", "."))
    print(f"  rebalanceamentos: {len(res['pesos'])}")
    print(f"  média por rebalanceamento: {len(ordens) / len(res['pesos']):.1f}")

    por_ativo = (ordens.groupby("ativo")
                 .agg(ordens=("lado", "size"), volume=("tamanho", "sum"))
                 .sort_values("ordens", ascending=False))
    por_ativo.to_csv(SAIDA / "ordens_por_ativo.csv")
    print(f"\n  Top 10 ativos por número de ordens:")
    print(por_ativo.head(10).to_string())

    ultima = res["pesos"].index[-1]
    carteira = res["pesos"].loc[ultima]
    carteira = carteira[carteira.abs() > 1e-6].sort_values(ascending=False)
    carteira.to_frame("peso").to_csv(SAIDA / "carteira_atual.csv")
    print(f"\n  CARTEIRA NA ÚLTIMA DATA ({ultima:%Y-%m-%d}) — {len(carteira)} posições")
    print(f"    comprado em {(carteira > 0).sum()}, vendido em {(carteira < 0).sum()}")
    print(f"    exposição bruta: {carteira.abs().sum():.2f}x")

    contrib.sort_values(ascending=False).to_frame("contrib_anual").to_csv(
        SAIDA / "contribuicao_por_ativo.csv")

    print(f"\nArquivos gravados em resultados/:")
    for f in ["dendrograma.png", "matriz_correlacao_40.png", "patrimonio_40.png",
              "desempenho_por_ativo.png", "ordens_completas.csv",
              "ordens_por_ativo.csv", "carteira_atual.csv",
              "contribuicao_por_ativo.csv"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
