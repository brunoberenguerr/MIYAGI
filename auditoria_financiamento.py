# -*- coding: utf-8 -*-
"""Quantifica a contabilidade de caixa no painel heterogêneo do MIYAGI.

O motor histórico soma CDI uma vez ao patrimônio e trata todas as pernas
arriscadas como overlays de futuros. Isso é coerente para retornos excedentes,
mas não para os ``Adjusted Close`` de ETFs, que já são retornos totais de
instrumentos financiados.

Este arquivo não escolhe o cenário com melhor desempenho. Ele mantém sinais,
volatilidades, universo e parâmetros e altera somente a contabilidade do caixa:

1. ``overlay_atual``: estimando histórico, sem encargo por ativo;
2. ``etfs_financiados``: desconta CDI da exposição líquida nos 11 ETFs;
3. ``etfs_indices_financiados``: stress adicional que trata também índices de
   preço como pernas financiadas. Não é uma correção completa, pois continuam
   faltando dividendos, FX, borrow, margem e instrumentos negociáveis.

O segundo cenário mede a dupla contagem nos ETFs *dentro da convenção atual de
numerário*: todas as pernas são financiadas a CDI, apesar de seus retornos ainda
estarem nas moedas de origem. Portanto ele é um stress contábil reproduzível,
não um retorno implementável em reais. O terceiro é apenas um limite de
sensibilidade e tampouco deve ser chamado de resultado correto.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest_miyagi import calcular_metricas, calcular_retornos, rodar_backtest
from dados_miyagi import AQUI, carregar_dados_oficiais, selecionar_etfs

SAIDA = AQUI / "resultados"


def classificar_ativos(universo: list[str]) -> tuple[list[str], list[str]]:
    """Separa ETFs e índices somente por convenções verificáveis dos tickers."""
    indices = sorted(a for a in universo if a.startswith("^"))
    etfs = sorted(selecionar_etfs(universo))
    return etfs, indices


def main() -> None:
    SAIDA.mkdir(exist_ok=True)
    precos, cdi, universo = carregar_dados_oficiais()
    retornos = calcular_retornos(precos)
    etfs, indices = classificar_ativos(universo)

    cenarios = [
        ("overlay_atual", set()),
        ("etfs_financiados", set(etfs)),
        ("etfs_indices_financiados", set(etfs + indices)),
    ]
    linhas_metricas: list[dict] = []
    linhas_exposicao: list[dict] = []

    for nome, financiados in cenarios:
        res = rodar_backtest(
            precos,
            retornos,
            cdi,
            ativos_financiados=financiados,
        )
        metricas = calcular_metricas(res["retornos"], cdi)
        anos = metricas["anos"]
        encargo_anual = float(res["encargo_financiamento"].sum() / anos)
        pesos = res["pesos_diarios"].reindex(res["retornos"].index).fillna(0.0)
        cols = sorted(financiados)
        liquida = pesos[cols].sum(axis=1) if cols else pd.Series(0.0, index=pesos.index)
        bruta = pesos[cols].abs().sum(axis=1) if cols else pd.Series(0.0, index=pesos.index)

        linhas_metricas.append({
            "cenario": nome,
            "ativos_financiados": len(financiados),
            "cagr": metricas["cagr"],
            "vol": metricas["vol"],
            "sharpe": metricas["sharpe"],
            "max_drawdown": metricas["max_drawdown"],
            "encargo_financiamento_anual_aritmetico": encargo_anual,
        })
        linhas_exposicao.append({
            "cenario": nome,
            "exposicao_liquida_media": float(liquida.mean()),
            "exposicao_liquida_minima": float(liquida.min()),
            "exposicao_liquida_maxima": float(liquida.max()),
            "exposicao_bruta_media": float(bruta.mean()),
            "exposicao_bruta_maxima": float(bruta.max()),
        })

    tabela = pd.DataFrame(linhas_metricas)
    exposicoes = pd.DataFrame(linhas_exposicao)
    tabela.to_csv(
        SAIDA / "auditoria_financiamento_metricas.csv", index=False,
        float_format="%.12g",
    )
    exposicoes.to_csv(
        SAIDA / "auditoria_financiamento_exposicoes.csv", index=False,
        float_format="%.12g",
    )

    print("MIYAGI — AUDITORIA DE FINANCIAMENTO")
    print("=" * 72)
    print(f"ETFs adjusted close: {len(etfs)}")
    print(f"Índices de preço:     {len(indices)}")
    print("\nMétricas por convenção:")
    print(tabela.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nExposição às pernas financiadas:")
    print(exposicoes.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nINTERPRETAÇÃO")
    print("- etfs_financiados isola a dupla contagem do CDI dentro da")
    print("  convenção simplificada de numerário usada pelo projeto;")
    print("- etfs_indices_financiados é stress, não correção completa;")
    print("- exposição líquida, e não a alavancagem bruta total, determina o")
    print("  encargo nesta convenção simplificada;")
    print("- sinais e universo permanecem fixos para não misturar financiamento")
    print("  com uma nova seleção pós-resultado.")
    print("- como os retornos seguem nas moedas de origem, nenhum cenário aqui")
    print("  deve ser apresentado como retorno implementável em reais.")


if __name__ == "__main__":
    main()
