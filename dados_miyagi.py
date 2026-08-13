# -*- coding: utf-8 -*-
"""Fonte única de dados e regras de qualidade do MIYAGI.

Este módulo existe para impedir que scripts diferentes rodem, sem perceber,
versões econômicas diferentes do mesmo painel. O painel oficial é o que contém
a correção de carrego cambial já produzida pelo projeto.

Nenhuma lacuna é preenchida sem limite. Buracos de até ``MAX_FFILL_DIAS``
podem representar calendários de negociação distintos e recebem o último
preço observado. Lacunas maiores permanecem ausentes e são reportadas.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
PAINEL_OFICIAL = AQUI / "dados" / "pool_carrego.csv"
UNIVERSO_OFICIAL = AQUI / "dados" / "universo_final.txt"
CDI_OFICIAL = AQUI / "dados" / "cdi.csv"
MAX_FFILL_DIAS = 5


def carregar_cdi() -> pd.Series:
    """Carrega o CDI diário em fração decimal."""
    cdi = pd.read_csv(CDI_OFICIAL, index_col=0, parse_dates=True)
    return (cdi.iloc[:, 0].sort_index() / 100.0).rename("cdi")


def carregar_pool_oficial() -> pd.DataFrame:
    """Carrega o painel com carrego corrigido, sem alterar seus dados brutos."""
    return pd.read_csv(PAINEL_OFICIAL, index_col=0, parse_dates=True).sort_index()


def carregar_universo_oficial() -> list[str]:
    """Lê a lista oficial e verifica se todos os símbolos existem no painel."""
    universo = UNIVERSO_OFICIAL.read_text(encoding="utf-8").split()
    colunas = set(carregar_pool_oficial().columns)
    ausentes = sorted(set(universo) - colunas)
    if ausentes:
        raise ValueError(f"Ativos do universo ausentes no painel oficial: {ausentes}")
    return universo


def alinhar_ao_calendario(
    precos: pd.DataFrame,
    calendario: pd.DatetimeIndex,
    limite: int = MAX_FFILL_DIAS,
) -> pd.DataFrame:
    """Alinha preços ao calendário do CDI com preenchimento curto e explícito."""
    return precos.ffill(limit=limite).reindex(calendario).ffill(limit=limite)


def carregar_dados_oficiais() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Devolve preços, CDI e universo usados por todo resultado oficial."""
    pool = carregar_pool_oficial()
    cdi = carregar_cdi()
    universo = carregar_universo_oficial()
    precos = alinhar_ao_calendario(pool[universo], cdi.index)
    cdi = cdi.reindex(precos.index).ffill().fillna(0.0)
    return precos, cdi, universo


def auditar_lacunas(precos: pd.DataFrame, limite: int = MAX_FFILL_DIAS) -> pd.DataFrame:
    """Lista lacunas internas maiores que o limite, sem fabricar observações.

    Lacunas antes da primeira ou depois da última observação não entram: são
    simplesmente períodos em que o instrumento ainda não existia ou deixou de
    estar disponível. O relatório trata apenas buracos dentro de sua cobertura.
    """
    linhas: list[dict] = []
    for ativo in precos.columns:
        serie = precos[ativo]
        primeira, ultima = serie.first_valid_index(), serie.last_valid_index()
        if primeira is None or ultima is None:
            continue
        interna = serie.loc[primeira:ultima]
        grupos = interna.isna().ne(interna.isna().shift()).cumsum()
        for _, trecho in interna.groupby(grupos):
            if not trecho.isna().all() or len(trecho) <= limite:
                continue
            linhas.append({
                "ativo": ativo,
                "inicio": trecho.index[0],
                "fim": trecho.index[-1],
                "dias_calendario_cdi": len(trecho),
            })
    colunas = ["ativo", "inicio", "fim", "dias_calendario_cdi"]
    return pd.DataFrame(linhas, columns=colunas).sort_values(
        ["dias_calendario_cdi", "ativo"], ascending=[False, True]
    ).reset_index(drop=True)
