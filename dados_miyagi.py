# -*- coding: utf-8 -*-
"""Fonte única de dados e regras de qualidade do MIYAGI.

Este módulo existe para impedir que scripts diferentes rodem, sem perceber,
versões econômicas diferentes do mesmo painel. O painel oficial é o que contém
o proxy de carrego cambial já produzido pelo projeto; ele não substitui forwards
nem séries vintage de taxas.

Nenhuma lacuna é preenchida sem limite. Buracos de até ``MAX_FFILL_DIAS``
podem representar calendários de negociação distintos e recebem o último
preço observado. Lacunas maiores permanecem ausentes e são reportadas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
PAINEL_OFICIAL = AQUI / "dados" / "pool_carrego.csv"
UNIVERSO_OFICIAL = AQUI / "dados" / "universo_final.txt"
CDI_OFICIAL = AQUI / "dados" / "cdi.csv"
MAX_FFILL_DIAS = 5


def eh_etf_adjusted_close(ativo: str) -> bool:
    """Identifica os tickers baixados como ETFs com ``auto_adjust=True``.

    As outras famílias têm marcadores explícitos no Yahoo: futuros ``=F``, FX
    ``=X`` e índices ``^``. Centralizar a regra impede que scripts de resultado
    e auditoria discordem sobre quais pernas exigem financiamento.
    """
    return not (
        ativo.endswith("=F") or ativo.endswith("=X") or ativo.startswith("^")
    )


def selecionar_etfs(ativos: list[str] | pd.Index) -> set[str]:
    """Devolve a família financiada de ETFs para uma lista de instrumentos."""
    return {ativo for ativo in ativos if eh_etf_adjusted_close(str(ativo))}


def carregar_cdi() -> pd.Series:
    """Carrega o CDI diário em fração decimal."""
    cdi = pd.read_csv(CDI_OFICIAL, index_col=0, parse_dates=True)
    return (cdi.iloc[:, 0].sort_index() / 100.0).rename("cdi")


def carregar_pool_oficial() -> pd.DataFrame:
    """Carrega o painel com proxy de carry, sem alterar seus dados brutos."""
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
    """Alinha ao CDI sem renovar artificialmente a idade de uma cotação.

    O limite conta datas do calendário oficial posteriores à última cotação
    *observada*. Uma implementação em duas chamadas de ``ffill(limit=...)``
    permitiria que valores já imputados fossem preenchidos outra vez, dobrando
    silenciosamente o horizonte máximo. A união dos índices preserva uma
    observação legítima em um dia fora do CDI (por exemplo, FX no domingo), mas
    ela nunca é confundida com uma nova observação durante o preenchimento.
    """
    if limite < 0:
        raise ValueError("limite de preenchimento deve ser não negativo")

    calendario = pd.DatetimeIndex(calendario).sort_values().unique()
    uniao = precos.index.union(calendario).sort_values()
    saida = pd.DataFrame(index=calendario, columns=precos.columns, dtype=float)
    datas_calendario = calendario.to_numpy()
    posicoes_atuais = np.arange(len(calendario))

    for ativo in precos.columns:
        serie = precos[ativo].reindex(uniao)
        observado = serie.notna()
        ultima_observacao = pd.Series(pd.NaT, index=uniao, dtype="datetime64[ns]")
        ultima_observacao.loc[observado] = uniao[observado]
        ultima_observacao = ultima_observacao.ffill().reindex(calendario)

        valores = serie.ffill().reindex(calendario)
        tem_observacao = ultima_observacao.notna().to_numpy()
        insercao = np.zeros(len(calendario), dtype=int)
        insercao[tem_observacao] = np.searchsorted(
            datas_calendario,
            ultima_observacao.to_numpy()[tem_observacao],
            side="right",
        )
        idade_no_calendario = posicoes_atuais - insercao + 1
        dentro_do_limite = tem_observacao & (idade_no_calendario <= limite)
        saida[ativo] = valores.where(dentro_do_limite)

    return saida


def carregar_dados_oficiais() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Devolve preços, CDI e universo usados pelos cenários auditados."""
    pool = carregar_pool_oficial()
    cdi = carregar_cdi()
    universo = carregar_universo_oficial()
    precos = alinhar_ao_calendario(pool[universo], cdi.index)
    cdi = cdi.reindex(precos.index).ffill().fillna(0.0)
    return precos, cdi, universo


def auditar_lacunas(precos: pd.DataFrame, limite: int = MAX_FFILL_DIAS) -> pd.DataFrame:
    """Lista sequências internas de NaN maiores que ``limite``.

    Lacunas antes da primeira ou depois da última observação não entram: são
    simplesmente períodos em que o instrumento ainda não existia ou deixou de
    estar disponível. Quando recebe o painel já alinhado e ``limite=0``, a
    função reporta cada lacuna que restou depois do preenchimento permitido,
    em datas do calendário oficial do CDI.
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
