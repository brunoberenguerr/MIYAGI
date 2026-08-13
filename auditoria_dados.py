# -*- coding: utf-8 -*-
"""Auditoria reproduzível de linhagem, lacunas e economia dos instrumentos."""

from __future__ import annotations

from pathlib import Path
import hashlib
import platform

import pandas as pd
import numpy as np
import scipy

from backtest_miyagi import calcular_retornos, rodar_backtest
from dados_miyagi import (AQUI, auditar_lacunas, carregar_dados_oficiais,
                          carregar_pool_oficial)

SAIDA = AQUI / "resultados"


def classificar_instrumento(ativo: str, corrigido_por_carrego: bool) -> dict:
    """Classifica apenas pelo identificador e pela diferença observada nos CSVs.

    A coluna ``conclusao`` não presume uma implementação inexistente: ela diz
    exatamente o que ainda impede tratar a série como retorno excedente de um
    futuro com collateral em CDI.
    """
    if ativo.endswith("=F"):
        tipo = "futuro_continuo_yahoo"
        conclusao = "roll e custo de rolagem não identificados"
    elif ativo.endswith("=X"):
        tipo = "fx_retorno_total" if corrigido_por_carrego else "fx_spot"
        conclusao = ("proxy de juros; falta vintage/lag de publicação"
                     if corrigido_por_carrego else "carrego cambial ausente")
    elif ativo.startswith("^"):
        tipo = "indice_de_preco"
        conclusao = "dividendos e conversão cambial não modelados"
    else:
        tipo = "etf_adjusted_close"
        conclusao = "retorno total financiado; não é retorno excedente de futuro"
    return {"ativo": ativo, "tipo_serie": tipo, "conclusao": conclusao}


def main() -> None:
    SAIDA.mkdir(exist_ok=True)
    precos, cdi, universo = carregar_dados_oficiais()
    pool_corrigido = carregar_pool_oficial()
    pool_spot = pd.read_csv(AQUI / "dados" / "pool_expandido.csv",
                            index_col=0, parse_dates=True).sort_index()

    diferenca = (pool_corrigido[universo] - pool_spot[universo]).abs()
    carrego_alterou = diferenca.fillna(0.0).max() > 1e-12

    instrumentos = pd.DataFrame([
        classificar_instrumento(a, bool(carrego_alterou.get(a, False)))
        for a in universo
    ]).sort_values(["tipo_serie", "ativo"])
    instrumentos.to_csv(SAIDA / "auditoria_instrumentos.csv", index=False)

    lacunas = auditar_lacunas(pool_corrigido[universo])
    lacunas.to_csv(SAIDA / "auditoria_lacunas.csv", index=False)

    retornos = calcular_retornos(precos)
    res = rodar_backtest(precos, retornos, cdi)
    pesos = res["pesos_diarios"].reindex(retornos.index).fillna(0.0)
    exposicao_ausente = pesos.abs().where(retornos.isna(), 0.0)
    por_ativo = pd.DataFrame({
        "dias_com_posicao_e_retorno_ausente": (exposicao_ausente > 0).sum(),
        "exposicao_abs_media_nesses_dias": exposicao_ausente.where(
            exposicao_ausente > 0
        ).mean(),
        "exposicao_abs_maxima": exposicao_ausente.max(),
    }).sort_values("dias_com_posicao_e_retorno_ausente", ascending=False)
    por_ativo.to_csv(SAIDA / "auditoria_exposicao_sem_retorno.csv")

    dias = res["exposicao_retorno_ausente"]
    resumo = pd.Series({
        "painel_oficial": "dados/pool_carrego.csv",
        "universo_oficial": "dados/universo_final.txt",
        "ativos": len(universo),
        "lacunas_internas_maiores_5_dias": len(lacunas),
        "dias_com_posicao_e_algum_retorno_ausente": int((dias > 0).sum()),
        "fracao_dos_dias_com_exposicao_ausente": float((dias > 0).mean()),
        "exposicao_abs_media_quando_ausente": float(dias[dias > 0].mean()),
        "exposicao_abs_maxima_quando_ausente": float(dias.max()),
    }, name="valor")
    resumo.to_csv(SAIDA / "auditoria_dados_resumo.csv")

    arquivos = [
        AQUI / "dados" / "pool_carrego.csv",
        AQUI / "dados" / "pool_expandido.csv",
        AQUI / "dados" / "cdi.csv",
        AQUI / "dados" / "universo_final.txt",
    ]
    proveniencia = []
    for caminho in arquivos:
        proveniencia.append({
            "arquivo": str(caminho.relative_to(AQUI)),
            "bytes": caminho.stat().st_size,
            "sha256": hashlib.sha256(caminho.read_bytes()).hexdigest(),
        })
    pd.DataFrame(proveniencia).to_csv(
        SAIDA / "auditoria_proveniencia_arquivos.csv", index=False
    )
    pd.Series({
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }, name="versao").to_csv(SAIDA / "auditoria_ambiente.csv")

    print(resumo.to_string())
    print("\nTipos econômicos no universo:")
    print(instrumentos.groupby("tipo_serie").size().to_string())
    print("\nCONCLUSÃO: o painel continua sendo um proxy de pesquisa, não uma série")
    print("homogênea de retornos excedentes de futuros. Os CSVs acima tornam a")
    print("limitação mensurável; nenhum dado ausente foi substituído por estimativa.")


if __name__ == "__main__":
    main()
