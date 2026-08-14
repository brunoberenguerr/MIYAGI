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
                          carregar_pool_oficial, eh_etf_adjusted_close,
                          selecionar_etfs)

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
        tipo = "fx_retorno_total_proxy" if corrigido_por_carrego else "fx_spot"
        conclusao = ("proxy de juros; falta vintage/lag de publicação"
                     if corrigido_por_carrego else "carrego cambial ausente")
    elif ativo.startswith("^"):
        tipo = "indice_de_preco"
        conclusao = "dividendos e conversão cambial não modelados"
    elif eh_etf_adjusted_close(ativo):
        tipo = "etf_adjusted_close"
        conclusao = "retorno total financiado; não é retorno excedente de futuro"
    else:  # defesa para uma futura família que não obedeça à convenção atual
        tipo = "nao_classificado"
        conclusao = "exige metadado econômico explícito"
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
    instrumentos.to_csv(
        SAIDA / "auditoria_instrumentos.csv", index=False,
        float_format="%.12g",
    )

    # Audita exatamente o painel que chega ao motor. No índice bruto do Yahoo,
    # a contagem de linhas não equivale a dias do calendário CDI; após o
    # alinhamento, cada linha é uma data oficial e todo NaN é uma lacuna que o
    # preenchimento permitido não resolveu.
    lacunas = auditar_lacunas(precos, limite=0)
    lacunas.to_csv(
        SAIDA / "auditoria_lacunas.csv", index=False,
        float_format="%.12g",
    )

    retornos = calcular_retornos(precos)
    etfs = selecionar_etfs(universo)
    res_overlay = rodar_backtest(precos, retornos, cdi)
    res = rodar_backtest(
        precos, retornos, cdi, ativos_financiados=etfs
    )
    pesos = res["pesos_diarios"].reindex(retornos.index).fillna(0.0)
    exposicao_ausente = pesos.abs().where(retornos.isna(), 0.0)
    por_ativo = pd.DataFrame({
        "dias_com_posicao_e_retorno_ausente": (exposicao_ausente > 0).sum(),
        "exposicao_abs_media_nesses_dias": exposicao_ausente.where(
            exposicao_ausente > 0
        ).mean(),
        "exposicao_abs_maxima": exposicao_ausente.max(),
    }).sort_values("dias_com_posicao_e_retorno_ausente", ascending=False)
    por_ativo.to_csv(
        SAIDA / "auditoria_exposicao_sem_retorno.csv",
        float_format="%.12g",
    )

    dias_overlay = res_overlay["exposicao_retorno_ausente"]
    dias = res["exposicao_retorno_ausente"]
    resumo = pd.Series({
        "painel_oficial": "dados/pool_carrego.csv",
        "universo_oficial": "dados/universo_final.txt",
        "ativos": len(universo),
        "lacunas_internas_apos_ffill_limitado": len(lacunas),
        "dias_com_posicao_e_algum_retorno_ausente": int((dias_overlay > 0).sum()),
        "fracao_dos_dias_com_exposicao_ausente": float((dias_overlay > 0).mean()),
        "exposicao_abs_media_quando_ausente": float(
            dias_overlay[dias_overlay > 0].mean()
        ),
        "exposicao_abs_maxima_quando_ausente": float(dias_overlay.max()),
        "exposicao_abs_maxima_etfs_financiados": float(dias.max()),
        "convencao_exposicao_por_ativo": "etfs_financiados",
    }, name="valor")
    resumo.map(
        lambda valor: f"{valor:.12g}"
        if isinstance(valor, (float, np.floating)) else valor
    ).to_csv(SAIDA / "auditoria_dados_resumo.csv")

    arquivos = [
        AQUI / "dados" / "pool_carrego.csv",
        AQUI / "dados" / "pool_expandido.csv",
        AQUI / "dados" / "cdi.csv",
        AQUI / "dados" / "universo_final.txt",
    ]
    proveniencia = []
    for caminho in arquivos:
        # Git pode materializar o mesmo arquivo como LF ou CRLF. O hash de
        # auditoria deve identificar o conteúdo, não a convenção do sistema.
        conteudo = caminho.read_bytes()
        conteudo_canonico = conteudo.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        proveniencia.append({
            "arquivo": caminho.relative_to(AQUI).as_posix(),
            "bytes_canonicos_lf": len(conteudo_canonico),
            "sha256_canonico_lf": hashlib.sha256(conteudo_canonico).hexdigest(),
        })
    pd.DataFrame(proveniencia).to_csv(
        SAIDA / "auditoria_proveniencia_arquivos.csv", index=False
    )
    pd.Series({
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }, name="versao").to_csv(SAIDA / "auditoria_ambiente_local.csv")

    print(resumo.to_string())
    print("\nTipos econômicos no universo:")
    print(instrumentos.groupby("tipo_serie").size().to_string())
    print("\nCONCLUSÃO: o painel continua sendo um proxy de pesquisa, não uma série")
    print("homogênea de retornos excedentes de futuros. Os CSVs acima tornam a")
    print("limitação mensurável; nenhum dado ausente foi substituído por estimativa.")


if __name__ == "__main__":
    main()
