# -*- coding: utf-8 -*-
"""
MIYAGI — proxy de carrego cambial (interest rate carry)
=======================================================

O ERRO QUE ESTE ARQUIVO CORRIGE
-------------------------------
O backtest usava APENAS a variação do preço à vista das moedas. Para câmbio,
isso está errado — e o erro é enorme em moedas de juro alto.

Para ficar comprado em USD/TRY (ou seja, vendido em lira), você precisa tomar
lira emprestada e PAGAR a taxa de juros turca. Durante o período essa taxa
chegou a 40-50% ao ano, contra 0-5,5% nos EUA.

O preço à vista da lira caiu 18% ao ano. Parecia lucro de 18% ao ano. Mas o
carrego que teríamos pago era da mesma ordem de grandeza — e, em vários anos,
MAIOR que a desvalorização.

A paridade COBERTA de juros liga o diferencial de juros ao preço a termo. A
paridade DESCOBERTA é uma hipótese sobre a desvalorização esperada e não uma
identidade; ela não é usada aqui como prova de retorno realizado.

O PROXY
-------
Para cada par de câmbio, constrói-se um índice de RETORNO TOTAL APROXIMADO:

    retorno_total  =  retorno_do_preço  +  carrego

onde, para um par cotado como USD/XXX (quantos XXX por dólar):

    carrego_diário  =  (juro_USD − juro_XXX) / 252

e para um par cotado como XXX/USD, o sinal se inverte.

O sinal de momentum passa a ler preço mais diferencial de taxas. Isso aproxima
a economia de um contrato a termo, mas não reconstrói forwards negociáveis nem
prova a paridade coberta com os retornos realizados.

FONTE DOS JUROS
---------------
FRED (Federal Reserve de St. Louis), séries do IMF/IFS. Cobertura verificada
antes do uso — ver a tabela impressa na execução.

LIMITAÇÃO DECLARADA
-------------------
Nem todo país tem série pública completa. Onde falta, o par fica SEM correção
e isso é reportado explicitamente. As moedas do G10 (AUD, CAD, GBP, SEK) têm
diferencial pequeno contra o dólar (0-5 p.p.), então o erro residual é de
ordem muito menor que o da lira — mas existe.

Taxas de política/redesconto são um proxy do custo real de financiamento, não
o custo exato. As séries públicas também não têm vintages nesta extração e são
alinhadas sem modelar o atraso de publicação. A ordem de grandeza é o que
importa aqui; o resultado não deve ser chamado de carry "correto".
"""

from __future__ import annotations

import io
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd

AQUI = Path(__file__).resolve().parent
DIAS_UTEIS_ANO = 252

# Séries do FRED, verificadas uma a uma quanto a período de cobertura.
JUROS = {
    "USD": ("DTB3", "T-bill 3 meses EUA"),
    "TRY": ("INTDSRTRM193N", "taxa de redesconto Turquia"),
    "BRL": ("INTDSRBRM193N", "taxa de redesconto Brasil"),
    "ZAR": ("INTGSTZAM193N", "T-bill Africa do Sul"),
    "MXN": ("INTGSTMXM193N", "T-bill Mexico"),
    "INR": ("INTDSRINM193N", "taxa de redesconto India"),
    "JPY": ("INTDSRJPM193N", "taxa de redesconto Japao"),
}

# Como cada par é cotado. "USD_BASE" significa USD/XXX (quantos XXX por dólar):
# comprar o par = comprar dólar e vender a moeda estrangeira.
PARES = {
    "TRY=X":    ("TRY", "USD_BASE"),
    "BRL=X":    ("BRL", "USD_BASE"),
    "ZAR=X":    ("ZAR", "USD_BASE"),
    "MXN=X":    ("MXN", "USD_BASE"),
    "INR=X":    ("INR", "USD_BASE"),
    "JPY=X":    ("JPY", "USD_BASE"),
    "CAD=X":    ("CAD", "USD_BASE"),      # sem dado -> fica sem correção
    "SEK=X":    ("SEK", "USD_BASE"),      # sem dado
    "GBPUSD=X": ("GBP", "USD_COTADA"),    # sem dado
    "AUDUSD=X": ("AUD", "USD_COTADA"),    # sem dado
    "EURUSD=X": ("EUR", "USD_COTADA"),    # sem dado
    "NZDUSD=X": ("NZD", "USD_COTADA"),    # sem dado
    "CHF=X":    ("CHF", "USD_BASE"),      # sem dado
    "NOK=X":    ("NOK", "USD_BASE"),      # sem dado
    "SGD=X":    ("SGD", "USD_BASE"),      # sem dado
    "PLN=X":    ("PLN", "USD_BASE"),      # sem dado
}


def busca_fred(serie: str) -> pd.Series | None:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={serie}"
    try:
        with urlopen(url, timeout=60) as resposta:  # noqa: S310 - domínio fixo FRED
            texto = resposta.read().decode("utf-8")
        if texto.startswith("<"):
            return None
        df = pd.read_csv(io.StringIO(texto))
        df.columns = ["data", "valor"]
        df["data"] = pd.to_datetime(df["data"])
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        s = df.dropna().set_index("data")["valor"].sort_index()
        return s if len(s) > 50 else None
    except (HTTPError, URLError, TimeoutError, ValueError, OSError):
        return None


def carrega_juros(calendario: pd.DatetimeIndex) -> tuple[dict, pd.DataFrame]:
    """Baixa as taxas e alinha ao calendário diário do backtest."""
    taxas, cobertura = {}, []
    for moeda, (serie, desc) in JUROS.items():
        s = busca_fred(serie)
        if s is None:
            cobertura.append({"moeda": moeda, "serie": serie, "status": "FALHOU"})
            continue
        fim_dado = s.index.max()
        # Reindexa para o calendário diário. ffill propaga o último valor
        # conhecido -- inclusive para além do fim da série, o que é registrado
        # como limitação abaixo.
        alinhada = s.reindex(calendario.union(s.index)).ffill().reindex(calendario)
        # Não fazemos bfill antes da primeira observação: isso aplicaria ao
        # passado uma taxa que só existiu depois. Datas sem dado permanecem NaN.
        taxas[moeda] = alinhada / 100.0                  # % a.a. -> fração
        dias_extrapolados = int((calendario > fim_dado).sum())
        cobertura.append({
            "moeda": moeda, "serie": serie, "status": "ok",
            "inicio": s.index.min().date(), "fim": fim_dado.date(),
            "dias_extrapolados": dias_extrapolados,
            "taxa_media": f"{s.mean():.1f}%",
        })
    return taxas, pd.DataFrame(cobertura)


def aplica_carrego(precos: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Substitui os preços por índices com proxy de retorno total.

    Devolve (painel com proxy, pares tratados, pares sem dado).
    """
    taxas, cobertura = carrega_juros(pd.DatetimeIndex(precos.index))

    print("COBERTURA DAS SÉRIES DE JUROS")
    print(cobertura.to_string(index=False))
    print()

    corrigidos, sem_dado = [], []
    saida = precos.copy()

    for par, (moeda, convencao) in PARES.items():
        if par not in precos.columns:
            continue
        if moeda not in taxas or "USD" not in taxas:
            sem_dado.append(par)
            continue

        r_est = taxas[moeda]
        r_usd = taxas["USD"]

        # Comprar USD/XXX = comprar dólar, vender a moeda estrangeira.
        # Você recebe o juro do dólar e paga o juro da moeda vendida.
        if convencao == "USD_BASE":
            carrego_anual = r_usd - r_est
        else:                                   # par cotado como XXX/USD
            carrego_anual = r_est - r_usd

        ret_preco = precos[par].pct_change(fill_method=None)
        ret_total = ret_preco + carrego_anual / DIAS_UTEIS_ANO

        # Reconstrói um índice de preço a partir do retorno total, para que o
        # resto do motor (sinal, volatilidade, P&L) continue funcionando sem
        # nenhuma alteração.
        base = precos[par].dropna()
        if base.empty:
            sem_dado.append(par)
            continue
        # O índice só começa quando preço e juros estão simultaneamente
        # disponíveis. Não supomos carrego zero onde a taxa não existe.
        primeira_valida = ret_total.first_valid_index()
        if primeira_valida is None:
            sem_dado.append(par)
            continue
        trecho = ret_total.loc[primeira_valida:]
        indice = pd.Series(pd.NA, index=precos.index, dtype="Float64")
        indice.loc[primeira_valida:] = (
            (1 + trecho.fillna(0.0)).cumprod() * float(base.loc[:primeira_valida].iloc[-1])
        )
        indice[ret_total.isna()] = pd.NA
        saida[par] = indice.astype(float)
        corrigidos.append(par)

    return saida, corrigidos, sem_dado


def main() -> None:
    print("=" * 78)
    print("PROXY DE CARREGO CAMBIAL")
    print("=" * 78)
    print("O backtest usava só a variação do preço à vista. Para câmbio isso")
    print("ignora os juros das duas pontas -- erro enorme em moedas de juro alto.\n")
    print("ATENÇÃO: esta rotina baixa a versão atualmente publicada pelo FRED.")
    print("Ela não cria vintages nem modela lag de publicação; o CSV versionado")
    print("é a entrada congelada para reproduzir os resultados da auditoria.\n")

    precos = pd.read_csv(AQUI / "dados" / "pool_expandido.csv",
                         index_col=0, parse_dates=True)
    corrigido, ok, faltando = aplica_carrego(precos)

    print(f"PARES COM PROXY ({len(ok)}): {', '.join(ok)}")
    print(f"PARES SEM DADO DE JUROS ({len(faltando)}): {', '.join(faltando)}")
    print("  (G10 majoritariamente -- diferencial pequeno contra o dólar,")
    print("   erro residual de ordem muito menor que o da lira)\n")

    # --- o impacto, par a par ------------------------------------------
    print("=" * 78)
    print("IMPACTO — retorno anualizado antes e depois do carrego")
    print("=" * 78)
    print(f"  {'par':<12}{'só preço':>12}{'com carrego':>14}{'diferença':>12}")
    print("  " + "-" * 52)
    for par in ok:
        a = precos[par].dropna()
        b = corrigido[par].dropna()
        anos = (a.index[-1] - a.index[0]).days / 365.25
        ra = (a.iloc[-1] / a.iloc[0]) ** (1 / anos) - 1
        rb = (b.iloc[-1] / b.iloc[0]) ** (1 / anos) - 1
        print(f"  {par:<12}{ra:>11.1%}{rb:>14.1%}{rb - ra:>12.1%}")

    saida = AQUI / "dados" / "pool_carrego.csv"
    corrigido.to_csv(saida)
    print(f"\nGravado: dados/pool_carrego.csv")
    print("\nPRÓXIMO PASSO: refazer o funil e o backtest sobre este painel.")


if __name__ == "__main__":
    main()
