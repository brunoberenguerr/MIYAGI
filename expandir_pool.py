# -*- coding: utf-8 -*-
"""
MIYAGI — expansão do pool de candidatos
========================================

POR QUE EXPANDIR
----------------
A tese do Miyagi é diversificação. A relação que a sustenta é:

    Sharpe  ≈  IR × raiz(nº de apostas INDEPENDENTES)

Hoje temos 6,1 apostas efetivas vindas de 8 ativos. Cada aposta genuinamente
nova (isto é, pouco correlacionada com as que já temos) empurra esse número
para cima — e o Sharpe sobe com a raiz dele.

Isto também ataca uma fraqueza medida: no teste de jackknife, retirar o SPY
derrubava o Sharpe de 0,41 para 0,19. Concentração demais em poucos nomes.

ONDE ESTÃO AS APOSTAS INDEPENDENTES MAIS BARATAS
------------------------------------------------
O pool antigo (42 nomes) tinha um buraco: commodities apareciam quase só como
CESTA (DBC). Mas o petróleo, o cobre, o café e o boi não andam juntos — cada um
tem sua própria oferta e demanda. Uma cesta mistura tudo isso numa aposta só e
joga fora a diversificação.

A literatura de trend following (Moskowitz et al. usam 58 mercados) negocia cada
FUTURO separadamente. É a maior fonte de apostas independentes disponível, e a
que mais faltava aqui.

O QUE ESTE SCRIPT FAZ
---------------------
Só baixa e mede cobertura. NÃO seleciona nada — a seleção é feita depois, pelo
funil de correlação, com critério mecânico. Separar as duas etapas é
deliberado: quem baixa dados olhando resultado acaba escolhendo por resultado.

LIMITAÇÃO DECLARADA
-------------------
Séries de futuros contínuos do Yahoo (sufixo =F) têm descontinuidade na rolagem
de contrato. O efeito pode ser material e impede interpretar diretamente essas
séries como retorno excedente negociável sem reconstruir contratos e rolagens.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
INICIO = "2000-01-01"
FIM = "2026-07-20"

# ---------------------------------------------------------------------------
# CANDIDATOS, por classe. Critério de entrada nesta lista: ser líquido, ter
# dado público gratuito e, idealmente, histórico desde ~2003. Nada aqui foi
# escolhido por desempenho — vários certamente serão descartados pelo funil.
# ---------------------------------------------------------------------------

CANDIDATOS = {
    # --- Índices de ações em MOEDA LOCAL --------------------------------
    # Usar o índice local (e não o ETF em dólar) evita misturar a aposta de
    # ações com uma aposta cambial embutida. É a convenção dos artigos de
    # momentum com futuros, e a mesma já adotada pelo Miyagi.
    "acoes_indice": [
        "^GSPC", "^NDX", "^RUT", "^DJI",          # EUA
        "^BVSP",                                    # Brasil
        "^GDAXI", "^FTSE", "^FCHI", "^STOXX50E",   # Europa
        "^IBEX", "^SSMI", "^AEX",                  # Europa (periferia/menores)
        "^N225", "^HSI", "^KS11", "^TWII",         # Ásia desenvolvida
        "^AXJO", "^NZ50",                          # Oceania
        "^BSESN", "^JKSE", "^STI",                 # Ásia emergente
        "^MXX", "^GSPTSE",                         # Américas
    ],

    # --- ETFs de ações (dólar) ------------------------------------------
    # Mantidos porque cobrem mercados sem índice limpo no Yahoo. O funil
    # decidirá se sobrevivem ao lado dos índices locais equivalentes.
    "acoes_etf": [
        "SPY", "QQQ", "IWM", "EFA", "EEM",
        "EWZ", "EWJ", "EWG", "EWU", "EWY", "EWT", "EWA",
        "EWC", "EWW", "EWH", "EWS", "EWL", "EWD", "EWN", "EWP", "EWI",
        "EPP", "ILF", "EZA", "FXI",
    ],

    # --- Setores dos EUA -------------------------------------------------
    "setores": ["XLF", "XLK", "XLI", "XLP", "XLY", "XLB", "XLE", "XLU", "XLV"],

    # --- Juros e crédito -------------------------------------------------
    # Futuros de Treasury cobrem a curva de forma mais limpa que ETFs.
    "juros": [
        "ZT=F", "ZF=F", "ZN=F", "ZB=F",            # 2, 5, 10 e 30 anos
        "SHY", "IEF", "TLT", "TIP", "LQD", "AGG",  # ETFs equivalentes
    ],

    # --- Câmbio ----------------------------------------------------------
    "cambio": [
        "BRL=X", "EURUSD=X", "JPY=X", "GBPUSD=X", "AUDUSD=X",
        "CAD=X", "CHF=X", "MXN=X", "NZDUSD=X",
        "SEK=X", "NOK=X", "ZAR=X", "SGD=X", "INR=X", "TRY=X", "PLN=X",
    ],

    # --- Commodities, uma a uma (a maior novidade) -----------------------
    # Metais, energia, grãos, softs e carnes têm ciclos próprios de oferta e
    # demanda. É aqui que moram as apostas menos correlacionadas do pool.
    "commodities": [
        "GC=F", "SI=F", "HG=F", "PL=F", "PA=F",              # metais
        "CL=F", "BZ=F", "NG=F", "HO=F", "RB=F",              # energia
        "ZC=F", "ZS=F", "ZW=F", "ZL=F",                      # grãos
        "KC=F", "SB=F", "CC=F", "CT=F", "OJ=F",              # softs
        "LE=F", "HE=F",                                       # carnes
        "GLD", "SLV", "DBC", "USO", "UNG", "DBA", "GDX",     # ETFs
    ],

    # --- Imobiliário -----------------------------------------------------
    "imobiliario": ["VNQ", "IYR", "RWR"],
}


def baixar(tickers: list[str]) -> pd.DataFrame:
    """Baixa preços de fechamento ajustado em lotes."""
    import yfinance as yf
    dados = {}
    lote = 25
    for i in range(0, len(tickers), lote):
        grupo = tickers[i:i + lote]
        print(f"  baixando {i+1}-{min(i+lote, len(tickers))} de {len(tickers)}...",
              flush=True)
        try:
            df = yf.download(grupo, start=INICIO, end=FIM, auto_adjust=True,
                             progress=False, threads=True)
        except Exception as exc:                       # noqa: BLE001
            print(f"    falhou o lote: {exc}")
            continue
        if df is None or df.empty:
            continue
        fech = df["Close"] if "Close" in df.columns.get_level_values(0) else df
        if isinstance(fech, pd.Series):
            fech = fech.to_frame(grupo[0])
        for c in fech.columns:
            s = fech[c].dropna()
            if len(s) > 250:                            # descarta série vazia
                dados[c] = s
    return pd.DataFrame(dados).sort_index()


def main() -> None:
    todos = sorted({t for lista in CANDIDATOS.values() for t in lista})
    print("=" * 76)
    print(f"EXPANSÃO DO POOL — {len(todos)} candidatos em {len(CANDIDATOS)} classes")
    print("=" * 76)
    for classe, lista in CANDIDATOS.items():
        print(f"  {classe:<16} {len(lista):>3} candidatos")

    print(f"\nBaixando {INICIO} a {FIM}...")
    precos = baixar(todos)

    print(f"\nRecebidos: {precos.shape[1]} de {len(todos)} tickers")

    # --- cobertura -------------------------------------------------------
    inicio = {c: precos[c].dropna().index.min() for c in precos.columns}
    ser = pd.Series(inicio).sort_values()
    corte = pd.Timestamp("2004-01-01")
    aptos = [c for c, d in ser.items() if d <= corte]

    print(f"\nCOM histórico até 2003 (aptos para backtest desde 2005): {len(aptos)}")
    for classe, lista in CANDIDATOS.items():
        na_classe = [t for t in lista if t in aptos]
        print(f"  {classe:<16} {len(na_classe):>3} de {len(lista):<3} "
              f"{', '.join(na_classe[:9])}{'...' if len(na_classe) > 9 else ''}")

    faltando = [t for t in todos if t not in precos.columns]
    if faltando:
        print(f"\nNão retornaram dado ({len(faltando)}): {', '.join(faltando)}")

    tardios = [(c, d.date()) for c, d in ser.items() if d > corte]
    if tardios:
        print(f"\nHistórico curto ({len(tardios)}) — entram no backtest só depois:")
        for c, d in tardios[:20]:
            print(f"   {c:<12} {d}")
        if len(tardios) > 20:
            print(f"   ... e mais {len(tardios) - 20}")

    saida = AQUI / "dados" / "pool_expandido.csv"
    precos.to_csv(saida)
    print(f"\nGravado: dados/pool_expandido.csv "
          f"({precos.shape[0]} linhas × {precos.shape[1]} ativos)")
    print("\nPRÓXIMO PASSO: rodar o funil de correlação sobre este pool.")
    print("Nada foi selecionado aqui — só medido.")


if __name__ == "__main__":
    main()
