# -*- coding: utf-8 -*-
"""
MIYAGI — análise diagnóstica de períodos
==========================================

A PERGUNTA
----------
"Qual período foi usado no teste? Não deveríamos treinar em 2007-2016 e
testar em 2017-2026?"

A RESPOSTA CURTA: O MIYAGI NÃO TEM PERÍODO DE TREINO
----------------------------------------------------
Dividir em treino/teste serve para um propósito específico: quando o modelo
APRENDE alguma coisa dos dados, você precisa de um pedaço que ele nunca viu
para checar se aprendeu de verdade ou só decorou.

Os parâmetros centrais do MIYAGI vieram de fora:

    sinal 12-1            Moskowitz, Ooi & Pedersen (2012)
    janela de vol 60d     convenção da literatura
    alvo de vol 10% a.a.  padrão da indústria de managed futures
    teto 3x               trava de segurança, não otimizada
    os 8 ativos           funil de correlação cego a retorno médio

Isso reduz data snooping de parâmetros, mas não torna os 21 anos fora da
amostra: elegibilidade, correlações e representantes do funil histórico foram
calculados com dados posteriores a 2005. A seleção anual pseudo-point-in-time
é tratada separadamente em ``auditoria_selecao.py``.

Era diferente no MARÉ: lá havia parâmetros calibrados, e por isso o design/
holdout era obrigatório.

MAS A PERGUNTA CONTINUA VALENDO — POR OUTRO MOTIVO
--------------------------------------------------
Existe um sentido real em que o começo da amostra é "in-sample": a estratégia
foi PUBLICADA em 2012, com dados até ~2009. Ou seja, os pesquisadores
originais viram o período até 2009 ao formular a hipótese.

O período posterior à publicação da literatura começa por volta de 2012, mas
não é holdout deste projeto. Existe um fenômeno documentado (McLean & Pontiff,
2016) de que anomalias
publicadas perdem parte do retorno depois de publicadas — o capital entra e
arbitra o prêmio.

Este script testa exatamente isso.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from backtest_miyagi import (
    ATIVOS, DIAS_UTEIS_ANO, calcular_metricas, carregar_dados,
    calcular_retornos, rodar_backtest,
)
from dados_miyagi import selecionar_etfs


def stats(r: pd.Series, cdi: pd.Series, rotulo: str) -> dict:
    """Métricas + significância estatística do Sharpe.

    O t-stat é o que separa "temos evidência" de "temos uma impressão".
    Para uma série de Sharpe S ao longo de N anos:

        t  ≈  S × raiz(N)          erro-padrão do Sharpe ≈ 1 / raiz(N)

    Com |t| < 2, o resultado NÃO é estatisticamente distinguível de zero pelas
    convenções usuais. Isso não quer dizer que a estratégia não funciona —
    quer dizer que a amostra é curta demais para provar que funciona.
    """
    m = calcular_metricas(r, cdi, rotulo)
    anos = m["anos"]
    s = m["sharpe"]
    erro_padrao = 1 / np.sqrt(anos)
    return {
        **m,
        "anos": anos,
        "t_stat": s * np.sqrt(anos),
        "ic_baixo": s - 1.96 * erro_padrao,
        "ic_alto": s + 1.96 * erro_padrao,
    }


def imprime_bloco(titulo: str, linhas: list[dict], cdi_por_bloco: dict) -> None:
    print(f"\n{titulo}")
    print(f"  {'periodo':<18}{'anos':>6}{'CAGR':>8}{'CDI':>8}{'Sharpe':>8}"
          f"{'t-stat':>8}{'IC 95% do Sharpe':>22}")
    print("  " + "-" * 78)
    for d in linhas:
        cdi_c = cdi_por_bloco[d["nome"]]
        marca = "" if d["cagr"] > cdi_c else "   <- perde do CDI"
        print(f"  {d['nome']:<18}{d['anos']:>6.1f}{d['cagr']:>8.1%}{cdi_c:>8.1%}"
              f"{d['sharpe']:>8.2f}{d['t_stat']:>8.2f}"
              f"   [{d['ic_baixo']:>5.2f} , {d['ic_alto']:>5.2f}]{marca}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--financiamento", choices=["overlay", "etfs"], default="etfs",
        help="etfs = stress financiado atual | overlay = estimando histórico",
    )
    args = ap.parse_args()
    print("=" * 88)
    print("MIYAGI — ANÁLISE DE PERÍODOS")
    print("=" * 88)

    precos, cdi = carregar_dados()
    retornos = calcular_retornos(precos)
    financiados = (
        selecionar_etfs(list(precos.columns))
        if args.financiamento == "etfs" else set()
    )
    res = rodar_backtest(
        precos, retornos, cdi, ativos_financiados=financiados
    )
    r = res["retornos"]
    m_total = calcular_metricas(r, cdi)

    print(f"Convenção de financiamento: {args.financiamento}")
    if financiados:
        print("Stress a CDI sobre ETFs com retornos ainda em moeda de origem;")
        print("não é uma carteira implementável em reais.")
    print(f"\nPERÍODO USADO NO BACKTEST: {r.index.min():%Y-%m-%d} a "
          f"{r.index.max():%Y-%m-%d}  ({(r.index[-1]-r.index[0]).days/365.25:.1f} anos)")
    print(f"Todos os {m_total['anos']:.1f} anos entram no Sharpe "
          f"{m_total['sharpe']:.2f}. A divisão")
    print("abaixo é antes/depois, não holdout; o funil completo usa o futuro.")

    def sub(ini, fim, nome):
        s = r.loc[ini:fim]
        return stats(s, cdi, nome) | {"nome": nome}

    def cdi_de(ini, fim):
        c = cdi.reindex(r.index).fillna(0.0).loc[ini:fim]
        return calcular_metricas(c, cdi)["cagr"]

    # ---------------------------------------------------------------- 1
    print("\n" + "=" * 88)
    print("1. O PERÍODO INTEIRO")
    linhas = [sub("2005", "2026", "2005-2026 (tudo)")]
    imprime_bloco("", linhas, {"2005-2026 (tudo)": cdi_de("2005", "2026")})
    t = linhas[0]["t_stat"]
    if abs(t) >= 2:
        leitura_t = "supera o limiar IID ingênuo de |t|=2"
    elif abs(t) >= 1.8:
        leitura_t = "fica próximo do limiar IID ingênuo de |t|=2"
    else:
        leitura_t = "fica bem abaixo do limiar IID ingênuo de |t|=2"
    print(f"\n  Leitura: com t = {t:.2f}, o resultado {leitura_t}.")
    print(f"  O intervalo de confiança do Sharpe é largo porque "
          f"{m_total['anos']:.1f} anos ainda")
    print("  é POUCO para medir")
    print("  um Sharpe pequeno com precisão. Isso é uma limitação honesta do")
    print("  trabalho, não um defeito do código.")

    # ---------------------------------------------------------------- 2
    print("\n" + "=" * 88)
    print("2. A DIVISÃO QUE VOCÊ SUGERIU (2005-2016 / 2017-2026)")
    print("   Não é treino/teste — nada foi treinado. É antes/depois.")
    linhas = [sub("2005", "2016", "2005-2016"), sub("2017", "2026", "2017-2026")]
    imprime_bloco("", linhas, {"2005-2016": cdi_de("2005", "2016"),
                               "2017-2026": cdi_de("2017", "2026")})

    # ---------------------------------------------------------------- 3
    print("\n" + "=" * 88)
    print("3. A DIVISÃO QUE FAZ MAIS SENTIDO TEÓRICO (antes/depois da publicação)")
    print("   Moskowitz et al. publicaram em 2012, com dados até ~2009.")
    print("   Anomalias publicadas tendem a perder retorno (McLean & Pontiff 2016).")
    linhas = [sub("2005", "2011", "2005-2011 (pré-pub)"),
              sub("2012", "2026", "2012-2026 (pós-pub)")]
    imprime_bloco("", linhas, {"2005-2011 (pré-pub)": cdi_de("2005", "2011"),
                               "2012-2026 (pós-pub)": cdi_de("2012", "2026")})

    # ---------------------------------------------------------------- 4
    print("\n" + "=" * 88)
    print("4. BLOCOS DE 5 ANOS — onde exatamente o resultado foi gerado")
    blocos = [("2005", "2010"), ("2011", "2015"), ("2016", "2020"), ("2021", "2026")]
    linhas = [sub(a, b, f"{a}-{b}") for a, b in blocos]
    imprime_bloco("", linhas, {f"{a}-{b}": cdi_de(a, b) for a, b in blocos})

    # ---------------------------------------------------------------- 5
    print("\n" + "=" * 88)
    print("5. CONTRIBUIÇÃO ANTES E DEPOIS DE 2016")
    print("   Reconstrói quanto cada ativo somou (ou tirou) do resultado.")

    # Exposições efetivas antes de cada retorno; elas derivam entre os
    # rebalanceamentos, em vez de restaurar o alvo gratuitamente todo dia.
    pesos_diarios = res["pesos_diarios"].reindex(r.index).fillna(0.0)
    contrib = pesos_diarios * retornos.reindex(r.index)[ATIVOS].fillna(0.0)
    if financiados:
        cols = sorted(financiados)
        contrib.loc[:, cols] = contrib[cols].sub(
            pesos_diarios[cols].mul(cdi.reindex(r.index).fillna(0.0), axis=0)
        )

    print(f"\n  {'ativo':<12}{'2005-2015':>14}{'2016-2026':>14}{'diferenca':>14}")
    print("  " + "-" * 54)
    antes = contrib.loc["2005":"2015"].sum()
    depois = contrib.loc["2016":"2026"].sum()
    # anualiza dividindo pelos anos de cada bloco
    n_antes = len(contrib.loc["2005":"2015"]) / DIAS_UTEIS_ANO
    n_depois = len(contrib.loc["2016":"2026"]) / DIAS_UTEIS_ANO
    for a in ATIVOS:
        va, vd = antes[a] / n_antes, depois[a] / n_depois
        print(f"  {a:<12}{va:>13.1%}{vd:>14.1%}{vd - va:>14.1%}")
    print("  " + "-" * 54)
    print(f"  {'TOTAL':<12}{antes.sum()/n_antes:>13.1%}{depois.sum()/n_depois:>14.1%}"
          f"{depois.sum()/n_depois - antes.sum()/n_antes:>14.1%}")
    print("\n  (retorno anualizado das posições, antes do CDI e dos custos)")

    # ---------------------------------------------------------------- 6
    print("\n" + "=" * 88)
    print("6. AS TENDÊNCIAS ENCURTARAM? — trocas de direção do sinal")
    print("   Trend following morre quando as tendências param de durar. Isso")
    print("   aparece como MAIS trocas de direção: o robô entra, a tendência")
    print("   reverte, ele sai no prejuízo e entra do outro lado. É o 'chicote'.")

    sinais = np.sign(res["pesos"])
    trocas = (sinais.diff().abs() > 0).sum(axis=1)
    print(f"\n  {'periodo':<14}{'trocas/ano':>14}{'meses ate trocar':>20}")
    print("  " + "-" * 48)
    for ini, fim in [("2005", "2010"), ("2011", "2015"),
                     ("2016", "2020"), ("2021", "2026")]:
        t_bloco = trocas.loc[ini:fim]
        if t_bloco.empty:
            continue
        anos_bloco = len(t_bloco) / 12
        por_ano = t_bloco.sum() / anos_bloco
        # duracao media de uma tendencia, por ativo
        meses = (len(t_bloco) * len(ATIVOS)) / max(t_bloco.sum(), 1)
        print(f"  {ini}-{fim:<9}{por_ano:>14.1f}{meses:>20.1f}")

    print("\n" + "=" * 88)
    print("RESUMO PARA O RELATÓRIO")
    print("=" * 88)
    print(f"""
  1. O backtest cobre 2005-2026 INTEIRO, mas não é integralmente fora da
     amostra: o funil histórico usa informação posterior a 2005.

  2. O Sharpe {m_total['sharpe']:.2f} é a MÉDIA de regimes diferentes. A média
     esconde a instabilidade entre blocos.

  3. O t-stat IID do período inteiro é {t:.2f}; HAC, múltiplos testes e bootstrap
     são reportados separadamente em auditoria_estatistica.py.

  4. Diferenças entre blocos e mudanças na duração estimada das tendências
     vêm do mesmo painel. A associação é descritiva e não identifica causa.
""")


if __name__ == "__main__":
    main()
