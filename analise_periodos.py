# -*- coding: utf-8 -*-
"""
MIYAGI — análise de períodos: por que o Sharpe é 0,41?
======================================================

A PERGUNTA
----------
"Qual período foi usado no teste? Não deveríamos treinar em 2007-2016 e
testar em 2017-2026?"

A RESPOSTA CURTA: O MIYAGI NÃO TEM PERÍODO DE TREINO
----------------------------------------------------
Dividir em treino/teste serve para um propósito específico: quando o modelo
APRENDE alguma coisa dos dados, você precisa de um pedaço que ele nunca viu
para checar se aprendeu de verdade ou só decorou.

O MIYAGI não aprende nada dos dados. Todos os parâmetros vieram de fora:

    sinal 12-1            Moskowitz, Ooi & Pedersen (2012)
    janela de vol 60d     convenção da literatura
    alvo de vol 10% a.a.  padrão da indústria de managed futures
    teto 3x               trava de segurança, não otimizada
    os 8 ativos           funil de correlação feito ANTES, por critério
                          estatístico, sem olhar retorno

Nenhum desses números foi escolhido olhando o resultado. Isso significa que
**os 21 anos inteiros já são out-of-sample** — não existe pedaço "contaminado"
por ajuste, porque não houve ajuste.

Era diferente no MARÉ: lá havia parâmetros calibrados, e por isso o design/
holdout era obrigatório.

MAS A PERGUNTA CONTINUA VALENDO — POR OUTRO MOTIVO
--------------------------------------------------
Existe um sentido real em que o começo da amostra é "in-sample": a estratégia
foi PUBLICADA em 2012, com dados até ~2009. Ou seja, os pesquisadores
originais viram o período até 2009 ao formular a hipótese.

O verdadeiro out-of-sample da literatura é, portanto, de ~2012 em diante.
E existe um fenômeno documentado (McLean & Pontiff, 2016) de que anomalias
publicadas perdem parte do retorno depois de publicadas — o capital entra e
arbitra o prêmio.

Este script testa exatamente isso.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_miyagi import (
    ATIVOS, DIAS_UTEIS_ANO, calcular_metricas, carregar_dados,
    calcular_retornos, rodar_backtest,
)


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
    print("=" * 88)
    print("MIYAGI — ANÁLISE DE PERÍODOS")
    print("=" * 88)

    precos, cdi = carregar_dados()
    retornos = calcular_retornos(precos)
    res = rodar_backtest(precos, retornos, cdi)
    r = res["retornos"]

    print(f"\nPERÍODO USADO NO BACKTEST: {r.index.min():%Y-%m-%d} a "
          f"{r.index.max():%Y-%m-%d}  ({(r.index[-1]-r.index[0]).days/365.25:.1f} anos)")
    print("Todos os 21,5 anos entram no resultado de Sharpe 0,41. Não há divisão")
    print("treino/teste porque NENHUM parâmetro foi estimado dos dados.")

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
    print(f"\n  Leitura: com t = {t:.2f}, o resultado fica no limite da")
    print("  significância estatística (a convenção pede |t| > 2). O intervalo de")
    print("  confiança do Sharpe é largo porque 21 anos ainda é POUCO para medir")
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
    print("5. POR QUE 2016+ FOI FRACO? — contribuição de cada ativo")
    print("   Reconstrói quanto cada ativo somou (ou tirou) do resultado.")

    # Pesos mensais -> diários (o robô segura a posição entre rebalanceamentos)
    pesos_diarios = res["pesos"].reindex(r.index).ffill().fillna(0.0)
    contrib = (pesos_diarios * retornos.reindex(r.index)[ATIVOS].fillna(0.0))

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
    print("""
  1. O backtest cobre 2005-2026 INTEIRO. Não há divisão treino/teste porque
     nenhum parâmetro foi estimado dos dados -- todos vieram da literatura.
     Os 21,5 anos já são, nesse sentido, out-of-sample.

  2. O Sharpe de 0,41 é a MÉDIA de dois regimes muito diferentes: forte até
     2015, fraco depois. A média esconde os dois.

  3. Com t-stat perto de 2, mesmo o resultado do período inteiro está no
     limite da significância. A amostra é curta para o tamanho do efeito.

  4. A queda pós-2016 coincide com (a) o período pós-publicação da estratégia
     e (b) uma década documentadamente ruim para trend following. Não
     conseguimos separar as duas causas com os dados que temos -- e dizer
     isso é mais defensável do que escolher uma.
""")


if __name__ == "__main__":
    main()
