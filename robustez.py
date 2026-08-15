# -*- coding: utf-8 -*-
"""
MIYAGI — testes de robustez
===========================

A PERGUNTA QUE ESTE ARQUIVO RESPONDE
------------------------------------
Um backtest produz UM número. Mas esse número saiu de dezenas de escolhas:
custo de 0,1%, janela de volatilidade de 60 dias, alavancagem máxima de 3x,
sinal de 12 meses, 8 ativos específicos, período de 2005 a 2026.

A pergunta que decide se o resultado vale alguma coisa é:

    "O resultado sobrevive se eu mexer nessas escolhas,
     ou ele existe APENAS na combinação exata que testamos?"

Se o Sharpe despenca ao trocar a janela de 60 para 90 dias, não encontramos uma
estratégia — encontramos uma coincidência.

ISTO NÃO É BUSCA DE PARÂMETRO
-----------------------------
A diferença é sutil e decide a credibilidade do trabalho inteiro:

    busca de parâmetro:  testo 20 configurações e ADOTO a melhor.
                         -> o resultado vira ficção (overfitting)

    teste de robustez:   testo 20 configurações, MANTENHO a original, e
                         reporto o espalhamento de todas.
                         -> o resultado ganha uma barra de erro honesta

A configuração base NÃO muda depois deste arquivo, aconteça o que acontecer.
Se algum teste mostrar algo melhor, isso é informação sobre a incerteza —
não um convite para trocar.

OS CRITÉRIOS, DECLARADOS ANTES DE RODAR
---------------------------------------
Escrevo aqui o que conta como aprovado, antes de ver qualquer número. Sem isso,
é fácil olhar o resultado e inventar o critério que ele satisfaz.

    A. Custos      aprovado se ainda supera o CDI com o DOBRO do custo base.
    B. Janela vol  aprovado se o Sharpe fica acima de 0,25 em toda a faixa
                   testada (20 a 250 dias).
    C. Sub-períodos aprovado se supera o CDI em pelo menos 3 dos 4 blocos.
    D. Alavancagem aprovado se não depende do teto de 3x — ou seja, se com
                   teto 2x ainda supera o CDI.
    E. Jackknife   com 8 ativos, retira um por vez; com 40, retira uma classe
                   inteira. Aprovado se o pior Sharpe fica acima de 0,25.
    F. Horizonte   aprovado se o sinal funciona numa FAIXA de horizontes
                   (6, 9, 12, 18 meses), e não só em 12.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_miyagi as bt
from backtest_miyagi import (
    calcular_metricas,
    carregar_dados,
    calcular_retornos,
    rodar_backtest,
)
from dados_miyagi import selecionar_etfs

SHARPE_MINIMO = 0.25          # piso declarado para os testes B e E
AQUI = Path(__file__).resolve().parent


def carregar_universo(qual: str):
    """Carrega o universo de 8 ativos (original) ou o de 40 (expandido)."""
    if qual == "8":
        precos, cdi = carregar_dados()
        return precos, cdi, list(precos.columns)
    from backtest_expandido import carregar_expandido
    precos, cdi, universo = carregar_expandido()
    return precos, cdi, universo


def classes_dos_ativos(universo: list[str]) -> dict[str, list[str]]:
    """Agrupa os ativos por classe, para o jackknife do universo grande.

    Com 40 ativos, retirar UM de cada vez tem pouca informação: a tese de
    diversificação já prevê que nenhum isolado importe, e seriam 40 backtests
    para confirmar o óbvio.

    A pergunta que ainda morde é outra: o resultado depende de uma CLASSE
    inteira? Se tirar todas as commodities derrubar tudo, a estratégia é uma
    aposta em commodities com enfeite -- e isso o teste por ativo não revelaria.
    """
    from expandir_pool import CANDIDATOS
    de_qual_classe = {t: cl for cl, lista in CANDIDATOS.items() for t in lista}
    grupos: dict[str, list[str]] = {}
    for a in universo:
        grupos.setdefault(de_qual_classe.get(a, "outros"), []).append(a)
    return grupos


@contextmanager
def parametro(**overrides):
    """Troca constantes do módulo temporariamente e SEMPRE devolve ao original.

    O `finally` não é decoração: sem ele, um erro no meio de um teste deixaria
    o parâmetro alterado e contaminaria todos os testes seguintes — um bug que
    produziria números plausíveis e errados, o pior tipo.
    """
    antigos = {k: getattr(bt, k) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(bt, k, v)
        yield
    finally:
        for k, v in antigos.items():
            setattr(bt, k, v)


def _metricas(
    precos, retornos, cdi, ativos_financiados: set[str] | None = None,
    **overrides,
) -> dict:
    """Roda um backtest com parâmetros alterados e devolve as métricas."""
    financiados_presentes = set(ativos_financiados or ()) & set(precos.columns)
    with parametro(**overrides):
        r = rodar_backtest(
            precos, retornos, cdi,
            ativos_financiados=financiados_presentes,
        )
    m = calcular_metricas(r["retornos"], cdi)
    m["exposicao_media"] = float(r["exposicao"].mean())
    m["custo_total"] = float(r["custos"].sum())
    m["retornos"] = r["retornos"]
    m["fracao_teto"] = float(
        np.isclose(r["exposicao"], bt.ALAVANCAGEM_MAX, rtol=0, atol=1e-8).mean()
    )
    return m


def _linha(rotulo, m, base_sharpe=None, marca_base=False) -> str:
    delta = ""
    if base_sharpe is not None and not marca_base:
        d = m["sharpe"] - base_sharpe
        delta = f"{d:+6.2f}"
    elif marca_base:
        delta = " (base)"
    return (f"  {rotulo:<22}{m['cagr']:>8.1%}{m['vol']:>9.1%}"
            f"{m['sharpe']:>9.2f}{m['max_drawdown']:>10.1%}{delta:>9}")


def _cabecalho() -> str:
    return (f"  {'':<22}{'CAGR':>8}{'Vol':>9}{'Sharpe':>9}{'Max DD':>10}"
            f"{'ΔSharpe':>9}\n  " + "-" * 66)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universo", choices=["8", "40"], default="8",
                    help="8 = universo original | 40 = universo expandido")
    ap.add_argument(
        "--financiamento", choices=["overlay", "etfs"], default="overlay",
        help=("overlay = convenção histórica | etfs = adjusted close "
              "financiado à taxa do CDI"),
    )
    args = ap.parse_args()

    print("=" * 78)
    print(f"MIYAGI — TESTES DE ROBUSTEZ (universo de {args.universo} ativos)")
    print("=" * 78)
    print(f"Convenção de financiamento: {args.financiamento}")
    if args.financiamento == "etfs":
        print("Stress a CDI sobre ETFs com retornos ainda em moeda de origem;")
        print("não é uma carteira implementável em reais.")
    print("A configuração base NÃO muda em função destes resultados.")
    print("Critérios declarados antes da execução (ver docstring do arquivo).\n")

    precos, cdi, universo = carregar_universo(args.universo)
    retornos = calcular_retornos(precos)
    ativos_financiados = (
        selecionar_etfs(universo) if args.financiamento == "etfs" else set()
    )

    # ---- linha de base -------------------------------------------------
    base = _metricas(
        precos, retornos, cdi, ativos_financiados=ativos_financiados
    )
    sb = base["sharpe"]
    cdi_cagr = calcular_metricas(cdi.reindex(base["patrimonio"].index).fillna(0.0),
                                 cdi)["cagr"]

    print(f"BASE: CAGR {base['cagr']:.1%} | vol {base['vol']:.1%} | "
          f"Sharpe {sb:.2f} | max DD {base['max_drawdown']:.1%}")
    print(f"CDI no mesmo período: {cdi_cagr:.1%} a.a.  "
          f"(superar isto é o mínimo para a estratégia existir)\n")

    veredito = {}

    # =================================================================== A
    print("=" * 78)
    print("A. SENSIBILIDADE A CUSTOS")
    print("   Se o resultado evapora com custo um pouco maior, ele não")
    print("   sobreviveria à execução real.")
    print(_cabecalho())
    custos = [0.0005, 0.001, 0.002, 0.005]
    res_custos = {}
    for c in custos:
        m = _metricas(
            precos, retornos, cdi, ativos_financiados=ativos_financiados,
            CUSTO_POR_TRADE=c,
        )
        res_custos[c] = m
        rotulo = f"{c*100:.2f}% por trade" + (" *" if c == 0.001 else "")
        print(_linha(rotulo, m, sb, marca_base=(c == 0.001)))
    print("  * configuração base")

    m2x = res_custos[0.002]
    veredito["A"] = m2x["cagr"] > cdi_cagr
    print(f"\n  Critério: superar o CDI ({cdi_cagr:.1%}) com o DOBRO do custo.")
    print(f"  Com 0,2%/trade: CAGR {m2x['cagr']:.1%}  ->  "
          f"{'APROVADO' if veredito['A'] else 'REPROVADO'}")

    # custo de break-even: onde a estratégia deixa de bater o CDI
    # Passo de 0,2% (e não 0,1%): com 40 ativos cada backtest custa ~90s, e
    # localizar o break-even com precisão de 0,1% não muda nenhuma conclusão.
    be = None
    for c in np.arange(0.002, 0.031, 0.002):
        m = _metricas(
            precos, retornos, cdi, ativos_financiados=ativos_financiados,
            CUSTO_POR_TRADE=float(c),
        )
        if m["cagr"] <= cdi_cagr:
            be = float(c)
            break
    if be:
        print(f"  Custo de break-even: ~{be*100:.1f}% por trade "
              f"({be/0.001:.0f}x o custo assumido) — acima disso, perde do CDI.")
    else:
        print("  Custo de break-even: acima de 3% por trade (folga ampla).")

    # =================================================================== B
    print("\n" + "=" * 78)
    print("B. JANELA DE VOLATILIDADE")
    print("   60 dias foi escolha de convenção. Se só ela funciona, a escolha")
    print("   estava fazendo trabalho que a estratégia deveria fazer.")
    print(_cabecalho())
    sharpes_b = []
    for j in (20, 40, 60, 90, 120, 250):
        m = _metricas(
            precos, retornos, cdi, ativos_financiados=ativos_financiados,
            JANELA_VOL_DIAS=j, JANELA_EWMA_DIAS=max(252, j),
        )
        sharpes_b.append(m["sharpe"])
        print(_linha(f"{j} dias" + (" *" if j == 60 else ""), m, sb,
                     marca_base=(j == 60)))
    veredito["B"] = min(sharpes_b) >= SHARPE_MINIMO
    print(f"\n  Critério: Sharpe >= {SHARPE_MINIMO} em toda a faixa.")
    print(f"  Mínimo observado: {min(sharpes_b):.2f}  ->  "
          f"{'APROVADO' if veredito['B'] else 'REPROVADO'}")

    # =================================================================== C
    print("\n" + "=" * 78)
    print("C. SUB-PERÍODOS")
    print("   Um bom resultado médio pode esconder um único período excepcional")
    print("   carregando duas décadas de mediocridade.")
    print(f"  {'':<22}{'CAGR':>8}{'Vol':>9}{'Sharpe':>9}{'Max DD':>10}{'CDI':>9}")
    print("  " + "-" * 66)
    r_base = base["retornos"]
    blocos = [("2005-2010", "2005", "2010"), ("2011-2015", "2011", "2015"),
              ("2016-2020", "2016", "2020"), ("2021-2026", "2021", "2026")]
    ganhou = 0
    for rotulo, ini, fim in blocos:
        sub = r_base.loc[ini:fim]
        if sub.empty:
            continue
        m = calcular_metricas(sub, cdi)
        c_sub = calcular_metricas(cdi.reindex(sub.index).fillna(0.0), cdi)
        bateu = m["cagr"] > c_sub["cagr"]
        ganhou += int(bateu)
        marca = "  ok" if bateu else "  --"
        print(f"  {rotulo:<22}{m['cagr']:>8.1%}{m['vol']:>9.1%}"
              f"{m['sharpe']:>9.2f}{m['max_drawdown']:>10.1%}"
              f"{c_sub['cagr']:>8.1%}{marca}")
    veredito["C"] = ganhou >= 3
    print(f"\n  Critério: superar o CDI em pelo menos 3 dos 4 blocos.")
    print(f"  Superou em {ganhou} de 4  ->  "
          f"{'APROVADO' if veredito['C'] else 'REPROVADO'}")

    # =================================================================== D
    print("\n" + "=" * 78)
    print("D. TETO DE ALAVANCAGEM")
    print(f"   A base bate no teto de 3x em {base['fracao_teto']:.1%} dos "
          "rebalanceamentos. Se o")
    print("   resultado depende disso, o teto virou parâmetro de retorno e")
    print("   não trava de segurança.")
    print(_cabecalho())
    res_d = {}
    for lev in (1.0, 2.0, 3.0, 5.0):
        m = _metricas(
            precos, retornos, cdi, ativos_financiados=ativos_financiados,
            ALAVANCAGEM_MAX=lev,
        )
        res_d[lev] = m
        print(_linha(f"teto {lev:.0f}x" + (" *" if lev == 3.0 else ""), m, sb,
                     marca_base=(lev == 3.0)))
    veredito["D"] = res_d[2.0]["cagr"] > cdi_cagr
    print(f"\n  Critério: com teto 2x, ainda superar o CDI ({cdi_cagr:.1%}).")
    print(f"  Com teto 2x: CAGR {res_d[2.0]['cagr']:.1%}  ->  "
          f"{'APROVADO' if veredito['D'] else 'REPROVADO'}")

    # =================================================================== E
    print("\n" + "=" * 78)
    if args.universo == "8":
        print("E. JACKKNIFE — retirando um ativo por vez")
        print("   A tese do trabalho é diversificação: apostas pouco")
        print("   correlacionadas. Se retirar UM ativo derruba tudo, a tese é")
        print("   falsa e o resultado era aposta concentrada disfarçada.")
        print(_cabecalho())
        sharpes_e = []
        for ativo in universo:
            restantes = [a for a in universo if a != ativo]
            m = _metricas(
                precos[restantes], retornos[restantes], cdi,
                ativos_financiados=ativos_financiados,
            )
            sharpes_e.append((ativo, m["sharpe"]))
            print(_linha(f"sem {ativo}", m, sb))
    else:
        print("E. JACKKNIFE POR CLASSE — retirando um bloco inteiro por vez")
        print("   Com 40 ativos, tirar UM de cada vez confirmaria o óbvio. A")
        print("   pergunta que ainda morde: o resultado depende de uma CLASSE")
        print("   inteira? Se tirar todas as commodities derruba tudo, isto é")
        print("   uma aposta em commodities com enfeite de diversificação.")
        print(_cabecalho())
        grupos = classes_dos_ativos(universo)
        sharpes_e = []
        for classe, membros in sorted(grupos.items(), key=lambda kv: -len(kv[1])):
            restantes = [a for a in universo if a not in membros]
            if len(restantes) < 5:
                continue
            m = _metricas(
                precos[restantes], retornos[restantes], cdi,
                ativos_financiados=ativos_financiados,
            )
            sharpes_e.append((classe, m["sharpe"]))
            print(_linha(f"sem {classe} ({len(membros)})", m, sb))

    pior_rot, pior_sharpe = min(sharpes_e, key=lambda x: x[1])
    veredito["E"] = pior_sharpe >= SHARPE_MINIMO
    print(f"\n  Critério: Sharpe >= {SHARPE_MINIMO} retirando qualquer "
          f"{'ativo' if args.universo == '8' else 'classe'}.")
    print(f"  Pior caso: sem {pior_rot} -> Sharpe {pior_sharpe:.2f}  ->  "
          f"{'APROVADO' if veredito['E'] else 'REPROVADO'}")

    # =================================================================== F
    print("\n" + "=" * 78)
    print("F. HORIZONTE DO SINAL")
    print("   12-1 veio da literatura, não dos nossos dados. O teste é se o")
    print("   efeito existe numa FAIXA de horizontes — momentum é um fenômeno")
    print("   amplo, não um truque que só funciona em 12 meses.")
    print(_cabecalho())
    sharpes_f = []
    for h in (6, 9, 12, 18):
        m = _metricas(
            precos, retornos, cdi, ativos_financiados=ativos_financiados,
            JANELA_SINAL_MESES=h,
        )
        sharpes_f.append(m["sharpe"])
        print(_linha(f"{h}-1 meses" + (" *" if h == 12 else ""), m, sb,
                     marca_base=(h == 12)))
    veredito["F"] = sum(s > 0 for s in sharpes_f) >= 3
    print(f"\n  Critério: Sharpe positivo em pelo menos 3 dos 4 horizontes.")
    print(f"  Positivo em {sum(s > 0 for s in sharpes_f)} de 4  ->  "
          f"{'APROVADO' if veredito['F'] else 'REPROVADO'}")

    # ================================================================ fim
    print("\n" + "=" * 78)
    print("VEREDITO GERAL")
    print("=" * 78)
    nomes = {"A": "Custos", "B": "Janela de volatilidade", "C": "Sub-períodos",
             "D": "Teto de alavancagem", "E": "Jackknife",
             "F": "Horizonte do sinal"}
    for k in "ABCDEF":
        print(f"  {k}. {nomes[k]:<26} {'APROVADO' if veredito[k] else 'REPROVADO'}")
    aprovados = sum(veredito.values())
    pd.DataFrame([
        {
            "teste": k,
            "nome": nomes[k],
            "aprovado": bool(veredito[k]),
            "universo": args.universo,
            "financiamento": args.financiamento,
            "cagr_base": base["cagr"],
            "sharpe_base": base["sharpe"],
        }
        for k in "ABCDEF"
    ]).to_csv(
        AQUI / "resultados"
        / f"auditoria_robustez_{args.universo}_{args.financiamento}.csv",
        index=False,
        float_format="%.12g",
    )
    print(f"\n  {aprovados} de 6 testes aprovados.")
    if aprovados == 6:
        print("  O resultado não depende de nenhuma escolha isolada testada.")
    else:
        reprovados = [nomes[k] for k in veredito if not veredito[k]]
        print(f"  ATENÇÃO — reprovou em: {', '.join(reprovados)}.")
        print("  Isso vai para o relatório como está. Um teste reprovado é")
        print("  informação sobre a estratégia, não motivo para trocar o teste.")
    print("=" * 78)


if __name__ == "__main__":
    main()
