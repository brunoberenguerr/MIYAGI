# -*- coding: utf-8 -*-
"""
MIYAGI — robô de tendência multi-ativo com alvo de risco
=========================================================

"Não prevê o golpe. Observa o movimento — e responde com técnica."

A IDEIA EM UMA FRASE
--------------------
Quando uma notícia boa sai, o preço não sobe tudo de uma vez: sobe aos poucos,
ao longo de meses, porque as pessoas demoram a acreditar e depois entram na onda.
Resultado: TENDÊNCIAS DURAM. O Miyagi não tenta adivinhar o futuro — ele olha
para onde cada mercado JÁ está andando e vai junto, comprado no que sobe e
vendido no que cai.

Isso é chamado de "time-series momentum" e está documentado em 58 mercados ao
longo de um século de dados (Moskowitz, Ooi & Pedersen 2012; Hurst, Ooi &
Pedersen 2017).

COMO LER ESTE ARQUIVO
---------------------
O código está dividido em 7 blocos numerados. Cada bloco começa com uma
explicação em português simples do QUE ele faz e POR QUE. Se você só quer
entender a estratégia, leia apenas os comentários — eles contam a história
inteira sem precisar entender Python.

    BLOCO 1 — Os parâmetros (as "regras do jogo")
    BLOCO 2 — Carregar e limpar os dados
    BLOCO 3 — O SINAL: para onde cada ativo está andando?
    BLOCO 4 — O TAMANHO: quanto apostar em cada um?
    BLOCO 5 — A DEFESA: controlar o risco da carteira inteira
    BLOCO 6 — A SIMULAÇÃO: rodar mês a mês pela história
    BLOCO 7 — As MÉTRICAS: o resultado foi bom?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dados_miyagi import alinhar_ao_calendario, carregar_cdi, carregar_pool_oficial

AQUI = Path(__file__).resolve().parent


# ==========================================================================
# BLOCO 1 — OS PARÂMETROS (as "regras do jogo")
# ==========================================================================
# Todos os números que definem a estratégia ficam AQUI, num lugar só. Isso é
# importante por dois motivos:
#
#   1. Transparência: qualquer pessoa vê todas as escolhas de uma vez.
#   2. Honestidade científica: estes valores vieram dos artigos acadêmicos,
#      NÃO de tentar várias combinações até achar a que dava o melhor resultado.
#      Testar 500 combinações e escolher a melhor é o erro clássico chamado
#      "overfitting" — o resultado fica lindo no passado e falha no futuro.
#
# Cada parâmetro abaixo diz de onde veio.

# Os 8 ativos do protótipo, escolhidos por um funil de correlação (ver
# selecao_universo/). O resultado oficial atual usa o universo expandido
# congelado em ``dados/universo_final.txt``; esta lista permanece como
# referência reduzida e diagnóstico, não como universo final.
# A lógica: quanto MENOS parecidos entre si, melhor. Oito apostas que sobem e
# descem juntas valem quase o mesmo que uma aposta só; oito apostas
# independentes valem oito. Os números históricos de correlação e apostas
# efetivas pertencem ao registro do funil e não são premissas do motor.
ATIVOS = [
    "^BVSP",      # Ações Brasil (Ibovespa)
    "SPY",        # Ações EUA (S&P 500)
    "IEF",        # Juros EUA 7-10 anos (títulos do tesouro americano)
    "BRL=X",      # Dólar/Real  <- protege o book: sobe quando o Ibovespa cai
    "EURUSD=X",   # Euro/Dólar
    "JPY=X",      # Dólar/Iene
    "GLD",        # Ouro
    "DBC",        # Commodities (cesta ampla)
]

# --- O sinal -------------------------------------------------------------
# Olhamos o retorno acumulado dos últimos 12 meses, mas IGNORANDO o mês mais
# recente. Por que ignorar? Porque no curtíssimo prazo os preços tendem a
# "quicar" de volta (efeito contrário ao que queremos capturar). Incluir o
# último mês só adicionaria ruído. Isso é tão padrão na literatura que tem
# nome próprio: sinal "12-1".
JANELA_SINAL_MESES = 12   # fonte: Moskowitz et al. (2012)
PULA_MESES = 1            # fonte: Moskowitz et al. (2012)

# --- O tamanho da posição ------------------------------------------------
# Medimos o quanto cada ativo balança (volatilidade) nos últimos 60 dias.
# Ativo calmo -> posição maior. Ativo agitado -> posição menor. Assim cada um
# contribui com uma dose PARECIDA de risco, e nenhum domina a carteira só por
# ser naturalmente mais nervoso.
JANELA_VOL_DIAS = 60      # fonte: convenção da literatura de trend following

# COMO medir essa volatilidade — a especificação do projeto permite duas formas:
#
#   "janela"  todos os últimos 60 dias contam IGUAL, e o dia 61 some de vez.
#             Simples, mas tem o "efeito penhasco": quando um dia de pânico
#             sai da janela, a volatilidade medida despenca de um dia para o
#             outro — não porque o mercado acalmou, mas porque um número saiu
#             de uma planilha. O robô então se alavanca de novo, possivelmente
#             no meio da crise.
#
#   "ewma"    a memória DESBOTA aos poucos: ontem pesa 100%, 10 dias atrás
#             pesa 54%, 30 dias atrás pesa 16%. Nada é esquecido de repente.
#             (EWMA = média móvel exponencialmente ponderada.)
#
# As duas estavam previstas na especificação. Trocar entre elas para investigar
# uma divergência é diagnóstico; trocar para melhorar o resultado seria
# overfitting. A distinção está no motivo, e ele está registrado aqui.
#
# ESCOLHA OFICIAL: "janela".
#
# O motivo NÃO é o desempenho observado. Escolher depois do backtest pelo maior
# Sharpe seria exatamente o overfitting contra o qual este arquivo adverte.
#
# O motivo real, em duas partes:
#   1. Os dois foram comparados historicamente (comparar_estimadores.py) sem
#      diferença material de drawdown naquela reconstrução. Os valores desse
#      registro não são métricas oficiais do painel auditado atual.
#   2. Quando duas opções são equivalentes no resultado, vence a mais simples
#      de explicar e auditar. "Desvio-padrão dos últimos 60 dias" cabe numa
#      frase; EWMA exige explicar decaimento exponencial e o parâmetro lambda.
#
# Registrado porque o critério de escolha importa mais que a escolha: um
# avaliador precisa conseguir verificar que não houve garimpo de resultado.
ESTIMADOR_VOL = "janela"  # "janela" | "ewma"
LAMBDA_EWMA = 0.94        # fonte: RiskMetrics (JP Morgan). Meia-vida ~11 dias.
JANELA_EWMA_DIAS = 252    # até onde olhar (após isso o peso é desprezível)

# --- A defesa ------------------------------------------------------------
# A carteira inteira é escalada para ter uma "velocidade de risco" fixa de
# 10% ao ano. É como o piloto automático de um carro: acelera na subida,
# freia na descida, mantendo o ritmo constante. Se tudo está calmo, aumenta as
# posições; se o mercado enlouquece, encolhe automaticamente.
VOL_ALVO_ANUAL = 0.10     # fonte: padrão da indústria de managed futures
ALAVANCAGEM_MAX = 3.0     # trava de segurança: nunca expor mais que 3x o capital

# --- Custos e execução ---------------------------------------------------
REBALANCEIA = "ME"        # "ME" = Month End (último dia útil de cada mês)
CUSTO_POR_TRADE = 0.001   # 0,1% sobre o valor negociado (não sobre a carteira)
DIAS_UTEIS_ANO = 252      # convenção de mercado

# --- Período -------------------------------------------------------------
# Começamos em 2005 para incluir DUAS crises grandes (2008 e 2020). Testar uma
# estratégia só em época de bonança não prova nada.
INICIO = "2005-01-01"

# Quantos dias de histórico um ativo precisa ter antes de podermos negociá-lo.
# O sinal de 12 meses precisa de ~252 pregões. Sem isso, não há sinal — e
# inventar um sinal sem dado é exatamente o tipo de trapaça que queremos evitar.
HISTORICO_MINIMO_DIAS = 252


# ==========================================================================
# BLOCO 2 — CARREGAR E LIMPAR OS DADOS
# ==========================================================================
# Aqui acontece um detalhe chato mas importante: cada mercado tem feriados
# diferentes. A bolsa brasileira fecha no Carnaval, a americana no
# Thanksgiving, e o câmbio quase nunca para. Então as séries têm "buracos" em
# dias diferentes.
#
# Solução: repetimos o último preço conhecido por até 5 dias (forward-fill
# limitado). Se um ativo ficou mais de 5 dias sem preço, deixamos o buraco —
# preencher demais seria inventar dado.

def carregar_dados() -> tuple[pd.DataFrame, pd.Series]:
    """Lê os 8 ativos do painel oficial e o CDI.

    Inclusive nesta versão reduzida, moedas vêm do painel corrigido por
    carrego. Assim, nenhum script volta silenciosamente às séries spot.
    """

    # --- Preços diários ---
    precos = carregar_pool_oficial()[ATIVOS]

    # --- CDI ---
    # O arquivo traz o CDI em PORCENTO ao dia (ex.: 0.088 significa 0,088%).
    # Dividimos por 100 para virar fração decimal (0.00088), que é o formato
    # com que a matemática financeira trabalha.
    cdi = carregar_cdi()

    # --- O CALENDÁRIO OFICIAL (detalhe que parece burocrático e não é) ------
    # O arquivo de preços tem ~290 datas por ano, não ~252. O motivo: o câmbio
    # (dólar, iene, euro) negocia em dias que a bolsa está fechada, então
    # entram sábados e feriados na planilha.
    #
    # Se deixarmos assim, três coisas quebram de uma vez:
    #   1. o CDI renderia em ~290 dias/ano em vez dos ~252 dias úteis reais,
    #      inflando o "dinheiro parado" artificialmente;
    #   2. a volatilidade seria subestimada (dias sem pregão viram retorno
    #      zero, e muitos zeros fazem a série parecer mais calma do que é);
    #   3. como o robô se alavanca quando o mercado parece calmo, ele acabaria
    #      tomando risco demais por causa de um artefato de planilha.
    #
    # Solução: adotar o calendário do CDI como oficial. Ele é, por definição,
    # o calendário de dias úteis (~252 por ano).
    calendario = cdi.index

    # Primeiro tapa buracos de feriado pontual, depois recorta no calendário
    # oficial. A ordem importa: assim um preço de sexta-feira sobrevive para
    # cobrir uma segunda de feriado, mas o sábado não vira um "pregão".
    precos = alinhar_ao_calendario(precos, calendario)
    cdi = cdi.reindex(calendario).ffill().fillna(0.0)

    return precos, cdi


def calcular_retornos(precos: pd.DataFrame) -> pd.DataFrame:
    """Transforma PREÇOS em RETORNOS diários.

    Retorno é a variação percentual de um dia para o outro. Se o preço vai de
    100 para 102, o retorno é +2% (ou 0,02 em fração decimal).

    Trabalhamos com retornos, e não com preços, porque retornos são comparáveis
    entre ativos: 2% de alta é 2% de alta tanto no Ibovespa (que vale ~170.000
    pontos) quanto no ouro (que vale ~360 dólares).

    NOTA DE SIMPLIFICAÇÃO (declarada no relatório): usamos o retorno de cada
    série na SUA moeda de origem, sem converter o resultado para reais. Isso é
    comum em estudos homogêneos de futuros, mas o painel atual também contém
    ETFs e índices; portanto seus resultados são cenários de pesquisa, não uma
    carteira implementável em reais.
    """
    # O argumento é deliberadamente explícito. O padrão antigo do pandas
    # preenchia NaN antes de calcular a variação, criando retornos de
    # "reabertura" enormes após lacunas longas. O padrão novo não deve mudar o
    # resultado quando a versão da biblioteca mudar.
    return precos.pct_change(fill_method=None)


# ==========================================================================
# BLOCO 3 — O SINAL: para onde cada ativo está andando?
# ==========================================================================
# Esta é a pergunta central do robô, feita uma vez por mês para cada ativo:
#
#     "Somando tudo, esse ativo subiu ou caiu no último ano
#      (ignorando o mês mais recente)?"
#
#     Subiu  -> COMPRADO  (+1): aposto que continua subindo
#     Caiu   -> VENDIDO   (-1): aposto que continua caindo
#
# "Vendido" merece explicação: significa que você GANHA quando o preço CAI.
# É isso que pode permitir ao robô ganhar dinheiro em crises — se o sinal já
# estiver vendido antes da queda. O resultado concreto de cada crise deve ser
# calculado no cenário auditado, não fixado como promessa neste comentário.
#
# Repare que o sinal é apenas +1 ou -1: não importa se subiu 5% ou 80%, a
# aposta tem o mesmo tamanho. Isso é deliberado — é mais robusto do que tentar
# calibrar a aposta pela intensidade da tendência.

def calcular_sinal(precos: pd.DataFrame, data: pd.Timestamp) -> pd.Series:
    """Devolve +1 (comprado), -1 (vendido) ou 0 (sem histórico) para cada ativo.

    REGRA ANTI-TRAPAÇA: esta função só recebe preços ATÉ `data`. Ela não tem
    como espiar o futuro, nem por acidente. Esse é o erro nº 1 em backtests
    ("look-ahead bias") e a defesa aqui é estrutural, não disciplinar.
    """
    # Fatia o histórico: nada depois da data do sinal existe para nós.
    ate_agora = precos.loc[:data]

    # Preço de HOJE menos 1 mês  (fim da janela do sinal)
    fim_janela = data - pd.DateOffset(months=PULA_MESES)
    # Preço de HOJE menos 13 meses (início da janela do sinal)
    inicio_janela = data - pd.DateOffset(months=JANELA_SINAL_MESES + PULA_MESES)

    sinais = {}
    for ativo in precos.columns:
        # Sem cotação utilizável na data da decisão, o instrumento não pode
        # receber uma nova posição. Não reutilizamos silenciosamente uma
        # cotação antiga além do limite definido na preparação dos dados.
        if data not in precos.index or pd.isna(precos.at[data, ativo]):
            sinais[ativo] = 0.0
            continue
        serie = ate_agora[ativo].dropna()

        # Ativo novo demais? Fica de fora até ter histórico suficiente.
        # (O DBC, por exemplo, só tem dados a partir de 2006 — então ele
        #  simplesmente não é negociado antes de 2007.)
        if len(serie) < HISTORICO_MINIMO_DIAS:
            sinais[ativo] = 0.0
            continue

        # Pega o último preço disponível em cada ponta da janela.
        preco_inicio = serie.loc[:inicio_janela]
        preco_fim = serie.loc[:fim_janela]
        if preco_inicio.empty or preco_fim.empty:
            sinais[ativo] = 0.0
            continue

        # O retorno acumulado no período: quanto R$1 teria virado.
        retorno_12m_menos_1m = preco_fim.iloc[-1] / preco_inicio.iloc[-1] - 1.0

        # np.sign devolve +1 se positivo, -1 se negativo, 0 se exatamente zero.
        sinais[ativo] = float(np.sign(retorno_12m_menos_1m))

    return pd.Series(sinais)


# ==========================================================================
# BLOCO 4 — O TAMANHO: quanto apostar em cada um?
# ==========================================================================
# Já sabemos a DIREÇÃO de cada aposta (bloco 3). Falta o TAMANHO.
#
# A regra é "peso proporcional a 1/volatilidade":
#
#     ativo calmo   (volatilidade baixa) -> posição GRANDE
#     ativo agitado (volatilidade alta)  -> posição PEQUENA
#
# Por quê? Imagine apostar o mesmo valor em títulos do tesouro americano
# (que oscilam ~5% ao ano) e em gás natural (que oscila ~50% ao ano). O gás
# sozinho decidiria o resultado da carteira inteira; os títulos seriam
# decoração. Dividindo pela volatilidade, cada ativo passa a contribuir com
# uma dose parecida de risco — e a diversificação realmente funciona.

def matriz_covariancia(retornos: pd.DataFrame, data: pd.Timestamp,
                       estimador: str | None = None) -> pd.DataFrame:
    """O 'mapa de risco' do momento: quanto cada ativo balança E quanto os
    pares andam juntos.

    Tudo que o robô sabe sobre risco sai daqui — tanto a volatilidade de cada
    ativo (a diagonal desta matriz) quanto a da carteira combinada. Ter uma
    fonte única evita a incoerência de medir cada coisa de um jeito.

    Os dois estimadores diferem SÓ no peso que cada dia recebe:
      "janela" -> os últimos 60 dias pesam igual; o dia 61 vale zero.
      "ewma"   -> o peso decai suavemente (ontem vale mais que anteontem).
    """
    estimador = estimador or ESTIMADOR_VOL

    if estimador == "janela":
        janela = retornos.loc[:data].tail(JANELA_VOL_DIAS)
    elif estimador == "ewma":
        janela = retornos.loc[:data].tail(JANELA_EWMA_DIAS)
    else:
        raise ValueError(f"estimador desconhecido: {estimador}")

    # Descarta ativos que ainda não existiam; trata feriado pontual como
    # "não houve movimento" (que é o correto para um mercado fechado).
    janela = janela.dropna(axis=1, how="all").fillna(0.0)
    if len(janela) < 20:          # amostra pequena demais para ser confiável
        return pd.DataFrame()

    if estimador == "janela":
        return janela.cov()

    # --- EWMA -----------------------------------------------------------
    # Cada dia recebe um peso: o mais recente vale 1, e cada dia anterior
    # vale LAMBDA (0,94) vezes o seguinte. Depois normalizamos para somarem 1.
    #
    #   ontem        -> 0.94^0  = 1.00  (100%)
    #   10 dias atrás -> 0.94^10 = 0.54  ( 54%)
    #   30 dias atrás -> 0.94^30 = 0.16  ( 16%)
    #
    # É esse decaimento suave que elimina o "efeito penhasco" da janela fixa.
    n = len(janela)
    idade = np.arange(n - 1, -1, -1)       # linha mais recente tem idade 0
    pesos_tempo = LAMBDA_EWMA ** idade
    pesos_tempo = pesos_tempo / pesos_tempo.sum()

    X = janela.to_numpy()
    # Convenção RiskMetrics: assume-se média zero (para horizontes curtos o
    # retorno médio diário é desprezível perto da oscilação).
    cov = (X * pesos_tempo[:, None]).T @ X
    return pd.DataFrame(cov, index=janela.columns, columns=janela.columns)


def calcular_volatilidade(retornos: pd.DataFrame, data: pd.Timestamp,
                          estimador: str | None = None) -> pd.Series:
    """O quanto cada ativo balança, em % ao ano.

    Volatilidade = desvio-padrão dos retornos diários. Traduzindo: se os
    retornos variam pouco em torno da média, a volatilidade é baixa (ativo
    calmo); se variam muito, é alta (ativo nervoso).

    Sai da diagonal da matriz de covariância: a "covariância de um ativo com
    ele mesmo" é exatamente a variância dele, e a raiz disso é a volatilidade.

    Multiplicamos por raiz de 252 para converter de "por dia" para "por ano" —
    convenção de mercado, e deixa o número comparável com o alvo de 10% a.a.
    """
    cov = matriz_covariancia(retornos, data, estimador)
    if cov.empty:
        return pd.Series(dtype=float)
    return pd.Series(np.sqrt(np.diag(cov)), index=cov.index) * np.sqrt(DIAS_UTEIS_ANO)


def calcular_pesos_brutos(sinal: pd.Series, vol: pd.Series) -> pd.Series:
    """Combina direção (+1/-1) com tamanho (1/volatilidade)."""
    # Ativos sem sinal (0) ou sem volatilidade medida ficam de fora.
    valido = (sinal != 0) & vol.notna() & (vol > 0)
    if not valido.any():
        return pd.Series(0.0, index=sinal.index)

    pesos = pd.Series(0.0, index=sinal.index)
    pesos[valido] = sinal[valido] / vol[valido]

    # Normaliza para que a soma dos tamanhos (em módulo) seja 1. Isso deixa a
    # escala neutra — quem controla a exposição final é o bloco 5, não este.
    soma = pesos.abs().sum()
    if soma > 0:
        pesos = pesos / soma
    return pesos


# ==========================================================================
# BLOCO 5 — A DEFESA: controlar o risco da carteira inteira
# ==========================================================================
# Este é o bloco mais importante do robô, e o que dá sentido ao nome "Miyagi":
# defesa antes do ataque.
#
# Os blocos 3 e 4 montaram uma carteira, mas ainda não sabemos o quanto ELA
# como um todo balança. Duas coisas mudam isso ao longo do tempo:
#
#   - a volatilidade de cada ativo sobe e desce;
#   - a correlação entre eles muda (em crise, tudo tende a andar junto,
#     e a diversificação encolhe justamente quando você mais precisa dela).
#
# Então medimos a volatilidade da CARTEIRA COMBINADA e a escalamos para bater
# o alvo de 10% ao ano:
#
#       fator = 10% / volatilidade_estimada_da_carteira
#
# Se a carteira está calma demais (5%), o fator é 2 e dobramos as posições.
# Se está agitada (30%), o fator é 0,33 e cortamos para um terço.
# A intenção é reduzir exposição quando o risco estimado sobe; o drawdown
# realizado continua sendo resultado do backtest, não consequência garantida.

def volatilidade_da_carteira(retornos: pd.DataFrame, pesos: pd.Series,
                             data: pd.Timestamp,
                             estimador: str | None = None) -> float:
    """Estima o quanto a carteira combinada deve balançar, em % ao ano.

    Não basta somar as volatilidades individuais: o que importa é como os
    ativos se movem EM CONJUNTO. Se um sobe quando o outro cai, eles se
    cancelam e a carteira balança menos que suas partes — é literalmente
    isto que a diversificação significa em números.

    Quem captura isso é a matriz de covariância: ela guarda, para cada par de
    ativos, o quanto eles costumam andar juntos. A fórmula w'Σw é a maneira
    padrão de combinar os pesos com essa matriz.
    """
    cov = matriz_covariancia(retornos, data, estimador)
    if cov.empty:
        return np.nan

    ativos_usados = [a for a in pesos[pesos != 0].index if a in cov.index]
    if not ativos_usados:
        return np.nan

    w = pesos[ativos_usados].to_numpy()
    variancia_diaria = float(w @ cov.loc[ativos_usados, ativos_usados].to_numpy() @ w)
    if variancia_diaria <= 0:
        return np.nan

    return float(np.sqrt(variancia_diaria) * np.sqrt(DIAS_UTEIS_ANO))


def aplicar_alvo_de_risco(pesos: pd.Series, vol_carteira: float) -> pd.Series:
    """Escala a carteira para a 'velocidade de risco' alvo de 10% ao ano."""
    if not np.isfinite(vol_carteira) or vol_carteira <= 0:
        return pesos * 0.0

    fator = VOL_ALVO_ANUAL / vol_carteira

    # Trava de segurança: nunca expor mais que 3x o capital, por mais calmo
    # que o mercado pareça. Mercado calmo é justamente quando as pessoas se
    # alavancam demais e quebram no primeiro susto.
    exposicao_atual = pesos.abs().sum()
    if exposicao_atual > 0:
        fator = min(fator, ALAVANCAGEM_MAX / exposicao_atual)

    return pesos * fator


# ==========================================================================
# BLOCO 6 — A SIMULAÇÃO: rodar mês a mês pela história
# ==========================================================================
# Agora juntamos tudo e "rebobinamos a fita" de 2005 até hoje, mês a mês,
# FINGINDO QUE NÃO SABEMOS O FUTURO.
#
# A cada fim de mês o robô:
#   1. calcula o sinal usando só dados até aquele dia         (bloco 3)
#   2. calcula os tamanhos                                     (bloco 4)
#   3. escala para o alvo de risco                             (bloco 5)
#   4. paga 0,1% sobre o que precisou negociar para chegar lá
#   5. carrega essas posições pelo mês seguinte, colhendo os retornos
#
# DUAS REGRAS ANTI-TRAPAÇA, que são o coração da credibilidade do backtest:
#
#   (a) O sinal do mês usa apenas dados ATÉ o fim daquele mês.
#   (b) Os retornos colhidos são do dia SEGUINTE em diante — nunca do próprio
#       dia da decisão. Na vida real você decide no fechamento e a posição só
#       passa a valer depois.
#
# Sem a regra (b), o backtest "compra" sabendo o resultado do dia — o que
# produz uma curva linda e completamente falsa.
#
# SOBRE O CAIXA: a representação pretendida é um overlay de futuros. Nesse caso
# você deposita margem e o restante fica rendendo CDI, de modo que o retorno
# total = CDI + resultado das posições - custos. O painel, porém, também inclui
# ETFs de retorno total. Para eles, ``ativos_financiados`` desconta a taxa do
# caixa da exposição líquida sob uma convenção simplificada e explicitamente
# auditável; FX, borrow, margem e moeda-base ainda exigem dados adicionais.
# (Esquecer o CDI foi um erro real da versão 1 deste projeto: sem ele, a
#  estratégia "perdia" para o CDI por um artefato contábil, não por desempenho.)


def derivar_pesos(
    pesos_atuais: pd.Series,
    retornos_do_dia: pd.Series,
    retorno_total: float,
    data: pd.Timestamp | None = None,
) -> pd.Series:
    """Atualiza pesos mantendo quantidades/notionais entre rebalanceamentos.

    Para patrimônio inicial igual a 1, o notional do ativo ``i`` passa de
    ``w_i`` para ``w_i * (1 + r_i)`` e o patrimônio passa para
    ``1 + retorno_total``. Retorno ausente mantém o notional inalterado; ele
    não é tratado como cotação observada e é registrado separadamente pelo
    chamador.

    Patrimônio nulo ou negativo encerra a simulação: depois de uma perda de
    100% não existem pesos economicamente definidos. Continuar com o vetor
    antigo produziria uma curva fictícia após a insolvência.
    """
    denominador = 1.0 + float(retorno_total)
    if not np.isfinite(denominador) or denominador <= 0.0:
        quando = f" em {pd.Timestamp(data).date()}" if data is not None else ""
        raise RuntimeError(
            "Patrimônio não positivo"
            f"{quando}: retorno diário total={float(retorno_total):.6f}."
        )

    retornos_validos = retornos_do_dia.reindex(pesos_atuais.index).fillna(0.0)
    return pesos_atuais * (1.0 + retornos_validos) / denominador


def calcular_resultado_posicoes(
    pesos: pd.Series,
    retornos_do_dia: pd.Series,
    taxa_caixa: float = 0.0,
    ativos_financiados: set[str] | None = None,
) -> tuple[float, float]:
    """Calcula P&L arriscado e encargo das pernas financiadas.

    Futuros e FX de retorno total são tratados como overlays: seu P&L é
    somado ao rendimento do caixa. Já uma série de retorno total de ETF exige
    capital. Sob a hipótese explícita de financiamento à taxa do caixa, sua
    contribuição excedente é ``w * (r - taxa_caixa)``.

    O encargo usa exposição *líquida assinada*. Uma posição vendida gera
    crédito de caixa neste modelo simplificado; borrow, margem e haircuts
    continuam fora do painel e devem ser reportados separadamente.
    """
    retornos_validos = retornos_do_dia.reindex(pesos.index).fillna(0.0)
    bruto = float((pesos * retornos_validos).sum())
    financiados = set(ativos_financiados or ())
    desconhecidos = financiados - set(pesos.index)
    if desconhecidos:
        raise ValueError(
            "Ativos financiados ausentes dos pesos: "
            + ", ".join(sorted(desconhecidos))
        )
    exposicao_liquida = float(pesos[list(financiados)].sum()) if financiados else 0.0
    encargo = float(taxa_caixa) * exposicao_liquida
    return bruto - encargo, encargo

def rodar_backtest(precos: pd.DataFrame, retornos: pd.DataFrame,
                   cdi: pd.Series, estimador: str | None = None,
                   universo_por_data: dict[pd.Timestamp, list[str]] | None = None,
                   inicio: str | pd.Timestamp | None = None,
                   ativos_financiados: set[str] | None = None) -> dict:
    """Roda a simulação completa e devolve as séries de resultado.

    ``universo_por_data`` é opcional e serve aos testes point-in-time. Cada
    lista passa a valer na sua data e continua válida até a próxima. O padrão
    mantém todas as colunas, reproduzindo a estratégia estática histórica.

    ``ativos_financiados`` permite auditar séries de retorno total que não são
    retornos excedentes de futuros. O padrão vazio preserva o estimando
    histórico; passar os ETFs desconta deles a taxa do caixa sem alterar
    silenciosamente a regra do sinal.
    """

    # Datas de rebalanceamento: último dia útil de cada mês, a partir de 2005.
    data_inicio = inicio if inicio is not None else INICIO
    datas_mes = retornos.loc[data_inicio:].resample(REBALANCEIA).last().index
    # Garante que cada data existe no calendário real de pregões.
    datas_rebal = [retornos.index[retornos.index <= d][-1]
                   for d in datas_mes if (retornos.index <= d).any()]
    datas_rebal = sorted(set(datas_rebal))

    # Exposição efetiva imediatamente antes de cada rebalanceamento. Ela
    # deriva com os preços entre rebalanceamentos; mantê-la igual ao alvo seria
    # simular, sem custo, um rebalanceamento diário que a regra não prevê.
    pesos_atuais = pd.Series(0.0, index=precos.columns)

    retorno_diario = {}      # resultado líquido, dia a dia
    log_pesos = {}           # pesos em cada rebalanceamento (para auditoria)
    log_pesos_pre_trade = {} # pesos derivados imediatamente antes da ordem
    log_giro = {}            # quanto foi negociado
    log_custo = {}           # quanto custou
    log_exposicao = {}       # exposição total (soma dos módulos dos pesos)
    log_pesos_diarios = {}   # exposição efetiva antes do retorno de cada dia
    log_faltantes = {}       # exposição cujo retorno diário estava ausente
    log_financiamento = {}   # encargo líquido das séries financiadas

    for i, data in enumerate(datas_rebal):
        # ---- 1, 2 e 3: decidir a carteira -----------------------------
        sinal = calcular_sinal(precos, data)
        if universo_por_data is not None:
            vigentes = [d for d in universo_por_data if d <= data]
            permitidos = set(universo_por_data[max(vigentes)]) if vigentes else set()
            sinal.loc[~sinal.index.isin(permitidos)] = 0.0
        vol = calcular_volatilidade(retornos, data, estimador)
        pesos_brutos = calcular_pesos_brutos(sinal, vol)
        vol_cart = volatilidade_da_carteira(retornos, pesos_brutos, data, estimador)
        pesos = aplicar_alvo_de_risco(pesos_brutos, vol_cart)
        pesos = pesos.reindex(precos.columns).fillna(0.0)

        # ---- 4: pagar o custo de chegar nessa carteira -----------------
        # "Giro" é o quanto mudou da carteira antiga para a nova. Se um peso
        # foi de 0,20 para 0,35, giramos 0,15 naquele ativo. Só pagamos pelo
        # que efetivamente mudou — manter posição não custa nada.
        log_pesos_pre_trade[data] = pesos_atuais.copy()
        giro = (pesos - pesos_atuais).abs().sum()
        custo = giro * CUSTO_POR_TRADE

        log_pesos[data] = pesos
        log_giro[data] = giro
        log_custo[data] = custo
        log_exposicao[data] = pesos.abs().sum()

        # O custo é debitado no próprio dia da execução.
        retorno_diario[data] = retorno_diario.get(data, 0.0) - custo
        pesos_atuais = pesos.copy()

        # ---- 5: carregar as posições até o próximo rebalanceamento -----
        proxima = datas_rebal[i + 1] if i + 1 < len(datas_rebal) else retornos.index[-1]

        # ATENÇÃO À REGRA ANTI-TRAPAÇA (b): o filtro é ">" e não ">=".
        # Colhemos o retorno a partir do dia SEGUINTE à decisão.
        periodo = retornos.loc[(retornos.index > data) & (retornos.index <= proxima)]

        for dia, retornos_do_dia in periodo.iterrows():
            log_pesos_diarios[dia] = pesos_atuais.copy()

            faltantes = retornos_do_dia.isna() & (pesos_atuais.abs() > 0)
            log_faltantes[dia] = float(pesos_atuais[faltantes].abs().sum())

            # Resultado das posições: peso × retorno de cada ativo, somado.
            # Em fechamento normal do mercado, retorno zero é a convenção
            # econômica usual. O mesmo fill também alcança lacunas longas, nas
            # quais zero é apenas hipótese de diagnóstico; por isso a exposição
            # afetada foi registrada acima e não é chamada de cotação observada.
            retornos_validos = retornos_do_dia.fillna(0.0)
            # Retorno total = CDI (dinheiro parado rendendo) + posições
            juros_do_dia = float(cdi.get(dia, 0.0))
            resultado_posicoes, encargo_financiamento = calcular_resultado_posicoes(
                pesos_atuais,
                retornos_validos,
                taxa_caixa=juros_do_dia,
                ativos_financiados=ativos_financiados,
            )
            log_financiamento[dia] = encargo_financiamento
            total_dia = resultado_posicoes + juros_do_dia
            retorno_diario[dia] = retorno_diario.get(dia, 0.0) + total_dia

            # Mantemos quantidades/notionais entre rebalanceamentos. Após a
            # oscilação do ativo e do patrimônio, os pesos efetivos mudam.
            pesos_atuais = derivar_pesos(
                pesos_atuais, retornos_do_dia, total_dia, data=dia
            )

    serie = pd.Series(retorno_diario).sort_index()

    return {
        "retornos": serie,
        "pesos": pd.DataFrame(log_pesos).T.sort_index(),
        "pesos_antes_rebalanceamento": pd.DataFrame(
            log_pesos_pre_trade
        ).T.sort_index(),
        "giro": pd.Series(log_giro).sort_index(),
        "custos": pd.Series(log_custo).sort_index(),
        "exposicao": pd.Series(log_exposicao).sort_index(),
        "pesos_diarios": pd.DataFrame(log_pesos_diarios).T.sort_index(),
        "exposicao_retorno_ausente": pd.Series(log_faltantes).sort_index(),
        "encargo_financiamento": pd.Series(log_financiamento).sort_index(),
    }


# ==========================================================================
# BLOCO 7 — AS MÉTRICAS: o resultado foi bom?
# ==========================================================================
# Aqui traduzimos a série de retornos em números que respondem perguntas
# humanas. As quatro que importam:
#
#   CAGR          "quanto rendeu por ano, em média?"
#   Volatilidade  "o quanto essa jornada balançou?"
#   Sharpe        "o retorno compensou o susto?"  <- a métrica mais importante
#   Max Drawdown  "qual foi a pior queda do topo até o fundo?"
#
# O Sharpe é a razão entre o retorno EXCEDENTE (o que rendeu ALÉM do CDI) e a
# volatilidade. Faz sentido: se você rende 12% ao ano mas o CDI paga 11%,
# você entregou muito pouco por todo o risco que correu.
#
# TESTE DE SANIDADE: a literatura de trend following reporta Sharpe entre
# 0,5 e 1,0. Se este código devolvesse 2,0, a conclusão correta NÃO seria
# "achamos algo genial" — seria "tem bug no código". Números bons demais são
# quase sempre erro, e é assim que se descobre um.

def calcular_metricas(retornos: pd.Series, cdi: pd.Series,
                      nome: str = "MIYAGI") -> dict:
    """Transforma a série diária de retornos nas métricas do relatório."""
    r = retornos.dropna()
    if r.empty:
        return {}

    # Tempo decorrido pelo CALENDÁRIO, não pela contagem de linhas. Contar
    # linhas e dividir por 252 só funciona se a série tiver exatamente 252
    # observações por ano — e qualquer descasamento de calendário quebra isso
    # silenciosamente (foi o que aconteceu na primeira versão deste código:
    # 21,5 anos reais viraram "24,7" e distorceram todo o resto).
    anos = (r.index[-1] - r.index[0]).days / 365.25

    # Patrimônio acumulado: quanto R$1 investido no início teria virado.
    patrimonio = (1 + r).cumprod()

    # CAGR: a taxa anual constante que levaria ao mesmo resultado final.
    cagr = float(patrimonio.iloc[-1] ** (1 / anos) - 1)

    vol = float(r.std() * np.sqrt(DIAS_UTEIS_ANO))

    # Excesso sobre o CDI, dia a dia.
    cdi_alinhado = cdi.reindex(r.index).fillna(0.0)
    excesso = r - cdi_alinhado
    sharpe = float(excesso.mean() / excesso.std() * np.sqrt(DIAS_UTEIS_ANO)) \
        if excesso.std() > 0 else np.nan

    # Max Drawdown: a maior queda do pico até o vale seguinte.
    # É a métrica do sofrimento: responde "qual o maior tombo que eu teria
    # que ter aguentado sem desistir?"
    pico = patrimonio.cummax()
    drawdown = patrimonio / pico - 1
    max_dd = float(drawdown.min())

    return {
        "nome": nome,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "anos": anos,
        "patrimonio": patrimonio,
        "drawdown": drawdown,
    }


def retorno_por_ano(retornos: pd.Series) -> pd.Series:
    """Retorno de cada ano-calendário — mostra a consistência (ou falta dela)."""
    return retornos.groupby(retornos.index.year).apply(lambda x: (1 + x).prod() - 1)


# ==========================================================================
# BLOCO 8 — OS GRÁFICOS
# ==========================================================================
# Duas figuras, porque são as duas perguntas que um avaliador faz:
#
#   "quanto rendeu?"        -> curva de patrimônio
#   "e quanto doeu?"        -> curva de drawdown ("submarino")
#
# A curva de patrimônio usa ESCALA LOGARÍTMICA. Parece detalhe técnico e não
# é: em escala normal, os últimos anos dominam visualmente e a crise de 2008
# vira um risquinho. Em escala log, uma queda de 50% tem o mesmo tamanho no
# gráfico esteja ela no começo ou no fim — que é o que permite comparar
# períodos honestamente.

def gerar_graficos(retornos: pd.Series, ibov: pd.Series, cdi: pd.Series,
                   destino: Path) -> None:
    """Salva as duas figuras do relatório."""
    import matplotlib
    matplotlib.use("Agg")          # sem janela: só grava arquivo
    import matplotlib.pyplot as plt

    COR_MIYAGI = "#1B4965"         # azul profundo
    COR_IBOV = "#8D99AE"           # cinza
    COR_CDI = "#C1121F"            # vermelho discreto

    patrimonio = (1 + retornos).cumprod()
    pat_ibov = (1 + ibov).cumprod()
    pat_cdi = (1 + cdi.reindex(retornos.index).fillna(0.0)).cumprod()

    # ---------------- Figura 1: patrimônio acumulado ----------------------
    fig, ax = plt.subplots(figsize=(11, 5.2))

    # Sombreia as duas crises: é onde a estratégia precisa se justificar.
    # O rótulo usa `get_xaxis_transform`: o x fica em coordenada de data, mas
    # o y em fração do eixo (0 = base, 1 = topo). Sem isso o texto é
    # posicionado antes das curvas existirem, e some para fora do gráfico.
    for inicio, fim, rotulo in [("2008-01-01", "2009-03-31", "crise 2008"),
                                ("2020-02-01", "2020-06-30", "COVID")]:
        ax.axvspan(pd.Timestamp(inicio), pd.Timestamp(fim),
                   color="#000000", alpha=0.055, zorder=0)
        ax.text(pd.Timestamp(inicio), 0.02, f" {rotulo}",
                transform=ax.get_xaxis_transform(),
                fontsize=7.5, color="#555555", va="bottom")

    ax.plot(patrimonio.index, patrimonio, color=COR_MIYAGI, lw=2.0, label="MIYAGI")
    ax.plot(pat_ibov.index, pat_ibov, color=COR_IBOV, lw=1.3, label="Ibovespa")
    ax.plot(pat_cdi.index, pat_cdi, color=COR_CDI, lw=1.3, ls="--", label="CDI")

    ax.set_yscale("log")
    ax.set_ylabel("patrimônio (escala log, base 1)")
    ax.set_title("Miyagi — patrimônio acumulado, 2005-2026",
                 fontsize=12, fontweight="bold", color=COR_MIYAGI, loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(alpha=0.22, lw=0.6)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    fig.tight_layout()
    fig.savefig(destino / "patrimonio.png", dpi=160)
    plt.close(fig)

    # ---------------- Figura 2: drawdown ("o sofrimento") -----------------
    # Drawdown responde: "se eu tivesse entrado no pior momento possível,
    # quanto eu estaria perdendo agora?" É a métrica que decide se um
    # investidor real aguentaria segurar a posição — ou desistiria no fundo.
    dd = patrimonio / patrimonio.cummax() - 1
    dd_ibov = pat_ibov / pat_ibov.cummax() - 1

    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.fill_between(dd_ibov.index, dd_ibov * 100, 0,
                    color=COR_IBOV, alpha=0.55, lw=0, label="Ibovespa")
    ax.fill_between(dd.index, dd * 100, 0,
                    color=COR_MIYAGI, alpha=0.85, lw=0, label="MIYAGI")

    ax.set_ylabel("queda do topo (%)")
    ax.set_title("Quanto doeu — drawdown", fontsize=12, fontweight="bold",
                 color=COR_MIYAGI, loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.grid(alpha=0.22, lw=0.6)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    fig.tight_layout()
    fig.savefig(destino / "drawdown.png", dpi=160)
    plt.close(fig)


# ==========================================================================
# EXECUÇÃO
# ==========================================================================

def main() -> None:
    print("=" * 74)
    print("MIYAGI — robô de tendência multi-ativo com alvo de risco")
    print('"Não prevê o golpe. Observa o movimento — e responde com técnica."')
    print("=" * 74)

    precos, cdi = carregar_dados()
    retornos = calcular_retornos(precos)
    print(f"\nDados: {precos.index.min():%Y-%m-%d} a {precos.index.max():%Y-%m-%d}"
          f"  |  {len(ATIVOS)} ativos")

    resultado = rodar_backtest(precos, retornos, cdi)
    r = resultado["retornos"]

    # --- Comparação com os benchmarks ---
    # Ibovespa: comprado e parado (a alternativa óbvia para um investidor BR).
    ibov = retornos["^BVSP"].reindex(r.index).fillna(0.0)
    cdi_serie = cdi.reindex(r.index).fillna(0.0)

    m_miyagi = calcular_metricas(r, cdi, "MIYAGI")
    m_ibov = calcular_metricas(ibov, cdi, "Ibovespa")
    m_cdi = calcular_metricas(cdi_serie, cdi, "CDI")

    print(f"Período do backtest: {r.index.min():%Y-%m-%d} a {r.index.max():%Y-%m-%d}"
          f"  ({m_miyagi['anos']:.1f} anos)")
    print(f"Rebalanceamentos: {len(resultado['giro'])}"
          f"  |  giro médio: {resultado['giro'].mean():.2f}"
          f"  |  custo total: {resultado['custos'].sum():.2%}")
    print(f"Exposição média: {resultado['exposicao'].mean():.2f}x"
          f"  (máx {resultado['exposicao'].max():.2f}x)")

    # --- Tabela principal ---
    print("\n" + "=" * 74)
    print(f"{'':<12}{'CAGR':>10}{'Vol a.a.':>11}{'Sharpe':>10}{'Max DD':>11}"
          f"{'2008':>10}{'2020':>10}")
    print("-" * 74)

    por_ano = {
        "MIYAGI": retorno_por_ano(r),
        "Ibovespa": retorno_por_ano(ibov),
        "CDI": retorno_por_ano(cdi_serie),
    }

    for m in (m_miyagi, m_ibov, m_cdi):
        nome = m["nome"]
        a08 = por_ano[nome].get(2008, np.nan)
        a20 = por_ano[nome].get(2020, np.nan)
        sharpe_txt = f"{m['sharpe']:>10.2f}" if nome != "CDI" else f"{'—':>10}"
        print(f"{nome:<12}{m['cagr']:>9.1%}{m['vol']:>11.1%}{sharpe_txt}"
              f"{m['max_drawdown']:>11.1%}{a08:>10.1%}{a20:>10.1%}")
    print("=" * 74)

    # --- Análise crítica: o que os números escondem ---
    anos_miyagi = por_ano["MIYAGI"]
    pior_ano = anos_miyagi.idxmin()
    print(f"\nPior ano: {pior_ano} ({anos_miyagi.min():.1%})")
    print(f"Anos positivos: {(anos_miyagi > 0).sum()} de {len(anos_miyagi)}")

    print("\nRetorno por ano:")
    for ano, valor in anos_miyagi.items():
        barra = "#" * max(0, int(abs(valor) * 100 / 3))
        sinal = "+" if valor >= 0 else "-"
        print(f"  {ano}  {valor:>7.1%}  {sinal}{barra}")

    # --- Teste de sanidade ---
    print("\n" + "-" * 74)
    s = m_miyagi["sharpe"]
    if s > 1.5:
        print(f"[!] ALERTA: Sharpe {s:.2f} está acima de 1,5 — a literatura reporta")
        print("    0,5 a 1,0 para trend following. Isso é forte indício de BUG ou")
        print("    viés no código. Investigue ANTES de acreditar no resultado.")
    elif 0.5 <= s <= 1.0:
        print(f"[ok] Sharpe {s:.2f} está DENTRO da faixa da literatura (0,5-1,0).")
        print("     Modesto o bastante para ser crível.")
    elif 0.3 <= s < 0.5:
        print(f"[~] Sharpe {s:.2f} está LIGEIRAMENTE ABAIXO da faixa da literatura")
        print("    (0,5-1,0). A estratégia entrega, mas menos do que os artigos")
        print("    reportam. Isso precisa ir para o relatório como está — não é")
        print("    um número para arredondar para cima nem para esconder.")
    elif s < 0.3:
        print(f"[!] Sharpe {s:.2f} é baixo demais para justificar a estratégia")
        print("    frente ao CDI. Vale reexaminar premissas antes de defender.")
    else:
        print(f"[?] Sharpe {s:.2f} está entre 1,0 e 1,5 — acima do típico.")
        print("    Não é alarme, mas vale conferir se não há viés escondido.")

    # --- Salva os resultados ---
    saida = AQUI / "resultados"
    saida.mkdir(exist_ok=True)
    pd.DataFrame({
        "retorno_diario": r,
        "patrimonio": m_miyagi["patrimonio"],
        "drawdown": m_miyagi["drawdown"],
    }).to_csv(saida / "serie_miyagi.csv")
    resultado["pesos"].to_csv(saida / "pesos_miyagi.csv")

    gerar_graficos(r, ibov, cdi, saida)
    print(f"\nResultados salvos em resultados/  "
          f"(serie_miyagi.csv, pesos_miyagi.csv, patrimonio.png, drawdown.png)")


if __name__ == "__main__":
    main()
