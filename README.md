# MIYAGI — robô de tendência multi-ativo com alvo de risco

> *"Não prevê o golpe. Observa o movimento — e responde com técnica."*

Estratégia de **time-series momentum** (trend following) sobre 8 ativos de 6
classes, com controle de risco por volatilidade alvo. Desafio Quant AI 2026.

---

## A ideia em um parágrafo

Quando uma notícia sai, o preço não incorpora tudo de uma vez — sobe aos poucos,
porque as pessoas demoram a acreditar e depois entram na onda. Resultado:
**tendências duram meses**. O Miyagi não tenta adivinhar o futuro; olha para onde
cada mercado já está andando e vai junto — comprado no que sobe, vendido no que
cai. É um prêmio documentado em 58 mercados e um século de dados
(Moskowitz, Ooi & Pedersen 2012; Hurst, Ooi & Pedersen 2017).

**Por que "Miyagi"?** Como o mestre: movimentos simples repetidos com disciplina,
e defesa antes do ataque. O sinal é deliberadamente simples; o controle de risco
vem primeiro.

## Como rodar

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install pandas numpy matplotlib
python backtest_miyagi.py
```

Produz as métricas no terminal e grava séries, pesos e as duas figuras em
`resultados/`.

## Estrutura

```
backtest_miyagi.py       o backtest completo, comentado bloco a bloco para leigos
comparar_estimadores.py  diagnóstico: janela de 60 dias vs. EWMA
dados/
  prices.csv             preços diários dos 8 ativos (2003-2026)
  cdi.csv                CDI diário (o "dinheiro parado" rende isso)
selecao_universo/        o funil 26 -> 8 ativos: código, matriz, heatmaps
docs/                    esboço inicial, emblema, edital, metodologia
resultados/              séries, pesos e figuras gerados pelo backtest
```

## As regras da estratégia

| decisão | regra | de onde veio |
|---|---|---|
| **Sinal** | retorno acumulado de 12 meses, ignorando o último mês; positivo → comprado, negativo → vendido | Moskowitz et al. (2012) |
| **Tamanho** | peso ∝ 1/volatilidade (janela 60 dias) | convenção da literatura |
| **Defesa** | carteira escalada para vol alvo de 10% a.a., alavancagem ≤ 3x | padrão de managed futures |
| **Rebalanceamento** | mensal, último dia útil | — |
| **Custos** | 0,1% sobre o valor negociado | premissa declarada |
| **Caixa** | rende CDI (é estratégia de futuros: você posta margem, o resto rende) | — |

**Zero grid search.** Nenhum parâmetro foi escolhido por dar o melhor resultado
no backtest — todos vêm dos artigos. Testar 500 combinações e ficar com a melhor
é o erro clássico que transforma um backtest lindo em prejuízo real.

Onde a especificação permitia duas opções (volatilidade por janela simples ou
EWMA), as duas foram testadas e o critério de escolha está registrado no código:
a diferença é imaterial (drawdown −20,4% vs −20,3%), e venceu a mais simples de
explicar — não a de melhor número. Ver `comparar_estimadores.py`.

## Os 8 ativos

Ibovespa · S&P 500 · Treasuries 7-10a · USD/BRL · EUR/USD · USD/JPY · Ouro ·
Commodities

Escolhidos por um funil de correlação (26 candidatos → clustering hierárquico →
8). Correlação média entre eles: **0,04** — o que equivale a ~6,1 apostas
realmente independentes. Diversificação é o motor: com o mesmo sinal, o Sharpe
cresce com a raiz do número de apostas independentes.

O USD/BRL é o diversificador-chave: correlação −0,36 com o Ibovespa, então sobe
justamente quando o book local sofre.

---

## Resultados (2005-2026, 21,5 anos)

| | CAGR | Vol a.a. | Sharpe\* | Max DD | 2008 | 2020 |
|---|---|---|---|---|---|---|
| **MIYAGI** | **15,0%** | **11,0%** | **0,41** | **−20,4%** | +3,9% | +6,8% |
| Ibovespa | 9,7% | 25,7% | 0,09 | −60,0% | −41,2% | +2,9% |
| CDI | 10,7% | 0,2% | — | 0,0% | +12,4% | +2,8% |

\* Sharpe do excesso sobre o CDI. Custos de 0,1%/trade cobrados. 259
rebalanceamentos, giro médio 0,89, custo acumulado 23,1%.

**19 anos positivos de 22.** Pior ano: 2021 (−5,2%).

### Leitura honesta

O robô entregou **mais de 50% a mais de retorno que o Ibovespa com 43% da
volatilidade**, e um tombo máximo três vezes menor. Ficou positivo nas duas
crises do período.

Mas o Sharpe de 0,41 está **abaixo** da faixa de 0,5-1,0 que a literatura
reporta. O código diz isso explicitamente na saída, e o número vai para o
relatório como está. Duas razões plausíveis: o CDI brasileiro é um piso alto
(10,7% a.a. no período — bater isso com folga é mais difícil do que bater a
taxa americana dos artigos), e o universo de 8 ativos é pequeno frente aos
50-60 dos estudos originais.

---

## Testes de robustez — 4 de 6 aprovados

`python robustez.py`. Seis testes com **critérios declarados antes da execução**,
para que não fosse possível olhar o resultado e inventar o critério que ele
satisfaz. A configuração base não mudou em função de nenhum deles.

| # | teste | critério declarado | resultado |
|---|---|---|---|
| A | Custos | superar o CDI com o dobro do custo | **aprovado** — break-even em ~0,5%/trade (5× o assumido) |
| B | Janela de volatilidade | Sharpe ≥ 0,25 de 20 a 250 dias | **aprovado** — mínimo 0,36 |
| C | Sub-períodos | superar o CDI em 3 dos 4 blocos | **REPROVADO** — 2 de 4 |
| D | Teto de alavancagem | com teto 2×, superar o CDI | **aprovado** — 14,3% vs 10,7% |
| E | Jackknife de ativos | Sharpe ≥ 0,25 retirando qualquer um | **REPROVADO** — sem SPY cai para 0,19 |
| F | Horizonte do sinal | positivo em 3 dos 4 horizontes | **aprovado** — 4 de 4 |

### C — a reprovação que mais importa

| período | CAGR | Sharpe | CDI | superou? |
|---|---|---|---|---|
| 2005–2010 | 16,2% | 0,32 | 12,9% | sim |
| 2011–2015 | 27,7% | **1,40** | 10,4% | sim |
| **2016–2020** | 6,7% | −0,04 | 7,8% | **não** |
| **2021–2026** | 11,0% | 0,03 | 11,3% | **não** |

**Os 21 anos de resultado são carregados por 2005–2015 — em particular pelo
bloco 2011–2015, com Sharpe 1,40. Depois de 2016 a estratégia não supera o CDI.**

Isso precisa estar no relatório com esse destaque. Duas leituras possíveis, e a
honesta é dizer que não sabemos separá-las com os dados que temos:

1. **Fenômeno conhecido:** a década de 2010 foi documentadamente difícil para
   trend following (volatilidade baixa, intervenção de bancos centrais,
   reversões frequentes). Não seria uma anomalia nossa.
2. **Fim do edge:** o prêmio de momentum pode ter sido arbitrado conforme o
   capital em managed futures cresceu.

### E — dependência do SPY

Retirar o SPY derruba o Sharpe de 0,41 para 0,19. A tese do trabalho é
diversificação com 8 apostas pouco correlacionadas, e o resultado depende mais
de um ativo do que essa tese sugeriria.

Nota: retirar EUR/USD **melhora** o Sharpe para 0,57, e retirar o Ibovespa
melhora para 0,52. **Isso não é um convite para removê-los.** Escolher os ativos
pelo desempenho no backtest é exatamente o overfitting que o funil de correlação
(feito ex-ante, por critério estatístico) foi desenhado para evitar. Fica como
diagnóstico, não como decisão.

### O que os testes B e F dizem a favor do trabalho

A configuração base **não é a melhor** em duas dimensões testadas:

- janela de volatilidade: 250 dias dá Sharpe 0,50 contra os 0,41 dos 60 dias;
- horizonte do sinal: 18-1 dá 0,73 e 9-1 dá 0,69, contra os 0,41 do 12-1.

Isso é evidência forte de que **não houve garimpo de parâmetro**: quem escolhe
parâmetros olhando o resultado não termina com a terceira melhor opção de quatro.
Os valores vieram da literatura e ficaram como estavam.

---

### O que o backtest NÃO prova

- **Não há garantia de que funcione daqui pra frente.** Trend following passou
  por décadas ruins (1990s, 2010s parciais).
- **O pior tombo é recente e demorado:** −20,4% entre abr/2020 e mar/2021, 338
  dias submerso. É o "chicote" clássico — o mercado despenca, o robô se
  posiciona para a queda, e a recuperação em V o pega na contramão.
- **Custos são premissa, não medição.** 0,1% por trade é razoável para ETFs e
  futuros líquidos, mas não foi verificado contra execução real.

---

## Nota sobre o esboço inicial

O pré-relatório em `docs/` foi um **esboço** da estratégia, com números
preliminares (CAGR 15,5%, Sharpe 0,57, max DD −14,5%). Este repositório é a
implementação de fato, e os números diferem.

A principal diferença foi rastreada até uma causa concreta: **um erro de
contagem de tempo**. O arquivo de preços tem ~290 datas por ano, não ~252 — o
câmbio negocia em dias sem pregão na bolsa, então sábados e feriados entram na
planilha. Contando anos como "linhas ÷ 252", 21,5 anos reais viram 24,7
aparentes, e o CDI acumulado dividido por um número inflado de anos cai de
10,68% para **9,21%** (o esboço reportava 9,2%).

Como o Sharpe mede o retorno **acima do CDI**, subestimar o CDI em 1,5 p.p.
infla o Sharpe em ~0,14 — quase exatamente a diferença entre 0,57 e 0,41.

O mesmo erro apareceu na primeira versão deste código e está corrigido: o
calendário oficial passou a ser o de dias úteis do CDI, e os anos são contados
pelo calendário e não por contagem de linhas. Fica registrado porque é o tipo
de erro que não aparece em teste de unidade — o backtest roda, produz números
plausíveis, e está errado.
