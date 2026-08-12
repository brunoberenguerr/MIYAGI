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

## Estrutura

```
backtest_miyagi.py    o backtest completo, comentado bloco a bloco para leigos
dados/
  prices.csv          preços diários dos 8 ativos (2003-2026)
  cdi.csv             CDI diário (o "dinheiro parado" rende isso)
selecao_universo/     o funil 26 -> 8 ativos: código, matriz de correlação, heatmaps
docs/                 pré-relatório, emblema, edital, metodologia
resultados/           séries e pesos gerados pelo backtest
```

## As regras da estratégia

| decisão | regra | de onde veio |
|---|---|---|
| **Sinal** | retorno acumulado de 12 meses, ignorando o último mês; positivo → comprado, negativo → vendido | Moskowitz et al. (2012) |
| **Tamanho** | peso ∝ 1/volatilidade (janela 60 dias) | convenção da literatura |
| **Defesa** | carteira escalada para vol alvo de 10% a.a., alavancagem ≤ 3x | padrão de managed futures |
| **Rebalanceamento** | mensal, último dia útil | — |
| **Custos** | 0,1% sobre o valor negociado | premissa declarada |
| **Caixa** | rende CDI (é uma estratégia de futuros: você posta margem, o resto rende) | — |

**Zero grid search.** Nenhum parâmetro foi escolhido por dar o melhor resultado
no backtest — todos vêm dos artigos. Testar 500 combinações e ficar com a melhor
é o erro clássico que faz um backtest lindo virar prejuízo real.

## Os 8 ativos

Ibovespa · S&P 500 · Treasuries 7-10a · USD/BRL · EUR/USD · USD/JPY · Ouro ·
Commodities

Escolhidos por um funil de correlação (26 candidatos → clustering hierárquico →
8). Correlação média entre eles: **0,04** — o que equivale a ~6,1 apostas
realmente independentes. Diversificação é o motor: com o mesmo sinal, o Sharpe
cresce com a raiz do número de apostas independentes.

O USD/BRL é o diversificador-chave: correlação −0,36 com o Ibovespa, então sobe
justamente quando o book local sofre.

## Resultados (2005-2026, 21,5 anos)

| | CAGR | Vol a.a. | Sharpe\* | Max DD | 2008 | 2020 |
|---|---|---|---|---|---|---|
| **MIYAGI** | 15,1% | 11,0% | **0,41** | −20,4% | +4,6% | +6,8% |
| Ibovespa | 9,7% | 25,7% | 0,09 | −60,0% | −41,2% | +2,9% |
| CDI | 10,7% | 0,2% | — | 0,0% | +12,4% | +2,8% |

\* Sharpe do excesso sobre o CDI. Custos de 0,1%/trade cobrados.

**Leitura honesta:** o robô entregou mais que o dobro do retorno do Ibovespa com
menos da metade da volatilidade, e um drawdown máximo três vezes menor. Ficou
positivo nas duas crises do período. Mas o Sharpe de 0,41 é modesto — está
ligeiramente abaixo da faixa de 0,5-1,0 da literatura, e isso precisa constar no
relatório em vez de ser maquiado.

### Teste de sanidade

A literatura reporta Sharpe de 0,5 a 1,0 para trend following. Se este código
devolvesse 2,0, a conclusão correta **não** seria "achamos algo genial" — seria
"tem bug". Números bons demais quase sempre são erro.

---

## ⚠️ Divergência com o pré-relatório — e o bug que a explica

O pré-relatório de julho reporta números diferentes dos acima:

| | pré-relatório | este código | causa |
|---|---|---|---|
| CAGR MIYAGI | 15,5% | 15,1% | ~igual |
| Vol | 10,9% | 11,0% | ~igual |
| **Sharpe** | **0,57** | **0,41** | ver abaixo |
| **CDI (benchmark)** | **9,2%** | **10,7%** | **bug de calendário** |
| Max DD | −14,5% | −20,4% | não reconciliado |
| 2008 | +9,9% | +4,6% | não reconciliado |

**A causa do gap no Sharpe está identificada.** O arquivo de preços tem ~290
datas por ano, não ~252: o câmbio negocia em dias que a bolsa fecha, então
sábados e feriados entram na planilha. Contando os anos como "linhas ÷ 252",
21,5 anos reais viram **24,7 anos aparentes** — e o CDI acumulado, dividido por
um número inflado de anos, cai de 10,68% para **9,21%**.

O pré-relatório reporta 9,2%. A reprodução é exata.

Como o Sharpe mede o retorno **acima do CDI**, subestimar o CDI em 1,5 p.p.
infla o Sharpe em ~1,5/10,9 ≈ 0,14 — que é quase exatamente a diferença entre
0,57 e 0,41.

Evidência de que os dados subjacentes são os mesmos: CDI de 2008 (+12,4%), CDI de
2020 (+2,8%) e Ibovespa de 2008 (−41,2%) batem **exatamente** entre as duas
versões. O que diverge é a contagem do tempo, não o dado.

### Hipótese testada e REFUTADA: o estimador de volatilidade

A especificação permitia medir volatilidade de duas formas ("vol EWMA ou janela
de 60 dias"). A hipótese natural era que o pré-relatório tivesse usado EWMA e
que isso explicasse o resto da divergência. Rode `comparar_estimadores.py`:

| | janela 60d | EWMA 0,94 | pré-relatório |
|---|---|---|---|
| CAGR | 15,0% | 14,6% | 15,5% |
| Volatilidade | 11,0% | 11,0% | 10,9% |
| Sharpe | 0,41 | 0,38 | 0,57 |
| Max Drawdown | −20,4% | −20,3% | −14,5% |
| 2008 | +3,9% | +3,9% | +9,9% |

**Os dois estimadores dão praticamente o mesmo resultado.** A hipótese foi
refutada — o que também é um resultado, e fica registrado.

### O que permanece não reconciliado

O drawdown máximo e o retorno de 2008. Três observações relevantes:

1. **O pior tombo da reconstrução é de abr/2020 a mar/2021** (−20,4%, 338 dias),
   não em 2008. É o "chicote" clássico do trend following: queda forte, robô se
   posiciona para ela, e a recuperação em V o pega na contramão.

2. **O posicionamento de 2008 está correto.** Vendido em S&P desde fevereiro,
   comprado em Treasuries o ano todo, e virando o Ibovespa para vendido em
   outubro. A virada tardia é o custo estrutural da estratégia no ponto de
   inflexão — não é defeito de implementação.

3. **Os drawdowns da reconstrução são consistentes** (17-20% em 2008, 2009,
   2010 e 2021), não um episódio isolado. Com volatilidade de 11% a.a., um
   drawdown máximo de 20% em 21 anos equivale a 1,9× a vol anual — dentro da
   faixa típica de trend followers (1,5× a 2,5×). Os −14,5% do pré-relatório
   seriam 1,3×, o que é excepcionalmente bom para um período que inclui 2008
   e 2020.

Somando ao bug do CDI já comprovado, a leitura mais provável é que o backtest
original contenha outros problemas de medição. **Isso não está provado** — sem o
código original não dá para fechar a conta, e a afirmação fica registrada como
hipótese, não como fato.
