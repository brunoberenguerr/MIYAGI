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

**O que permanece não reconciliado:** o drawdown máximo e o retorno de 2008. Não
são explicados pelo calendário e provavelmente vêm de escolhas de implementação
(estimador de volatilidade EWMA vs. janela simples; como a volatilidade da
carteira combinada é estimada). Sem o código original não dá para fechar essa
conta — e isso está declarado aqui em vez de escondido.
