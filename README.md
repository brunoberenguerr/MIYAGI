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

### Por que não há divisão treino/teste

`python analise_periodos.py`

Dividir em treino/teste serve para quando o modelo **aprende** algo dos dados —
você separa um pedaço que ele nunca viu para checar se aprendeu ou decorou.

**O Miyagi não aprende nada dos dados.** Sinal 12-1, janela de 60 dias, alvo de
10%, teto de 3× e os 8 ativos vieram todos de fora: da literatura ou de um funil
de correlação feito ex-ante. Nenhum foi escolhido olhando retorno. Nesse sentido,
**os 21,5 anos inteiros já são out-of-sample** — não existe pedaço contaminado
por ajuste porque não houve ajuste.

(Era diferente no MARÉ: lá havia parâmetros calibrados, então o design/holdout
era obrigatório.)

Ainda assim, a divisão foi feita — não como treino/teste, mas como antes/depois:

| divisão | período | Sharpe | t-stat | IC 95% |
|---|---|---|---|---|
| tudo | 2005–2026 | 0,41 | 1,88 | [−0,02 , 0,83] |
| antes | 2005–2016 | 0,61 | 2,10 | [0,04 , 1,18] |
| depois | 2017–2026 | 0,16 | 0,50 | [−0,47 , 0,80] |

**O intervalo de confiança do período inteiro inclui o zero.** Com t = 1,88, o
resultado está no limite da significância — 21 anos ainda é amostra curta para
medir um Sharpe pequeno. Isso é limitação honesta do trabalho, não defeito do
código.

### Hipótese testada e REFUTADA: decaimento pós-publicação

A explicação natural para a queda seria o efeito documentado por McLean & Pontiff
(2016): anomalias publicadas perdem retorno quando o capital entra e as arbitra.
Moskowitz et al. publicaram em 2012. Se a hipótese valesse, a quebra estaria lá.

| período | Sharpe |
|---|---|
| 2005–2011 (pré-publicação) | 0,35 |
| 2012–2026 (pós-publicação) | **0,43** |

**O pós-publicação foi melhor.** A hipótese não se sustenta — a quebra não está
em 2012, está por volta de 2016.

### A causa mecânica: as tendências encurtaram 40%

| período | trocas de direção/ano | duração média da tendência |
|---|---|---|
| 2005–2010 | 8,3 | **11,5 meses** |
| 2011–2015 | 11,0 | 8,7 meses |
| 2016–2020 | 14,0 | **6,9 meses** |
| 2021–2026 | 12,7 | 7,5 meses |

Esta é a explicação econômica, e ela é coerente: o sinal olha 12 meses para trás.
Quando as tendências duravam ~11,5 meses, o sinal chegava a tempo. Quando passaram
a durar ~7, o robô entra sistematicamente atrasado — a tendência já virou quando
ele se posiciona. É o "chicote", medido.

A queda é ampla, não concentrada num ativo (contribuição anualizada das posições):

| | 2005–2015 | 2016–2026 | Δ |
|---|---|---|---|
| Ibovespa | +0,7% | −1,6% | −2,3% |
| Commodities | +1,2% | −1,1% | −2,2% |
| Ouro | +2,5% | +1,3% | −1,3% |
| USD/BRL | +1,0% | −0,1% | −1,1% |
| S&P 500 | +2,5% | +2,7% | +0,2% |
| **total** | **+9,8%** | **+2,3%** | **−7,5%** |

---

## Parâmetros treinados: walk-forward vs treino/teste

`python treino_parametros.py`

O Miyagi base não treina nada — 12-1 fixo, da literatura. Faz sentido dar a ele
parâmetros que aprendem? E qual é a forma honesta de fazer isso?

**Por que testar o horizonte do sinal, e não outra coisa:** o diagnóstico de
períodos mediu que as tendências encurtaram de 11,5 para 6,9 meses. O sinal olha
12 meses para trás; se a tendência dura 7, o robô entra atrasado por construção.
A hipótese econômica veio **antes** do teste — e essa ordem é o que separa
pesquisa de garimpo.

### Resultado no holdout (2017–2026), CDI de 9,1%

| abordagem | CAGR | Sharpe | t | Max DD |
|---|---|---|---|---|
| Base (12-1 fixo, sem treino) | 10,4% | 0,16 | 0,50 | −20,4% |
| **1. Walk-forward (causal)** | **13,0%** | **0,38** | 1,16 | −20,1% |
| 2. Treino/teste (18-1 congelado) | 16,8% | 0,69 | 2,14 | −19,4% |

Treinar melhorou nos dois casos. Mas **os dois números não valem o mesmo.**

### A tese econômica foi confirmada de forma independente

O walk-forward escolhe o horizonte mês a mês, olhando só o passado. O que ele
escolheu ao longo do tempo:

| período | horizonte médio | mais escolhido |
|---|---|---|
| 2005–2009 | 13,7 meses | 18-1 |
| 2010–2014 | 11,8 meses | 9-1 |
| 2015–2019 | 12,6 meses | 12-1 |
| **2020–2024** | **9,9 meses** | **6-1** |
| 2025–2026 | 9,0 meses | 9-1 |

**O robô migrou sozinho para horizontes mais curtos depois de 2020** — sem que
ninguém programasse isso, e usando apenas dados passados. É exatamente o que a
tese das "tendências encurtando" previa. Duas medições independentes apontando
para o mesmo mecanismo é a evidência mais forte deste trabalho.

### Por que o 0,69 do treino/teste NÃO deve ser reportado como out-of-sample

Três problemas, em ordem de gravidade:

**1. Contaminação de processo.** Nós já tínhamos rodado `robustez.py` e visto
que 18-1 era o melhor horizonte do período inteiro (Sharpe 0,73) **antes** de
rodar este "treino". O treino então "descobriu" 18-1. O código respeita as datas;
o processo de pesquisa não. Nenhuma implementação conserta isso.

**2. Múltiplos testes.** Cinco horizontes testados, com erro-padrão do Sharpe de
~0,29 no período de treino. O espalhamento observado no treino (0,15 a 0,77) é
compatível com ruído puro. Escolher o máximo de cinco sorteios ruidosos e
apresentá-lo como habilidade é o erro clássico.

**3. O holdout já tinha sido visto.** Analisamos 2017–2026 em detalhe — Sharpe,
ativos, duração das tendências — antes deste teste. Um holdout observado deixa
de ser holdout.

### A distinção que vale para o relatório

> **Disciplina de código** — o motor garante que o sinal só vê o passado.
> **Disciplina de processo** — depende de quando o pesquisador olhou os dados.
>
> A primeira é verificável. A segunda depende de honestidade, e é justamente
> por isso que precisa ser declarada.

O walk-forward tem as duas. O treino/teste tem só a primeira.

### DECISÃO: o walk-forward é a versão adotada (v2)

Decidido em 13/08/2026. O walk-forward passa a ser a versão principal da
estratégia, apresentada **ao lado** da base — não no lugar dela.

| | período inteiro (2005–2026) | recente (2017–2026) |
|---|---|---|
| Base (12-1 fixo) | 0,41 (t=1,88) | 0,16 |
| **Walk-forward (adotado)** | **0,51 (t=2,36)** | **0,38** |

Três razões, em ordem de peso:

1. **Cruza o limiar de significância.** Com t = 2,36 o resultado é
   estatisticamente distinguível de zero pela convenção usual (t > 2); a base,
   com 1,88, não é. É uma diferença qualitativa, não só um número maior.
2. **Entra na faixa da literatura** (0,5–1,0), onde a base ficava de fora.
3. **A justificativa econômica veio antes do teste** e foi confirmada por um
   caminho independente (o robô migrou sozinho para horizontes curtos quando as
   tendências encurtaram).

**Por que a base continua sendo reportada junto:** ela tem zero parâmetros
ajustados, o que é o argumento mais forte possível para "tratamento adequado de
vieses". A comparação entre as duas *é* a análise crítica — mostra o ganho de
adaptar e o custo em complexidade.

*Ressalva honesta:* a família de horizontes {3, 6, 9, 12, 18} e a janela de
aprendizado de 36 meses foram escolhas nossas. São todas convenções da
literatura de momentum, mas foram fixadas por nós — e isso também é uma
liberdade que, em princípio, poderia ser explorada.

---

## Universo expandido: 8 → 40 ativos

`python expandir_pool.py` → `python funil_expandido.py` → `python backtest_expandido.py`

O pool de candidatos foi ampliado de 42 para **114** — a maior lacuna era
commodities, que apareciam quase só como cesta (DBC). Petróleo, cobre, café e
boi têm ciclos próprios; uma cesta funde tudo numa aposta só. Foram incluídos 20
futuros de commodities individuais, 4 futuros de Treasury, 22 índices de bolsa
em moeda local e 14 pares de câmbio.

O funil aplicado foi **idêntico** ao original (correlação de retornos mensais
log, clustering hierárquico average linkage sobre 1−ρ, corte em 0,35, histórico
≥15 anos). O representante de cada cluster é o **medoide**, com desempate por
histórico — regra cega a retorno.

| | ativos | ρ médio | apostas efetivas |
|---|---|---|---|
| atual | 8 | 0,020 | 7,0 |
| expandido | 40 | 0,065 | **11,3** |

### A previsão foi registrada antes do backtest

Commit `88392ef`, anterior à existência de `backtest_expandido.py`:

> `Sharpe_novo = 0,51 × √(11,3 / 7,0) = 0,65`

### O resultado — todas as quatro combinações

| | Sharpe | t | CAGR | Vol | Max DD |
|---|---|---|---|---|---|
| 8 ativos · 12-1 fixo | 0,41 | 1,88 | 15,0% | 11,0% | −20,4% |
| 8 ativos · walk-forward | 0,51 | 2,36 | 16,3% | 11,0% | −20,1% |
| **40 ativos · 12-1 fixo** | **0,60** | **2,78** | 17,2% | 10,4% | −21,6% |
| 40 ativos · walk-forward | 0,48 | 2,22 | 15,6% | 10,3% | −24,2% |

**A previsão de 0,65 não foi batida pelo walk-forward (0,48).** Mas o quadro é
mais interessante que um simples erro:

- Para a **configuração base**, a previsão equivalente era 0,41 × 1,273 = 0,52.
  O realizado foi **0,60** — a expansão entregou o que a teoria previa, e um
  pouco mais.
- Para o **walk-forward**, a previsão era 0,65 e o realizado 0,48. **A adaptação
  de horizonte, que ajudava com 8 ativos, atrapalha com 40.**

### Por que o walk-forward deixou de ajudar

A explicação econômica: **adaptar o horizonte e diversificar ativos são
substitutos, não complementos.** Ambos servem para não depender de um único
regime de tendência. Com 8 ativos, trocar de horizonte era a única defesa
disponível contra tendências que encurtavam. Com 40 ativos espalhados por
commodities, câmbios, juros e bolsas, sempre há algum mercado em tendência
longa — a diversificação já faz esse trabalho, e a troca de horizonte vira
giro extra sem ganho.

O drawdown confirma: piora de −21,6% (fixo) para −24,2% (walk-forward).

### O controle de risco segurou

Preocupação legítima: com 40 ativos e janela de 60 dias, q = 0,67, e a
covariância amostral tende a **subestimar** o risco — o que viraria alavancagem
excessiva. Medido explicitamente:

| | vol realizada | alvo |
|---|---|---|
| 40 ativos · 12-1 | 10,4% | 10% |
| 40 ativos · walk-forward | 10,3% | 10% |

O alvo segurou mesmo com 5× mais ativos.

### Sub-períodos: o perfil mudou de forma relevante

| período | 8 ativos (WF) | 40 ativos (WF) | CDI |
|---|---|---|---|
| 2005–2010 | 0,32 | 0,37 | 12,6% |
| 2011–2015 | **1,40** | 0,60 | 10,4% |
| 2016–2020 | −0,04 | −0,31 | 7,8% |
| **2021–2026** | **0,03** | **1,21** | 11,3% |

O universo expandido inverte o problema recente: o bloco 2021–2026 salta de
0,03 para **1,21**. Economicamente coerente — 2021–22 teve tendências fortes e
duradouras em energia, grãos e metais, exatamente o que 20 commodities
individuais capturam e uma cesta única dilui.

**Mas não extrapole isso.** Um Sharpe de 1,21 num bloco de 5 anos, concentrado
num choque inflacionário específico, não é previsão de futuro. A fraqueza de
2016–2020 continua lá, e piorou.

### Robustez no universo de 40: 6 de 6 aprovados

`python robustez.py --universo 40`

| # | teste | 8 ativos | 40 ativos |
|---|---|---|---|
| A | Custos | aprovado | **aprovado** |
| B | Janela de volatilidade | aprovado | **aprovado** (mín. 0,54) |
| C | Sub-períodos | **REPROVADO** (2/4) | **aprovado** (3/4) |
| D | Teto de alavancagem | aprovado | **aprovado** |
| E | Jackknife | **REPROVADO** | **aprovado** |
| F | Horizonte do sinal | aprovado | **aprovado** (4/4) |

**As duas reprovações do universo de 8 foram corrigidas pela expansão.** Isso é
consistente com a tese: as duas falhas eram sintomas de diversificação
insuficiente.

**C — sub-períodos.** Passou de 2/4 para 3/4:

| período | Sharpe | CDI | superou? |
|---|---|---|---|
| 2005–2010 | 0,40 | 12,9% | sim |
| 2011–2015 | 1,04 | 10,4% | sim |
| **2016–2020** | **−0,23** | 7,8% | **não** |
| 2021–2026 | 1,20 | 11,3% | sim |

O buraco de 2016–2020 **continua** — e é o único que resiste a tudo que
tentamos. Ele precisa aparecer no relatório.

**E — jackknife por classe.** Nenhuma classe é indispensável:

| retirando | Sharpe | Δ |
|---|---|---|
| commodities (17) | **0,67** | **+0,07** |
| câmbio (10) | 0,43 | −0,17 |
| setores (4) | 0,56 | −0,04 |
| índices de bolsa (4) | 0,58 | −0,02 |
| ETFs de ações (3) | 0,56 | −0,04 |

### Correção a uma atribuição anterior

Ao ver o salto de 2021–2026, escrevi que ele vinha das commodities (energia,
grãos e metais em 2021–22). **O jackknife contradiz isso:** retirar as 17
commodities *melhora* o Sharpe do período inteiro, de 0,60 para 0,67.

As duas coisas podem coexistir — commodities ajudando muito em 2021–22 e
atrapalhando em outros períodos (o colapso do petróleo em 2014–16, por
exemplo). Mas a atribuição que fiz era especulação, não medição, e o dado
disponível aponta no sentido contrário. O que sustenta o resultado com mais
força é o **câmbio** (pior classe para se retirar, −0,17).

### A configuração base segue não sendo a melhor

| dimensão | base | melhor testado |
|---|---|---|
| horizonte | 12-1 → 0,60 | 9-1 → **0,68** |
| janela de vol | 60d → 0,60 | 20d → **0,61** |
| teto de alavancagem | 3× → 0,60 | 5× → **0,63** |

Nenhuma foi adotada. Quem garimpa parâmetro não termina com a configuração que
perde em três de três dimensões testadas.

---

## ⚠️ CORREÇÃO MAIOR: o carrego cambial (interest rate carry)

`python carrego.py` → `python resultado_final.py`

### O erro

O backtest usava **apenas a variação do preço à vista** das moedas. Para câmbio
isso está errado, e o erro é enorme em moedas de juro alto.

Para ficar comprado em USD/TRY (vendido em lira) é preciso tomar lira
emprestada e **pagar a taxa turca** — que chegou a 40–50% ao ano, contra 0–5,5%
nos EUA. O preço à vista da lira caiu 18% ao ano e isso parecia lucro. O
carrego que teríamos pago era da mesma ordem.

Há um argumento teórico que fecha o caso: pela paridade descoberta de juros,
moedas de juro alto se desvalorizam aproximadamente pelo diferencial. **O fato
de a lira ter caído 18% ao ano é, ele mesmo, evidência de que o diferencial era
dessa ordem.**

### A correção, com dados reais de juros

Séries do FRED (IMF/IFS). Cada par virou um índice de **retorno total**:
`retorno_total = retorno_do_preço + carrego`.

| par | só preço | com carrego | diferença |
|---|---|---|---|
| **TRY=X** | +18,0% a.a. | **−1,6% a.a.** | **−19,6 p.p.** |
| BRL=X | +2,5% | −10,3% | −12,8 |
| ZAR=X | +4,3% | −0,7% | −5,0 |
| MXN=X | +1,9% | −2,8% | −4,7 |
| INR=X | +3,4% | −1,5% | −4,9 |
| JPY=X | +1,8% | **+3,3%** | **+1,6** |

A paridade de juros se confirma quase exatamente: a lira sai de +18% para
−1,6%. O iene vai na direção **oposta** (+1,6%), porque os juros japoneses eram
menores que os americanos — ficar comprado em USD/JPY *recebe* carrego.

<em>Limitação:</em> G10 (AUD, CAD, GBP, SEK, EUR) não tem série pública completa
no FRED e ficou sem correção. O diferencial dessas moedas contra o dólar é de
0–5 p.p., ordem de grandeza muito menor que o da lira, mas o erro residual
existe e está declarado.

### O RESULTADO FINAL — comparação justa

| universo | sem carrego | **com carrego (correto)** |
|---|---|---|
| 8 ativos | 0,41 (t=1,88) | **0,36 (t=1,64)** |
| **40 ativos** | 0,60 (t=2,78) | **0,50 (t=2,30)** |

**A expansão do universo ajudou de verdade: 0,36 → 0,50.** E só a versão de 40
ativos mantém significância estatística (t > 2).

### A previsão de diversificação se confirma nos dados limpos

A metodologia registrada em `88392ef` era escalar o Sharpe pela raiz das
apostas efetivas. Aplicada aos dados corrigidos:

```
previsto = 0,36 × raiz(11,2 / 7,0) = 0,36 × 1,265 = 0,455
realizado                                        = 0,50
erro                                             = +0,045
```

Dentro da banda de ±0,10 que foi **declarada como "confirmada" antes de
qualquer resultado existir**. O ganho de diversificação era real; o que estava
errado era o instrumento, não a teoria.

<small class="note">Ressalva: a previsão original (0,65) usava o walk-forward
sobre dados não corrigidos. O número acima é a mesma fórmula recalculada sobre
a base limpa — a metodologia foi registrada antes, o insumo mudou.</small>

### Concentração depois da correção

| | Sharpe | t |
|---|---|---|
| universo completo (40) | 0,50 | 2,30 |
| sem TRY=X | 0,41 | 1,89 |
| sem todos os câmbios emergentes | 0,40 | 1,87 |
| sem câmbio nenhum | 0,43 | 1,97 |

A dependência da lira **caiu pela metade** (impacto de −0,18 para −0,09), mas
ela segue como maior contribuidora (+1,08% a.a. contra +1,92% antes). Mesmo sem
nenhuma moeda, o universo de 40 rende 0,43 — ainda acima dos 0,36 do universo
de 8.

### Sub-períodos, corrigidos

| período | CAGR | Sharpe | CDI | |
|---|---|---|---|---|
| 2005–2010 | 19,8% | 0,58 | 12,9% | ok |
| 2011–2015 | 16,8% | 0,62 | 10,4% | ok |
| **2016–2020** | **2,4%** | **−0,45** | 7,8% | **não** |
| 2021–2026 | 24,4% | 1,15 | 11,3% | ok |

O buraco de 2016–2020 **piorou** com a correção (−0,23 → −0,45). Continua sendo
a fraqueza que resiste a tudo.

---

## Histórico: a análise que levou à correção do carrego

<small class="note">Mantido porque documenta como o erro foi encontrado. Os
números abaixo são os <strong>não corrigidos</strong> e foram superados pela
seção acima.</small>

### ⚠️ Concentração: o resultado depende da lira turca

O gráfico de contribuição por ativo revelou o que o jackknife por classe tinha
escondido: **TRY=X sozinho contribui 1,92% ao ano**, quatro vezes mais que o
segundo colocado, num excesso total sobre o CDI de ~6,5%.

| | Sharpe | t | CAGR |
|---|---|---|---|
| base (40 ativos) | 0,60 | 2,78 | 17,2% |
| **sem TRY=X** | **0,42** | **1,95** | 15,1% |
| sem os 4 câmbios emergentes | 0,40 | 1,86 | 15,0% |
| sem os 3 maiores contribuidores | 0,36 | 1,68 | 14,4% |

**Sem a lira, o universo de 40 rende 0,42 — praticamente igual aos 0,41 do
universo de 8.** O ganho aparente da expansão veio quase todo de um ativo.

E o t-stat cai de 2,78 para 1,95, cruzando de volta para baixo do limiar
convencional de significância.

#### O que isso faz com a previsão registrada

| | Sharpe |
|---|---|
| previsto pela teoria de diversificação | 0,52 |
| realizado **com** TRY=X | 0,60 |
| realizado **sem** TRY=X | **0,42** |

Descontando o ativo excepcional, o resultado fica **abaixo** do previsto — que
era exatamente o terceiro cenário pré-registrado: *"a correlação média
superestima a diversificação real"*. Em regimes de estresse os ativos se movem
juntos, e a correlação média não captura isso.

**Conclusão revisada, mais honesta que a anterior:** expandir o universo **não**
entregou o ganho de diversificação que a teoria previa. O que entregou resultado
foi a inclusão acidental de uma tendência cambial excepcional — a lira turca
saiu de ~1,5 para ~40 por dólar ao longo do período, o cenário perfeito para
trend following, e improvável de se repetir.

#### Por que TRY=X permanece no universo

Ela foi selecionada pelo funil mecânico (medoide do seu cluster), com critério
puramente de correlação, antes de qualquer backtest. **Removê-la agora, porque
sabemos que rendeu bem, seria overfitting ao contrário** — a mesma falha
metodológica com o sinal invertido.

Ela fica, e a sensibilidade fica reportada ao lado. Quem lê decide o quanto
descontar.

### Ressalva de múltiplos testes

Já foram testadas quatro combinações de universo × parametrização. Com
erro-padrão do Sharpe de ~0,22 em 21 anos, o intervalo observado (0,41 a 0,60)
é compatível com o que puro ruído produziria ao se escolher o máximo de quatro
tentativas. O 0,60 deve ser lido com esse desconto — e a tabela completa das
quatro células está acima justamente para que ninguém precise adivinhar quantas
foram testadas.

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
