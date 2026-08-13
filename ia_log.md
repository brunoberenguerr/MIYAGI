# Log de uso de IA generativa — MIYAGI

> Critério 6 do edital (15% da nota): *como a IA foi utilizada, valor agregado
> ao projeto, exemplos concretos de aplicação, limitações encontradas.*

Este documento registra os usos concretos de IA generativa no projeto, com o
resultado de cada um e o que deu errado. Está organizado do uso de maior valor
para o de menor — e o de maior valor não foi escrever código.

---

## 1. IA como revisora adversarial — o uso mais valioso

Foi aqui que a IA mudou o resultado do projeto, não apenas acelerou o trabalho.
Em quatro ocasiões ela encontrou erros que produziam números plausíveis e
errados — o tipo mais perigoso, porque não quebra nada e não aparece em teste.

### 1.1 Vazamento temporal no backtest do MARÉ

**O que foi pedido:** revisar o motor de backtest antes de confiar nos números.

**O que a IA encontrou:** na última perna de acumulação, o código usava o fim de
*todo* o painel de dados em vez do fim do período pedido:

```python
next_trade = (plan.trade_date.iloc[i + 1] if i + 1 < len(plan)
              else market.dates[-1])   # <- deveria ser o `end` do split
```

**Como foi provado:** um teste de fronteira — pedir um backtest de 6 meses
(jan–jun/2008) e medir até onde a série ia.

| | antes | depois da correção |
|---|---|---|
| Último retorno (pedido: 30/06/2008) | 17/07/2026 | 30/06/2008 |
| Pregões na série | 4.659 | 120 |

**Impacto:** 49% da série de "design" era, na verdade, período de validação. Com
a correção, a série ativa do MARÉ passou de Sharpe +1,008 no design para
−0,698 na validação — não é o sinal enfraquecendo, é ele invertendo. Foi o que
levou ao abandono da estratégia.

### 1.2 Erro de calendário no MIYAGI — encontrado no próprio código da IA

**Contexto:** a IA reconstruiu o backtest do Miyagi e, ao revisar a própria
saída, notou "24,7 anos" para um período de 2005 a 2026 (que tem 21,5).

**A causa:** o arquivo de preços tem ~290 datas por ano, não ~252 — o câmbio
negocia em dias sem pregão na bolsa. Contar anos como "linhas ÷ 252" inflava o
tempo decorrido, e três coisas quebravam juntas: o CDI rendia em dias
inexistentes, a volatilidade era subestimada (dias sem pregão viram retorno
zero) e o robô se alavancava demais por achar o mercado calmo.

**A confirmação:** o mesmo erro estava no backtest original da equipe. Aplicando
a contagem inflada ao CDI real, obtém-se 9,21% — exatamente o 9,2% reportado
antes. Como o Sharpe mede retorno *acima* do CDI, subestimá-lo inflava o Sharpe
em ~0,14.

### 1.3 Dois defeitos no código que a própria IA escreveu

Encontrados por ela ao revisar a saída, sem que ninguém apontasse:

- **Um teste de sanidade que afirmava algo falso.** O código imprimia
  "Sharpe 0,41 está na faixa esperada (0,5–1,0)" — mas 0,41 não está nessa
  faixa. A condição (`0.4 <= s <= 1.2`) não batia com o texto. Num arquivo cujo
  propósito é honestidade de medição, era um erro grave.
- **Rótulos de gráfico invisíveis.** As legendas das crises eram posicionadas
  antes das curvas existirem e caíam fora da área visível.

**Valor:** um revisor que audita o próprio trabalho com o mesmo rigor que aplica
ao alheio.

---

## 2. Três hipóteses testadas e refutadas

Resultados negativos foram registrados com o mesmo cuidado que os positivos.
Todos estão no histórico do repositório.

| hipótese | teste | veredito |
|---|---|---|
| O resíduo do MARÉ ainda carrega estrutura setorial | correlação intra vs. entre setores | **refutada** — a PCA já remove 97,7% do efeito |
| A divergência do MIYAGI vem do estimador de volatilidade | janela de 60 dias vs. EWMA | **refutada** — diferença imaterial (−20,4% vs −20,3%) |
| A queda pós-2016 é decaimento pós-publicação | Sharpe antes vs. depois de 2012 | **refutada** — o pós-publicação foi *melhor* (0,43 vs 0,35) |

Cada refutação economizou trabalho: a primeira evitou implementar neutralização
setorial inútil; a segunda encerrou a caça a um estimador melhor; a terceira
redirecionou a investigação para a causa correta.

---

## 3. Descoberta de mecanismo — a IA encontrou a explicação, não só o número

**Pergunta:** por que a estratégia parou de bater o CDI depois de 2016?

**O que a IA mediu:** a duração média das tendências.

| período | duração média |
|---|---|
| 2005–2010 | 11,5 meses |
| 2016–2020 | **6,9 meses** |

O sinal olha 12 meses para trás. Com tendências de 7 meses, o robô entra
sistematicamente atrasado — a tendência já virou quando ele se posiciona.

**A confirmação independente:** um robô que escolhe o horizonte mês a mês,
usando apenas dados passados, migrou sozinho de 18 meses (2005–2009) para
6 meses (2020–2024). Duas medições por caminhos diferentes apontando o mesmo
mecanismo.

---

## 4. Previsão registrada antes do teste

Antes de expandir o universo de 8 para 40 ativos, a IA registrou uma previsão
quantitativa no histórico do repositório (commit `88392ef`, anterior à
existência do script de backtest):

> `Sharpe_novo = 0,51 × √(11,3 / 7,0) = 0,65`

com os critérios de julgamento declarados junto: perto → mecanismo entendido;
muito acima → sinal de erro; muito abaixo → a correlação média superestima a
diversificação real.

**Resultado:** 0,60 na configuração base (previsão equivalente: 0,52) e 0,48 no
walk-forward (previsão: 0,65). Parcialmente confirmada, e a discrepância gerou
o achado de que adaptar horizonte e diversificar ativos são **substitutos**.

O valor aqui não é acertar — é que o timestamp do git torna impossível ter
escrito a previsão depois de ver o resultado.

---

## 5. Seleção do universo

**Etapa inicial:** funil de 26 candidatos → clustering hierárquico de
correlações → 8 ativos, com correlação média de 0,04.

**Etapa de expansão:** a IA propôs e executou a ampliação do pool para 114
candidatos, identificando a maior lacuna do universo antigo: commodities
apareciam quase só como cesta (DBC), quando petróleo, cobre, café e boi têm
ciclos independentes. Foram acrescentados 20 futuros de commodities
individuais, 4 futuros de Treasury, 22 índices de bolsa em moeda local e 14
pares de câmbio.

**Salvaguarda aplicada:** o critério de seleção permaneceu idêntico ao original
e cego a retorno — o representante de cada cluster é o medoide (o ativo mais
correlacionado com os próprios colegas), com desempate por histórico.

**Resultado:** 7,0 → 11,3 apostas efetivas, e as duas reprovações de robustez do
universo antigo foram corrigidas.

---

## 6. Geração de código

Todo o backtest do MIYAGI foi escrito por IA a partir da especificação da
equipe, com uma exigência explícita: **comentar cada bloco em linguagem
acessível a quem não programa**. O arquivo `backtest_miyagi.py` está dividido
em 8 blocos numerados, e é possível entender a estratégia inteira lendo apenas
os comentários.

Também foram gerados por IA: a suíte de robustez com critérios declarados antes
da execução, o funil de seleção expandido, a análise de períodos e a comparação
walk-forward vs. treino/teste.

---

## 7. Limitações encontradas

Registradas porque o edital pede, e porque omiti-las tornaria o resto menos
crível.

**A IA erra, e erra de forma plausível.** Dois dos quatro bugs encontrados
estavam em código que ela mesma escreveu. O erro de calendário produzia números
perfeitamente razoáveis — CAGR de 15,2%, Sharpe de 0,42 — e só foi detectado
porque alguém estranhou um "24,7 anos" numa linha de diagnóstico. **Código
gerado por IA precisa da mesma auditoria que código humano.**

**Ela também especula com confiança excessiva.** Ao ver o salto de desempenho
em 2021–2026, a IA atribuiu o ganho às commodities. O teste de jackknife
contradisse: retirá-las *melhora* o resultado do período inteiro (0,60 → 0,67).
A atribuição era plausível e não medida — e foi corrigida no repositório.

**Ambiente sem acesso a dados de mercado.** O ambiente de execução não tinha
conexão a fontes pagas; contornado usando dados públicos do Yahoo Finance
baixados previamente. Limitação residual: séries de futuros contínuos têm
descontinuidade na rolagem de contrato, o que não é corrigido.

**Ela tende a concordar.** Quando questionada sobre a necessidade de dividir os
dados em treino e teste, a IA precisou ser explicitamente instruída a verificar
o edital antes de responder — e a leitura mostrou que a exigência não existia.
A resposta útil veio de checar a fonte, não de aceitar a premissa.

---

## 8. O que a IA não fez

- **Não escolheu a estratégia.** A tese de trend following foi decisão da equipe,
  fundamentada na literatura.
- **Não otimizou parâmetros.** Todos vieram de artigos publicados. Quando testes
  mostraram configurações melhores (horizonte 9-1 daria Sharpe 0,68 contra os
  0,60 do 12-1), **nenhuma foi adotada** — trocar após ver o resultado é o
  overfitting que invalidaria o trabalho.
- **Não decidiu o que vai no relatório final.** As escolhas de conteúdo e
  ênfase são da equipe.
