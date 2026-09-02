# Checkpoint 01 — Modelagem Linear para Aprendizado de Máquina

Aplicação de classificação supervisionada com `scikit-learn`: previsão de contratação de depósito a prazo em uma campanha de telemarketing bancário.

**Disciplina:** Modelagem Linear para Aprendizado de Máquina — MLAM
**Professor:** André Tritiack
**Turma:** *(preencher)*

---

## Objetivo

Desenvolver um fluxo completo de classificação — da carga de um arquivo CSV até a interpretação da matriz de confusão — demonstrando as decisões técnicas tomadas em cada etapa e não apenas a métrica final.

O projeto responde a uma pergunta de negócio concreta: **para quais clientes o banco deve ligar?** Cada ligação tem custo e a campanha original converteu apenas 11,7% dos contatos. Um modelo útil aqui não é o de maior acurácia, e sim o que encontra o maior número de clientes interessados dentro de uma lista de contato viável.

---

## Integrantes do grupo

| Nome | RM |
|---|---|
| *(preencher)* | *(preencher)* |
| *(preencher)* | *(preencher)* |
| *(preencher)* | *(preencher)* |
| *(preencher)* | *(preencher)* |

---

## Tema e dataset escolhidos

**Tema (Exercício 3): Economia** — *Bank Marketing*.

| | |
|---|---|
| **Dataset** | Bank Marketing Data Set (arquivo `bank-full.csv`) |
| **Fonte** | UCI Machine Learning Repository — <https://archive.ics.uci.edu/dataset/222/bank+marketing> |
| **Referência** | Moro, S., Cortez, P. e Rita, P. *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems, Elsevier, 62:22-31, 2014. |
| **Licença** | Creative Commons Attribution 4.0 (CC BY 4.0) |
| **Dimensões** | 45.211 linhas × 17 colunas (16 preditoras + alvo) |
| **Formato** | CSV com separador `;` |
| **Contexto** | Campanhas de telemarketing de um banco português entre maio de 2008 e novembro de 2010 |

Cada linha representa **um contato telefônico**, não um cliente — a mesma pessoa pode aparecer mais de uma vez.

O notebook também resolve o **Exercício 1** (Breast Cancer Wisconsin, exemplo do enunciado) e o **Exercício 2** (Adult / Census Income, <https://archive.ics.uci.edu/dataset/2/adult>).

---

## Descrição da variável-alvo

**`y` — o cliente contratou o depósito a prazo após o contato?**

| Valor original | Codificação | Significado | Frequência |
|---|---|---|---|
| `yes` | **1** | Contratou o depósito a prazo (**classe de interesse**) | 5.289 (11,7%) |
| `no` | **0** | Não contratou | 39.922 (88,3%) |

A base é **fortemente desbalanceada**. Isso define a régua da avaliação: um modelo que respondesse sempre `no` já alcançaria **88,3% de acurácia** sem ter aprendido nada. Por isso todas as métricas são reportadas ao lado da precisão e da revocação da classe `yes`.

**Custo dos erros:**

- **Falso positivo** (ligar para quem não contrata): custo de uma ligação sem retorno. Erro mais barato.
- **Falso negativo** (não ligar para quem contrataria): venda perdida que nunca aparece no relatório da campanha. Erro mais caro — e o que orientou a escolha final do modelo.

---

## Etapas realizadas

1. **Carga a partir de CSV** — leitura com `sep=";"`; o Exercício 2 exige ainda `names`, `skipinitialspace` e `na_values="?"`, pois o arquivo original do Adult não tem cabeçalho.
2. **Inspeção inicial** — dimensões, tipos, amostra e uso de memória.
3. **Qualidade dos dados** — verificação de ausentes, duplicados e valores inconsistentes.
4. **Análise exploratória** — distribuição das classes e cinco visualizações (distribuição do alvo, taxa de contratação por categoria, variáveis numéricas por classe, sazonalidade mensal, matriz de correlação).
5. **Tratamento de codificações especiais** — `pdays = -1` é código de ausência, não uma quantidade: criamos `contactado_antes` e zeramos o `-1`.
6. **Remoção de vazamento** — exclusão da variável `duration` (detalhes abaixo).
7. **Separação `X` / `y`** e **divisão treino/teste** 75/25 estratificada, com `random_state=42`.
8. **Pré-processamento** com `ColumnTransformer` dentro de `Pipeline`: `StandardScaler` nas 7 numéricas, `OneHotEncoder` nas 9 categóricas (16 → 51 colunas).
9. **Treinamento** de cinco modelos, incluindo uma linha de base.
10. **Avaliação** — acurácia, precisão, revocação, F1 e matriz de confusão, com interpretação em Markdown a cada etapa.
11. **Conclusão** com limitações e melhorias propostas.

### Decisões de pré-processamento e suas justificativas

| Decisão | Justificativa |
|---|---|
| Manter `unknown` como categoria própria | A ausência é informativa. Dos 36.959 registros com `poutcome = unknown`, 36.954 têm `pdays = -1`: não é dado perdido, significa "não houve campanha anterior". Imputar pela moda inventaria um dado inexistente. |
| Manter `balance` negativo | Saldo devedor é um valor válido, não erro de digitação. |
| Não remover outliers | Valores extremos de `balance`, `campaign` e `duration` são plausíveis; removê-los sem regra de negócio descartaria informação real. |
| Padronizar as numéricas | `balance` chega a dezenas de milhares; `campaign` fica abaixo de 10. Sem padronizar, a regressão logística converge mal. |
| One-Hot nas categóricas | `job`, `month` e `poutcome` não têm ordem natural; codificá-las como inteiros criaria uma grandeza artificial. |
| Todo o pré-processamento dentro do `Pipeline` | Garante que médias, desvios e categorias sejam calculados **somente no treino**, evitando vazamento do conjunto de teste. |

### Por que `duration` foi removida

`duration` (duração da ligação em segundos) é a variável mais correlacionada com o alvo (0,40; mediana de 164 s em `no` contra 426 s em `yes`) — e é justamente a que não pode ser usada.

A relação é circular: a conversa só se estende porque o cliente está aceitando a oferta. A duração só é conhecida **depois** que a ligação termina, ou seja, depois que o desfecho já ocorreu. Como o modelo precisa decidir **para quem ligar**, esse valor não existe no momento da decisão. Mantê-la produziria uma métrica inflada que desapareceria em produção. A própria documentação do UCI recomenda descartá-la em modelos com intenção preditiva realista.

O notebook **demonstra o efeito numericamente** (seção 3.3.12): reincluindo `duration`, a revocação da classe `yes` quase dobra — um ganho inteiramente ilusório.

---

## Algoritmos utilizados

Todos do `scikit-learn`, com `random_state=42` e a mesma divisão treino/teste:

- `DummyClassifier(strategy="most_frequent")` — linha de base
- `LogisticRegression` — modelo linear, versão padrão
- `LogisticRegression(class_weight="balanced")` — penaliza mais o erro na classe rara
- `DecisionTreeClassifier(max_depth=6)`
- `RandomForestClassifier(n_estimators=300, min_samples_leaf=3)`

---

## Principais resultados

### Comparação dos modelos (conjunto de teste, 11.303 contatos)

| Modelo | Acurácia | Precisão (yes) | Revocação (yes) | F1 (yes) | Positivos previstos |
|---|---:|---:|---:|---:|---:|
| Linha de base (sempre `no`) | 0,883 | 0,000 | 0,000 | 0,000 | 0 |
| Regressão logística | 0,894 | 0,683 | 0,184 | 0,290 | 356 |
| **Regressão logística (balanceada)** | **0,752** | **0,266** | **0,636** | **0,375** | **3.164** |
| Árvore de decisão (prof. 6) | 0,895 | 0,643 | 0,222 | 0,331 | 457 |
| Random Forest | 0,897 | 0,694 | 0,210 | 0,322 | 399 |

### O resultado central

**A acurácia não distingue os modelos.** Regressão logística, árvore e Random Forest ficam todos a menos de 1,5 ponto percentual da linha de base que nunca prevê contratação. Os três são conservadores: preveem `yes` raramente porque errar na classe rara custa pouco no erro total.

**A revocação da classe `yes` distingue.** Os três encontram entre 18% e 22% dos clientes interessados. A versão balanceada encontra **63,6%**.

### Modelo escolhido pelo grupo

**Regressão logística com `class_weight="balanced"`** — o modelo de **menor acurácia** entre todos os treinados.

A escolha é deliberada e decorre do custo dos erros definido no início do projeto. Matriz de confusão:

| | Previsto: não contrata | Previsto: contrata |
|---|---:|---:|
| **Real: não contratou** | 7.658 | 2.323 (falsos positivos) |
| **Real: contratou** | 481 (falsos negativos) | 841 |

Em termos operacionais: o modelo indica 3.164 ligações (28% da base) e captura 841 dos 1.322 clientes que contratariam. A taxa de acerto entre os indicados é de **26,6%, contra 11,7% de uma lista aleatória** — mais que o dobro da eficiência por ligação, cortando 72% do esforço da campanha.

### Resultados do Exercício 2 (Adult)

| Modelo | Acurácia | Revocação (>50K) | Precisão (>50K) | F1 (>50K) |
|---|---:|---:|---:|---:|
| Linha de base | 0,761 | 0,000 | 0,000 | 0,000 |
| Regressão logística | 0,854 | 0,612 | 0,733 | 0,667 |
| Random Forest | 0,867 | 0,624 | 0,776 | 0,692 |

Mesmo padrão: a classe majoritária é bem prevista (revocação acima de 0,92) e a classe de interesse fica em torno de 0,62.

---

## Estrutura do repositório

```
.
├── README.md
├── Checkpoint_01_Modelagem_Linear.ipynb   # notebook preenchido e executado
└── dados/
    ├── adult.csv                          # Exercício 2 — sem cabeçalho, ausentes como '?'
    └── bank-full.csv                      # Exercício 3 — separador ';'
```

O arquivo `dados/dados_saude_tumores.csv` (Exercício 1) é gerado automaticamente pelo notebook a partir do `scikit-learn`.

---

## Instruções para executar o notebook

### Requisitos

Python 3.9+ e as bibliotecas:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Execução local

```bash
git clone <URL-DESTE-REPOSITORIO>
cd <pasta-do-repositorio>
jupyter notebook Checkpoint_01_Modelagem_Linear.ipynb
```

No menu, use **Kernel → Restart & Run All**. A execução completa leva cerca de 2 a 4 minutos, sendo o Random Forest do Exercício 2 a etapa mais demorada.

### Execução no Google Colab

1. Abra <https://colab.research.google.com> e faça upload do notebook.
2. Crie uma pasta `dados/` e envie os dois CSVs, **ou** simplesmente rode as células: as de carga baixam automaticamente os arquivos caso não os encontrem localmente.

### Obtenção dos dados

Os CSVs estão versionados na pasta `dados/`. Para baixá-los da fonte original:

- **Bank Marketing:** <https://archive.ics.uci.edu/dataset/222/bank+marketing> → descompacte `bank.zip` → use `bank-full.csv`
- **Adult:** <https://archive.ics.uci.edu/dataset/2/adult> → use `adult.data`

### Reprodutibilidade

Todos os algoritmos usam `random_state=42` e a divisão treino/teste é estratificada, então os números deste README são reproduzíveis. Pequenas variações de última casa decimal podem ocorrer entre versões diferentes do `scikit-learn`.

---

## Conclusão do grupo

**O que a análise exploratória mostrou.** A base não tem valores nulos nem duplicatas, mas codifica desconhecidos como a categoria `unknown` em quatro colunas. A taxa de contratação é de 11,7%. Convertem acima da média: `student` (28,7%), `retired` (22,8%), escolaridade `tertiary` (15,0%), clientes sem financiamento imobiliário (16,7%) e, sobretudo, quem já havia aceitado uma oferta anterior (`poutcome = success`, 64,7%). Há sazonalidade marcante — maio concentra 13.766 contatos com apenas 6,7% de conversão, enquanto março converte 52% em 477 contatos. Todas essas são **associações observadas na amostra**, não relações de causa e efeito.

**O aprendizado principal do trabalho.** Em uma base desbalanceada, **a acurácia é uma métrica enganosa**. Ela é dominada pela classe majoritária e não distingue um modelo útil de um que nunca prevê a classe de interesse. Ao acrescentar precisão e revocação da classe `yes`, os modelos passaram a se diferenciar com clareza — e a decisão do grupo mudou: escolhemos deliberadamente o modelo de menor acurácia, porque é o que atende ao objetivo declarado no início do projeto.

**Limitações.**

- Dados de **um único banco português entre 2008 e 2010**, período que inclui a crise financeira. Não se transferem automaticamente para outro país, outra instituição ou o presente.
- A base descreve **contatos, não clientes**. Como não há identificador, o mesmo cliente pode estar em treino e teste, o que viola a premissa de independência da divisão aleatória.
- **`duration` foi removida** por vazamento, e era a variável mais associada ao alvo. O modelo é honesto, mas necessariamente mais fraco do que resultados publicados que mantêm essa coluna.
- Avaliamos **uma única divisão treino/teste**, sem validação cruzada e sem ajuste de hiperparâmetros. As diferenças pequenas entre modelos estão dentro da margem de variação esperada.
- A escolha do modelo foi baseada em um raciocínio **qualitativo** sobre o custo dos erros. Sem os custos reais de ligação e a receita média por contratação, continua sendo uma decisão de critério, não de otimização.
- O modelo **descreve padrões estatísticos observados nesta base**. Não comprova nem garante que um cliente específico contratará, e não deve ser usado como justificativa isolada para decidir sobre pessoas.

**Próximos passos.** Validação cruzada estratificada; ajuste do limiar de decisão a partir de `predict_proba` em vez do corte padrão de 0,5; métricas independentes de limiar (PR AUC); busca de hiperparâmetros otimizando F1 da classe `yes`; discretização de `age` para capturar o efeito não linear; algoritmos de boosting; análise dos coeficientes para explicabilidade; e curva de ganho acumulado, que traduz o modelo na linguagem da operação.
