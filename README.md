<h1 align="center">🦷 Workshop Hands-On Databricks — Odontoprev</h1>
<h3 align="center">Detecção de Fraude em Reembolsos: do Batch ao Tempo Real</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white" alt="Databricks">
  <img src="https://img.shields.io/badge/Unity_Catalog-00A972?style=for-the-badge&logo=databricks&logoColor=white" alt="Unity Catalog">
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" alt="MLflow">
  <img src="https://img.shields.io/badge/Model_Serving-FF3621?style=for-the-badge&logo=databricks&logoColor=white" alt="Model Serving">
  <img src="https://img.shields.io/badge/Lakebase-003366?style=for-the-badge&logo=postgresql&logoColor=white" alt="Lakebase">
  <img src="https://img.shields.io/badge/Databricks_Apps-FF3621?style=for-the-badge&logo=databricks&logoColor=white" alt="Apps">
</p>

<p align="center">
  Workshop prático de <strong>Machine Learning, Model Serving, Lakebase (feature store online),
  Databricks Apps e GenAI (Agent Bricks)</strong> aplicados à detecção de fraude no canal de
  <strong>reembolso (livre escolha)</strong> de uma operadora de planos odontológicos.
</p>

---

## 🎯 O caso de negócio

A Odontoprev — líder em planos odontológicos na América Latina — paga reembolso quando o
beneficiário usa a **livre escolha**: procura um dentista fora da rede credenciada, paga do
próprio bolso e pede o valor de volta apresentando **nota fiscal**.

É o canal de **maior risco de fraude** de toda a operação, por um motivo estrutural: a
operadora não viu o atendimento. Ela só vê o documento.

**A regra que existe hoje:** pedido acima de **R$ 1.500** vai para análise manual.

O resultado é o pior dos dois mundos — e no Lab 01 você vai **medir** isso:

| | Regra do teto |
|---|---|
| Precision | **~2%** — de cada 100 pedidos analisados, ~98 são legítimos |
| Recall | **~5%** — deixa passar 95% das fraudes |

Por que falha? Porque **valor alto é normal em odontologia**: implante, prótese e coroa
custam milhares de reais legitimamente. E as fraudes bem feitas são desenhadas para ficar
**logo abaixo** do teto.

### E a lacuna que o batch não resolve

Um modelo em batch entrega a fila de manhã. Mas a fraude de reembolso se decide **no instante
em que o pedido entra** — se o score chega no dia seguinte, o dinheiro já saiu. Por isso o
workshop não termina no modelo: ele vai até o **app que decide em milissegundos**.

### E a pergunta que o modelo não responde

O score diz **onde olhar**. Ele não diz **o que fazer**. Antes de glosar, o auditor precisa da
norma: esse plano cobre implante? cabe recurso? qual o teto? Nenhuma dessas respostas está nos
40 mil pedidos históricos — estão num PDF de política interna.

Por isso o Lab 05 fecha o outro lado com um **Knowledge Assistant**: o modelo prioriza, o
agente fundamenta com citação da seção, o auditor decide.

### E a causa que ninguém trata

Todos os labs acima agem **depois** que o pedido de reembolso entrou. Mas por que o
beneficiário usou reembolso? Muitas vezes porque **não achou um credenciado perto** — e nas
UFs de rede rala isso é legítimo (o Amazonas tem 3 credenciados em 3 especialidades; São
Paulo tem 136 em 10).

O Lab 06 constrói um **agente de busca de dentistas** com tool calling. Cada beneficiário que
encontra credenciado perto é um reembolso que não acontece. E quando a busca volta **vazia**, o
mesmo dado vira a evidência auditável de que aquele reembolso é devido.

---

## 📋 Agenda

| # | Atividade | Duração | Descrição |
|---|-----------|---------|-----------|
| ⚙️ | **Setup Inicial** | 15 min | Catálogo + schema isolado + geração dos dados sintéticos |
| 1️⃣ | **Lab 1 — ML para Detecção de Fraude** | 80 min | Baseline, target leakage, feature engineering relacional, PR AUC, threshold econômico, UC Registry, fila em batch |
| 2️⃣ | **Lab 2 — Model Serving** | 30 min | Alias → versão, endpoint REST pela UI, latência cold vs warm, validação online × batch |
| 3️⃣ | **Lab 3 — Lakebase (Feature Store Online)** | 35 min | Tabelas de agregado com PK+CDF, instância Lakebase, synced tables, lookup por chave |
| 4️⃣ | **Lab 4 — Databricks App** | 30 min | Simulador de inferência em tempo real (Streamlit) + deploy guiado |
| 5️⃣ | **Lab 5 — Knowledge Assistant (GenAI)** | 30 min | Agente que responde sobre a norma interna a partir de um PDF no Volume (Agent Bricks, pela UI) |
| 6️⃣ | **Lab 6 — Agente de Busca de Dentistas** | 40 min | Tool calling: 3 UC Functions + busca geoespacial + agente no AI Playground |
| 🧹 | **Cleanup** | 5 min | Remove app, endpoint, Lakebase e schema (o agente do Lab 5 sai pela UI) |
| | **Total** | **~4h25** | |

> ⏱️ **Ajuste de agenda:** o **Lab 3 é o cortável** se o tempo apertar. O app do Lab 4 tem
> fallback (`USE_LAKEBASE=false`) que lê os mesmos agregados via SQL Warehouse. Os **Labs 5 e
> 6 são independentes** dos demais — só precisam do Setup, então podem virar uma sessão
> separada de GenAI (~70 min) sem os labs de ML.
>
> 💡 **Duas trilhas possíveis:** Labs 1–4 formam a trilha de **ML/MLOps** (o que a operadora
> faz com o sinistro). Labs 5–6 formam a trilha de **GenAI** (como o agente ajuda auditor e
> beneficiário). Nenhum dos dois grupos precisa do outro para rodar.

---

## 🧠 O que este workshop ensina (e que a maioria dos labs de ML não ensina)

| Conceito | Onde | Por que importa |
|---|---|---|
| **Medir o baseline primeiro** | Lab 1 | Sem a precision de 2% da regra atual, não há como provar que o ML valeu a pena |
| **Target leakage** | Lab 1 | 4 colunas (`status`, `valor_reembolsado`, `motivo_glosa`, `data_analise`) só existem **depois** da decisão |
| **Feature relacional** | Lab 1 | Um pedido isolado quase nunca parece fraude. O sinal está **entre linhas** |
| **PR AUC, não acurácia** | Lab 1 | Com 5,5% de fraude, "nunca é fraude" acerta 94,5% |
| **Threshold pelo dinheiro** | Lab 1 | 0,5 não tem significado de negócio; o corte vem do custo dos dois erros |
| **`predict` vs `predict_proba`** | Lab 1 | Logar errado faz o endpoint devolver **classe (0/1)** e mata o threshold — sem quebrar nada |
| **Signature = contrato** | Lab 1/2 | Ordem trocada das 20 features dá score errado **silenciosamente** |
| **Alias não serve endpoint** | Lab 2 | `@champion` é governança; o endpoint exige número de versão — promover **não** faz deploy |
| **Training/serving skew** | Lab 2 | O mesmo pedido tem que dar o mesmo score online e em batch |
| **Separar calcular de servir** | Lab 3 | 7 das 20 features são agregados de 2 anos: Spark calcula 1x/dia, Postgres serve em ms |
| **Requisitos de synced table** | Lab 3 | PK não-nula + CDF para Triggered/Continuous |
| **O score é auditável** | Lab 4 | As features viram a **justificativa** que o auditor lê |
| **RAG sem escrever RAG** | Lab 5 | Chunking, embedding, índice e serving saem prontos — o que sobra é curadoria, descrição da fonte e avaliação |
| **Recusar bem é requisito** | Lab 5 | Um agente que inventa um teto de reembolso é pior que um que não responde |
| **Sync não é automático** | Lab 5 | Trocar o PDF no Volume **não** atualiza o índice do agente |
| **A resposta é limitada pelo retrieval** | Lab 5 | O parsing preserva as tabelas em HTML — sem isso, "carência de ortodontia no Dental Essencial" é impossível de responder |
| **O `COMMENT` é a interface** | Lab 6 | O LLM escolhe a ferramenta lendo a descrição, não o corpo. Comentário ruim = ferramenta ignorada |
| **Regra crítica no SQL, não no prompt** | Lab 6 | Filtro por plano e `LIMIT` não se burlam conversando; guardrail de prompt sim |
| **O agente menos capaz é o mais correto** | Lab 6 | Deduzir especialidade a partir de sintoma é exercer atividade clínica — o agente tem que recusar |
| **UC Function não tem egress** | Lab 6 | Notebook serverless acessa internet; o sandbox da função **não** (`TimeoutError`) — medido |

### Os números que fecham o argumento (medidos, não estimados)

| Abordagem | Precision | Recall | PR AUC |
|---|---|---|---|
| Regra `valor > R$ 1.500` | 0,02 | 0,05 | — |
| ML só com features do próprio pedido | — | — | 0,66 |
| **ML com features relacionais** | 0,69 | 0,77 | **0,82–0,85** |

> Baseline de um modelo aleatório = 0,055 (a prevalência) → **~15x de lift**. A diferença
> entre as duas últimas linhas **é** o valor do feature engineering.

E a fila resultante, no fim do Lab 1:

| Prioridade | Pedidos | % fraude real |
|---|---|---|
| CRÍTICA | ~1.500 | **99%** |
| ALTA | ~1.560 | 36% |
| MÉDIA | ~730 | 5% |
| BAIXA | ~36.700 | 0,1% |

---

## 🏗️ Arquitetura

```
   Sistemas transacionais (simulados)
   beneficiários · rede credenciada · procedimentos · pedidos de reembolso
                          │
   ┌──────────────────────▼───────────────────────┐
   │  Unity Catalog — workshop_databricks.<você>  │   LAB 01
   │  silver_*  ──►  gold_features_fraude (20 ft) │
   │            ──►  gold_fila_auditoria          │
   │  modelo_fraude_reembolso  @champion          │
   └──────────┬─────────────────────┬─────────────┘
              │                     │
       LAB 02 │              LAB 03 │
              ▼                     ▼
    Serving Endpoint      gold_features_prestador   (PK prestador_id)
    scale_to_zero=True    gold_features_beneficiario (PK beneficiario_id)
    ~50ms warm                      │ synced table (Snapshot)
              │                     ▼
              │            Lakebase (Postgres) — lookup por PK
              │                     │
              └──────────┬──────────┘
                         ▼                                    LAB 04
              Streamlit App (Databricks Apps)
       [pedido novo] → 13 features do pedido
                     → 7 agregados do Lakebase
                     → vetor de 20 → POST /invocations
                     → score + prioridade + "por quê" + latência


   ┌────────────────────────────────────────────────┐        LAB 05
   │  Volume UC: politicas_internas                 │
   │  politica_autorizacao_odontoprev.pdf           │
   └────────────────────┬───────────────────────────┘
                        ▼
            Knowledge Assistant (Agent Bricks)
       "cabe recurso?" → resposta + citação da seção
                        │
                        └─► endpoint ka-<id>-endpoint (REST)


   ┌────────────────────────────────────────────────┐        LAB 06
   │  gold_rede_credenciada    (lat/long, planos)   │
   │  gold_beneficiarios_geo   (CPF, lat/long)      │
   └────────────────────┬───────────────────────────┘
                        ▼
        3 UC Functions = ferramentas do agente
   buscar_beneficiario_por_cpf → encontrar_dentistas_proximos
                        ▼
              AI Playground (tool calling)
     "endodontista perto de casa?" → 5 opções por distância
```

Os Labs 05 e 06 correm em paralelo aos de ML, não em série: o score diz **onde olhar**, a
norma diz **o que fazer**, e o agente de busca **evita o pedido acontecer**.

---

## 📊 Modelo de Dados

### Tabelas do Setup (Lab 01)

| Tabela | Registros | Papel |
|--------|-----------|-------|
| `silver_planos` | 12 | `percentual_reembolso` por produto |
| `silver_beneficiarios` | 8.000 | UF, plano, segmento, `data_adesao` |
| `silver_prestadores` | 700 | `credenciado` — quem está **fora da rede** |
| `silver_procedimentos_rede` | ~38.000 | Cruzamento: duplicidade e dente já extraído |
| `silver_pedidos_reembolso` | **~40.500** | **Tabela alvo** — ~5,5% de fraude confirmada |
| `_gabarito_personas_fraude` | ~40.500 | Qual **tipo** de fraude era cada pedido (diagnóstico final) |

### Tabelas criadas nos labs

| Tabela | Lab | Chave | Papel |
|---|---|---|---|
| `gold_features_fraude` | 01 | `pedido_id` | 20 features para treino |
| `gold_fila_auditoria` | 01 | `pedido_id` | fila priorizada (batch) |
| `gold_features_prestador` | 03 | `prestador_id` | 4 agregados (700 linhas) |
| `gold_features_beneficiario` | 03 | `beneficiario_id` | 3 agregados (~7.600 linhas) |
| `online_features_*` | 03 | idem | synced tables no Lakebase |
| `gold_rede_credenciada` | 06 | `prestador_id` | rede com endereço, bairro, lat/long e `planos_aceitos` |
| `gold_beneficiarios_geo` | 06 | `beneficiario_id` | beneficiários com CPF, endereço, plano e lat/long |

### As 20 features, por origem

Essa divisão é a espinha dorsal dos Labs 3 e 4:

| Origem | Qtd | Exemplos | Calculável na hora? |
|---|---|---|---|
| **Do próprio pedido** | 13 | `valor_solicitado`, `prox_teto`, `dente_ja_extraido` | ✅ sim |
| **Agregado do prestador** | 4 | `prest_n_benef`, `prest_pct_prox_teto` | ❌ vem do Lakebase |
| **Agregado do beneficiário** | 3 | `benef_valor_total`, `benef_n_prestadores` | ❌ vem do Lakebase |

### As 6 personas de fraude plantadas nos dados

Nenhuma aparece como coluna. O sinal está na **relação entre linhas**:

| Persona | Padrão | Feature que a captura |
|---|---|---|
| **Reembolso assistido** | Clínica de fachada concentra dezenas de beneficiários | `prest_n_benef`, `prest_pct_prox_teto` |
| **Fracionamento** | Tratamento quebrado em 3-6 pedidos sob o teto | `pedidos_par_30d` |
| **Duplicidade rede × reembolso** | Mesmo procedimento cobrado na rede **e** como reembolso | `existe_na_rede` |
| **Adesão oportunista** | Uso intenso nos primeiros 120 dias, cancela depois | `carencia_recente` |
| **Documento reciclado** | Mesma nota fiscal reapresentada | `hash_repetido` |
| **Implausibilidade clínica** | Restauração/canal em dente já extraído | `dente_ja_extraido` |

### E os *hard negatives* — de propósito

Para que soluções ingênuas **falhem** (é o que gera a discussão em sala):

- **Ortodontia legítima** gera muitos pedidos do mesmo prestador por meses;
- **Prótese e implante legítimos** custam R$ 1.200 – R$ 4.000;
- **UF com rede rala** (PA, AM, MT) legitimamente usa muito reembolso;
- **Documento reenviado de boa-fé**, **tratamento em várias sessões** e **implante no dente
  extraído** — casos honestos que acionam as mesmas flags das fraudes.

> Resultado medido: `valor_solicitado` sozinho tem **AUC de apenas 0,74**, e nenhuma flag
> isolada resolve o caso (`hash_repetido` tem precision de 0,16).

---

## 📁 Estrutura do Repositório

```
odontoprev-ai-workshop/
├── 📄 README.md
│
├── 📂 00_Setup/
│   ├── 00_configuracao_catalogo.py         ← Schema isolado via current_user()
│   └── 01_dados_sinteticos.py              ← Dados + 6 personas + hard negatives
│
├── 📂 01_Lab_ML_Fraude/
│   ├── 02a_ml_to_do.py                     ← Exercícios (9 TO-DOs)
│   └── 02b_ml_completo.py                  ← Gabarito
│
├── 📂 02_Lab_Serving/
│   ├── 03a_serving_to_do.py                ← Exercícios (3 TO-DOs)
│   └── 03b_serving_completo.py             ← Gabarito
│
├── 📂 03_Lab_Lakebase/
│   ├── 04a_lakebase_to_do.py               ← Exercícios (3 TO-DOs)
│   └── 04b_lakebase_completo.py            ← Gabarito
│
├── 📂 04_Lab_App/
│   ├── app.py                              ← Streamlit (formulário, gauge, explicação)
│   ├── features.py                          ← 13 features + vetor na ordem da signature
│   ├── db.py                                ← Conexão Lakebase (psycopg2 + OAuth)
│   ├── app.yaml · requirements.txt
│   └── 05_deploy_app.py                    ← Deploy guiado (UI primeiro)
│
├── 📂 05_Lab_Knowledge_Assistant/
│   ├── 06_knowledge_assistant.py           ← Passo a passo guiado (UI primeiro)
│   └── documentos/
│       └── politica_autorizacao_odontoprev.pdf   ← Base de conhecimento do agente
│
├── 📂 06_Lab_Agente_Busca/
│   ├── 07a_dados_geo.py                    ← Geo para prestadores e beneficiários
│   ├── 07b_criar_ferramentas.py            ← As 3 UC Functions (tools)
│   └── 07c_agente_playground.py            ← Agente no AI Playground (UI) + system prompt
│
├── 📂 resources/                            ← JSONs de synced tables e app resources
├── 📂 99_Cleanup/ 99_cleanup.py             ← Remove app, endpoint, Lakebase e schema
└── 📂 .tools/                               ← Fonte única dos notebooks (ver abaixo)
```

### 🔧 Fonte única dos notebooks

Os pares `*_to_do` / `*_completo` **não são editados à mão** — são gerados de uma fonte
única, o que garante que a narrativa seja idêntica nos dois:

```bash
python3 .tools/gerar_notebooks.py              # gera todos
python3 .tools/gerar_notebooks.py fraude       # gera um lab
```

O gerador valida que o markdown ficou byte-a-byte igual (só o título difere). A única
diferença entre os dois notebooks: o código entre `# @@TAREFA` e `# @@FIM` fica **comentado**
na versão de exercícios.

> ⚠️ Para alterar um lab, edite `.tools/lab_<x>_source.py` e rode o gerador.

---

## 🔧 Pré-requisitos

### Workspace Databricks
- ✅ Unity Catalog habilitado
- ✅ **Serverless Compute** (ou DBR 14.3 LTS ML+)
- ✅ SQL Warehouse serverless
- ✅ **Model Serving** habilitado (Lab 02)
- ✅ **Lakebase** disponível — *Public Preview* (Lab 03)
- ✅ **Databricks Apps** habilitado (Lab 04)
- ✅ **Agent Bricks** habilitado (Lab 05) — menu **Agents** > **Create agent** deve mostrar o
  tile **Knowledge Assistant**
- ✅ **AI Playground** disponível **com um modelo `Tools enabled`** (Lab 06) — menu **AI/ML** >
  **Playground**; sem tool calling o Lab 06 não funciona

### Preparação do instrutor (uma vez, antes do workshop)

```sql
CREATE CATALOG IF NOT EXISTS workshop_databricks
  MANAGED LOCATION 'abfss://<container>@<storage>.dfs.core.windows.net/';

GRANT USE CATALOG, CREATE SCHEMA ON CATALOG workshop_databricks TO `account users`;
```

> ⚠️ O `MANAGED LOCATION` é **obrigatório** quando o metastore não tem storage root próprio —
> sem ele o `CREATE CATALOG` falha com `Metastore storage root URL does not exist`.

### Permissões do participante

| Lab | Permissões |
|-----|-----------|
| ⚙️ Setup | `USE CATALOG`, `CREATE SCHEMA`, `CREATE VOLUME` |
| 1️⃣ ML | `CREATE TABLE`, `CREATE MODEL`, criar experimento MLflow |
| 2️⃣ Serving | criar serving endpoint |
| 3️⃣ Lakebase | criar database instance e synced tables |
| 4️⃣ App | criar Databricks App e adicionar resources |
| 5️⃣ Knowledge Assistant | criar agente no Agent Bricks, `CREATE VOLUME`, `READ VOLUME` |
| 6️⃣ Agente de busca | `CREATE FUNCTION`, `EXECUTE`, acesso ao AI Playground |

---

## 🚀 Como executar

1. ⚙️ Importe a pasta para o seu Workspace Databricks
2. ⚙️ `00_Setup/00_configuracao_catalogo` → cria o **seu** schema automaticamente
3. ⚙️ `00_Setup/01_dados_sinteticos` → gera as 6 tabelas (~2 min)
4. 1️⃣ `01_Lab_ML_Fraude/02a_ml_to_do` → resolva os **9 TO-DOs**
5. 2️⃣ `02_Lab_Serving/03a_serving_to_do` → crie o endpoint (**dispare no início**: leva 10-25 min)
6. 3️⃣ `03_Lab_Lakebase/04a_lakebase_to_do` → instância + synced tables + lookup
7. 4️⃣ `04_Lab_App/05_deploy_app` → publique o simulador e rode os 6 presets
8. 5️⃣ `05_Lab_Knowledge_Assistant/06_knowledge_assistant` → crie o agente pela UI
9. 6️⃣ `06_Lab_Agente_Busca/07a` → `07b` → `07c` → monte o agente no Playground
10. 🧹 `99_Cleanup/99_cleanup` — **não esqueça**

> Travou em algum TO-DO? Cada lab tem o gabarito (`*_completo`) com o mesmo texto.
> O Lab 05 é notebook único (todo o trabalho é na UI), então não tem par to_do/gabarito.

### 💰 Custo — o que cobra parado

| Recurso | Cobra em idle? | Removido pelo `99_Cleanup`? |
|---|---|---|
| Serving endpoint | não, se `scale to zero` estiver ligado | ✅ |
| **Instância Lakebase** | **sim** | ✅ |
| **Databricks App** | **sim** | ✅ |
| **Knowledge Assistant** (Lab 05) | **sim** — mantém um endpoint provisionado | ❌ **apague pela UI** |

> ⚠️ O KA sai só pela UI (**Agents > ⋮ > Delete**): o Agent Bricks não expõe API pública de
> listagem (`GET /api/2.0/custom-llms` devolve `ENDPOINT_NOT_FOUND`), então o script de
> cleanup não tem como encontrá-lo. Está no Passo 6 do Lab 05.

---

## 👤 Isolamento por Participante

| Aspecto | Como funciona |
|---------|--------------|
| **Catálogo** | `workshop_databricks` (compartilhado pela turma) |
| **Schema** | Derivado de `current_user()` → ex. `gabriel_rangel` (✅ automático) |
| **Modelo** | `workshop_databricks.<schema>.modelo_fraude_reembolso` |
| **Endpoint** | `fraude-reembolso-<seu-nome>` |
| **Lakebase** | `lakebase-<seu-nome>` |
| **App** | `fraude-sim-<seu-nome>` |
| **Knowledge Assistant** | `Politica Odontoprev <seu_nome>` (nome digitado na UI — ⚠️ não é automático) |
| **Experimento MLflow** | `/Users/<seu_email>/odontoprev_workshop_fraude` |

> ⚠️ Nomes de endpoint, instância e app **não aceitam `_`** — use `-`.
> E **nome de app tem no máximo 30 caracteres** (por isso `fraude-sim-`, não
> `fraude-simulador-`).

---

## 🛠️ Tecnologias

| Categoria | Tecnologias |
|-----------|------------|
| **Governança** | Unity Catalog (catálogo, schemas, volumes, PK/CDF, lineage) |
| **Storage** | Delta Lake + Change Data Feed |
| **Feature Engineering** | PySpark Window Functions (`partitionBy`, `rangeBetween` de 30 dias, `collect_set`) |
| **ML** | scikit-learn (`HistGradientBoostingClassifier`, `RandomForestClassifier` com `class_weight`) |
| **MLOps** | MLflow, UC Model Registry, aliases `champion`/`challenger`, signature, threshold como tag, wrapper pyfunc |
| **Serving** | Model Serving (REST, scale-to-zero), Spark UDF para batch |
| **OLTP / Online store** | Lakebase (Postgres gerenciado), synced tables (Snapshot/Triggered/Continuous), credencial OAuth |
| **App** | Databricks Apps (Streamlit + Plotly), resources via `valueFrom` |
| **GenAI** | Agent Bricks — Knowledge Assistant (RAG gerenciado), `ai_parse_document`, exemplos com guidelines |
| **Agentes / Tool calling** | UC Functions como ferramentas (`COMMENT` como interface), AI Playground, busca geoespacial (haversine + bounding box) |
| **Métricas** | PR AUC, curva Precision-Recall, matriz de custo em R$, latência cold/warm |

---

## ⚠️ Sobre os dados

Todos os dados são **100% sintéticos**, gerados proceduralmente com seed fixa (`SEED = 42`).
Nomes, CPFs, CROs, clínicas e valores são fictícios e não correspondem a pessoas, prestadores
ou operações reais. Os padrões de fraude são reproduções **conceituais** de tipologias
públicas discutidas no setor de saúde suplementar.

---

<p align="center">
  <img src="https://img.shields.io/badge/Databricks-FF3621?style=flat-square&logo=databricks&logoColor=white" alt="Databricks">
  <br>
  <em>Workshop desenvolvido por Gabriel Rangel — Solutions Engineer</em>
</p>
