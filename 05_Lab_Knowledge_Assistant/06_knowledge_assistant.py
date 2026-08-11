# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 05 — Knowledge Assistant: o agente que le a norma interna
# MAGIC ## Passo a passo guiado (Agent Bricks, pela UI)
# MAGIC
# MAGIC **Objetivo:** criar um **Knowledge Assistant (KA)** que responde duvidas da equipe
# MAGIC de auditoria e autorizacoes a partir de um **documento interno** da Odontoprev —
# MAGIC sem escrever uma linha de codigo de RAG.
# MAGIC
# MAGIC ### Onde este lab se encaixa
# MAGIC
# MAGIC Os Labs 01 a 04 respondem **"este pedido e suspeito?"**. Nenhum deles responde
# MAGIC **"e a norma, o que diz?"** — e essa e a pergunta que o auditor faz em seguida.
# MAGIC
# MAGIC | | Labs 01–04 (ML preditivo) | Lab 05 (GenAI) |
# MAGIC |---|---|---|
# MAGIC | Pergunta | "qual a probabilidade de fraude?" | "qual a regra que se aplica?" |
# MAGIC | Entrada | 20 features numericas | pergunta em portugues |
# MAGIC | Fonte | 40 mil pedidos historicos | PDF de norma interna |
# MAGIC | Saida | score de 0 a 1 | resposta com **citacao da fonte** |
# MAGIC
# MAGIC Juntos, formam o fluxo real do auditor:
# MAGIC
# MAGIC ```
# MAGIC   gold_fila_auditoria (Lab 01)     ->  "pedido 18432, score 0.94, CRITICA"
# MAGIC             |
# MAGIC             v
# MAGIC   Knowledge Assistant (Lab 05)     ->  "implante em plano Corp Essencial NAO tem
# MAGIC                                         cobertura -> negativa automatica, sem recurso
# MAGIC                                         (Politica de Autorizacao, Secao 4.1)"
# MAGIC             |
# MAGIC             v
# MAGIC   decisao documentada e auditavel
# MAGIC ```
# MAGIC
# MAGIC ### O documento deste lab
# MAGIC
# MAGIC **`politica_autorizacao_odontoprev.pdf`** — Politica de Autorizacao e Cobertura,
# MAGIC Normas Internas, Revisao Mar/2026. Esta na pasta `documentos/` ao lado deste
# MAGIC notebook. Quatro secoes:
# MAGIC
# MAGIC | Secao | Conteudo | Pergunta tipica |
# MAGIC |---|---|---|
# MAGIC | **1** | 12 codigos TUSS que exigem autorizacao previa, prazo e alcada | "quem aprova um implante?" |
# MAGIC | **2** | carencias por linha de produto (6 planos x 10 grupos) | "qual a carencia de protese no Corp Plus?" |
# MAGIC | **3** | tetos e percentuais de reembolso, prazos de deposito | "qual o teto de reembolso de implante?" |
# MAGIC | **4** | 7 regras especiais e excecoes | "cabe recurso numa negativa automatica?" |
# MAGIC
# MAGIC ### O que vamos fazer
# MAGIC
# MAGIC | Passo | O que faz | Onde |
# MAGIC |---|---|---|
# MAGIC | 1 | Criar o Volume e publicar o PDF nele | codigo |
# MAGIC | 2 | Criar o Knowledge Assistant | **UI** |
# MAGIC | 3 | Testar no chat e inspecionar as fontes citadas | **UI** |
# MAGIC | 4 | Melhorar a qualidade com exemplos e guidelines | **UI** |
# MAGIC | 5 | Consumir o KA por codigo (endpoint REST) | codigo |
# MAGIC | 6 | Apagar o agente | **UI** |
# MAGIC
# MAGIC ## 🔧 O que personalizar neste lab
# MAGIC
# MAGIC | Valor | Onde | Sugestao |
# MAGIC |---|---|---|
# MAGIC | **Nome do agente** | Passo 2 (UI) | `Politica Odontoprev <seu_nome>` |
# MAGIC | **Volume de origem** | Passo 2 (UI) | o caminho impresso no Passo 1 |
# MAGIC
# MAGIC Procure por **`👉 ALTERE`** nas celulas. O resto pode rodar como esta.
# MAGIC
# MAGIC > **Pre-requisito:** ter rodado `00_Setup/00_configuracao_catalogo` (o schema
# MAGIC > isolado precisa existir). Os Labs 01–04 **nao** sao necessarios para este lab.
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dependencias
# MAGIC
# MAGIC Instalamos tudo **aqui, no inicio**, de proposito: o `dbutils.library.restartPython()`
# MAGIC reinicia o interpretador e **apaga todas as variaveis**. Fazendo isso antes de
# MAGIC definir qualquer coisa, nada se perde no meio do lab.

# COMMAND ----------

# MAGIC %pip install -q 'databricks-sdk>=0.59'

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Passo 1 — Publicar o documento no Volume
# MAGIC
# MAGIC O KA le arquivos de um **Volume do Unity Catalog**. Isso e importante: o documento
# MAGIC nao sai da governanca. Quem nao tem `READ VOLUME` nao ve o conteudo — nem pelo
# MAGIC agente.
# MAGIC
# MAGIC Formatos aceitos: `pdf`, `txt`, `md`, `doc/docx`, `ppt/pptx`.
# MAGIC
# MAGIC > 💡 Num cenario real o documento ja chegaria ao Volume por um pipeline — vindo do
# MAGIC > SharePoint, do GED ou do portal de normas. Aqui ele viaja junto do repositorio,
# MAGIC > na pasta `documentos/`, e nos so o copiamos.

# COMMAND ----------

usuario = spark.sql("SELECT current_user()").collect()[0][0]
nome = usuario.split("@")[0].replace(".", "_").replace("-", "_").lower()

CATALOGO = "workshop_databricks"
SCHEMA = nome
VOLUME = "politicas_internas"

spark.sql(f"USE CATALOG {CATALOGO}")
spark.sql(f"USE SCHEMA {SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOGO}.{SCHEMA}.{VOLUME}")

CAMINHO_VOLUME = f"/Volumes/{CATALOGO}/{SCHEMA}/{VOLUME}"
print(f"Volume pronto: {CAMINHO_VOLUME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Copiar o PDF da pasta `documentos/` para o Volume
# MAGIC
# MAGIC O notebook descobre sozinho a sua propria pasta no Workspace, entao nao ha caminho
# MAGIC para editar. O PDF esta em `documentos/` — um **workspace file**, que da para ler
# MAGIC com `open()` normal.

# COMMAND ----------

import os
import shutil

# Caminho deste notebook no Workspace -> a pasta do lab fica ao lado.
caminho_notebook = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
PASTA_LAB = "/Workspace" + os.path.dirname(caminho_notebook)
PASTA_DOCS = f"{PASTA_LAB}/documentos"

ARQUIVO = "politica_autorizacao_odontoprev.pdf"
origem = f"{PASTA_DOCS}/{ARQUIVO}"

print(f"Pasta do lab: {PASTA_LAB}")

if not os.path.exists(origem):
    raise FileNotFoundError(
        f"Nao encontrei {origem}.\n"
        "Confira se a pasta 'documentos/' foi importada junto com o notebook "
        "(no Databricks: Workspace > seu repo > 05_Lab_Knowledge_Assistant/documentos)."
    )

shutil.copy(origem, f"{CAMINHO_VOLUME}/{ARQUIVO}")
print(f"Copiado: {ARQUIVO}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conferir o que chegou ao Volume

# COMMAND ----------

print(f"Base de conhecimento em {CAMINHO_VOLUME}\n")
for a in dbutils.fs.ls(CAMINHO_VOLUME):
    print(f"  {a.name:45s} {a.size / 1024:6.1f} KB")

print(f"\n{'=' * 70}")
print("  👉 COPIE ESTE CAMINHO — voce vai cola-lo na UI no Passo 2:")
print(f"     {CAMINHO_VOLUME}")
print(f"{'=' * 70}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### (Opcional, mas recomendado) Ver o que o agente vai ler
# MAGIC
# MAGIC O KA faz o parsing do PDF por conta propria. Ainda assim, vale espiar o texto
# MAGIC extraido antes: **se o texto sai ruim aqui, a resposta do agente vai sair ruim
# MAGIC tambem.**
# MAGIC
# MAGIC E o teste de qualidade de fonte mais barato que existe — e o primeiro lugar para
# MAGIC olhar quando um KA responde mal. Um PDF escaneado sem OCR, por exemplo, chega
# MAGIC praticamente vazio: nenhuma instruction conserta isso.
# MAGIC
# MAGIC > A funcao `ai_parse_document` faz o parsing em SQL, sem instalar nada. Ela devolve
# MAGIC > um `VARIANT` com um **elemento por bloco** (titulo, paragrafo, tabela), cada um com
# MAGIC > `type` e `content`. Por ser `VARIANT`, o jeito limpo de abrir e em SQL, com a
# MAGIC > sintaxe de acesso `:` e um `explode`.

# COMMAND ----------

df_elementos = spark.sql(f"""
    WITH parsed AS (
      SELECT ai_parse_document(content) AS doc
      FROM READ_FILES('{CAMINHO_VOLUME}/{ARQUIVO}', format => 'binaryFile')
    )
    SELECT
      el:type::string    AS tipo,
      el:content::string AS conteudo
    FROM parsed
    LATERAL VIEW explode(CAST(doc:document:elements AS ARRAY<VARIANT>)) t AS el
""")

elementos = df_elementos.collect()
print(f"{len(elementos)} elementos extraidos\n{'=' * 78}")

for el in elementos:
    conteudo = " ".join(str(el["conteudo"]).split())
    print(f"\n[{el['tipo']}]")
    print(conteudo[:700] + ("..." if len(conteudo) > 700 else ""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### O que conferir na saida acima
# MAGIC
# MAGIC No nosso teste sairam **18 elementos**, assim distribuidos:
# MAGIC
# MAGIC ```
# MAGIC title 1 | section_header 4 | text 5 | table 4 | page_header 2 | page_footer 2
# MAGIC ```
# MAGIC
# MAGIC | Verificacao | Por que importa |
# MAGIC |---|---|
# MAGIC | Os **valores** aparecem? (`R$ 2.500,00`, `88000031`) | se nao, nenhuma pergunta sobre teto ou TUSS vai funcionar |
# MAGIC | Sairam as **4 tabelas**? | as Secoes 1, 2 e 3 **sao** tabelas — e onde estao TUSS, carencias e tetos |
# MAGIC | Os paragrafos da **Secao 4** vieram inteiros? | e onde estao as excecoes: o "nao cabe recurso" e o "12 meses, nao 6" |
# MAGIC
# MAGIC ### ⭐ Repare em como a tabela foi extraida
# MAGIC
# MAGIC O parser nao achatou a tabela em texto corrido — devolveu **HTML com a estrutura
# MAGIC preservada**:
# MAGIC
# MAGIC ```html
# MAGIC <table>
# MAGIC   <tr><th>Codigo TUSS</th><th>Procedimento</th><th>Prazo</th><th>Aprovacao</th></tr>
# MAGIC   <tr><td>88000031</td><td>Implante osseointegrado</td><td>10 dias uteis</td>
# MAGIC       <td>Junta medica (3 auditores)</td></tr>
# MAGIC ```
# MAGIC
# MAGIC **Isso e o que faz a pergunta 3 do Passo 3 funcionar.** "Carencia de ortodontia no
# MAGIC plano Dental Essencial" exige cruzar uma **linha** com uma **coluna** de uma tabela de
# MAGIC 6 planos. Se a extracao tivesse achatado tudo em texto — `Ortodontia 30 dias 90 dias
# MAGIC Nao coberto 120 dias 120 dias Nao coberto` — nenhum modelo saberia qual "Nao coberto"
# MAGIC pertence a qual plano.
# MAGIC
# MAGIC > 💡 **A licao pratica:** quando um KA erra respostas de tabela, o problema quase
# MAGIC > nunca esta no prompt — esta aqui, na extracao. E por isso que vale rodar esta
# MAGIC > celula **antes** de culpar as instructions.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Passo 2 (UI) — Criar o Knowledge Assistant
# MAGIC
# MAGIC > ⏱️ **Faca isto agora e siga lendo.** O endpoint do agente leva
# MAGIC > **2 a 5 minutos** para provisionar (`PROVISIONING` → `ONLINE`).
# MAGIC
# MAGIC ### 2.1 — Abrir o Agent Bricks
# MAGIC
# MAGIC 1. Menu lateral > **Agents**.
# MAGIC 2. **Create agent**.
# MAGIC 3. Escolha o tile **Knowledge Assistant**.
# MAGIC
# MAGIC ### 2.2 — Identificar o agente
# MAGIC
# MAGIC | Campo | O que preencher |
# MAGIC |---|---|
# MAGIC | **Name** | `Politica Odontoprev <seu_nome>` &nbsp; 👉 **ALTERE** |
# MAGIC | **Description** | `Responde duvidas da equipe de auditoria e autorizacoes sobre a Politica de Autorizacao e Cobertura da Odontoprev: procedimentos que exigem autorizacao previa, carencias por plano, tetos de reembolso e regras especiais.` |
# MAGIC
# MAGIC > ⚠️ Use o **seu nome** no final. Como todos na sala criam no mesmo workspace,
# MAGIC > sem isso voce nao vai achar o seu agente na lista.
# MAGIC
# MAGIC ### 2.3 — Adicionar a fonte de conhecimento
# MAGIC
# MAGIC | Campo | O que preencher |
# MAGIC |---|---|
# MAGIC | **Type** | `Files in a Volume` |
# MAGIC | **Source** | o caminho impresso na celula acima &nbsp; 👉 **ALTERE** |
# MAGIC | **Name** | `politica_autorizacao_cobertura` |
# MAGIC | **Describe the content** | veja o texto abaixo |
# MAGIC
# MAGIC Cole em **Describe the content**:
# MAGIC
# MAGIC ```
# MAGIC Politica de Autorizacao e Cobertura da Odontoprev (documento interno confidencial,
# MAGIC Revisao Mar/2026), com quatro secoes:
# MAGIC (1) Procedimentos que exigem autorizacao previa: codigo TUSS, prazo de autorizacao
# MAGIC     e tipo de aprovacao (automatica, auditor clinico ou junta medica).
# MAGIC (2) Carencias por plano em dias corridos, para as linhas Corp Executive, Corp Plus,
# MAGIC     Corp Essencial, PME Completo, Dental Premium e Dental Essencial, por grupo de
# MAGIC     procedimento (urgencia, consultas, restauracoes, endodontia, periodontia,
# MAGIC     cirurgia, protese, implante, ortodontia, clareamento).
# MAGIC (3) Regras de reembolso: valor maximo por grupo, percentual da tabela Odontoprev,
# MAGIC     prazo de deposito e condicoes para solicitar (raio de 30 km sem rede, prazo de
# MAGIC     30 dias, documentacao exigida).
# MAGIC (4) Regras especiais e excecoes: negativa automatica, segunda opiniao, clareamento
# MAGIC     dental, urgencia fora do horario, portabilidade interna, limite de procedimentos
# MAGIC     e canal de escalacao interna (ESC-AUDIT).
# MAGIC ```
# MAGIC
# MAGIC > 💡 **A descricao nao e enfeite.** E com ela que o agente decide **se** esta fonte
# MAGIC > e relevante para a pergunta. Uma descricao vaga ("documentos da empresa") degrada
# MAGIC > a recuperacao — e o efeito fica bem visivel quando ha mais de uma fonte. Repare
# MAGIC > que a descricao acima nomeia os **planos** e os **grupos de procedimento**: sao
# MAGIC > exatamente os termos que aparecem nas perguntas reais.
# MAGIC
# MAGIC ### 2.4 — Instructions
# MAGIC
# MAGIC Campo **Instructions** (opcional na UI, decisivo na pratica). Cole:
# MAGIC
# MAGIC ```
# MAGIC Voce e um assistente interno de apoio a equipe de atendimento, autorizacoes e
# MAGIC auditoria da Odontoprev.
# MAGIC
# MAGIC Regras de resposta:
# MAGIC 1. Responda EXCLUSIVAMENTE com base na Politica de Autorizacao e Cobertura
# MAGIC    fornecida. Nunca use conhecimento geral sobre planos odontologicos nem invente
# MAGIC    codigos TUSS, prazos, percentuais ou valores.
# MAGIC 2. Cite sempre a secao de onde veio a resposta (ex.: "Secao 2 - Carencias",
# MAGIC    "Secao 4.1 - Negativa Automatica") e, quando houver, o codigo TUSS.
# MAGIC 3. Se a informacao nao estiver no documento, diga isso com clareza e oriente a
# MAGIC    consultar a auditoria clinica. Nao preencha lacunas com suposicoes.
# MAGIC 4. Carencias e coberturas mudam por plano. Se a pergunta nao informar o plano do
# MAGIC    beneficiario, pergunte antes de responder - ou apresente a linha completa da
# MAGIC    tabela para todos os planos.
# MAGIC 5. Use tabelas ou listas curtas para prazos, valores e carencias, reproduzindo
# MAGIC    exatamente os numeros do documento.
# MAGIC 6. Nunca de orientacao clinica de tratamento nem diagnostico.
# MAGIC 7. Este e um documento interno confidencial: suas respostas sao para a equipe
# MAGIC    interna, nao para beneficiarios.
# MAGIC 8. Responda sempre em portugues do Brasil, em tom objetivo e operacional.
# MAGIC
# MAGIC Frase de fallback quando a informacao nao constar:
# MAGIC "Essa informacao nao consta na Politica de Autorizacao e Cobertura (Rev. Mar/2026).
# MAGIC Recomendo consultar a area de auditoria clinica para orientacao."
# MAGIC ```
# MAGIC
# MAGIC ### 2.5 — Criar
# MAGIC
# MAGIC 1. **Create agent**.
# MAGIC 2. O agente abre na aba **Build**, com o endpoint em `PROVISIONING`.
# MAGIC 3. Aguarde chegar a **ONLINE** (2–5 min). Enquanto isso, leia o Passo 3.
# MAGIC
# MAGIC > 📄 Doc: https://docs.databricks.com/aws/en/generative-ai/agent-bricks/knowledge-assistant

# COMMAND ----------

# MAGIC %md
# MAGIC ### O que o Databricks fez por voce nesses 5 minutos
# MAGIC
# MAGIC Vale parar um instante nisto, porque e o argumento do Agent Bricks. Um pipeline de
# MAGIC RAG construido a mao exige:
# MAGIC
# MAGIC | Etapa | Feito a mao | Com Knowledge Assistant |
# MAGIC |---|---|---|
# MAGIC | Extrair texto do PDF | `ai_parse_document` ou lib de parsing | automatico |
# MAGIC | Fatiar em chunks | escolher tamanho e overlap, testar | automatico |
# MAGIC | Gerar embeddings | escolher modelo, criar pipeline | automatico |
# MAGIC | Indexar | criar Vector Search endpoint + index | automatico |
# MAGIC | Recuperar | escrever a query, definir top-k, reranking | automatico |
# MAGIC | Gerar resposta | prompt, citacoes, guardrails | **Instructions** |
# MAGIC | Servir | criar e manter serving endpoint | automatico |
# MAGIC | Avaliar | montar dataset, escolher juizes | aba **Examples** |
# MAGIC
# MAGIC O que **continua sendo seu trabalho** — e e onde esta a qualidade:
# MAGIC
# MAGIC 1. **A curadoria do documento.** Norma desatualizada no Volume = resposta errada com
# MAGIC    aparencia de certeza. E por isso que este PDF tem **vigencia e revisao no
# MAGIC    cabecalho** — e que a Secao 4.2 diz explicitamente *"a cada 12 meses (nao 6 meses
# MAGIC    como praticado anteriormente)"*. O documento carrega a propria correcao.
# MAGIC 2. **A descricao da fonte.** E o que roteia a pergunta.
# MAGIC 3. **As Instructions.** Especialmente a regra "se nao esta no documento, diga que
# MAGIC    nao esta".
# MAGIC 4. **Os exemplos com guidelines** (Passo 4) — o unico jeito de saber se melhorou.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Passo 3 (UI) — Testar e inspecionar
# MAGIC
# MAGIC Com o endpoint **ONLINE**, teste na aba **Build**. Rode as perguntas nesta ordem:
# MAGIC a progressao e proposital — vai de fato simples ate os casos que quebram um RAG
# MAGIC mal configurado.
# MAGIC
# MAGIC | # | Pergunta | O que observar |
# MAGIC |---|---|---|
# MAGIC | 1 | `Qual o teto de reembolso para implante e qual o percentual da tabela?` | fato direto: `R$ 2.500,00`, `50%`, `30 dias uteis`. Confira contra o PDF |
# MAGIC | 2 | `Qual o codigo TUSS do implante osseointegrado e quem aprova?` | `88000031`, junta medica de 3 auditores, 10 dias uteis (Secao 1) |
# MAGIC | 3 | `Qual a carencia de ortodontia no plano Dental Essencial?` | `Nao coberto` — exige cruzar **linha e coluna** de uma tabela de 6 planos |
# MAGIC | 4 | `Um beneficiario Corp Essencial pediu autorizacao de implante. O que devo fazer?` | ⭐ deve citar a **Secao 4.1** e dizer que **NAO se abre recurso** |
# MAGIC | 5 | `De quanto em quanto tempo o beneficiario tem direito a segunda opiniao?` | ⭐ **teste de armadilha** — a resposta e **12 meses**. Se vier "6 meses", o agente leu a mencao a pratica antiga e ignorou a regra vigente |
# MAGIC | 6 | `Um beneficiario migrou de outro plano Odontoprev. Ele tem carencia zero para protese?` | ⭐ exige **combinar Secao 2 e Secao 4.5**: carencia zero (*) so vale para migracao interna, e a portabilidade preserva carencia cumprida se a interrupcao foi de ate 30 dias |
# MAGIC | 7 | `Qual o teto de reembolso para clareamento dental estetico?` | ⭐ **teste de recusa** — nao existe linha de clareamento na Secao 3, e a 4.3 diz que estetico nao e coberto. Deve dizer isso, **nao** inventar um valor |
# MAGIC | 8 | `A Odontoprev cobre cirurgia bariatrica?` | ⭐ **fora de escopo** — deve usar a frase de fallback |
# MAGIC
# MAGIC ### Inspecionar cada resposta — os tres botoes
# MAGIC
# MAGIC | Botao | Mostra | Por que importa |
# MAGIC |---|---|---|
# MAGIC | **View sources** | os trechos recuperados | e a **auditabilidade**: da para provar de onde veio a resposta |
# MAGIC | **View thoughts** | o raciocinio do agente | mostra se ele entendeu a pergunta |
# MAGIC | **View trace** | o trace MLflow completo | latencia por etapa, tokens, chamadas |
# MAGIC
# MAGIC > ⭐ **Faca isto ao menos uma vez:** rode a pergunta 6 e abra **View sources**. Se o
# MAGIC > trecho recuperado for so a tabela de carencias (Secao 2), **sem** o paragrafo 4.5
# MAGIC > da portabilidade, voce acabou de ver ao vivo o problema central de RAG: **a
# MAGIC > resposta e limitada pelo que foi recuperado**, nao pelo que o documento diz. E o
# MAGIC > motivo de existir o Passo 4.
# MAGIC
# MAGIC ### Playground
# MAGIC
# MAGIC **Open in Playground** abre a mesma conversa no Playground, onde da para comparar o
# MAGIC KA com outros modelos e ajustar parametros.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Passo 4 (UI) — Melhorar a qualidade: aba Examples
# MAGIC
# MAGIC Aqui esta a diferenca entre uma demo e algo que a operacao usa. O KA aprende com
# MAGIC **exemplos rotulados**: pergunta + a *guideline* que a resposta boa tem que cumprir.
# MAGIC
# MAGIC ### Como funciona
# MAGIC
# MAGIC 1. Aba **Examples** > **+ Add**.
# MAGIC 2. Escreva a **pergunta**.
# MAGIC 3. Em **Guidelines**, descreva o que a resposta correta precisa conter — nao a
# MAGIC    resposta em si, mas o **criterio**.
# MAGIC 4. Repita para os exemplos abaixo e clique em **Sync**.
# MAGIC
# MAGIC > 💡 Guideline e criterio de avaliacao, nao gabarito. "Deve informar R$ 2.500,00 e
# MAGIC > citar a Secao 3" e uma boa guideline; colar a resposta inteira nao e.
# MAGIC
# MAGIC ### Os 8 exemplos deste lab
# MAGIC
# MAGIC | Pergunta | Guideline |
# MAGIC |---|---|
# MAGIC | `Qual o teto de reembolso para implante?` | `Deve informar R$ 2.500,00, o percentual de 50% da tabela Odontoprev e o prazo de deposito de 30 dias uteis, citando a Secao 3.` |
# MAGIC | `Qual o codigo TUSS do implante osseointegrado e quem aprova?` | `Deve informar o TUSS 88000031, prazo de 10 dias uteis e aprovacao por junta medica de 3 auditores, citando a Secao 1.` |
# MAGIC | `Qual a carencia de protese no plano Dental Essencial?` | `Deve informar 300 dias corridos e citar a Secao 2. Nao deve confundir com a carencia de protese de outros planos.` |
# MAGIC | `Beneficiario Corp Essencial quer implante. Posso abrir recurso?` | `Deve dizer que o plano nao tem cobertura para implante, que a negativa e automatica e que o atendente NAO deve abrir recurso, citando a Secao 4.1.` |
# MAGIC | `De quanto em quanto tempo cabe segunda opiniao clinica?` | `Deve informar uma vez por especialidade a cada 12 meses, citando a Secao 4.2. NAO deve responder 6 meses - esse era o criterio anterior, revogado em Jan/2026.` |
# MAGIC | `Migrei de plano dentro da Odontoprev. Tenho carencia zero para protese?` | `Deve explicar que a carencia zero (*) da Secao 2 vale apenas para migracao dentro da propria operadora, e que pela Secao 4.5 a carencia ja cumprida e preservada desde que a interrupcao entre planos nao passe de 30 dias. Deve pedir ou considerar o plano de destino.` |
# MAGIC | `Qual o teto de reembolso para clareamento dental estetico?` | `Deve dizer que nao consta valor de reembolso para clareamento na tabela da Secao 3 e que clareamento estetico nao e coberto (Secao 4.3), exigindo indicacao clinica documentada. NAO deve inventar valor.` |
# MAGIC | `A Odontoprev cobre cirurgia bariatrica?` | `Deve usar a frase de fallback: a informacao nao consta na Politica de Autorizacao e Cobertura e recomenda-se consultar a auditoria clinica. NAO deve tentar responder.` |
# MAGIC
# MAGIC ### Import / Export
# MAGIC
# MAGIC | Botao | Para que serve |
# MAGIC |---|---|
# MAGIC | **Import** | carregar um conjunto de exemplos de uma tabela do Unity Catalog |
# MAGIC | **Export** | salvar os exemplos rotulados numa tabela nova |
# MAGIC
# MAGIC O **Export** e o que transforma isto em processo: a auditoria vai acumulando casos
# MAGIC reais, e o conjunto de exemplos passa a ser um **ativo governado no UC** —
# MAGIC versionado, com lineage, reaproveitavel em avaliacao com MLflow.
# MAGIC
# MAGIC ### Depois de sincronizar, volte ao Passo 3
# MAGIC
# MAGIC Rode as perguntas 5, 6, 7 e 8 de novo e compare. **Avaliar antes e depois e o
# MAGIC lab** — sem isso voce so tem a impressao de que melhorou.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Passo 4b (UI) — Manter o documento atualizado: aba Sources
# MAGIC
# MAGIC Norma muda — e este PDF ja prova o ponto: a Secao 4.2 registra que a segunda
# MAGIC opiniao passou de 6 para 12 meses em Jan/2026. Numa operadora real isso acontece
# MAGIC varias vezes por ano.
# MAGIC
# MAGIC Na aba **Sources**:
# MAGIC
# MAGIC | Acao | Como |
# MAGIC |---|---|
# MAGIC | Adicionar fonte (ate 10) | **+ Add** |
# MAGIC | Editar nome / descricao | icone de **lapis** |
# MAGIC | Remover fonte | icone de **lixeira** (aparece ao passar o mouse) |
# MAGIC | **Reindexar apos mudar arquivos** | icone de **sync** |
# MAGIC | Aplicar as mudancas | **Save and Update** |
# MAGIC
# MAGIC > ⚠️ **Trocar o PDF no Volume nao atualiza o agente sozinho.** Sem o **sync**, o
# MAGIC > agente continua respondendo com a versao antiga do indice — com toda a confianca.
# MAGIC > Em producao, o sync entra no mesmo job que publica a norma.
# MAGIC
# MAGIC ### Experimento (3 min) — vale fazer
# MAGIC
# MAGIC 1. Rode a celula abaixo (descomentando as duas ultimas linhas). Ela adiciona ao
# MAGIC    Volume um **comunicado que altera o teto de implante**.
# MAGIC 2. Pergunte ao agente: `Qual o teto de reembolso de implante?` → ainda `R$ 2.500,00`.
# MAGIC 3. Va em **Sources** > icone de **sync** > **Save and Update** e espere reindexar.
# MAGIC 4. Pergunte de novo. Agora deve aparecer `R$ 3.000,00` **a partir de 01/09/2026**,
# MAGIC    e o agente precisa lidar com **dois documentos que discordam** — o comunicado
# MAGIC    prevalece sobre a Secao 3 apenas para implante.
# MAGIC
# MAGIC E a demonstracao mais barata que existe de por que governanca de conteudo importa
# MAGIC mais que escolha de modelo.

# COMMAND ----------

# Comunicado de alteracao — para o experimento de sync do Passo 4b.
# Descomente as duas ultimas linhas para gravar no Volume.
COMUNICADO = """COMUNICADO DE ALTERACAO DE NORMA - CN-2026-014
Odontoprev | Diretoria de Operacoes | Emitido em 05/08/2026
Classificacao: Interno Confidencial

ASSUNTO: Reajuste do teto de reembolso de implante osseointegrado (TUSS 88000031).

1. A partir de 01/09/2026, o valor maximo de reembolso para implante osseointegrado
   passa de R$ 2.500,00 para R$ 3.000,00. O percentual da tabela Odontoprev permanece
   em 50% e o prazo de deposito permanece em 30 dias uteis.

2. Pedidos protocolados ate 31/08/2026 seguem o teto anterior de R$ 2.500,00,
   independentemente da data de analise.

3. Este comunicado prevalece sobre a Secao 3 da Politica de Autorizacao e Cobertura
   (Revisao Mar/2026) no que se refere EXCLUSIVAMENTE ao teto de implante. As demais
   linhas da tabela de reembolso permanecem inalteradas.

4. A alcada de aprovacao nao muda: implante segue exigindo junta medica de 3 auditores
   e prazo de autorizacao de 10 dias uteis (Politica de Autorizacao, Secao 1).

5. As carencias de implante por plano permanecem inalteradas (Secao 2), incluindo a
   ausencia de cobertura nos planos Corp Essencial e Dental Essencial.

Duvidas: auditoria clinica regional.
"""

# with open(f"{CAMINHO_VOLUME}/comunicado_CN-2026-014.txt", "w") as f:
#     f.write(COMUNICADO)
# print(f"Gravado em {CAMINHO_VOLUME} -> agora faca o SYNC na aba Sources")

print("Celula pronta. Descomente as duas ultimas linhas para rodar o experimento.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Passo 5 — Consumir o KA por codigo
# MAGIC
# MAGIC O KA nao vive so na UI: ele **e** um serving endpoint. Isso e o que permite plugar
# MAGIC o agente no app do Lab 04, num fluxo de CRM ou num job.
# MAGIC
# MAGIC ### 5.1 (UI) — Descobrir o endpoint
# MAGIC
# MAGIC 1. Na pagina do agente, abra a aba **Endpoint**.
# MAGIC 2. Copie o **nome do endpoint** — o padrao e `ka-<tile_id>-endpoint`.
# MAGIC 3. **Open in playground** > **Get code** traz o snippet pronto em `curl` e Python.
# MAGIC
# MAGIC A celula abaixo tenta achar o endpoint sozinha, o que e mais rapido em sala.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

endpoints_ka = [e for e in w.serving_endpoints.list() if e.name and e.name.startswith("ka-")]

print(f"Knowledge Assistants neste workspace: {len(endpoints_ka)}\n")
for e in endpoints_ka:
    print(f"  {e.name:50s} {str(e.state.ready) if e.state else '?'}")

ENDPOINT_KA = endpoints_ka[0].name if len(endpoints_ka) == 1 else None

if ENDPOINT_KA:
    print(f"\nUsando: {ENDPOINT_KA}")
else:
    print("\n👉 ALTERE a celula abaixo com o nome do SEU endpoint (aba Endpoint do agente).")

# COMMAND ----------

# 👉 ALTERE se houver mais de um KA no workspace (turma inteira criando)
# ENDPOINT_KA = "ka-xxxxxxxxxxxx-endpoint"

if not ENDPOINT_KA:
    raise ValueError(
        "ENDPOINT_KA nao definido.\n\n"
        "Isto e esperado se voce ainda nao criou o agente (Passo 2) ou se ha mais de um "
        "KA no workspace.\n"
        "Pegue o nome na pagina do SEU agente > aba Endpoint, descomente a linha acima "
        "e rode de novo."
    )

print(f"Endpoint: {ENDPOINT_KA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 — Fazer a pergunta por REST
# MAGIC
# MAGIC Agentes do Agent Bricks podem expor duas formas de payload, dependendo da versao:
# MAGIC
# MAGIC | Formato | Corpo |
# MAGIC |---|---|
# MAGIC | **ResponsesAgent** (atual) | `{"input": [{"role": "user", "content": "..."}]}` |
# MAGIC | **ChatCompletion** (legado) | `{"messages": [{"role": "user", "content": "..."}]}` |
# MAGIC
# MAGIC A funcao abaixo tenta o primeiro e cai para o segundo — assim o lab funciona nas
# MAGIC duas. Se ambos falharem, use o snippet exato de **Endpoint > Open in playground >
# MAGIC Get code**, que sempre reflete a versao do seu workspace.

# COMMAND ----------

import json
import time

import requests
from databricks.sdk.core import Config

cfg = Config()
host = cfg.host if cfg.host.startswith("http") else f"https://{cfg.host}"
url = f"{host.rstrip('/')}/serving-endpoints/{ENDPOINT_KA}/invocations"


def extrair_texto(resposta: dict) -> str:
    """Le a resposta nos dois formatos possiveis."""
    if "output" in resposta:  # ResponsesAgent
        partes = [
            c["text"]
            for item in resposta["output"]
            for c in item.get("content", [])
            if c.get("type") in ("output_text", "text") and c.get("text")
        ]
        if partes:
            return "\n".join(partes)
    if "choices" in resposta:  # ChatCompletion
        return resposta["choices"][0]["message"]["content"]
    if "messages" in resposta:
        return resposta["messages"][-1].get("content", "")
    return json.dumps(resposta, ensure_ascii=False)[:2000]


def perguntar(pergunta: str, timeout: int = 180) -> tuple[str, float]:
    """Devolve (resposta, latencia_ms). Tenta ResponsesAgent, cai para ChatCompletion."""
    headers = cfg.authenticate()
    headers["Content-Type"] = "application/json"
    mensagem = [{"role": "user", "content": pergunta}]

    inicio = time.perf_counter()
    resp = requests.post(url, headers=headers, json={"input": mensagem}, timeout=timeout)
    if resp.status_code >= 400:
        resp = requests.post(url, headers=headers, json={"messages": mensagem}, timeout=timeout)
    latencia_ms = (time.perf_counter() - inicio) * 1000
    resp.raise_for_status()
    return extrair_texto(resp.json()), latencia_ms


texto, ms = perguntar(
    "Qual o teto de reembolso para implante e qual o percentual da tabela Odontoprev?"
)
print(f"[{ms:.0f} ms]\n")
print(texto)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 — Rodar a bateria de perguntas de uma vez
# MAGIC
# MAGIC O mesmo conjunto do Passo 3, agora por codigo. E assim que se compara duas versoes
# MAGIC do agente sem depender de clicar na UI — o primeiro passo em direcao a uma
# MAGIC avaliacao automatizada com MLflow.

# COMMAND ----------

PERGUNTAS = [
    "Qual o codigo TUSS do implante osseointegrado e quem aprova?",
    "Qual a carencia de ortodontia no plano Dental Essencial?",
    "Um beneficiario Corp Essencial pediu autorizacao de implante. O que devo fazer?",
    "De quanto em quanto tempo o beneficiario tem direito a segunda opiniao clinica?",
    "Um beneficiario migrou de outro plano Odontoprev. Ele tem carencia zero para protese?",
    "Qual o teto de reembolso para clareamento dental estetico?",
    "A Odontoprev cobre cirurgia bariatrica?",
]

for i, p in enumerate(PERGUNTAS, 1):
    try:
        texto, ms = perguntar(p)
        print(f"\n{'=' * 78}")
        print(f"[{i}] {p}   ({ms:.0f} ms)")
        print(f"{'=' * 78}")
        print(texto)
    except Exception as e:  # noqa: BLE001
        print(f"\n[{i}] FALHOU: {type(e).__name__}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4 — Onde isso se conecta ao resto do workshop
# MAGIC
# MAGIC Com o endpoint na mao, o KA entra em qualquer um dos labs anteriores:
# MAGIC
# MAGIC | Integracao | Como | Beneficio |
# MAGIC |---|---|---|
# MAGIC | **App do Lab 04** | adicionar o endpoint do KA como **resource** e um campo "consultar a norma" | o auditor ve o score **e** a regra na mesma tela |
# MAGIC | **Supervisor Agent** | um agente que roteia entre o KA (norma) e um Genie Space (dados) | "quantos pedidos de implante glosamos em julho?" vai para o Genie; "cabe recurso?" vai para o KA |
# MAGIC | **Job em batch** | para cada pedido CRITICA da fila, pedir ao KA a norma aplicavel | anexa a fundamentacao normativa a fila **antes** do auditor abrir |
# MAGIC
# MAGIC O terceiro e o mais interessante: fecha o ciclo entre o preditivo e o generativo —
# MAGIC o modelo prioriza, o agente fundamenta, o auditor decide.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Passo 6 (UI) — Apagar o agente
# MAGIC
# MAGIC > ⚠️ **Faca isto no fim do workshop.** O KA mantem um **serving endpoint
# MAGIC > provisionado** enquanto existir, e ele consome recurso serverless — mesma
# MAGIC > disciplina de custo do Lab 02, do Lab 03 e do Lab 04.
# MAGIC
# MAGIC 1. Menu lateral > **Agents**.
# MAGIC 2. Ache o **seu** agente (`Politica Odontoprev <seu_nome>`).
# MAGIC 3. Menu **kebab** (⋮) > **Delete**.
# MAGIC 4. Confirme que o endpoint `ka-...-endpoint` desapareceu de **Serving**.
# MAGIC
# MAGIC O Volume `politicas_internas` sai junto com o schema quando voce roda o
# MAGIC `99_Cleanup/99_cleanup` (`DROP SCHEMA ... CASCADE`) — nao precisa apagar a mao.
# MAGIC
# MAGIC > 💡 O `99_Cleanup` **nao** apaga o agente: nao existe API publica de listagem para
# MAGIC > Agent Bricks (`GET /api/2.0/custom-llms` devolve `ENDPOINT_NOT_FOUND`), so a UI.
# MAGIC > Este passo e manual de proposito.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Resumo
# MAGIC
# MAGIC | Conceito | O que voce praticou |
# MAGIC |----------|--------------------|
# MAGIC | **O documento e o produto** | a base de conhecimento vive num **Volume do UC** — governanca, lineage e permissao valem para o agente tambem |
# MAGIC | **RAG sem escrever RAG** | parsing, chunking, embedding, indice, retrieval e serving saem prontos |
# MAGIC | **Verificar o parsing antes** | `ai_parse_document` mostra o que o agente vai realmente ler |
# MAGIC | **A descricao da fonte roteia** | e ela que decide se a fonte e relevante para a pergunta |
# MAGIC | **Instructions definem o comportamento** | a frase de fallback e o que separa util de perigoso |
# MAGIC | **Citacao = auditabilidade** | **View sources** e o que permite defender a decisao |
# MAGIC | **A resposta e limitada pelo retrieval** | a pergunta 6 (carencia + portabilidade) mostra isso ao vivo |
# MAGIC | **Exemplos + guidelines** | avaliar antes e depois, em vez de confiar na impressao |
# MAGIC | **Sync nao e automatico** | trocar o PDF no Volume **nao** atualiza o indice |
# MAGIC | **KA e um endpoint** | consumivel por REST, plugavel em app, job ou Supervisor Agent |
# MAGIC
# MAGIC ### As duas perguntas que fecham este lab
# MAGIC
# MAGIC **1. Por que a pergunta do clareamento estetico e a mais importante das oito?**
# MAGIC Um agente que inventa `R$ 200,00` com tom seguro e pior que um agente que nao
# MAGIC responde — porque a glosa sai fundamentada num valor que nao existe. Recusar bem e
# MAGIC requisito, nao limitacao.
# MAGIC
# MAGIC **2. Se o documento estivesse desatualizado, quem descobriria?**
# MAGIC Nem o modelo nem o KA. A Secao 4.2 deste PDF existe justamente porque alguem
# MAGIC precisou registrar que a regra dos 6 meses caiu. Vigencia, versao e sync sao parte
# MAGIC da arquitetura — nao detalhe de operacao.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### O workshop completo
# MAGIC
# MAGIC | Lab | Pergunta que responde |
# MAGIC |---|---|
# MAGIC | **01 — ML** | "este pedido e suspeito?" (batch, fila priorizada) |
# MAGIC | **02 — Serving** | "e agora, em milissegundos?" |
# MAGIC | **03 — Lakebase** | "e o historico do prestador, como chega em tempo real?" |
# MAGIC | **04 — App** | "como o auditor usa isso?" |
# MAGIC | **05 — Knowledge Assistant** | "e a norma, o que diz?" |
