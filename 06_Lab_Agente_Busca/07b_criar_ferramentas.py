# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 06 — Agente de Busca de Dentistas
# MAGIC ## Parte 2 de 3 — Criar as ferramentas (UC Functions)
# MAGIC
# MAGIC **Objetivo:** criar as **3 ferramentas** que o agente vai chamar. Elas sao
# MAGIC `FUNCTION` do Unity Catalog — objetos governados, com `GRANT`, lineage e versao.
# MAGIC
# MAGIC ### A ideia central: o LLM nao sabe nada, e isso e bom
# MAGIC
# MAGIC Um LLM nao tem a menor ideia de onde ha dentista credenciado. Se voce perguntar
# MAGIC direto, ele **inventa** um endereco plausivel — e isso e pior que nao responder.
# MAGIC
# MAGIC A saida nao e treinar o modelo com a rede credenciada. E dar a ele **ferramentas**:
# MAGIC
# MAGIC ```
# MAGIC   "Quero um endodontista perto de casa"
# MAGIC              |
# MAGIC              v
# MAGIC   [LLM decide QUAIS ferramentas chamar e em que ORDEM]
# MAGIC              |
# MAGIC              +--> buscar_beneficiario_por_cpf(cpf)
# MAGIC              |      devolve: nome, plano, endereco, LAT, LONG
# MAGIC              |
# MAGIC              +--> encontrar_dentistas_proximos(lat, long, especialidade, plano)
# MAGIC              |      devolve: lista ordenada por distancia
# MAGIC              |
# MAGIC              +--> listar_especialidades()   (quando o pedido e ambiguo)
# MAGIC              |
# MAGIC              v
# MAGIC   [LLM redige a resposta com os dados REAIS que recebeu]
# MAGIC ```
# MAGIC
# MAGIC O LLM faz o que sabe fazer (entender linguagem, decidir, redigir). O SQL faz o
# MAGIC que sabe fazer (filtrar e ordenar dado governado). **Ninguem faz o trabalho do
# MAGIC outro.**
# MAGIC
# MAGIC ### O que voce vai construir
# MAGIC
# MAGIC | # | Ferramenta | Entrada | Devolve |
# MAGIC |---|---|---|---|
# MAGIC | 1 | `buscar_beneficiario_por_cpf` | CPF | nome, plano, endereco, lat, long |
# MAGIC | 2 | `encontrar_dentistas_proximos` | lat, long, especialidade, plano, raio | ate 5 credenciados por distancia |
# MAGIC | 3 | `listar_especialidades` | — | especialidades validas |
# MAGIC
# MAGIC > **Pre-requisito:** `07a_dados_geo` executado (as tabelas
# MAGIC > `gold_rede_credenciada` e `gold_beneficiarios_geo` precisam existir).
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configuracao

# COMMAND ----------

usuario = spark.sql("SELECT current_user()").collect()[0][0]
nome = usuario.split("@")[0].replace(".", "_").replace("-", "_").lower()

CATALOGO = "workshop_databricks"
SCHEMA = nome

spark.sql(f"USE CATALOG {CATALOGO}")
spark.sql(f"USE SCHEMA {SCHEMA}")

TB_REDE = f"{CATALOGO}.{SCHEMA}.gold_rede_credenciada"
TB_BENEF = f"{CATALOGO}.{SCHEMA}.gold_beneficiarios_geo"

print(f"Rede:          {TB_REDE}")
print(f"Beneficiarios: {TB_BENEF}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## ⭐ Antes de escrever: o `COMMENT` E a interface
# MAGIC
# MAGIC Este e o conceito mais importante do lab, e o mais facil de ignorar.
# MAGIC
# MAGIC Quando voce registra uma UC Function como ferramenta, o Databricks converte a
# MAGIC assinatura num **JSON schema** que vai no prompt do modelo. O que o LLM ve e:
# MAGIC
# MAGIC - o **nome** da funcao;
# MAGIC - o `COMMENT` **da funcao** — para decidir **se** usa;
# MAGIC - o `COMMENT` **de cada parametro** — para decidir **o que** passar;
# MAGIC - os tipos.
# MAGIC
# MAGIC Ele **nao ve** o corpo da funcao. Nao ve o SQL. Nao ve os dados.
# MAGIC
# MAGIC | Comentario | Consequencia |
# MAGIC |---|---|
# MAGIC | `COMMENT 'busca'` | o modelo nao sabe se serve; **pode nao chamar** |
# MAGIC | `COMMENT 'Retorna dentistas CREDENCIADOS proximos a uma coordenada, filtrando por especialidade e plano. Use depois de obter lat/long via buscar_beneficiario_por_cpf.'` | o modelo chama na hora certa, com os argumentos certos |
# MAGIC
# MAGIC > 💰 **Em GenAI, escrever documentacao deixou de ser boa educacao e passou a ser
# MAGIC > programacao.** O `COMMENT` nao e comentario — e o codigo que orienta o modelo.
# MAGIC
# MAGIC Repare, nas funcoes abaixo, que os comentarios **dizem quando usar** e **como se
# MAGIC encadeiam**, nao apenas o que a coluna significa.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Ferramenta 1 — `buscar_beneficiario_por_cpf`
# MAGIC
# MAGIC Traduz o CPF em tudo que a busca precisa: **plano** (para filtrar quem aceita) e
# MAGIC **lat/long** (de onde medir distancia).
# MAGIC
# MAGIC ### Por que ela devolve lat/long direto
# MAGIC
# MAGIC No projeto de referencia havia uma etapa a mais: uma funcao `geocode_endereco`
# MAGIC convertia o endereco em coordenada chamando uma API externa. **Aqui isso nao
# MAGIC funciona** — o sandbox de UC Function nao tem saida para internet
# MAGIC (`TimeoutError`), como medimos na Parte 1.
# MAGIC
# MAGIC A solucao e melhor do que o contorno sugere: a coordenada vive **no cadastro**.
# MAGIC Georreferencia-se uma vez, na ingestao, e nao a cada pergunta de beneficiario.
# MAGIC Isso remove uma chamada externa do caminho critico, elimina dependencia de
# MAGIC servico de terceiro em producao, e corta uma ferramenta do agente — **menos
# MAGIC ferramenta, menos chance de o modelo errar a ordem**.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOGO}.{SCHEMA}.buscar_beneficiario_por_cpf(
  cpf_beneficiario STRING
    COMMENT 'CPF do beneficiario, somente digitos, sem pontos ou tracos. Fornecido pelo sistema — NUNCA solicite ao usuario.'
)
RETURNS TABLE(
  nome       STRING COMMENT 'Nome completo do beneficiario',
  plano      STRING COMMENT 'Nome do plano contratado. Use este valor como filtro na ferramenta encontrar_dentistas_proximos.',
  endereco   STRING COMMENT 'Logradouro e numero do beneficiario',
  bairro     STRING COMMENT 'Bairro do beneficiario',
  cidade     STRING COMMENT 'Cidade do beneficiario',
  uf         STRING COMMENT 'Unidade federativa (sigla) do beneficiario',
  latitude   DOUBLE COMMENT 'Latitude do endereco. Passe para o parametro lat_usuario de encontrar_dentistas_proximos.',
  longitude  DOUBLE COMMENT 'Longitude do endereco. Passe para o parametro long_usuario de encontrar_dentistas_proximos.',
  status     STRING COMMENT 'Situacao do contrato: ativo ou cancelado'
)
COMMENT 'Retorna os dados cadastrais de um beneficiario da Odontoprev a partir do CPF, incluindo o plano contratado e a latitude/longitude do endereco. Use esta ferramenta PRIMEIRO, antes de buscar dentistas, porque a busca de dentistas exige o plano e as coordenadas que esta funcao devolve. Se nao retornar nenhuma linha, o CPF nao existe na base.'
RETURN
  SELECT nome, plano, endereco, bairro, cidade, uf, latitude, longitude, status
  FROM {TB_BENEF}
  WHERE cpf = cpf_beneficiario
  LIMIT 1
""")

print("Ferramenta 1 criada: buscar_beneficiario_por_cpf")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Testar

# COMMAND ----------

spark.sql(f"SELECT * FROM {CATALOGO}.{SCHEMA}.buscar_beneficiario_por_cpf('11111111111')").display()

# COMMAND ----------

spark.sql(f"SELECT * FROM {CATALOGO}.{SCHEMA}.buscar_beneficiario_por_cpf('22222222222')").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Ferramenta 2 — `encontrar_dentistas_proximos`
# MAGIC
# MAGIC O coracao do lab. Recebe coordenada, especialidade, plano e raio; devolve os
# MAGIC credenciados mais proximos.
# MAGIC
# MAGIC ### Tres decisoes de implementacao que valem explicar
# MAGIC
# MAGIC **1. Filtro barato antes de calculo caro (*bounding box*).**
# MAGIC A haversine tem quatro funcoes trigonometricas por linha. Rodar em 700
# MAGIC prestadores nao doi, mas em 200 mil doi muito. Entao primeiro cortamos com um
# MAGIC **retangulo** — comparacao numerica simples, que usa indice:
# MAGIC
# MAGIC ```
# MAGIC latitude BETWEEN lat - (raio/111.0) AND lat + (raio/111.0)
# MAGIC ```
# MAGIC
# MAGIC 1 grau de latitude ~ 111 km em qualquer lugar. Para longitude, os meridianos
# MAGIC convergem nos polos, entao divide-se por `111 * cos(latitude)`. O retangulo pega
# MAGIC um pouco **mais** que o circulo (os cantos), e o filtro final de raio corrige.
# MAGIC
# MAGIC **2. `array_contains` para o plano.** `planos_aceitos` e `ARRAY<STRING>`;
# MAGIC `array_contains(planos_aceitos, plano)` resolve sem `explode` nem join.
# MAGIC
# MAGIC **3. `LIMIT 5`, nao 50.** O resultado vai para dentro do **prompt** do modelo.
# MAGIC Devolver 50 dentistas gasta contexto, aumenta custo e latencia, e piora a
# MAGIC resposta — ninguem escolhe entre 50 opcoes. **Em ferramenta de agente, o limite
# MAGIC de linhas e decisao de produto, nao detalhe tecnico.**

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOGO}.{SCHEMA}.encontrar_dentistas_proximos(
  lat_usuario DOUBLE
    COMMENT 'Latitude do beneficiario, obtida na ferramenta buscar_beneficiario_por_cpf.',
  long_usuario DOUBLE
    COMMENT 'Longitude do beneficiario, obtida na ferramenta buscar_beneficiario_por_cpf.',
  especialidade_desejada STRING
    COMMENT 'Especialidade odontologica exata. Use listar_especialidades para ver os valores validos. NUNCA deduza a especialidade a partir de sintomas relatados pelo usuario.',
  plano_usuario STRING
    COMMENT 'Nome do plano do beneficiario, obtido na ferramenta buscar_beneficiario_por_cpf. Filtra apenas dentistas que aceitam esse plano.',
  raio_km INT DEFAULT 15
    COMMENT 'Raio maximo de busca em quilometros. Padrao 15. Se a busca nao retornar resultados, tente novamente com um raio maior, por exemplo 50.'
)
RETURNS TABLE(
  nome_prestador  STRING COMMENT 'Nome do dentista ou da clinica',
  especialidade   STRING COMMENT 'Especialidade do dentista',
  cro             STRING COMMENT 'Registro no Conselho Regional de Odontologia',
  endereco        STRING COMMENT 'Logradouro e numero do consultorio',
  bairro          STRING COMMENT 'Bairro do consultorio',
  cidade          STRING COMMENT 'Cidade do consultorio',
  uf              STRING COMMENT 'UF do consultorio',
  telefone        STRING COMMENT 'Telefone de contato do consultorio',
  nota_avaliacao  DOUBLE COMMENT 'Nota media de avaliacao dos beneficiarios, de 0 a 5',
  aceita_urgencia BOOLEAN COMMENT 'Indica se o consultorio atende casos de urgencia',
  distancia_km    DOUBLE COMMENT 'Distancia em linha reta, em quilometros, entre o beneficiario e o consultorio'
)
COMMENT 'Retorna ate 5 dentistas CREDENCIADOS da rede Odontoprev proximos a uma coordenada, ordenados do mais perto para o mais longe. Filtra apenas dentistas da especialidade indicada que aceitam o plano do beneficiario. Requer latitude, longitude e plano — obtenha-os antes com buscar_beneficiario_por_cpf. IMPORTANTE: se retornar zero linhas, significa que NAO existe dentista credenciado dessa especialidade que aceite esse plano dentro do raio; nesse caso informe o beneficiario e sugira ampliar o raio ou usar reembolso por livre escolha. Esta funcao busca somente rede credenciada, nunca prestadores fora da rede.'
RETURN
  WITH candidatos AS (
    -- Passo 1: filtros baratos primeiro (especialidade, plano, bounding box)
    SELECT
      nome_prestador, especialidade, cro, endereco, bairro, cidade, uf,
      telefone, nota_avaliacao, aceita_urgencia, latitude, longitude
    FROM {TB_REDE}
    WHERE credenciado = true
      AND upper(especialidade) = upper(especialidade_desejada)
      AND array_contains(planos_aceitos, plano_usuario)
      AND latitude  BETWEEN lat_usuario  - (raio_km / 111.0)
                        AND lat_usuario  + (raio_km / 111.0)
      AND longitude BETWEEN long_usuario - (raio_km / (111.0 * cos(radians(lat_usuario))))
                        AND long_usuario + (raio_km / (111.0 * cos(radians(lat_usuario))))
  ),
  com_distancia AS (
    -- Passo 2: haversine apenas nos candidatos que sobraram
    SELECT
      *,
      6371 * acos(
        least(1.0,
          cos(radians(lat_usuario)) * cos(radians(latitude)) *
          cos(radians(longitude) - radians(long_usuario)) +
          sin(radians(lat_usuario)) * sin(radians(latitude))
        )
      ) AS distancia_km
    FROM candidatos
  )
  -- Passo 3: raio exato, ordena e limita
  SELECT
    nome_prestador, especialidade, cro, endereco, bairro, cidade, uf,
    telefone, nota_avaliacao, aceita_urgencia,
    round(distancia_km, 2) AS distancia_km
  FROM com_distancia
  WHERE distancia_km <= raio_km
  ORDER BY distancia_km ASC
  LIMIT 5
""")

print("Ferramenta 2 criada: encontrar_dentistas_proximos")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 💡 Por que `least(1.0, ...)` dentro do `acos`
# MAGIC
# MAGIC Detalhe pequeno e traicoeiro. `acos` so aceita argumento entre -1 e 1. Quando os
# MAGIC dois pontos sao **quase identicos**, o arredondamento de ponto flutuante pode
# MAGIC produzir `1.0000000000000002` — e `acos` disso devolve `NaN`.
# MAGIC
# MAGIC O resultado seria um dentista com `distancia_km = NaN` que **desaparece do
# MAGIC `WHERE`** silenciosamente. Justamente o dentista no mesmo endereco do
# MAGIC beneficiario — o mais perto de todos.
# MAGIC
# MAGIC O `least(1.0, ...)` trava o teto e resolve. Vale mencionar em sala: e o tipo de
# MAGIC bug que nao levanta erro, so entrega resultado errado.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Ferramenta 3 — `listar_especialidades`
# MAGIC
# MAGIC Parece trivial, mas resolve um problema real: **o modelo nao sabe quais
# MAGIC especialidades existem na sua base.**
# MAGIC
# MAGIC Sem ela, o LLM chuta com vocabulario generico — "Odontologia Geral", "Dentista
# MAGIC de canal" — e o `upper(especialidade) = upper(...)` nao casa. A busca volta
# MAGIC vazia e parece que nao ha rede, quando na verdade o filtro nao bateu.
# MAGIC
# MAGIC Com ela, o agente pode **listar as opcoes** para o beneficiario escolher — que e
# MAGIC exatamente o comportamento exigido pelo guardrail clinico (Parte 3): quando o
# MAGIC pedido vem por sintoma, o agente pergunta em vez de deduzir.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOGO}.{SCHEMA}.listar_especialidades()
RETURNS TABLE(
  especialidade    STRING COMMENT 'Nome exato da especialidade, como deve ser passado para encontrar_dentistas_proximos',
  qtd_credenciados INT    COMMENT 'Quantos dentistas credenciados existem nessa especialidade em toda a rede'
)
COMMENT 'Retorna a lista das especialidades odontologicas validas na rede credenciada Odontoprev, com a quantidade de dentistas em cada uma. Use esta ferramenta quando o beneficiario nao disser claramente a especialidade, ou quando descrever apenas sintomas, para apresentar as opcoes disponiveis e pedir que ele escolha. Os valores retornados sao os unicos aceitos pelo parametro especialidade_desejada de encontrar_dentistas_proximos.'
RETURN
  SELECT especialidade, CAST(COUNT(*) AS INT) AS qtd_credenciados
  FROM {TB_REDE}
  WHERE credenciado = true
  GROUP BY especialidade
  ORDER BY qtd_credenciados DESC
""")

print("Ferramenta 3 criada: listar_especialidades")

# COMMAND ----------

spark.sql(f"SELECT * FROM {CATALOGO}.{SCHEMA}.listar_especialidades()").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Testar o encadeamento completo — sem LLM
# MAGIC
# MAGIC Antes de envolver o modelo, provamos que as ferramentas funcionam encadeadas.
# MAGIC **Se falhar aqui, o problema nao e o agente.** Depurar SQL e barato; depurar
# MAGIC "por que o agente respondeu estranho" e caro.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cenario A — Ana, em Sao Paulo (rede densa)

# COMMAND ----------

ana = spark.sql(
    f"SELECT * FROM {CATALOGO}.{SCHEMA}.buscar_beneficiario_por_cpf('11111111111')"
).collect()[0]

print(f"Beneficiaria: {ana['nome']}")
print(f"Plano:        {ana['plano']}")
print(f"Endereco:     {ana['endereco']}, {ana['bairro']}, {ana['cidade']}-{ana['uf']}")
print(f"Coordenada:   {ana['latitude']}, {ana['longitude']}")

# COMMAND ----------

ESPECIALIDADE_TESTE = "Endodontia"

spark.sql(f"""
    SELECT * FROM {CATALOGO}.{SCHEMA}.encontrar_dentistas_proximos(
      {ana['latitude']}, {ana['longitude']},
      '{ESPECIALIDADE_TESTE}', '{ana['plano']}', 15
    )
""").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cenario B — Carlos, em Manaus (rede rala)
# MAGIC
# MAGIC ⭐ Aqui esta o resultado mais importante do lab. Provavelmente vem **vazio**.

# COMMAND ----------

carlos = spark.sql(
    f"SELECT * FROM {CATALOGO}.{SCHEMA}.buscar_beneficiario_por_cpf('22222222222')"
).collect()[0]

print(f"Beneficiario: {carlos['nome']}")
print(f"Plano:        {carlos['plano']}")
print(f"Endereco:     {carlos['endereco']}, {carlos['bairro']}, {carlos['cidade']}-{carlos['uf']}")
print()

for raio in (15, 50, 200):
    n = spark.sql(f"""
        SELECT count(*) AS n FROM {CATALOGO}.{SCHEMA}.encontrar_dentistas_proximos(
          {carlos['latitude']}, {carlos['longitude']},
          '{ESPECIALIDADE_TESTE}', '{carlos['plano']}', {raio}
        )
    """).collect()[0]["n"]
    print(f"  raio {raio:4d} km -> {n} credenciado(s) de {ESPECIALIDADE_TESTE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### ⭐ Leia isto com a turma
# MAGIC
# MAGIC Resultado medido no ambiente de validacao:
# MAGIC
# MAGIC ```
# MAGIC Ana   (Bela Vista, SP)  raio  15 km -> 5 endodontistas (0,78 a 5,44 km)
# MAGIC Carlos (Adrianopolis, AM) raio  15 km -> 0
# MAGIC                            raio  50 km -> 0
# MAGIC                            raio 200 km -> 0
# MAGIC ```
# MAGIC
# MAGIC E nao e falha de filtro. O Amazonas tem, na base inteira, **5 prestadores** —
# MAGIC 3 credenciados, distribuidos em **3 especialidades** (Cirurgia, Periodontia,
# MAGIC Odontopediatria). **Endodontia nao existe em AM.** Compare com SP: 217
# MAGIC prestadores, 136 credenciados, 10 especialidades.
# MAGIC
# MAGIC Se o Carlos nao tem credenciado em raio de 200 km, ele **vai** pedir reembolso.
# MAGIC E vai ser **legitimo**.
# MAGIC
# MAGIC Feche o circulo do workshop:
# MAGIC
# MAGIC | Lab | O que faz com esse mesmo beneficiario |
# MAGIC |---|---|
# MAGIC | **01** | aprende que reembolso alto em AM **nao** e fraude (hard negative) |
# MAGIC | **05** | consulta a norma: reembolso vale quando nao ha rede em 30 km (Secao 3) |
# MAGIC | **06** | **prova, com dado, que nao ha rede** — e a evidencia da excecao |
# MAGIC
# MAGIC Ou seja: esta ferramenta nao serve so para o beneficiario achar dentista. Ela
# MAGIC produz a **evidencia auditavel** de que a condicao da Secao 3 da politica foi
# MAGIC atendida. O mesmo dado serve ao beneficiario e ao auditor.
# MAGIC
# MAGIC E a acao de negocio e clara: **credenciar em Manaus** custa menos que pagar
# MAGIC reembolso a 50% da tabela indefinidamente.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cenario C — o que acontece com especialidade invalida
# MAGIC
# MAGIC Simulando o erro que o LLM comete quando **nao** tem a ferramenta 3.

# COMMAND ----------

for esp in ("Endodontia", "Dentista de canal", "Odontologia Geral"):
    n = spark.sql(f"""
        SELECT count(*) AS n FROM {CATALOGO}.{SCHEMA}.encontrar_dentistas_proximos(
          {ana['latitude']}, {ana['longitude']}, '{esp}', '{ana['plano']}', 15
        )
    """).collect()[0]["n"]
    marca = "OK " if n > 0 else "VAZIO"
    print(f"  [{marca}] '{esp}' -> {n} resultado(s)")

print("\n👉 'Dentista de canal' e 'Odontologia Geral' NAO existem na base.")
print("   Vazio aqui NAO significa 'sem rede' — significa 'filtro nao bateu'.")
print("   E por isso que listar_especialidades existe.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Permissoes — quem pode chamar a ferramenta
# MAGIC
# MAGIC A ferramenta e um objeto do Unity Catalog, entao vale a mesma governanca das
# MAGIC tabelas. Isso tem uma consequencia forte:
# MAGIC
# MAGIC > **Um agente nao tem privilegio proprio.** Ele executa com a identidade de quem
# MAGIC > o chama (ou do service principal que o serve). Se essa identidade nao tem
# MAGIC > `EXECUTE` na funcao, a ferramenta simplesmente nao roda.
# MAGIC
# MAGIC No Playground (Parte 3) voce e a identidade, e como criou as funcoes, ja tem
# MAGIC acesso. A celula abaixo fica **comentada** porque so e necessaria ao publicar o
# MAGIC agente para outras pessoas ou como app.

# COMMAND ----------

# Descomente e ajuste ao publicar o agente para terceiros.
#
# PRINCIPAL = "conta.usuario@empresa.com"     # ou o client_id do service principal
#
# spark.sql(f"GRANT USE CATALOG ON CATALOG {CATALOGO} TO `{PRINCIPAL}`")
# spark.sql(f"GRANT USE SCHEMA ON SCHEMA {CATALOGO}.{SCHEMA} TO `{PRINCIPAL}`")
# for f in ("buscar_beneficiario_por_cpf", "encontrar_dentistas_proximos",
#           "listar_especialidades"):
#     spark.sql(f"GRANT EXECUTE ON FUNCTION {CATALOGO}.{SCHEMA}.{f} TO `{PRINCIPAL}`")
# # A funcao le as tabelas, entao o chamador tambem precisa de SELECT nelas:
# spark.sql(f"GRANT SELECT ON TABLE {TB_REDE} TO `{PRINCIPAL}`")
# spark.sql(f"GRANT SELECT ON TABLE {TB_BENEF} TO `{PRINCIPAL}`")

print("Celula de GRANTs pronta (comentada). Necessaria so ao publicar o agente.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conferir as ferramentas registradas

# COMMAND ----------

spark.sql(f"SHOW FUNCTIONS IN {CATALOGO}.{SCHEMA} LIKE '*'").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Apendice — e se eu **precisar** chamar uma API externa?
# MAGIC
# MAGIC O caminho governado e `http_request()` sobre uma **UC Connection** do tipo
# MAGIC `HTTP`. A conexao e um objeto do Unity Catalog: tem dono, tem `GRANT`, guarda a
# MAGIC credencial fora do codigo e registra quem chamou. Bem diferente de um
# MAGIC `requests.get` solto.
# MAGIC
# MAGIC ```sql
# MAGIC CREATE CONNECTION geocoding_api TYPE HTTP
# MAGIC OPTIONS (
# MAGIC   host 'https://api.exemplo.com',
# MAGIC   port '443',
# MAGIC   base_path '/v1/',
# MAGIC   bearer_token 'SEU_TOKEN'
# MAGIC );
# MAGIC
# MAGIC SELECT http_request(
# MAGIC   conn   => 'geocoding_api',
# MAGIC   method => 'GET',
# MAGIC   path   => '/geocode',
# MAGIC   params => map('endereco', 'Avenida Paulista, 1000')
# MAGIC );
# MAGIC -- devolve STRUCT<status_code INT, text STRING>
# MAGIC ```
# MAGIC
# MAGIC ### Duas ressalvas que medimos / lemos na doc
# MAGIC
# MAGIC **1. A conexao tenta descoberta OIDC no host.** Tentamos criar uma conexao para
# MAGIC o Nominatim (API publica, sem OAuth) e falhou:
# MAGIC
# MAGIC ```
# MAGIC DCR registration failed for host 'https://nominatim.openstreetmap.org':
# MAGIC Request to .../.well-known/openid-configuration failed. HTTP 404
# MAGIC ```
# MAGIC
# MAGIC Ou seja: o caminho governado espera uma API que se comporte como servico
# MAGIC autenticado. API publica anonima nao e o caso de uso previsto.
# MAGIC
# MAGIC **2. `http_request` e *rate limited*.** A propria doc diz que foi desenhada para
# MAGIC uso **interativo e de agente**, nao para varrer muitas linhas: rodar em volume
# MAGIC resulta em throttling. Para novos projetos, a recomendacao e o *connections
# MAGIC proxy endpoint* com o SDK do provedor.
# MAGIC
# MAGIC > 📌 **A licao de arquitetura:** enriquecimento em massa (geocodificar 700
# MAGIC > prestadores) pertence ao **pipeline**. Chamada externa pontual dentro do fluxo
# MAGIC > do agente (consultar um CEP, criar um ticket) e que pertence ao `http_request`.
# MAGIC > E o mesmo principio do Lab 03: **separar calcular de servir**.
# MAGIC >
# MAGIC > 📄 Doc: https://docs.databricks.com/aws/en/sql/language-manual/functions/http_request
# MAGIC
# MAGIC ---
# MAGIC ## Resumo desta parte
# MAGIC
# MAGIC | Conceito | O que voce praticou |
# MAGIC |---|---|
# MAGIC | **Ferramenta = UC Function** | objeto governado, com `GRANT`, lineage e versao |
# MAGIC | **O `COMMENT` e a interface** | o LLM le a descricao, nao o corpo. Comentario ruim = ferramenta ignorada |
# MAGIC | **Comentario que orienta encadeamento** | "obtenha antes com `buscar_beneficiario_por_cpf`" |
# MAGIC | **Filtro barato antes de calculo caro** | bounding box antes da haversine |
# MAGIC | **`LIMIT` e decisao de produto** | o resultado vai para o prompt: 5, nao 50 |
# MAGIC | **`least(1.0, ...)` no `acos`** | evita `NaN` que sumiria com o dentista mais proximo |
# MAGIC | **Vazio e resposta valida** | e a evidencia de que nao ha rede — nao um bug |
# MAGIC | **UC Function nao acessa internet** | enriquecimento externo pertence ao pipeline |
# MAGIC
# MAGIC > **Proximo:** `07c_agente_playground` — montar o agente na **UI**, com o system
# MAGIC > prompt e os guardrails.
