# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop Hands-On Databricks — Odontoprev
# MAGIC ## Notebook 00 — Configuracao do Ambiente
# MAGIC
# MAGIC **Objetivo:** preparar um ambiente **isolado por participante** no Unity Catalog.
# MAGIC
# MAGIC O schema de cada participante e criado automaticamente a partir do seu
# MAGIC **usuario Databricks** — voce nao precisa digitar nada.
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Passo 1: Identificar o participante automaticamente

# COMMAND ----------

# Extrai o nome do usuario a partir do email (antes do @)
# Ex: gabriel.rangel@databricks.com -> gabriel_rangel
usuario = spark.sql("SELECT current_user()").collect()[0][0]
nome = usuario.split("@")[0].replace(".", "_").replace("-", "_").lower()

print(f"Usuario:        {usuario}")
print(f"Nome derivado:  {nome}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Passo 2: Criar o Catalogo (compartilhado) e o seu Schema
# MAGIC
# MAGIC ```
# MAGIC workshop_databricks                   <-- Catalogo (compartilhado pela turma)
# MAGIC   |-- gabriel_rangel                  <-- Schema derivado do seu email
# MAGIC   |     |-- bronze_reembolsos
# MAGIC   |     |-- silver_reembolsos
# MAGIC   |     |-- gold_features_fraude
# MAGIC   |     `-- ...
# MAGIC   |-- maria_silva
# MAGIC   `-- ...
# MAGIC ```
# MAGIC
# MAGIC Cada participante escreve **somente no seu schema**, então ninguem sobrescreve
# MAGIC o trabalho de ninguem — mesmo todos usando o mesmo catalogo.

# COMMAND ----------

CATALOGO = "workshop_databricks"
SCHEMA = nome

# O catalogo e criado UMA VEZ pelo instrutor, antes do workshop:
#
#   CREATE CATALOG workshop_databricks
#     MANAGED LOCATION 'abfss://<container>@<storage>.dfs.core.windows.net/';
#
# O MANAGED LOCATION e obrigatorio quando o metastore nao tem storage root
# proprio. Os participantes apenas criam o seu schema dentro dele.
spark.sql(f"USE CATALOG {CATALOGO}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOGO}.{SCHEMA}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print(f"Catalogo: {CATALOGO} (compartilhado)")
print(f"Schema:   {SCHEMA} (seu espaco isolado)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Passo 3: Criar o Volume da landing zone
# MAGIC
# MAGIC Os arquivos de origem (como chegariam dos sistemas transacionais da operadora)
# MAGIC vao para um **Volume** do Unity Catalog. O Lab de ML le as tabelas geradas a
# MAGIC partir desses arquivos.

# COMMAND ----------

VOLUMES = ["landing_reembolsos"]

for vol in VOLUMES:
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOGO}.{SCHEMA}.{vol}")
    print(f"  /Volumes/{CATALOGO}/{SCHEMA}/{vol}/")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verificacao

# COMMAND ----------

print(f"{'='*64}")
print(f"  AMBIENTE CONFIGURADO COM SUCESSO")
print(f"{'='*64}")
print(f"  Usuario:   {usuario}")
print(f"  Catalogo:  {CATALOGO}")
print(f"  Schema:    {SCHEMA}")
print(f"  Volume:    {' | '.join(VOLUMES)}")
print(f"{'='*64}")
print()
print("Proximo passo: execute 00_Setup/01_dados_sinteticos")
