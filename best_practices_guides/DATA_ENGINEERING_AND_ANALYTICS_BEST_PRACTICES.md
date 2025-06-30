# The Definitive Guide to Modern Data Engineering & Analytics (Mid-2025)

This guide provides a production-grade blueprint for building scalable, reliable, and cost-effective data platforms using the latest tools and patterns. It synthesizes real-world lessons from operating petabyte-scale systems and moves beyond basic tutorials to address the complexities of enterprise data engineering.

### Prerequisites & Technology Stack
- **Ingestion/Streaming**: Kafka 3.8+, Debezium 2.7+, Confluent Schema Registry 7.7+
- **Storage**: Apache Iceberg 1.6+ or Delta Lake 3.2+, S3/MinIO, Snowflake/BigQuery/Databricks
- **Processing**: Apache Spark 4.0+, Apache Flink 2.0+, dbt 1.9+, DuckDB 1.1+
- **Orchestration**: Apache Airflow 3.0+, Prefect 3.0+, or Dagster 2.0+
- **Observability**: Great Expectations 0.19+, Monte Carlo, OpenLineage 1.20+
- **Languages**: Python 3.12+, SQL, Scala 3.4+ (for Spark/Flink)

### Key Architectural Principles
1. **Lake-house First**: Combine the flexibility of data lakes with warehouse capabilities
2. **Streaming by Default**: Design for real-time with batch as a special case
3. **Schema Evolution**: Plan for changing data structures from day one
4. **Cost-Aware Design**: Optimize for both performance and cloud spend
5. **Observability Native**: Build quality checks and lineage into every pipeline

---

## 1. Modern Lake-house Architecture

The lake-house pattern has matured into the dominant architecture, offering warehouse-like performance on data lake storage. Choose your table format based on your ecosystem.

### ✅ DO: Choose the Right Table Format

**Apache Iceberg** - Best for multi-engine environments:
```python
# Environment setup for Iceberg
from pyspark.sql import SparkSession
import os

spark = SparkSession.builder \
    .appName("IcebergLakehouse") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog") \
    .config("spark.sql.catalog.spark_catalog.type", "hive") \
    .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.iceberg.type", "hadoop") \
    .config("spark.sql.catalog.iceberg.warehouse", "s3a://your-bucket/warehouse") \
    .config("spark.sql.defaultCatalog", "iceberg") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .getOrCreate()

# Create partitioned table with Z-ordering for query optimization
spark.sql("""
    CREATE TABLE IF NOT EXISTS iceberg.raw.events (
        event_id BIGINT,
        user_id STRING,
        event_type STRING,
        properties MAP<STRING, STRING>,
        event_time TIMESTAMP,
        processing_time TIMESTAMP,
        year INT,
        month INT,
        day INT
    ) USING iceberg
    PARTITIONED BY (year, month, day)
    TBLPROPERTIES (
        'write.format.default' = 'parquet',
        'write.parquet.compression-codec' = 'zstd',
        'write.metadata.delete-after-commit.enabled' = 'true',
        'write.metadata.previous-versions-max' = '100',
        'write.spark.fanout.enabled' = 'true',
        'write.distribution-mode' = 'hash'
    )
""")

# Optimize with Z-ordering for common query patterns
spark.sql("""
    CALL iceberg.system.rewrite_data_files(
        table => 'iceberg.raw.events',
        strategy => 'sort',
        sort_order => 'user_id, event_time'
    )
""")
```

**Delta Lake** - Best for Databricks-centric environments:
```python
from delta import configure_spark_with_delta_pip, DeltaTable

# Configure for optimal performance
spark = configure_spark_with_delta_pip(
    SparkSession.builder
    .appName("DeltaLakehouse")
    .config("spark.databricks.delta.preview.enabled", "true")
    .config("spark.databricks.delta.retentionDurationCheck.enabled", "false")
    .config("spark.databricks.delta.optimizeWrite.enabled", "true")
    .config("spark.databricks.delta.autoCompact.enabled", "true")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
).getOrCreate()

# Create table with liquid clustering (Delta 3.2+)
spark.sql("""
    CREATE TABLE IF NOT EXISTS bronze.raw_events (
        event_id BIGINT,
        user_id STRING,
        event_type STRING,
        properties MAP<STRING, STRING>,
        event_time TIMESTAMP,
        _ingestion_time TIMESTAMP DEFAULT current_timestamp()
    ) USING DELTA
    CLUSTER BY (user_id, date(event_time))
    TBLPROPERTIES (
        'delta.enableDeletionVectors' = 'true',
        'delta.columnMapping.mode' = 'name',
        'delta.autoOptimize.optimizeWrite' = 'true',
        'delta.autoOptimize.autoCompact' = 'true',
        'delta.tuneFileSizesForRewrites' = 'true'
    )
""")
```

### ✅ DO: Implement a Medallion Architecture

Structure your lake-house in bronze/silver/gold layers for clarity and maintainability:

```python
# Bronze Layer - Raw data ingestion with minimal transformation
def ingest_to_bronze(source_df, table_name, partition_cols=['year', 'month', 'day']):
    """Ingest raw data to bronze layer with deduplication"""
    
    # Add metadata columns
    bronze_df = source_df \
        .withColumn("_ingestion_time", current_timestamp()) \
        .withColumn("_source_file", input_file_name()) \
        .withColumn("_row_hash", sha2(concat_ws("||", *source_df.columns), 256))
    
    # Write with merge for deduplication
    if DeltaTable.isDeltaTable(spark, f"bronze.{table_name}"):
        delta_table = DeltaTable.forName(spark, f"bronze.{table_name}")
        
        delta_table.alias("target").merge(
            bronze_df.alias("source"),
            "target._row_hash = source._row_hash"
        ).whenNotMatchedInsertAll().execute()
    else:
        bronze_df.write \
            .mode("overwrite") \
            .partitionBy(*partition_cols) \
            .saveAsTable(f"bronze.{table_name}")

# Silver Layer - Cleaned and conformed data
def transform_to_silver(bronze_table, silver_table):
    """Clean and conform data for silver layer"""
    
    silver_df = spark.table(f"bronze.{bronze_table}") \
        .filter(col("_ingestion_time") >= last_processed_time) \
        .dropDuplicates(["event_id"]) \
        .withColumn("event_date", to_date("event_time")) \
        .withColumn("properties_parsed", from_json(col("properties"), schema)) \
        .select(
            "event_id",
            "user_id",
            "event_type",
            "properties_parsed.*",
            "event_time",
            "event_date"
        )
    
    # SCD Type 2 implementation for dimension tables
    if "dimension" in silver_table:
        apply_scd_type2(silver_df, silver_table)
    else:
        silver_df.write \
            .mode("overwrite") \
            .partitionBy("event_date") \
            .saveAsTable(f"silver.{silver_table}")

# Gold Layer - Business-level aggregates
@dlt.table(
    name="gold_user_metrics",
    comment="Daily user engagement metrics",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
@dlt.expect_all_or_drop(
    {"valid_user_id": "user_id IS NOT NULL",
     "valid_date": "metric_date >= '2020-01-01'"}
)
def gold_user_metrics():
    return (
        dlt.read("silver_events")
        .groupBy("user_id", "event_date")
        .agg(
            count("*").alias("total_events"),
            countDistinct("session_id").alias("session_count"),
            sum(when(col("event_type") == "purchase", col("revenue")).otherwise(0)).alias("daily_revenue")
        )
        .withColumnRenamed("event_date", "metric_date")
    )
```

### ❌ DON'T: Ignore Table Maintenance

Lake-house tables require active maintenance for optimal performance:

```python
# Bad - Never maintaining tables
# Tables grow unbounded, queries slow down, storage costs explode

# Good - Automated maintenance
from datetime import datetime, timedelta

class LakehouseMaintenanceJob:
    def __init__(self, catalog="iceberg", database="raw"):
        self.catalog = catalog
        self.database = database
        
    def run_daily_maintenance(self):
        """Run daily maintenance tasks"""
        tables = spark.sql(f"SHOW TABLES IN {self.catalog}.{self.database}").collect()
        
        for table in tables:
            table_name = f"{self.catalog}.{self.database}.{table.tableName}"
            
            # 1. Expire old snapshots (Iceberg)
            if self.catalog == "iceberg":
                spark.sql(f"""
                    CALL {self.catalog}.system.expire_snapshots(
                        table => '{table_name}',
                        older_than => TIMESTAMP '{(datetime.now() - timedelta(days=7)).isoformat()}',
                        retain_last => 3
                    )
                """)
                
                # 2. Remove orphan files
                spark.sql(f"""
                    CALL {self.catalog}.system.remove_orphan_files(
                        table => '{table_name}',
                        older_than => TIMESTAMP '{(datetime.now() - timedelta(days=3)).isoformat()}'
                    )
                """)
                
                # 3. Rewrite small files
                spark.sql(f"""
                    CALL {self.catalog}.system.rewrite_data_files(
                        table => '{table_name}',
                        options => map(
                            'target-file-size-bytes', '134217728',
                            'min-file-size-bytes', '67108864'
                        )
                    )
                """)
            
            # For Delta Lake
            elif self.catalog == "delta":
                spark.sql(f"OPTIMIZE {table_name}")
                spark.sql(f"VACUUM {table_name} RETAIN 168 HOURS")
                
                # Z-order by commonly filtered columns
                if "user_id" in [f.name for f in spark.table(table_name).schema.fields]:
                    spark.sql(f"OPTIMIZE {table_name} ZORDER BY (user_id)")
```

---

## 2. Streaming-First Data Ingestion

Modern data platforms must handle real-time data as a first-class concern. Kafka 3.8's improvements in exactly-once semantics and KRaft mode have made it even more reliable.

### ✅ DO: Design for Exactly-Once Processing

```python
# Kafka producer configuration for exactly-once semantics
from confluent_kafka import Producer, SerializingProducer
from confluent_kafka.serialization import StringSerializer, JSONSerializer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
import uuid

class ExactlyOnceProducer:
    def __init__(self, bootstrap_servers, schema_registry_url):
        # Schema Registry configuration
        schema_registry_conf = {'url': schema_registry_url}
        schema_registry_client = SchemaRegistryClient(schema_registry_conf)
        
        # Avro serializer configuration
        avro_serializer = AvroSerializer(
            schema_registry_client,
            self.get_schema(),
            to_dict=lambda obj, ctx: obj.to_dict()
        )
        
        # Producer configuration for exactly-once
        self.producer = SerializingProducer({
            'bootstrap.servers': bootstrap_servers,
            'key.serializer': StringSerializer('utf_8'),
            'value.serializer': avro_serializer,
            
            # Exactly-once semantics configuration
            'enable.idempotence': True,
            'transactional.id': f'producer-{uuid.uuid4()}',
            'max.in.flight.requests.per.connection': 5,
            'acks': 'all',
            'retries': 2147483647,  # Max retries
            
            # Performance optimizations
            'compression.type': 'zstd',
            'linger.ms': 10,
            'batch.size': 32768,
            
            # Observability
            'statistics.interval.ms': 60000,
            'interceptor.classes': 'io.confluent.monitoring.clients.interceptor.MonitoringProducerInterceptor'
        })
        
        # Initialize transactions
        self.producer.init_transactions()
    
    def send_batch(self, messages):
        """Send a batch of messages within a transaction"""
        try:
            self.producer.begin_transaction()
            
            for msg in messages:
                self.producer.produce(
                    topic=msg['topic'],
                    key=msg['key'],
                    value=msg['value'],
                    on_delivery=self.delivery_report,
                    headers={
                        'source': 'data-platform',
                        'version': '1.0',
                        'timestamp': str(datetime.now().timestamp())
                    }
                )
            
            self.producer.commit_transaction()
        except Exception as e:
            self.producer.abort_transaction()
            raise e
    
    @staticmethod
    def get_schema():
        return """
        {
            "type": "record",
            "name": "Event",
            "namespace": "com.company.events",
            "fields": [
                {"name": "event_id", "type": "string"},
                {"name": "user_id", "type": "string"},
                {"name": "event_type", "type": "string"},
                {"name": "properties", "type": {"type": "map", "values": "string"}},
                {"name": "event_time", "type": {"type": "long", "logicalType": "timestamp-millis"}}
            ]
        }
        """
```

### ✅ DO: Implement Robust CDC Pipelines with Debezium

```yaml
# docker-compose.yml for Debezium stack
version: '3.8'
services:
  postgres:
    image: debezium/postgres:16
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: inventory
    command:
      - postgres
      - -c
      - wal_level=logical
      - -c
      - max_wal_senders=10
      - -c
      - max_replication_slots=10
    volumes:
      - postgres_data:/var/lib/postgresql/data

  kafka:
    image: confluentinc/cp-kafka:7.7.0
    depends_on:
      - zookeeper
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_KRAFT_MODE: "true"
      KAFKA_PROCESS_ROLES: controller,broker
      KAFKA_NODE_ID: 1
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka:9093
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_LOG_DIRS: /var/lib/kafka/data
      CLUSTER_ID: MkU3OEVBNTcwNTJENDM2Qk

  connect:
    image: debezium/connect:2.7
    depends_on:
      - kafka
      - postgres
    environment:
      BOOTSTRAP_SERVERS: kafka:9092
      GROUP_ID: 1
      CONFIG_STORAGE_TOPIC: connect_configs
      OFFSET_STORAGE_TOPIC: connect_offsets
      STATUS_STORAGE_TOPIC: connect_statuses
      ENABLE_DEBEZIUM_SCRIPTING: "true"
```

```python
# Debezium connector configuration with advanced features
import requests
import json

class DebeziumConnectorManager:
    def __init__(self, connect_url="http://localhost:8083"):
        self.connect_url = connect_url
        
    def create_postgres_connector(self, name, config_overrides=None):
        """Create a Postgres CDC connector with production-ready configuration"""
        
        config = {
            "name": name,
            "config": {
                "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
                "database.hostname": "postgres",
                "database.port": "5432",
                "database.user": "postgres",
                "database.password": "postgres",
                "database.dbname": "inventory",
                "database.server.name": "dbserver1",
                "topic.prefix": "cdc",
                
                # Capture configuration
                "table.include.list": "public.orders,public.customers,public.products",
                "column.exclude.list": "public.customers.ssn,public.orders.internal_notes",
                
                # Performance and reliability
                "plugin.name": "pgoutput",
                "publication.name": "debezium_publication",
                "publication.autocreate.mode": "filtered",
                "snapshot.mode": "initial",
                "snapshot.isolation.mode": "repeatable_read",
                "heartbeat.interval.ms": "10000",
                "slot.drop.on.stop": "false",
                
                # Schema handling
                "schema.history.internal.kafka.topic": "schema-changes.inventory",
                "schema.history.internal.kafka.bootstrap.servers": "kafka:9092",
                "include.schema.changes": "true",
                
                # Transformations
                "transforms": "unwrap,addMetadata,route",
                "transforms.unwrap.type": "io.debezium.transforms.ExtractNewRecordState",
                "transforms.unwrap.drop.tombstones": "false",
                "transforms.unwrap.delete.handling.mode": "rewrite",
                "transforms.unwrap.add.fields": "op,source.ts_ms",
                
                # Add metadata
                "transforms.addMetadata.type": "org.apache.kafka.connect.transforms.InsertField$Value",
                "transforms.addMetadata.static.field": "_cdc_source",
                "transforms.addMetadata.static.value": "postgres-inventory",
                
                # Topic routing
                "transforms.route.type": "org.apache.kafka.connect.transforms.RegexRouter",
                "transforms.route.regex": "([^.]+)\\.([^.]+)\\.([^.]+)",
                "transforms.route.replacement": "cdc.$2.$3",
                
                # Error handling
                "errors.tolerance": "all",
                "errors.log.enable": "true",
                "errors.log.include.messages": "true",
                "errors.deadletterqueue.topic.name": "cdc-dlq",
                "errors.deadletterqueue.topic.replication.factor": "1",
                
                # Exactly-once semantics
                "exactly.once.support": "required",
                "transaction.boundary": "poll"
            }
        }
        
        # Apply any overrides
        if config_overrides:
            config["config"].update(config_overrides)
        
        response = requests.post(
            f"{self.connect_url}/connectors",
            headers={"Content-Type": "application/json"},
            data=json.dumps(config)
        )
        
        if response.status_code != 201:
            raise Exception(f"Failed to create connector: {response.text}")
        
        return response.json()
    
    def monitor_connector_health(self, connector_name):
        """Monitor connector health and lag"""
        status = requests.get(f"{self.connect_url}/connectors/{connector_name}/status").json()
        
        # Check overall health
        if status["connector"]["state"] != "RUNNING":
            raise Exception(f"Connector not running: {status}")
        
        # Check task health
        for task in status["tasks"]:
            if task["state"] != "RUNNING":
                raise Exception(f"Task {task['id']} not running: {task}")
        
        # Get lag metrics from JMX or Prometheus
        # This would integrate with your monitoring stack
        return status
```

### ✅ DO: Use Flink 2.0 for Complex Stream Processing

Apache Flink 2.0's unified batch/streaming and improved state management make it ideal for complex event processing:

```python
# PyFlink job for real-time feature engineering
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment, EnvironmentSettings
from pyflink.table.udf import udf
from pyflink.table.expressions import col, lit, call
import pandas as pd

def create_feature_engineering_job():
    # Environment setup with performance optimizations
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(4)
    env.enable_checkpointing(60000)  # 1-minute checkpoints
    
    # Configure state backend for large state
    env.set_state_backend(
        "filesystem",
        "s3://your-bucket/flink-checkpoints"
    )
    
    # Table environment with modern configuration
    settings = EnvironmentSettings.new_instance() \
        .in_streaming_mode() \
        .with_configuration({
            "table.exec.mini-batch.enabled": "true",
            "table.exec.mini-batch.allow-latency": "5 s",
            "table.exec.mini-batch.size": "5000",
            "table.exec.state.ttl": "1 d",
            "table.optimizer.agg-phase-strategy": "TWO_PHASE"
        }) \
        .build()
    
    t_env = StreamTableEnvironment.create(env, settings)
    
    # Register Kafka source with latest format
    t_env.execute_sql("""
        CREATE TABLE user_events (
            user_id STRING,
            event_type STRING,
            properties MAP<STRING, STRING>,
            event_time TIMESTAMP(3),
            processing_time AS PROCTIME(),
            WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
        ) WITH (
            'connector' = 'kafka',
            'topic' = 'user-events',
            'properties.bootstrap.servers' = 'kafka:9092',
            'properties.group.id' = 'flink-feature-engineering',
            'scan.startup.mode' = 'latest-offset',
            'format' = 'avro-confluent',
            'avro-confluent.url' = 'http://schema-registry:8081',
            'properties.isolation.level' = 'read_committed'
        )
    """)
    
    # Feature computation with windowing
    features = t_env.sql_query("""
        SELECT 
            user_id,
            window_start,
            window_end,
            COUNT(*) as event_count,
            COUNT(DISTINCT event_type) as unique_event_types,
            
            -- Session analysis
            COUNT(DISTINCT properties['session_id']) as session_count,
            AVG(CAST(properties['duration'] AS DOUBLE)) as avg_duration,
            
            -- Temporal features
            STDDEV_POP(
                UNIX_TIMESTAMP(event_time) - 
                LAG(UNIX_TIMESTAMP(event_time)) OVER (
                    PARTITION BY user_id 
                    ORDER BY event_time
                )
            ) as event_time_stddev,
            
            -- Advanced aggregations
            LISTAGG(DISTINCT event_type, ',') as event_sequence,
            
            -- ML-ready features
            CASE 
                WHEN COUNT(*) > 100 THEN 'high'
                WHEN COUNT(*) > 10 THEN 'medium'
                ELSE 'low'
            END as engagement_level
            
        FROM TABLE(
            TUMBLE(TABLE user_events, DESCRIPTOR(event_time), INTERVAL '1' HOUR)
        )
        GROUP BY user_id, window_start, window_end
    """)
    
    # Write to feature store (Delta Lake)
    t_env.execute_sql("""
        CREATE TABLE feature_store (
            user_id STRING,
            window_start TIMESTAMP(3),
            window_end TIMESTAMP(3),
            event_count BIGINT,
            unique_event_types BIGINT,
            session_count BIGINT,
            avg_duration DOUBLE,
            event_time_stddev DOUBLE,
            event_sequence STRING,
            engagement_level STRING,
            PRIMARY KEY (user_id, window_start) NOT ENFORCED
        ) WITH (
            'connector' = 'delta',
            'table-path' = 's3://your-bucket/features/user_engagement',
            'checkpointing' = 'true',
            'compact.file-size' = '128MB',
            'write.merge.max-concurrent-file-rewrites' = '2'
        )
    """)
    
    # Execute the streaming pipeline
    features.execute_insert("feature_store")
    
# Custom UDF for complex logic
@udf(result_type="ROW<risk_score DOUBLE, risk_factors ARRAY<STRING>>")
def calculate_risk_score(user_history: pd.DataFrame) -> pd.DataFrame:
    """Complex risk scoring logic"""
    # Implementation here
    pass

```

---

## 3. Modern SQL Transformation with dbt

dbt 1.9's native Python support and multi-project deployments have made it the standard for SQL transformations. Here's how to structure a production-grade dbt project.

### ✅ DO: Structure dbt Projects for Scale

```yaml
# dbt_project.yml - Main configuration
name: 'analytics'
version: '1.0.0'
config-version: 2

profile: 'analytics_prod'

model-paths: ["models"]
analysis-paths: ["analyses"]
test-paths: ["tests"]
seed-paths: ["data"]
macro-paths: ["macros"]
snapshot-paths: ["snapshots"]

target-path: "target"
clean-targets:
  - "target"
  - "dbt_packages"

# Model configurations with best practices
models:
  analytics:
    # Staging layer - 1:1 with source systems
    staging:
      +materialized: view
      +schema: staging
      +tags: ['staging', 'daily']
      
    # Intermediate layer - business logic
    intermediate:
      +materialized: ephemeral
      +schema: intermediate
      
    # Marts - final business-facing models
    marts:
      +materialized: table
      +schema: analytics
      +persist_docs:
        relation: true
        columns: true
        
      finance:
        +materialized: incremental
        +unique_key: ['date', 'account_id']
        +on_schema_change: "sync_all_columns"
        +incremental_strategy: merge
        +cluster_by: ['date', 'region']
        
      marketing:
        +materialized: incremental
        +partition_by:
          field: event_date
          data_type: date
          granularity: day
        +cluster_by: ['user_id', 'campaign_id']

# Advanced configurations
vars:
  # Use vars for environment-specific settings
  start_date: '2020-01-01'
  
  # Feature flags for gradual rollouts
  enable_advanced_attribution: false
  
  # Data quality thresholds
  acceptable_null_rate: 0.05
  expected_row_count_change: 0.1

# Semantic layer configuration (dbt 1.6+)
semantic-models:
  - name: financial_metrics
    model: ref('fct_revenue')
    entities:
      - name: date
        type: time
        expr: date
      - name: customer
        type: foreign
        expr: customer_id
    measures:
      - name: revenue
        agg: sum
        expr: revenue_amount
      - name: mrr
        agg: sum
        expr: monthly_recurring_revenue

# Python model configuration (dbt 1.9+)
python-models:
  +py_version: "3.12"
  +memory: "4g"
  +timeout: 300
```

### ✅ DO: Implement Robust Incremental Models

```sql
-- models/marts/finance/fct_revenue.sql
{{
    config(
        materialized='incremental',
        unique_key=['date', 'customer_id', 'product_id'],
        on_schema_change='sync_all_columns',
        partition_by={
            "field": "date",
            "data_type": "date",
            "granularity": "month"
        },
        cluster_by=['region', 'customer_segment'],
        incremental_strategy='merge',
        incremental_predicates=[
            "date >= date_sub(current_date(), 7)"
        ],
        tags=['finance', 'critical'],
        pre_hook=[
            "{{ log_model_start() }}",
            "CREATE TEMP TABLE _revenue_staging AS {{ revenue_staging_query() }}"
        ],
        post_hook=[
            "{{ create_revenue_indexes() }}",
            "{{ analyze_table() }}",
            "{{ log_model_end() }}"
        ]
    )
}}

WITH source_data AS (
    SELECT
        {{ dbt_utils.generate_surrogate_key(['order_id', 'line_item_id']) }} as revenue_id,
        date(order_date) as date,
        customer_id,
        product_id,
        region,
        customer_segment,
        
        -- Revenue calculations with currency conversion
        CASE
            WHEN currency != 'USD' THEN
                revenue_amount * {{ get_exchange_rate('currency', 'order_date') }}
            ELSE revenue_amount
        END as revenue_amount_usd,
        
        -- Advanced metrics
        CASE
            WHEN subscription_id IS NOT NULL THEN revenue_amount
            ELSE 0
        END as recurring_revenue,
        
        -- Audit columns
        current_timestamp() as _dbt_inserted_at,
        '{{ invocation_id }}' as _dbt_invocation_id
        
    FROM {{ ref('stg_orders') }}
    WHERE 1=1
    
    {% if is_incremental() %}
        -- Incremental filter with lookback window
        AND order_date >= (
            SELECT COALESCE(
                date_sub(MAX(date), {{ var('incremental_lookback_days', 3) }}),
                '{{ var("start_date") }}'
            )
            FROM {{ this }}
        )
        
        -- Handle late-arriving data
        AND (
            order_date >= date_sub(current_date(), 7)
            OR updated_at >= (
                SELECT MAX(_dbt_inserted_at) 
                FROM {{ this }}
            )
        )
    {% endif %}
),

-- Data quality checks inline
quality_checked AS (
    SELECT *,
        -- Flag data quality issues
        CASE
            WHEN revenue_amount_usd < 0 THEN 'negative_revenue'
            WHEN revenue_amount_usd > 1000000 THEN 'unusual_high_revenue'
            WHEN customer_id IS NULL THEN 'missing_customer'
            ELSE 'passed'
        END as quality_check
    FROM source_data
)

SELECT * FROM quality_checked
WHERE quality_check = 'passed'  -- Filter out bad data

-- Or keep with flags for monitoring
-- SELECT * FROM quality_checked
```

### ✅ DO: Leverage dbt's Python Models for ML

```python
# models/ml/customer_lifetime_value.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

def model(dbt, session):
    # dbt configuration
    dbt.config(
        materialized="table",
        packages=["pandas", "scikit-learn"],
        schema="ml_models",
        tags=["ml", "python"]
    )
    
    # Load training data
    revenue_df = dbt.ref("fct_revenue").to_pandas()
    customer_df = dbt.ref("dim_customers").to_pandas()
    
    # Feature engineering
    features = revenue_df.groupby('customer_id').agg({
        'revenue_amount_usd': ['sum', 'mean', 'std', 'count'],
        'date': ['min', 'max'],
        'product_id': 'nunique',
        'recurring_revenue': 'sum'
    }).reset_index()
    
    # Flatten column names
    features.columns = ['_'.join(col).strip() for col in features.columns.values]
    features.rename(columns={'customer_id_': 'customer_id'}, inplace=True)
    
    # Calculate derived features
    features['customer_age_days'] = (
        pd.Timestamp.now() - pd.to_datetime(features['date_min'])
    ).dt.days
    
    features['avg_days_between_purchase'] = (
        (pd.to_datetime(features['date_max']) - pd.to_datetime(features['date_min'])).dt.days /
        features['revenue_amount_usd_count'].clip(lower=1)
    )
    
    # Merge with customer attributes
    features = features.merge(
        customer_df[['customer_id', 'segment', 'acquisition_channel', 'region']],
        on='customer_id',
        how='left'
    )
    
    # Prepare for modeling
    categorical_features = ['segment', 'acquisition_channel', 'region']
    features_encoded = pd.get_dummies(features, columns=categorical_features)
    
    # Split features and target
    target = features_encoded['revenue_amount_usd_sum']
    feature_columns = [col for col in features_encoded.columns 
                      if col not in ['customer_id', 'revenue_amount_usd_sum']]
    X = features_encoded[feature_columns]
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, target)
    
    # Generate predictions
    features_encoded['predicted_ltv'] = model.predict(X)
    features_encoded['model_version'] = 'rf_v1.0'
    features_encoded['prediction_date'] = datetime.now().date()
    
    # Feature importance for explainability
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Log feature importance (would go to MLflow in production)
    print(feature_importance.head(10))
    
    return features_encoded[['customer_id', 'predicted_ltv', 'model_version', 'prediction_date']]
```

### ✅ DO: Implement Comprehensive Testing

```sql
-- tests/assert_revenue_completeness.sql
-- Custom test for data completeness
WITH daily_revenue AS (
    SELECT 
        date,
        COUNT(DISTINCT customer_id) as customer_count,
        SUM(revenue_amount_usd) as total_revenue
    FROM {{ ref('fct_revenue') }}
    WHERE date >= date_sub(current_date(), 30)
    GROUP BY date
),

expected_ranges AS (
    SELECT
        date,
        customer_count,
        total_revenue,
        AVG(customer_count) OVER (
            ORDER BY date 
            ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
        ) as avg_customers_last_week,
        AVG(total_revenue) OVER (
            ORDER BY date 
            ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
        ) as avg_revenue_last_week
    FROM daily_revenue
)

SELECT 
    date,
    customer_count,
    total_revenue,
    CASE
        WHEN customer_count < avg_customers_last_week * 0.5 
        THEN 'Customer count dropped by >50%'
        WHEN total_revenue < avg_revenue_last_week * 0.5 
        THEN 'Revenue dropped by >50%'
        WHEN customer_count > avg_customers_last_week * 2 
        THEN 'Customer count increased by >100%'
        WHEN total_revenue > avg_revenue_last_week * 2 
        THEN 'Revenue increased by >100%'
    END as anomaly_reason
FROM expected_ranges
WHERE 
    customer_count < avg_customers_last_week * {{ var('min_acceptable_ratio', 0.5) }}
    OR customer_count > avg_customers_last_week * {{ var('max_acceptable_ratio', 2.0) }}
    OR total_revenue < avg_revenue_last_week * {{ var('min_acceptable_ratio', 0.5) }}
    OR total_revenue > avg_revenue_last_week * {{ var('max_acceptable_ratio', 2.0) }}
```

```yaml
# models/schema.yml - Model documentation and tests
version: 2

models:
  - name: fct_revenue
    description: |
      Daily revenue fact table containing all completed transactions.
      This is the source of truth for financial reporting.
    
    meta:
      owner: finance-team
      tier: 1  # Critical
      sla: "08:00 UTC"
      
    # Model-level tests
    tests:
      - dbt_expectations.expect_table_row_count_to_be_between:
          min_value: 10000
          max_value: 1000000
          config:
            severity: error
            
      - dbt_utils.recency:
          datepart: hour
          field: cast(_dbt_inserted_at as timestamp)
          interval: 6
          config:
            severity: warn
            
    columns:
      - name: revenue_id
        description: Surrogate key for the revenue record
        tests:
          - unique
          - not_null
          
      - name: date
        description: Date of the transaction
        tests:
          - not_null
          - dbt_expectations.expect_column_values_to_be_between:
              min_value: '2020-01-01'
              max_value: 'current_date()'
              
      - name: customer_id
        description: Foreign key to dim_customers
        tests:
          - not_null
          - relationships:
              to: ref('dim_customers')
              field: customer_id
              config:
                where: "date >= date_sub(current_date(), 7)"
                
      - name: revenue_amount_usd
        description: Revenue amount in USD after currency conversion
        tests:
          - not_null
          - dbt_expectations.expect_column_values_to_be_between:
              min_value: 0
              max_value: 1000000
              config:
                severity: warn
                
      - name: quality_check
        description: Data quality flag
        tests:
          - accepted_values:
              values: ['passed']
              quote: true

# Sources with freshness checks
sources:
  - name: raw_kafka
    database: "{{ env_var('KAFKA_CONNECT_DB') }}"
    schema: raw
    
    tables:
      - name: orders
        description: Raw order events from Kafka
        
        # Freshness SLA
        freshness:
          warn_after: {count: 2, period: hour}
          error_after: {count: 6, period: hour}
          
        loaded_at_field: _kafka_timestamp
        
        columns:
          - name: order_id
            tests:
              - not_null
              - unique
          
          - name: _kafka_timestamp
            description: Timestamp when message was produced to Kafka
            tests:
              - not_null
              - dbt_expectations.expect_column_max_to_be_between:
                  min_value: "dateadd(hour, -2, current_timestamp())"
                  max_value: "current_timestamp()"
```

---

## 4. Orchestration at Scale

While Apache Airflow 3.0 remains the most mature option, Prefect 3.0 and Dagster 2.0 offer compelling alternatives with better developer experience.

### ✅ DO: Design Idempotent and Retryable Pipelines

```python
# Airflow 3.0 with TaskFlow API and dynamic task generation
from airflow.decorators import dag, task, task_group
from airflow.models import Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.snowflake.operators.snowflake import SnowflakeOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List

# Default arguments with production-ready settings
default_args = {
    'owner': 'data-platform-team',
    'depends_on_past': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(hours=1),
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['data-alerts@company.com'],
    'sla': timedelta(hours=4),
}

@dag(
    'advanced_data_pipeline',
    default_args=default_args,
    description='Production data pipeline with dynamic tasks',
    schedule='0 2 * * *',  # 2 AM daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['production', 'critical'],
    doc_md=__doc__,
    # Airflow 3.0 features
    render_template_as_native_obj=True,
    params={
        "source_tables": ["orders", "customers", "products"],
        "processing_date": "{{ ds }}",
        "full_refresh": False
    }
)
def advanced_data_pipeline():
    """
    # Advanced Data Pipeline
    
    This DAG orchestrates the daily data processing pipeline with:
    - Dynamic task generation based on configuration
    - Parallel processing with task groups
    - Data quality validation
    - Automatic retry and recovery
    """
    
    @task(
        pool='data_extraction_pool',
        pool_slots=2,
        trigger_rule='all_success'
    )
    def get_source_tables(**context) -> List[Dict[str, str]]:
        """Dynamically determine which tables to process"""
        processing_date = context['params']['processing_date']
        
        # In production, this would query a metadata table
        tables = []
        for table_name in context['params']['source_tables']:
            tables.append({
                'name': table_name,
                'source': f's3://raw-data/{table_name}/date={processing_date}/',
                'destination': f'raw.{table_name}',
                'partition_column': 'processing_date',
                'merge_keys': ['id']
            })
        
        return tables
    
    @task_group(group_id='data_extraction')
    def extract_data_group(table_configs: List[Dict[str, str]]):
        """Task group for parallel data extraction"""
        
        @task.short_circuit  # Skip if no data
        def check_source_data(table_config: Dict[str, str]) -> bool:
            """Check if source data exists"""
            s3 = S3Hook(aws_conn_id='aws_default')
            
            exists = s3.check_for_key(
                key=table_config['source'],
                bucket_name='raw-data'
            )
            
            if not exists:
                print(f"No data found for {table_config['name']}")
                
            return exists
        
        @task(
            retries=5,
            execution_timeout=timedelta(hours=2)
        )
        def extract_to_staging(table_config: Dict[str, str]) -> Dict[str, any]:
            """Extract data from S3 to staging"""
            import awswrangler as wr
            
            # Read data with schema inference
            df = wr.s3.read_parquet(
                path=table_config['source'],
                dataset=True,
                use_threads=True,
                ray_args={
                    "num_cpus": 4,
                }
            )
            
            # Add metadata
            df['_extracted_at'] = datetime.utcnow()
            df['_source_file'] = table_config['source']
            
            # Write to staging with partitioning
            staging_path = f"s3://staging/{table_config['name']}"
            wr.s3.to_parquet(
                df=df,
                path=staging_path,
                dataset=True,
                partition_cols=[table_config['partition_column']],
                mode='overwrite_partitions',
                compression='snappy'
            )
            
            return {
                'table_name': table_config['name'],
                'row_count': len(df),
                'staging_path': staging_path
            }
        
        # Dynamic task generation
        extraction_results = []
        for table_config in table_configs:
            if check_source_data(table_config):
                result = extract_to_staging(table_config)
                extraction_results.append(result)
        
        return extraction_results
    
    @task_group(group_id='data_quality')
    def data_quality_checks(extraction_results: List[Dict[str, any]]):
        """Run data quality checks on extracted data"""
        
        @task
        def validate_schema(result: Dict[str, any]) -> bool:
            """Validate schema matches expectations"""
            import awswrangler as wr
            from great_expectations import DataContext
            
            # Load data sample
            df = wr.s3.read_parquet(
                path=result['staging_path'],
                dataset=True,
                max_rows=10000
            )
            
            # Run Great Expectations validation
            context = DataContext()
            batch = context.sources.pandas_default.read_dataframe(
                dataframe=df,
                asset_name=result['table_name']
            )
            
            checkpoint_result = context.run_checkpoint(
                checkpoint_name=f"{result['table_name']}_staging_checkpoint",
                batch_request=batch
            )
            
            return checkpoint_result.success
        
        @task
        def check_data_freshness(result: Dict[str, any]) -> bool:
            """Ensure data is recent"""
            import awswrangler as wr
            
            df = wr.s3.read_parquet(
                path=result['staging_path'],
                columns=['_extracted_at'],
                dataset=True,
                max_rows=1
            )
            
            if df.empty:
                return False
                
            max_extracted = pd.to_datetime(df['_extracted_at'].max())
            age_hours = (datetime.utcnow() - max_extracted).total_seconds() / 3600
            
            return age_hours < 24
        
        # Run quality checks in parallel
        quality_results = []
        for result in extraction_results:
            schema_valid = validate_schema(result)
            data_fresh = check_data_freshness(result)
            quality_results.append(schema_valid & data_fresh)
        
        return quality_results
    
    @task
    def load_to_warehouse(
        extraction_results: List[Dict[str, any]], 
        quality_results: List[bool]
    ):
        """Load validated data to warehouse"""
        
        for result, quality_passed in zip(extraction_results, quality_results):
            if not quality_passed:
                raise ValueError(f"Quality checks failed for {result['table_name']}")
        
        # Snowflake loading with COPY INTO
        for result in extraction_results:
            copy_sql = f"""
            COPY INTO {result['table_name']}
            FROM '{result['staging_path']}'
            FILE_FORMAT = (TYPE = PARQUET)
            MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE
            ON_ERROR = 'SKIP_FILE'
            """
            
            SnowflakeOperator(
                task_id=f"load_{result['table_name']}",
                sql=copy_sql,
                snowflake_conn_id='snowflake_default'
            ).execute(context={})
    
    # DAG structure
    source_tables = get_source_tables()
    extraction_results = extract_data_group(source_tables)
    quality_results = data_quality_checks(extraction_results)
    load_to_warehouse(extraction_results, quality_results)

# Instantiate the DAG
dag = advanced_data_pipeline()
```

### ✅ DO: Use Modern Orchestrators for Better Developer Experience

```python
# Prefect 3.0 - Cloud-native with better testing
from prefect import flow, task, get_run_logger
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from prefect.tasks import task_input_hash
from datetime import timedelta
import pandas as pd

@task(
    name="Extract Data",
    retries=3,
    retry_delay_seconds=60,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
    log_prints=True,
    tags=["extraction", "io-heavy"]
)
def extract_data(table_name: str, date: str) -> pd.DataFrame:
    """Extract data with caching and retries"""
    logger = get_run_logger()
    logger.info(f"Extracting {table_name} for {date}")
    
    # Extraction logic here
    df = pd.read_parquet(f"s3://raw/{table_name}/date={date}")
    
    return df

@task(
    name="Transform Data",
    retries=2,
    log_prints=True,
    tags=["transformation", "cpu-heavy"]
)
def transform_data(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Apply transformation rules"""
    logger = get_run_logger()
    
    # Apply transformations
    for column, rule in rules.items():
        if rule['type'] == 'mapping':
            df[column] = df[column].map(rule['mapping'])
        elif rule['type'] == 'calculation':
            df[column] = eval(rule['expression'])
    
    logger.info(f"Transformed {len(df)} records")
    return df

@flow(
    name="ETL Pipeline",
    description="Production ETL pipeline with Prefect",
    log_prints=True,
    retries=1,
    retry_delay_seconds=300,
    persist_result=True,
    result_storage_key="etl-pipeline-{flow_run.id}"
)
def etl_pipeline(
    tables: list[str] = ["orders", "customers"],
    date: str = None,
    parallel: bool = True
):
    """Main ETL flow with parallel processing"""
    logger = get_run_logger()
    
    if date is None:
        date = pd.Timestamp.now().strftime("%Y-%m-%d")
    
    logger.info(f"Starting ETL for {date}")
    
    # Define transformation rules
    transform_rules = {
        "orders": {
            "status": {"type": "mapping", "mapping": {"1": "pending", "2": "completed"}},
            "total": {"type": "calculation", "expression": "df['quantity'] * df['price']"}
        },
        "customers": {
            "segment": {"type": "mapping", "mapping": {"A": "premium", "B": "standard"}}
        }
    }
    
    # Process tables
    if parallel:
        # Parallel processing with Dask
        from prefect_dask import DaskTaskRunner
        
        with DaskTaskRunner(
            cluster_kwargs={"n_workers": 4, "threads_per_worker": 2}
        ):
            futures = []
            for table in tables:
                future = extract_data.submit(table, date)
                transformed = transform_data.submit(
                    future, 
                    transform_rules.get(table, {})
                )
                futures.append(transformed)
            
            results = [f.result() for f in futures]
    else:
        # Sequential processing
        results = []
        for table in tables:
            df = extract_data(table, date)
            transformed = transform_data(df, transform_rules.get(table, {}))
            results.append(transformed)
    
    logger.info(f"Processed {len(results)} tables")
    return results

# Deployment configuration
deployment = Deployment.build_from_flow(
    flow=etl_pipeline,
    name="production-etl",
    schedule=CronSchedule(cron="0 2 * * *", timezone="UTC"),
    work_queue_name="data-pipelines",
    infrastructure="kubernetes-job",
    infra_overrides={
        "namespace": "prefect",
        "image": "company/etl-pipeline:latest",
        "resources": {
            "requests": {"memory": "2Gi", "cpu": "1000m"},
            "limits": {"memory": "4Gi", "cpu": "2000m"}
        }
    },
    parameters={
        "tables": ["orders", "customers", "products"],
        "parallel": True
    },
    tags=["production", "etl"],
    description="Main ETL pipeline for production data"
)

if __name__ == "__main__":
    deployment.apply()
```

```python
# Dagster 2.0 - Software-defined assets approach
from dagster import (
    asset, 
    multi_asset,
    AssetExecutionContext,
    AssetMaterialization,
    AssetKey,
    DailyPartitionsDefinition,
    MetadataValue,
    Output,
    DynamicPartitionsDefinition,
    define_asset_job,
    ScheduleDefinition,
    sensor,
    RunRequest,
    SkipReason,
    Definitions,
    ConfigurableResource,
    EnvVar
)
from dagster_aws.s3 import S3Resource
from dagster_dbt import DbtCliResource, dbt_assets
from dagster_spark import spark_resource
import pandas as pd
from typing import Iterator

# Partitions for time-based processing
daily_partitions = DailyPartitionsDefinition(start_date="2024-01-01")

# Resources with configuration
class WarehouseResource(ConfigurableResource):
    """Configurable warehouse connection"""
    host: str
    database: str
    user: str
    password: str = EnvVar("WAREHOUSE_PASSWORD")
    
    def get_connection(self):
        # Connection logic here
        pass

# Define software-defined assets
@asset(
    partitions_def=daily_partitions,
    group_name="bronze",
    description="Raw order data from source systems",
    metadata={
        "source": "kafka",
        "topic": "orders",
        "sla_hours": 2
    },
    retry_policy=RetryPolicy(max_retries=3, delay=60),
    code_version="1.2.0"
)
def bronze_orders(
    context: AssetExecutionContext,
    kafka: KafkaResource
) -> pd.DataFrame:
    """Ingest raw orders from Kafka"""
    
    partition_date = context.partition_key
    
    # Read from Kafka with exactly-once semantics
    orders = kafka.consume_topic(
        topic="orders",
        start_date=partition_date,
        end_date=partition_date
    )
    
    # Add metadata
    orders['_ingestion_time'] = pd.Timestamp.now()
    orders['_partition_date'] = partition_date
    
    # Log metrics
    context.add_output_metadata({
        "row_count": len(orders),
        "null_count": orders.isnull().sum().sum(),
        "preview": MetadataValue.md(orders.head().to_markdown())
    })
    
    return orders

@multi_asset(
    outs={
        "silver_orders": AssetOut(
            key=AssetKey(["silver", "orders"]),
            description="Cleaned and validated orders"
        ),
        "data_quality_metrics": AssetOut(
            key=AssetKey(["metrics", "orders_quality"]),
            description="Quality metrics for orders"
        )
    },
    partitions_def=daily_partitions,
    group_name="silver",
    retry_policy=RetryPolicy(max_retries=2)
)
def process_orders(
    context: AssetExecutionContext,
    bronze_orders: pd.DataFrame
) -> Iterator[Output]:
    """Process orders with quality checks"""
    
    # Data cleaning
    cleaned_orders = bronze_orders.copy()
    
    # Remove duplicates
    cleaned_orders = cleaned_orders.drop_duplicates(subset=['order_id'])
    
    # Validate required fields
    required_fields = ['order_id', 'customer_id', 'amount']
    cleaned_orders = cleaned_orders.dropna(subset=required_fields)
    
    # Type conversions
    cleaned_orders['amount'] = pd.to_numeric(cleaned_orders['amount'], errors='coerce')
    cleaned_orders['order_date'] = pd.to_datetime(cleaned_orders['order_date'])
    
    # Calculate quality metrics
    quality_metrics = {
        'total_records': len(bronze_orders),
        'cleaned_records': len(cleaned_orders),
        'duplicate_rate': 1 - (len(cleaned_orders) / len(bronze_orders)),
        'null_rate': bronze_orders.isnull().sum().sum() / bronze_orders.size,
        'validation_timestamp': pd.Timestamp.now()
    }
    
    # Output both assets
    yield Output(
        cleaned_orders,
        output_name="silver_orders",
        metadata={
            "row_count": len(cleaned_orders),
            "columns": list(cleaned_orders.columns)
        }
    )
    
    yield Output(
        pd.DataFrame([quality_metrics]),
        output_name="data_quality_metrics"
    )

# dbt integration
dbt_project_dir = "/path/to/dbt/project"

@dbt_assets(
    manifest=dbt_project_dir + "/target/manifest.json",
    project=dbt_project_dir,
    partitions_def=daily_partitions,
    exclude="tag:deprecated"
)
def analytics_dbt_assets(context: AssetExecutionContext, dbt: DbtCliResource):
    """Run dbt models as Dagster assets"""
    yield from dbt.cli(
        ["run", "--vars", f"{{run_date: {context.partition_key}}}"],
        context=context
    ).stream()

# Define a job that materializes assets
daily_pipeline = define_asset_job(
    name="daily_data_pipeline",
    selection=[
        "bronze_orders",
        "silver_orders",
        "analytics_dbt_assets"
    ],
    partitions_def=daily_partitions,
    tags={"team": "data-platform", "priority": "high"}
)

# Schedule the job
daily_schedule = ScheduleDefinition(
    job=daily_pipeline,
    cron_schedule="0 2 * * *",
    execution_timezone="UTC",
    default_status=DefaultScheduleStatus.RUNNING
)

# Sensor for event-driven processing
@sensor(
    job=daily_pipeline,
    minimum_interval_seconds=300  # Check every 5 minutes
)
def orders_sensor(context):
    """Trigger pipeline when new orders arrive"""
    
    # Check for new files in S3
    s3 = context.resources.s3
    new_files = s3.list_new_files("orders/incoming/")
    
    if not new_files:
        return SkipReason("No new order files found")
    
    # Create run requests for each date
    run_requests = []
    for file in new_files:
        date = extract_date_from_filename(file)
        run_requests.append(
            RunRequest(
                partition_key=date,
                tags={"source_file": file}
            )
        )
    
    return run_requests

# Combine everything into Definitions
defs = Definitions(
    assets=[bronze_orders, process_orders, analytics_dbt_assets],
    jobs=[daily_pipeline],
    schedules=[daily_schedule],
    sensors=[orders_sensor],
    resources={
        "warehouse": WarehouseResource(
            host="warehouse.company.com",
            database="analytics",
            user="dagster"
        ),
        "s3": S3Resource(
            region_name="us-east-1",
            bucket="data-lake"
        ),
        "dbt": DbtCliResource(
            project_dir=dbt_project_dir,
            profiles_dir="/path/to/profiles"
        ),
        "spark": spark_resource.configured({
            "master": "spark://spark-master:7077",
            "app_name": "dagster-pipeline"
        })
    }
)
```

---

## 5. Data Quality & Observability

Data quality is not an afterthought—it must be built into every layer of your pipeline.

### ✅ DO: Implement Comprehensive Data Quality Checks

```python
# Great Expectations 0.19 with modern patterns
from great_expectations.data_context import DataContext
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
from great_expectations.checkpoint import Checkpoint
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.core.yaml_handler import YAMLHandler
import pandas as pd

class DataQualityFramework:
    def __init__(self, ge_context_root: str):
        self.context = DataContext(context_root_dir=ge_context_root)
        
    def create_expectation_suite_for_table(
        self, 
        table_name: str,
        schema: dict,
        business_rules: dict
    ) -> ExpectationSuite:
        """Create comprehensive expectation suite"""
        
        suite = self.context.add_or_update_expectation_suite(
            expectation_suite_name=f"{table_name}_suite"
        )
        
        # Schema validation expectations
        for column, dtype in schema.items():
            # Column existence
            suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": column}
                )
            )
            
            # Data type validation
            if dtype == "integer":
                suite.add_expectation(
                    ExpectationConfiguration(
                        expectation_type="expect_column_values_to_be_of_type",
                        kwargs={"column": column, "type_": "int64"}
                    )
                )
            elif dtype == "string":
                suite.add_expectation(
                    ExpectationConfiguration(
                        expectation_type="expect_column_values_to_be_of_type",
                        kwargs={"column": column, "type_": "object"}
                    )
                )
            elif dtype == "timestamp":
                suite.add_expectation(
                    ExpectationConfiguration(
                        expectation_type="expect_column_values_to_be_dateutil_parseable",
                        kwargs={"column": column}
                    )
                )
        
        # Business rule validations
        for rule_name, rule_config in business_rules.items():
            if rule_config['type'] == 'unique':
                suite.add_expectation(
                    ExpectationConfiguration(
                        expectation_type="expect_column_values_to_be_unique",
                        kwargs={"column": rule_config['column']}
                    )
                )
            
            elif rule_config['type'] == 'not_null':
                suite.add_expectation(
                    ExpectationConfiguration(
                        expectation_type="expect_column_values_to_not_be_null",
                        kwargs={
                            "column": rule_config['column'],
                            "mostly": rule_config.get('mostly', 0.95)
                        }
                    )
                )
            
            elif rule_config['type'] == 'regex':
                suite.add_expectation(
                    ExpectationConfiguration(
                        expectation_type="expect_column_values_to_match_regex",
                        kwargs={
                            "column": rule_config['column'],
                            "regex": rule_config['pattern']
                        }
                    )
                )
            
            elif rule_config['type'] == 'value_set':
                suite.add_expectation(
                    ExpectationConfiguration(
                        expectation_type="expect_column_values_to_be_in_set",
                        kwargs={
                            "column": rule_config['column'],
                            "value_set": rule_config['values']
                        }
                    )
                )
            
            elif rule_config['type'] == 'range':
                suite.add_expectation(
                    ExpectationConfiguration(
                        expectation_type="expect_column_values_to_be_between",
                        kwargs={
                            "column": rule_config['column'],
                            "min_value": rule_config['min'],
                            "max_value": rule_config['max']
                        }
                    )
                )
            
            elif rule_config['type'] == 'relationship':
                suite.add_expectation(
                    ExpectationConfiguration(
                        expectation_type="expect_column_pair_values_to_be_equal",
                        kwargs={
                            "column_A": rule_config['column_a'],
                            "column_B": rule_config['column_b'],
                            "mostly": rule_config.get('mostly', 1.0)
                        }
                    )
                )
            
            elif rule_config['type'] == 'custom_sql':
                suite.add_expectation(
                    ExpectationConfiguration(
                        expectation_type="expect_query_to_return_no_rows",
                        kwargs={
                            "query": rule_config['query'],
                            "description": rule_config.get('description', '')
                        }
                    )
                )
        
        # Statistical expectations for anomaly detection
        suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={
                    "min_value": business_rules.get('min_rows', 100),
                    "max_value": business_rules.get('max_rows', 1000000)
                }
            )
        )
        
        self.context.save_expectation_suite(suite)
        return suite
    
    def create_checkpoint(
        self,
        checkpoint_name: str,
        suite_name: str,
        datasource_name: str,
        data_asset_name: str,
        action_list: list = None
    ) -> Checkpoint:
        """Create checkpoint with actions"""
        
        if action_list is None:
            action_list = [
                {
                    "name": "store_validation_result",
                    "action": {
                        "class_name": "StoreValidationResultAction"
                    }
                },
                {
                    "name": "store_evaluation_params",
                    "action": {
                        "class_name": "StoreEvaluationParametersAction"
                    }
                },
                {
                    "name": "update_data_docs",
                    "action": {
                        "class_name": "UpdateDataDocsAction"
                    }
                },
                {
                    "name": "send_slack_notification",
                    "action": {
                        "class_name": "SlackNotificationAction",
                        "webhook": "${SLACK_WEBHOOK_URL}",
                        "notify_on": "failure",
                        "notification_type": "detailed"
                    }
                }
            ]
        
        checkpoint_config = {
            "name": checkpoint_name,
            "config_version": 1.0,
            "class_name": "Checkpoint",
            "run_name_template": "%Y%m%d-%H%M%S",
            "expectation_suite_name": suite_name,
            "action_list": action_list,
            "validations": [
                {
                    "batch_request": {
                        "datasource_name": datasource_name,
                        "data_asset_name": data_asset_name,
                        "options": {
                            "year": "${year}",
                            "month": "${month}",
                            "day": "${day}"
                        }
                    }
                }
            ]
        }
        
        self.context.add_or_update_checkpoint(**checkpoint_config)
        return self.context.get_checkpoint(checkpoint_name)
    
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        suite_name: str,
        run_name: str = None
    ) -> dict:
        """Validate a pandas DataFrame"""
        
        # Create batch request
        batch_request = RuntimeBatchRequest(
            datasource_name="pandas_datasource",
            data_connector_name="runtime_data_connector",
            data_asset_name="runtime_df",
            runtime_parameters={"batch_data": df},
            batch_identifiers={"run_id": run_name or "manual_run"}
        )
        
        # Run validation
        validator = self.context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=suite_name
        )
        
        validation_result = validator.validate()
        
        # Extract summary
        summary = {
            "success": validation_result.success,
            "statistics": validation_result.statistics,
            "failed_expectations": [
                exp for exp in validation_result.results 
                if not exp.success
            ]
        }
        
        return summary

# Example usage
dq_framework = DataQualityFramework("/path/to/ge/context")

# Define schema and rules for orders table
orders_schema = {
    "order_id": "string",
    "customer_id": "string",
    "order_date": "timestamp",
    "amount": "float",
    "status": "string",
    "region": "string"
}

orders_business_rules = {
    "order_id_unique": {
        "type": "unique",
        "column": "order_id"
    },
    "customer_id_not_null": {
        "type": "not_null",
        "column": "customer_id",
        "mostly": 0.99
    },
    "amount_positive": {
        "type": "range",
        "column": "amount",
        "min": 0,
        "max": 1000000
    },
    "status_values": {
        "type": "value_set",
        "column": "status",
        "values": ["pending", "processing", "completed", "cancelled"]
    },
    "no_future_orders": {
        "type": "custom_sql",
        "query": "SELECT * FROM orders WHERE order_date > CURRENT_DATE",
        "description": "Orders should not have future dates"
    }
}

# Create suite
orders_suite = dq_framework.create_expectation_suite_for_table(
    "orders",
    orders_schema,
    orders_business_rules
)
```

### ✅ DO: Implement Data Lineage and Observability

```python
# OpenLineage integration for data lineage
from openlineage.client import OpenLineageClient
from openlineage.client.event import (
    RunEvent, 
    RunState, 
    Job, 
    Run, 
    Dataset,
    DatasetFacets,
    SchemaField,
    SchemaDatasetFacet,
    DataQualityMetricsInputDatasetFacet,
    ColumnLineageDatasetFacet
)
from datetime import datetime
import uuid

class DataLineageTracker:
    def __init__(self, openlineage_url: str):
        self.client = OpenLineageClient(url=openlineage_url)
        self.namespace = "data-platform"
        
    def track_job_run(
        self,
        job_name: str,
        input_datasets: list,
        output_datasets: list,
        transformation_logic: dict = None
    ):
        """Track a complete job run with lineage"""
        
        run_id = str(uuid.uuid4())
        job = Job(namespace=self.namespace, name=job_name)
        run = Run(runId=run_id)
        
        # Start event
        start_event = RunEvent(
            eventTime=datetime.now().isoformat(),
            producer="data-engineering-platform",
            job=job,
            run=run,
            eventType=RunState.START,
            inputs=self._create_input_datasets(input_datasets),
            outputs=self._create_output_datasets(output_datasets)
        )
        
        self.client.emit(start_event)
        
        # Add column-level lineage if available
        if transformation_logic:
            lineage_event = self._create_column_lineage_event(
                job, run, transformation_logic
            )
            self.client.emit(lineage_event)
        
        return run_id
    
    def _create_input_datasets(self, datasets: list) -> list:
        """Create input dataset objects with quality metrics"""
        
        input_datasets = []
        for ds in datasets:
            # Schema facet
            schema_facet = SchemaDatasetFacet(
                fields=[
                    SchemaField(name=col['name'], type=col['type'])
                    for col in ds.get('schema', [])
                ]
            )
            
            # Quality metrics facet
            quality_facet = DataQualityMetricsInputDatasetFacet(
                rowCount=ds.get('row_count'),
                bytes=ds.get('size_bytes'),
                columnMetrics={
                    col_name: {
                        "nullCount": metrics.get('null_count'),
                        "distinctCount": metrics.get('distinct_count'),
                        "min": metrics.get('min'),
                        "max": metrics.get('max')
                    }
                    for col_name, metrics in ds.get('column_metrics', {}).items()
                }
            )
            
            dataset = Dataset(
                namespace=self.namespace,
                name=ds['name'],
                facets={
                    "schema": schema_facet,
                    "dataQualityMetrics": quality_facet
                }
            )
            
            input_datasets.append(dataset)
        
        return input_datasets
    
    def _create_column_lineage_event(
        self, 
        job: Job, 
        run: Run, 
        transformation_logic: dict
    ) -> RunEvent:
        """Create column-level lineage event"""
        
        column_lineage = ColumnLineageDatasetFacet(
            fields={
                output_col: {
                    "inputFields": [
                        {
                            "namespace": self.namespace,
                            "name": input_dataset,
                            "field": input_col
                        }
                        for input_dataset, input_col in mappings
                    ],
                    "transformationDescription": desc,
                    "transformationType": transform_type
                }
                for output_col, mappings, desc, transform_type 
                in transformation_logic['column_mappings']
            }
        )
        
        return RunEvent(
            eventTime=datetime.now().isoformat(),
            producer="data-engineering-platform",
            job=job,
            run=run,
            eventType=RunState.RUNNING,
            outputs=[
                Dataset(
                    namespace=self.namespace,
                    name=transformation_logic['output_dataset'],
                    facets={"columnLineage": column_lineage}
                )
            ]
        )
    
    def complete_job_run(
        self, 
        run_id: str, 
        job_name: str,
        success: bool,
        error_message: str = None
    ):
        """Mark job run as complete"""
        
        job = Job(namespace=self.namespace, name=job_name)
        run = Run(runId=run_id)
        
        complete_event = RunEvent(
            eventTime=datetime.now().isoformat(),
            producer="data-engineering-platform",
            job=job,
            run=run,
            eventType=RunState.COMPLETE if success else RunState.FAIL,
            message=error_message
        )
        
        self.client.emit(complete_event)

# Monte Carlo integration for automated monitoring
from pymontecarloapi import MonteCarloApi

class DataObservabilityPlatform:
    def __init__(self, monte_carlo_api_key: str):
        self.mc_client = MonteCarloApi(api_key=monte_carlo_api_key)
        
    def setup_table_monitoring(
        self,
        table_name: str,
        warehouse: str,
        monitors: dict
    ):
        """Configure automated monitoring for a table"""
        
        # Freshness monitor
        if 'freshness' in monitors:
            self.mc_client.create_freshness_monitor(
                table_name=table_name,
                warehouse=warehouse,
                timestamp_field=monitors['freshness']['timestamp_field'],
                expected_freshness_hours=monitors['freshness']['sla_hours'],
                severity='SEV-2'
            )
        
        # Volume monitor
        if 'volume' in monitors:
            self.mc_client.create_volume_monitor(
                table_name=table_name,
                warehouse=warehouse,
                min_expected_rows=monitors['volume']['min_rows'],
                max_expected_rows=monitors['volume']['max_rows'],
                lookback_days=monitors['volume'].get('lookback_days', 7)
            )
        
        # Schema change detection
        if 'schema' in monitors:
            self.mc_client.create_schema_monitor(
                table_name=table_name,
                warehouse=warehouse,
                alert_on_schema_change=True,
                allowed_schema_changes=monitors['schema'].get('allowed_changes', [])
            )
        
        # Custom SQL monitors
        for custom_monitor in monitors.get('custom', []):
            self.mc_client.create_custom_sql_monitor(
                name=custom_monitor['name'],
                query=custom_monitor['query'],
                warehouse=warehouse,
                expected_result=custom_monitor['expected'],
                comparison_type=custom_monitor['comparison']
            )
    
    def create_circuit_breaker(
        self,
        pipeline_name: str,
        conditions: list
    ):
        """Create circuit breaker to stop bad data propagation"""
        
        circuit_breaker_config = {
            "name": f"{pipeline_name}_circuit_breaker",
            "conditions": conditions,
            "actions": [
                {
                    "type": "pause_pipeline",
                    "target": pipeline_name
                },
                {
                    "type": "send_alert",
                    "channels": ["slack", "pagerduty"],
                    "severity": "critical"
                }
            ]
        }
        
        self.mc_client.create_circuit_breaker(circuit_breaker_config)

# Example monitoring configuration
monitoring_config = {
    "orders": {
        "freshness": {
            "timestamp_field": "created_at",
            "sla_hours": 2
        },
        "volume": {
            "min_rows": 1000,
            "max_rows": 100000,
            "lookback_days": 7
        },
        "schema": {
            "allowed_changes": ["ADD_COLUMN"]
        },
        "custom": [
            {
                "name": "duplicate_check",
                "query": """
                    SELECT COUNT(*) as duplicates
                    FROM (
                        SELECT order_id, COUNT(*) as cnt
                        FROM orders
                        WHERE created_at >= CURRENT_DATE - 1
                        GROUP BY order_id
                        HAVING COUNT(*) > 1
                    )
                """,
                "expected": 0,
                "comparison": "equals"
            }
        ]
    }
}
```

---

## 6. Real-time Analytics with DuckDB

DuckDB has emerged as the SQLite of analytics, perfect for local development and embedded analytics.

### ✅ DO: Use DuckDB for Local Analytics and Development

```python
import duckdb
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq

class DuckDBAnalytics:
    def __init__(self, db_path: str = ":memory:"):
        self.conn = duckdb.connect(db_path)
        self._setup_extensions()
        
    def _setup_extensions(self):
        """Install and load useful extensions"""
        extensions = ['httpfs', 'parquet', 'json', 'fts', 'sqlite']
        
        for ext in extensions:
            self.conn.execute(f"INSTALL {ext}")
            self.conn.execute(f"LOAD {ext}")
        
        # Configure S3 access
        self.conn.execute("""
            SET s3_region='us-east-1';
            SET s3_access_key_id='${AWS_ACCESS_KEY_ID}';
            SET s3_secret_access_key='${AWS_SECRET_ACCESS_KEY}';
        """)
    
    def create_lakehouse_views(self, catalog_path: str):
        """Create views over Iceberg/Delta tables"""
        
        # Iceberg tables
        self.conn.execute(f"""
            CREATE OR REPLACE VIEW iceberg_orders AS
            SELECT * FROM iceberg_scan('{catalog_path}/orders')
        """)
        
        # Delta tables
        self.conn.execute(f"""
            CREATE OR REPLACE VIEW delta_customers AS
            SELECT * FROM delta_scan('s3://datalake/delta/customers')
        """)
        
        # Parquet files
        self.conn.execute("""
            CREATE OR REPLACE VIEW parquet_products AS
            SELECT * FROM read_parquet('s3://datalake/products/*.parquet')
        """)
    
    def run_complex_analytics(self) -> pd.DataFrame:
        """Run complex analytical queries efficiently"""
        
        # Advanced window functions and CTEs
        result = self.conn.execute("""
            WITH customer_cohorts AS (
                SELECT 
                    customer_id,
                    DATE_TRUNC('month', MIN(order_date)) as cohort_month,
                    COUNT(DISTINCT order_id) as lifetime_orders,
                    SUM(amount) as lifetime_value
                FROM iceberg_orders
                GROUP BY customer_id
            ),
            
            monthly_retention AS (
                SELECT
                    c.cohort_month,
                    DATE_DIFF('month', c.cohort_month, DATE_TRUNC('month', o.order_date)) as months_since_first,
                    COUNT(DISTINCT c.customer_id) as active_customers,
                    COUNT(DISTINCT c.customer_id) * 100.0 / 
                        FIRST_VALUE(COUNT(DISTINCT c.customer_id)) OVER (
                            PARTITION BY c.cohort_month 
                            ORDER BY DATE_DIFF('month', c.cohort_month, DATE_TRUNC('month', o.order_date))
                        ) as retention_rate
                FROM customer_cohorts c
                JOIN iceberg_orders o ON c.customer_id = o.customer_id
                WHERE o.order_date >= c.cohort_month
                GROUP BY 1, 2
            ),
            
            -- Advanced statistical functions
            customer_segments AS (
                SELECT
                    customer_id,
                    lifetime_value,
                    NTILE(10) OVER (ORDER BY lifetime_value) as value_decile,
                    lifetime_value > PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY lifetime_value) OVER () as is_top_10_percent,
                    
                    -- K-means clustering simulation
                    CASE
                        WHEN lifetime_orders > 10 AND lifetime_value > 1000 THEN 'champions'
                        WHEN lifetime_orders > 5 AND lifetime_value > 500 THEN 'loyal'
                        WHEN lifetime_orders <= 2 THEN 'new'
                        ELSE 'regular'
                    END as segment
                FROM customer_cohorts
            )
            
            -- Main query with PIVOT
            SELECT * FROM (
                SELECT 
                    cohort_month,
                    months_since_first,
                    retention_rate
                FROM monthly_retention
                WHERE months_since_first <= 12
            )
            PIVOT (
                AVG(retention_rate) FOR months_since_first IN (0, 1, 2, 3, 6, 12)
            ) AS p
            ORDER BY cohort_month DESC
        """).df()
        
        return result
    
    def create_ml_features(self, output_path: str):
        """Generate ML-ready features using DuckDB"""
        
        self.conn.execute(f"""
            COPY (
                WITH user_features AS (
                    SELECT
                        u.user_id,
                        u.registration_date,
                        
                        -- Behavioral features
                        COUNT(DISTINCT e.session_id) as total_sessions,
                        COUNT(DISTINCT DATE(e.event_time)) as active_days,
                        SUM(CASE WHEN e.event_type = 'purchase' THEN 1 ELSE 0 END) as purchase_count,
                        
                        -- Time-based features
                        AVG(e.event_duration) as avg_session_duration,
                        STDDEV(e.event_duration) as stddev_session_duration,
                        
                        -- Advanced aggregations
                        MODE() WITHIN GROUP (ORDER BY e.device_type) as most_common_device,
                        LISTAGG(DISTINCT e.category, ',') WITHIN GROUP (ORDER BY e.category) as categories_viewed,
                        
                        -- Window functions for trends
                        SUM(CASE WHEN e.event_time >= CURRENT_DATE - INTERVAL 7 DAY THEN 1 ELSE 0 END) as events_last_7d,
                        SUM(CASE WHEN e.event_time >= CURRENT_DATE - INTERVAL 30 DAY THEN 1 ELSE 0 END) as events_last_30d,
                        
                        -- Text analysis with FTS
                        GROUP_CONCAT(e.search_query, ' ') as all_searches
                        
                    FROM users u
                    LEFT JOIN events e ON u.user_id = e.user_id
                    GROUP BY u.user_id, u.registration_date
                ),
                
                feature_engineering AS (
                    SELECT
                        *,
                        -- Derived features
                        events_last_7d::FLOAT / NULLIF(events_last_30d, 0) as recent_activity_ratio,
                        total_sessions::FLOAT / NULLIF(active_days, 0) as sessions_per_day,
                        
                        -- One-hot encoding
                        CASE WHEN most_common_device = 'mobile' THEN 1 ELSE 0 END as is_mobile_user,
                        CASE WHEN most_common_device = 'desktop' THEN 1 ELSE 0 END as is_desktop_user,
                        
                        -- Feature hashing for high cardinality
                        HASH(categories_viewed) % 1000 as category_hash
                        
                    FROM user_features
                )
                
                SELECT * FROM feature_engineering
            ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
    
    def streaming_aggregation(self, input_stream):
        """Process streaming data with DuckDB"""
        
        # DuckDB can process data as it arrives
        for batch in input_stream:
            # Convert batch to Arrow table for zero-copy
            arrow_table = batch.to_arrow()
            
            # Register as temporary view
            self.conn.register("stream_batch", arrow_table)
            
            # Incremental aggregation
            self.conn.execute("""
                INSERT INTO streaming_metrics
                SELECT
                    DATE_TRUNC('minute', event_time) as minute,
                    COUNT(*) as event_count,
                    COUNT(DISTINCT user_id) as unique_users,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time) as p95_latency
                FROM stream_batch
                GROUP BY 1
                ON CONFLICT (minute) DO UPDATE SET
                    event_count = streaming_metrics.event_count + EXCLUDED.event_count,
                    unique_users = streaming_metrics.unique_users + EXCLUDED.unique_users,
                    p95_latency = EXCLUDED.p95_latency
            """)
            
            # Unregister to free memory
            self.conn.unregister("stream_batch")
```

---

## 7. Production Deployment Patterns

### ✅ DO: Use Infrastructure as Code

```yaml
# kubernetes/data-platform/spark-operator.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: spark-operator
---
apiVersion: sparkoperator.k8s.io/v1beta2
kind: SparkApplication
metadata:
  name: data-processing-job
  namespace: spark-operator
spec:
  type: Python
  pythonVersion: "3"
  mode: cluster
  image: "company/spark:4.0-python3.12"
  imagePullPolicy: Always
  mainApplicationFile: "s3a://data-platform/jobs/process_data.py"
  sparkVersion: "4.0.0"
  
  # Spark configuration for production
  sparkConf:
    # Performance
    "spark.sql.adaptive.enabled": "true"
    "spark.sql.adaptive.coalescePartitions.enabled": "true"
    "spark.sql.adaptive.skewJoin.enabled": "true"
    "spark.sql.adaptive.localShuffleReader.enabled": "true"
    
    # S3 optimization
    "spark.hadoop.fs.s3a.fast.upload": "true"
    "spark.hadoop.fs.s3a.fast.upload.buffer": "bytebuffer"
    "spark.hadoop.fs.s3a.multipart.size": "128M"
    "spark.hadoop.fs.s3a.connection.maximum": "100"
    
    # Iceberg/Delta configuration
    "spark.sql.catalog.spark_catalog": "org.apache.iceberg.spark.SparkSessionCatalog"
    "spark.sql.catalog.spark_catalog.type": "hive"
    "spark.sql.extensions": "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions"
    
    # Memory management
    "spark.memory.offHeap.enabled": "true"
    "spark.memory.offHeap.size": "16g"
    
  # Dynamic allocation
  dynamicAllocation:
    enabled: true
    initialExecutors: 2
    minExecutors: 2
    maxExecutors: 100
    
  # Driver specification
  driver:
    cores: 4
    coreLimit: "4000m"
    memory: "8g"
    serviceAccount: spark-operator-spark
    env:
      - name: AWS_REGION
        value: "us-east-1"
    
  # Executor specification  
  executor:
    cores: 4
    instances: 2
    memory: "16g"
    memoryOverhead: "2g"
    
  # Monitoring
  monitoring:
    exposeDriverMetrics: true
    exposeExecutorMetrics: true
    prometheus:
      jmxExporterJar: "/opt/spark/jars/jmx_prometheus_javaagent.jar"
      port: 8090
```

```python
# terraform/modules/data-platform/main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.27"
    }
  }
}

# S3 buckets for data lake
resource "aws_s3_bucket" "data_lake" {
  bucket = "${var.environment}-data-lake"
  
  tags = {
    Environment = var.environment
    Team        = "data-platform"
  }
}

# Bucket versioning for data recovery
resource "aws_s3_bucket_versioning" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

# Intelligent tiering for cost optimization
resource "aws_s3_bucket_intelligent_tiering_configuration" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id
  name   = "EntireBucket"
  
  tiering {
    access_tier = "ARCHIVE_ACCESS"
    days        = 90
  }
  
  tiering {
    access_tier = "DEEP_ARCHIVE_ACCESS"
    days        = 180
  }
}

# Glue catalog database
resource "aws_glue_catalog_database" "main" {
  name = "${var.environment}_data_catalog"
  
  description = "Data catalog for ${var.environment} environment"
  
  location_uri = "s3://${aws_s3_bucket.data_lake.id}/catalog/"
}

# EMR Serverless for Spark
resource "aws_emrserverless_application" "spark" {
  name          = "${var.environment}-spark-serverless"
  release_label = "emr-7.0.0"
  type          = "SPARK"
  
  initial_capacity {
    initial_capacity_type = "Driver"
    
    initial_capacity_config {
      worker_count = 1
      worker_configuration {
        cpu    = "4 vCPU"
        memory = "16 GB"
      }
    }
  }
  
  initial_capacity {
    initial_capacity_type = "Executor"
    
    initial_capacity_config {
      worker_count = 4
      worker_configuration {
        cpu    = "4 vCPU"
        memory = "16 GB"
      }
    }
  }
  
  auto_stop_configuration {
    enabled              = true
    idle_timeout_minutes = 15
  }
  
  network_configuration {
    security_group_ids = [aws_security_group.emr_serverless.id]
    subnet_ids         = var.private_subnet_ids
  }
}

# Kafka MSK Serverless
resource "aws_msk_serverless_cluster" "main" {
  cluster_name = "${var.environment}-kafka-serverless"
  
  vpc_config {
    subnet_ids         = var.private_subnet_ids
    security_group_ids = [aws_security_group.kafka.id]
  }
  
  client_authentication {
    sasl {
      iam {
        enabled = true
      }
    }
  }
}

# Airflow on EKS
module "airflow" {
  source = "./modules/airflow-eks"
  
  cluster_name        = module.eks.cluster_id
  namespace          = "airflow"
  airflow_version    = "2.8.1"
  
  # Configuration
  airflow_config = {
    AIRFLOW__CORE__PARALLELISM                = 128
    AIRFLOW__CORE__DAG_CONCURRENCY            = 64
    AIRFLOW__CORE__MAX_ACTIVE_RUNS_PER_DAG   = 16
    AIRFLOW__KUBERNETES__WORKER_PODS_CREATION_BATCH_SIZE = 16
    AIRFLOW__KUBERNETES__WORKER_PODS_QUEUED_CHECK_INTERVAL = 30
  }
  
  # Autoscaling
  autoscaling_enabled = true
  min_workers        = 2
  max_workers        = 50
  target_cpu_utilization = 70
}
```

### ✅ DO: Implement Cost Optimization

```python
# cost_optimization/s3_lifecycle_manager.py
import boto3
from datetime import datetime, timedelta
import pandas as pd

class S3CostOptimizer:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.cloudwatch = boto3.client('cloudwatch')
        
    def analyze_access_patterns(self, bucket_name: str) -> pd.DataFrame:
        """Analyze S3 access patterns for optimization"""
        
        # Get S3 Inventory data
        inventory_data = self._get_s3_inventory(bucket_name)
        
        # Analyze access patterns
        analysis = inventory_data.groupby(['storage_class', 'age_days']).agg({
            'size_bytes': ['sum', 'count'],
            'last_accessed_days_ago': ['mean', 'min', 'max']
        }).reset_index()
        
        # Calculate potential savings
        analysis['potential_savings'] = analysis.apply(
            lambda row: self._calculate_savings(row), axis=1
        )
        
        return analysis
    
    def create_intelligent_lifecycle_policy(self, bucket_name: str):
        """Create lifecycle policy based on access patterns"""
        
        lifecycle_policy = {
            'Rules': [
                {
                    'ID': 'IntelligentTransition',
                    'Status': 'Enabled',
                    'Transitions': [
                        {
                            # Hot to Warm (IA) after 30 days
                            'Days': 30,
                            'StorageClass': 'STANDARD_IA'
                        },
                        {
                            # Warm to Cool (Glacier) after 90 days
                            'Days': 90,
                            'StorageClass': 'GLACIER_IR'
                        },
                        {
                            # Cool to Cold (Deep Archive) after 180 days
                            'Days': 180,
                            'StorageClass': 'DEEP_ARCHIVE'
                        }
                    ],
                    # Delete old data
                    'Expiration': {
                        'Days': 730  # 2 years
                    },
                    # Handle incomplete multipart uploads
                    'AbortIncompleteMultipartUpload': {
                        'DaysAfterInitiation': 7
                    }
                }
            ]
        }
        
        self.s3_client.put_bucket_lifecycle_configuration(
            Bucket=bucket_name,
            LifecycleConfiguration=lifecycle_policy
        )
    
    def optimize_compute_costs(self, job_history: pd.DataFrame) -> dict:
        """Recommend optimal compute configurations"""
        
        recommendations = {}
        
        # Analyze Spark job patterns
        spark_analysis = job_history[job_history['engine'] == 'spark'].groupby('job_name').agg({
            'duration_minutes': ['mean', 'std'],
            'data_processed_gb': 'mean',
            'cpu_hours': 'mean',
            'memory_gb_hours': 'mean',
            'cost_usd': 'mean'
        })
        
        for job_name, stats in spark_analysis.iterrows():
            duration_mean = stats[('duration_minutes', 'mean')]
            data_size = stats[('data_processed_gb', 'mean')]
            
            # Recommend instance types
            if duration_mean < 15 and data_size < 100:
                recommendations[job_name] = {
                    'current_cost': stats[('cost_usd', 'mean')],
                    'recommendation': 'Use Spark Serverless',
                    'estimated_savings': stats[('cost_usd', 'mean')] * 0.4
                }
            elif duration_mean > 60:
                recommendations[job_name] = {
                    'current_cost': stats[('cost_usd', 'mean')],
                    'recommendation': 'Use Spot Instances',
                    'estimated_savings': stats[('cost_usd', 'mean')] * 0.7
                }
        
        return recommendations
```

---

## 8. Security and Compliance

### ✅ DO: Implement End-to-End Encryption and Access Control

```python
# security/data_encryption.py
from cryptography.fernet import Fernet
import boto3
from functools import wraps
import hashlib
import json

class DataSecurityManager:
    def __init__(self, kms_key_id: str):
        self.kms_client = boto3.client('kms')
        self.kms_key_id = kms_key_id
        self.secrets_manager = boto3.client('secretsmanager')
        
    def encrypt_pii_columns(self, df: pd.DataFrame, pii_columns: list) -> pd.DataFrame:
        """Encrypt PII columns using format-preserving encryption"""
        
        # Get data encryption key
        dek = self._get_data_encryption_key()
        
        for column in pii_columns:
            if column in df.columns:
                df[f'{column}_encrypted'] = df[column].apply(
                    lambda x: self._format_preserving_encrypt(x, dek) if pd.notna(x) else None
                )
                # Hash original for lookups
                df[f'{column}_hash'] = df[column].apply(
                    lambda x: hashlib.sha256(str(x).encode()).hexdigest() if pd.notna(x) else None
                )
                # Remove original
                df = df.drop(columns=[column])
        
        return df
    
    def _format_preserving_encrypt(self, value: str, key: bytes) -> str:
        """Encrypt while preserving format (e.g., SSN: XXX-XX-XXXX)"""
        # Implementation depends on specific requirements
        # This is a simplified example
        if '-' in str(value):
            parts = str(value).split('-')
            encrypted_parts = [
                self._encrypt_part(part, key) for part in parts
            ]
            return '-'.join(encrypted_parts)
        else:
            return self._encrypt_part(str(value), key)
    
    def create_data_access_policy(self, table_name: str, policy: dict):
        """Create fine-grained access control policy"""
        
        # Example policy structure
        policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": policy['allowed_roles']},
                    "Action": ["s3:GetObject"],
                    "Resource": f"arn:aws:s3:::data-lake/{table_name}/*",
                    "Condition": {
                        "StringEquals": {
                            "s3:ExistingObjectTag/Classification": policy['allowed_classifications']
                        }
                    }
                }
            ]
        }
        
        # Apply column-level security in Glue
        for column, access_list in policy.get('column_access', {}).items():
            self._apply_column_security(table_name, column, access_list)
        
        return policy_document
    
    def audit_data_access(self, query: str, user: str, accessed_tables: list):
        """Log data access for compliance"""
        
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user': user,
            'query_hash': hashlib.sha256(query.encode()).hexdigest(),
            'accessed_tables': accessed_tables,
            'query_type': self._classify_query(query),
            'ip_address': self._get_user_ip(),
            'session_id': self._get_session_id()
        }
        
        # Log to CloudWatch
        self._log_to_cloudwatch(audit_entry)
        
        # Check for anomalies
        if self._is_anomalous_access(audit_entry):
            self._trigger_security_alert(audit_entry)

# Privacy-preserving analytics
class PrivacyPreservingAnalytics:
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon  # Differential privacy parameter
        
    def add_differential_privacy(self, df: pd.DataFrame, numeric_columns: list) -> pd.DataFrame:
        """Add noise for differential privacy"""
        
        for column in numeric_columns:
            if column in df.columns:
                # Calculate sensitivity
                sensitivity = df[column].max() - df[column].min()
                
                # Add Laplace noise
                noise = np.random.laplace(0, sensitivity / self.epsilon, size=len(df))
                df[f'{column}_private'] = df[column] + noise
        
        return df
    
    def k_anonymize(self, df: pd.DataFrame, quasi_identifiers: list, k: int = 5) -> pd.DataFrame:
        """Implement k-anonymity"""
        
        # Group by quasi-identifiers
        grouped = df.groupby(quasi_identifiers).size().reset_index(name='count')
        
        # Suppress groups smaller than k
        suppress_groups = grouped[grouped['count'] < k]
        
        # Generalize or suppress
        for _, row in suppress_groups.iterrows():
            mask = True
            for qi in quasi_identifiers:
                mask &= (df[qi] == row[qi])
            
            # Suppress these records
            df.loc[mask, quasi_identifiers] = np.nan
        
        return df
```

---

## 9. Advanced Patterns and Future-Proofing

### ✅ DO: Build for Change with Event-Driven Architecture

```python
# event_driven/schema_evolution.py
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
import json

class SchemaEvolutionManager:
    def __init__(self, schema_registry_url: str):
        self.sr_client = SchemaRegistryClient({'url': schema_registry_url})
        
    def register_backwards_compatible_schema(self, subject: str, new_schema: dict) -> int:
        """Register schema with backward compatibility check"""
        
        # Check compatibility
        is_compatible = self.sr_client.test_compatibility(
            subject_name=subject,
            schema=json.dumps(new_schema)
        )
        
        if not is_compatible:
            raise ValueError("Schema is not backward compatible")
        
        # Register new version
        schema_id = self.sr_client.register_schema(
            subject_name=subject,
            schema=json.dumps(new_schema)
        )
        
        return schema_id
    
    def create_migration_path(self, subject: str, from_version: int, to_version: int):
        """Create migration path between schema versions"""
        
        from_schema = self.sr_client.get_schema(subject, from_version)
        to_schema = self.sr_client.get_schema(subject, to_version)
        
        migration_code = self._generate_migration_code(from_schema, to_schema)
        
        return {
            'from_version': from_version,
            'to_version': to_version,
            'migration_type': self._determine_migration_type(from_schema, to_schema),
            'migration_code': migration_code
        }

# Future-proof data contracts
class DataContract:
    def __init__(self, contract_definition: dict):
        self.contract = contract_definition
        self.version = contract_definition['version']
        
    def validate_data(self, df: pd.DataFrame) -> tuple[bool, list]:
        """Validate data against contract"""
        
        violations = []
        
        # Check schema
        for field in self.contract['schema']['fields']:
            if field['name'] not in df.columns:
                if field.get('required', True):
                    violations.append(f"Missing required field: {field['name']}")
            else:
                # Validate data type
                expected_type = field['type']
                actual_type = str(df[field['name']].dtype)
                
                if not self._types_compatible(expected_type, actual_type):
                    violations.append(
                        f"Type mismatch for {field['name']}: "
                        f"expected {expected_type}, got {actual_type}"
                    )
        
        # Check quality rules
        for rule in self.contract.get('quality_rules', []):
            rule_result = self._evaluate_rule(df, rule)
            if not rule_result['passed']:
                violations.append(f"Quality rule failed: {rule['name']} - {rule_result['message']}")
        
        # Check SLAs
        sla_check = self._check_sla(df)
        if not sla_check['passed']:
            violations.append(f"SLA violation: {sla_check['message']}")
        
        return len(violations) == 0, violations
    
    def generate_contract_tests(self) -> str:
        """Generate automated tests from contract"""
        
        test_code = f"""
import pytest
import pandas as pd
from data_contracts import DataContract

class TestDataContract_{self.contract['name'].replace('-', '_')}:
    
    def setup_method(self):
        self.contract = DataContract('{self.contract['name']}')
    
"""
        
        # Generate test for each field
        for field in self.contract['schema']['fields']:
            test_code += f"""
    def test_field_{field['name']}(self, sample_data):
        assert '{field['name']}' in sample_data.columns
        assert sample_data['{field['name']}'].dtype == '{field['type']}'
"""
        
        # Generate test for each quality rule
        for rule in self.contract.get('quality_rules', []):
            test_code += f"""
    def test_quality_rule_{rule['name'].replace('-', '_')}(self, sample_data):
        result = self.contract._evaluate_rule(sample_data, {rule})
        assert result['passed'], result['message']
"""
        
        return test_code
```

### ✅ DO: Implement ML-Ops for Data Pipelines

```python
# mlops/feature_store_integration.py
from feast import FeatureStore, Entity, Feature, FeatureView, FileSource
from feast.types import Float32, Int64, String
from datetime import timedelta

class FeatureEngineeringPipeline:
    def __init__(self, feature_store_repo: str):
        self.fs = FeatureStore(repo_path=feature_store_repo)
        
    def create_feature_definitions(self):
        """Define features in the feature store"""
        
        # Define entities
        customer = Entity(
            name="customer_id",
            value_type=String,
            description="Customer identifier"
        )
        
        # Define feature views
        customer_features = FeatureView(
            name="customer_features",
            entities=["customer_id"],
            ttl=timedelta(days=1),
            features=[
                Feature(name="lifetime_value", dtype=Float32),
                Feature(name="churn_probability", dtype=Float32),
                Feature(name="days_since_last_purchase", dtype=Int64),
                Feature(name="preferred_category", dtype=String),
            ],
            online=True,
            batch_source=FileSource(
                path="s3://feature-store/customer_features.parquet",
                event_timestamp_column="event_timestamp",
            ),
            tags={"team": "data-science", "tier": "production"}
        )
        
        # Apply to feature store
        self.fs.apply([customer, customer_features])
    
    def generate_training_dataset(self, entity_df: pd.DataFrame, features: list) -> pd.DataFrame:
        """Generate point-in-time correct training dataset"""
        
        # Get historical features
        training_df = self.fs.get_historical_features(
            entity_df=entity_df,
            features=features,
            full_feature_names=True
        ).to_df()
        
        # Add data quality checks
        self._validate_training_data(training_df)
        
        return training_df
    
    def serve_features_online(self, entity_keys: dict) -> dict:
        """Serve features for online inference"""
        
        feature_vector = self.fs.get_online_features(
            features=[
                "customer_features:lifetime_value",
                "customer_features:churn_probability",
                "customer_features:days_since_last_purchase"
            ],
            entity_rows=[entity_keys]
        ).to_dict()
        
        return feature_vector

# Automated model monitoring
class ModelMonitoring:
    def __init__(self, model_name: str, baseline_metrics: dict):
        self.model_name = model_name
        self.baseline_metrics = baseline_metrics
        
    def detect_drift(self, current_data: pd.DataFrame, reference_data: pd.DataFrame) -> dict:
        """Detect data and prediction drift"""
        
        from alibi_detect.cd import KSDrift, ChiSquareDrift
        
        drift_results = {}
        
        # Numerical features - KS test
        numerical_features = current_data.select_dtypes(include=[np.number]).columns
        for feature in numerical_features:
            detector = KSDrift(
                reference_data[feature].values,
                p_val=0.05
            )
            
            drift_detected = detector.predict(current_data[feature].values)
            drift_results[feature] = {
                'drift_detected': bool(drift_detected['data']['is_drift']),
                'p_value': float(drift_detected['data']['p_val']),
                'threshold': 0.05
            }
        
        # Categorical features - Chi-square test
        categorical_features = current_data.select_dtypes(include=['object']).columns
        for feature in categorical_features:
            detector = ChiSquareDrift(
                reference_data[feature].values,
                p_val=0.05
            )
            
            drift_detected = detector.predict(current_data[feature].values)
            drift_results[feature] = {
                'drift_detected': bool(drift_detected['data']['is_drift']),
                'p_value': float(drift_detected['data']['p_val']),
                'threshold': 0.05
            }
        
        return drift_results
    
    def monitor_model_performance(self, predictions_df: pd.DataFrame) -> dict:
        """Monitor model performance metrics"""
        
        metrics = {}
        
        # Calculate performance metrics
        if 'actual' in predictions_df.columns:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics['mse'] = mean_squared_error(predictions_df['actual'], predictions_df['predicted'])
            metrics['mae'] = mean_absolute_error(predictions_df['actual'], predictions_df['predicted'])
            metrics['r2'] = r2_score(predictions_df['actual'], predictions_df['predicted'])
            
            # Compare with baseline
            for metric, value in metrics.items():
                baseline = self.baseline_metrics.get(metric)
                if baseline:
                    degradation = abs(value - baseline) / baseline
                    if degradation > 0.1:  # 10% degradation threshold
                        self._trigger_alert(
                            f"Model {self.model_name} performance degraded: "
                            f"{metric} changed from {baseline} to {value}"
                        )
        
        # Monitor prediction distribution
        prediction_stats = {
            'mean': predictions_df['predicted'].mean(),
            'std': predictions_df['predicted'].std(),
            'min': predictions_df['predicted'].min(),
            'max': predictions_df['predicted'].max(),
            'nulls': predictions_df['predicted'].isna().sum()
        }
        
        metrics['prediction_distribution'] = prediction_stats
        
        return metrics
```

---

## 10. Conclusion and Best Practices Summary

### Key Takeaways

1. **Lake-house Architecture**: Combine the best of data lakes and warehouses using Iceberg or Delta Lake
2. **Streaming-First**: Design for real-time data with batch as a special case
3. **SQL Transformation**: dbt has become the standard for maintainable data transformations
4. **Observability**: Build quality checks and monitoring into every layer
5. **Cost Optimization**: Use lifecycle policies and right-size compute resources
6. **Security**: Implement encryption, access control, and privacy preservation from the start

### Migration Checklist

When modernizing your data platform:

- [ ] Assess current data volume and growth projections
- [ ] Choose table format (Iceberg vs Delta Lake) based on your ecosystem
- [ ] Implement CDC for real-time data capture
- [ ] Migrate transformations to dbt
- [ ] Set up comprehensive monitoring and alerting
- [ ] Implement cost optimization strategies
- [ ] Ensure security and compliance requirements are met
- [ ] Create data contracts for critical interfaces
- [ ] Build feature stores for ML workloads
- [ ] Plan for schema evolution and backwards compatibility

### Future Trends to Watch

- **Lakehouse Formats**: Continued convergence of Iceberg, Delta, and Hudi
- **Streaming SQL**: More powerful streaming SQL capabilities in Flink and Spark
- **Zero-ETL**: Direct query federation across systems
- **AI-Powered Optimization**: Automatic query and pipeline optimization
- **Privacy-Preserving Analytics**: More sophisticated differential privacy implementations

Remember: The best architecture is one that evolves with your needs. Start simple, measure everything, and iterate based on real usage patterns.