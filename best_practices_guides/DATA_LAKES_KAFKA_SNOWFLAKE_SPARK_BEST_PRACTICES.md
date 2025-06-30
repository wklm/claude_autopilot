# The Definitive Guide to Modern Data Lakes: Kafka, Iceberg, Spark, and dbt (2025)

This guide synthesizes production-grade best practices for building scalable, cost-efficient data lakes using the modern streaming-first architecture. It covers the complete data lifecycle from ingestion through transformation, addressing the real challenges of petabyte-scale operations.

### Prerequisites & Technology Stack
This guide assumes **Kafka 3.8+**, **Apache Spark 4.0+**, **Apache Flink 2.0+**, **dbt 1.9+**, and **Apache Iceberg 1.6+** or **Delta Lake 3.2+**. All examples use **Python 3.13+** and **Java 17+** (for JVM components).

### Core Architecture Overview

```yaml
# docker-compose.yml for local development
version: '3.9'
services:
  kafka:
    image: confluentinc/cp-kafka:7.6.0
    environment:
      KAFKA_NODE_ID: 1
      KAFKA_KRAFT_MODE: "true"
      KAFKA_PROCESS_ROLES: 'broker,controller'
      KAFKA_CONTROLLER_QUORUM_VOTERS: '1@kafka:29093'
      KAFKA_LISTENERS: 'PLAINTEXT://kafka:29092,CONTROLLER://kafka:29093'
      
  schema-registry:
    image: confluentinc/cp-schema-registry:7.6.0
    depends_on:
      - kafka
      
  connect:
    image: debezium/connect:2.6
    environment:
      BOOTSTRAP_SERVERS: kafka:29092
      GROUP_ID: 1
      CONFIG_STORAGE_TOPIC: connect_configs
      
  minio:
    image: minio/minio:RELEASE.2025-01-15
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
      
  spark-iceberg:
    build: ./spark-iceberg
    environment:
      AWS_ACCESS_KEY_ID: minioadmin
      AWS_SECRET_ACCESS_KEY: minioadmin
      AWS_ENDPOINT: http://minio:9000
```

> **Note**: Kafka 3.8 defaults to KRaft mode (no ZooKeeper). The `kafka.Kraft` protocol provides 3x faster metadata operations and eliminates the ZooKeeper dependency.

---

## 1. Foundational Data Lake Architecture

Modern data lakes follow a medallion architecture (Bronze → Silver → Gold) with clear separation between raw ingestion, cleaned data, and business-ready datasets.

### ✅ DO: Implement a Multi-Zone Architecture

```
data-lake/
├── raw/                    # Bronze: Raw, immutable data
│   ├── kafka_topics/       # Streaming data landed from Kafka
│   ├── database_cdc/       # CDC snapshots from Debezium
│   └── file_drops/         # Batch file ingestion
├── cleaned/                # Silver: Deduplicated, typed, partitioned
│   ├── events/            # Event streams with schema evolution
│   └── dimensions/        # Slowly changing dimensions
├── curated/               # Gold: Business-ready datasets
│   ├── analytics/         # Aggregated fact tables
│   └── ml_features/       # Feature store tables
└── metadata/              # Iceberg/Delta metadata and manifests
    ├── iceberg/
    └── delta/
```

### ✅ DO: Choose the Right Table Format

**Apache Iceberg** for maximum flexibility and multi-engine support:

```python
# iceberg_setup.py
from pyspark.sql import SparkSession
import os

def create_iceberg_spark_session():
    """Create Spark session configured for Iceberg on S3"""
    
    # Use Spark 4.0's improved Iceberg integration
    spark = SparkSession.builder \
        .appName("DataLakeProcessor") \
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
        .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog") \
        .config("spark.sql.catalog.iceberg.type", "rest") \
        .config("spark.sql.catalog.iceberg.uri", os.getenv("ICEBERG_REST_URI", "http://localhost:8181")) \
        .config("spark.sql.catalog.iceberg.io-impl", "org.apache.iceberg.aws.s3.S3FileIO") \
        .config("spark.sql.catalog.iceberg.warehouse", "s3a://data-lake/warehouse") \
        .config("spark.sql.catalog.iceberg.s3.endpoint", os.getenv("S3_ENDPOINT", "http://minio:9000")) \
        .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID")) \
        .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY")) \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    # Enable Iceberg's row-level operations
    spark.sql("SET spark.sql.iceberg.merge-on-read.enabled=true")
    
    return spark
```

**Delta Lake** for Databricks-centric environments:

```python
# delta_setup.py
def create_delta_spark_session():
    """Create Spark session configured for Delta Lake"""
    
    spark = SparkSession.builder \
        .appName("DeltaLakeProcessor") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.databricks.delta.properties.defaults.enableChangeDataFeed", "true") \
        .config("spark.databricks.delta.optimizeWrite.enabled", "true") \
        .config("spark.databricks.delta.autoCompact.enabled", "true") \
        .getOrCreate()
    
    return spark
```

### ❌ DON'T: Mix Table Formats in the Same Zone

This creates operational complexity and prevents unified governance. Pick one format per zone and stick with it.

```python
# Bad - Mixing formats in silver zone
df.write.format("iceberg").save("s3a://data-lake/cleaned/users")
df2.write.format("delta").save("s3a://data-lake/cleaned/transactions")  # Don't mix!
```

---

## 2. Streaming-First Ingestion with Kafka & Debezium

The modern data lake treats batch as a special case of streaming. Everything flows through Kafka first, providing a unified ingestion layer with replay capability.

### ✅ DO: Implement Schema Registry from Day One

Schema evolution is inevitable. Confluent Schema Registry with Avro provides backward/forward compatibility.

```java
// KafkaProducerConfig.java
import io.confluent.kafka.serializers.KafkaAvroSerializer;
import org.apache.kafka.clients.producer.ProducerConfig;
import java.util.Properties;

public class KafkaProducerConfig {
    public static Properties createProducerProps() {
        Properties props = new Properties();
        
        // Kafka 3.8 producer defaults are optimized
        props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, "kafka:29092");
        props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, 
                  "org.apache.kafka.common.serialization.StringSerializer");
        props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, KafkaAvroSerializer.class);
        
        // Schema Registry configuration
        props.put("schema.registry.url", "http://schema-registry:8081");
        props.put("auto.register.schemas", false); // Require explicit schema registration
        
        // Idempotent producer for exactly-once semantics
        props.put(ProducerConfig.ENABLE_IDEMPOTENCE_CONFIG, true);
        props.put(ProducerConfig.ACKS_CONFIG, "all");
        props.put(ProducerConfig.MAX_IN_FLIGHT_REQUESTS_PER_CONNECTION, 5);
        
        // Compression for efficiency
        props.put(ProducerConfig.COMPRESSION_TYPE_CONFIG, "zstd"); // Better than snappy
        
        return props;
    }
}
```

### ✅ DO: Configure Debezium for Optimal CDC

```json
{
  "name": "postgres-connector",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "postgres",
    "database.port": "5432",
    "database.user": "debezium",
    "database.password": "${file:/opt/kafka/external-configuration/connector-password:password}",
    "database.dbname": "production",
    "database.server.name": "prod-db",
    "plugin.name": "pgoutput",
    "publication.name": "dbz_publication",
    "slot.name": "debezium_slot",
    
    "table.include.list": "public.users,public.orders,public.products",
    
    "transforms": "unwrap,addTopicPrefix",
    "transforms.unwrap.type": "io.debezium.transforms.ExtractNewRecordState",
    "transforms.unwrap.drop.tombstones": "false",
    "transforms.unwrap.delete.handling.mode": "rewrite",
    "transforms.addTopicPrefix.type": "org.apache.kafka.connect.transforms.RegexRouter",
    "transforms.addTopicPrefix.regex": "prod-db.public.(.*)",
    "transforms.addTopicPrefix.replacement": "cdc.postgres.$1",
    
    "decimal.handling.mode": "string",
    "time.precision.mode": "adaptive_time_microseconds",
    "interval.handling.mode": "numeric",
    
    "heartbeat.interval.ms": "10000",
    "heartbeat.action.query": "UPDATE debezium_heartbeat SET last_heartbeat = NOW()",
    
    "snapshot.mode": "initial",
    "snapshot.isolation.mode": "repeatable_read",
    "incremental.snapshot.chunk.size": "10240",
    
    "topic.creation.default.replication.factor": "3",
    "topic.creation.default.partitions": "6",
    "topic.creation.default.cleanup.policy": "compact",
    "topic.creation.default.compression.type": "zstd"
  }
}
```

### Advanced Debezium Patterns

#### Handling Schema Evolution

```python
# schema_evolution_handler.py
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer
import json

class SchemaEvolutionHandler:
    def __init__(self, registry_url: str):
        self.sr_client = SchemaRegistryClient({'url': registry_url})
        self.schema_cache = {}
    
    async def handle_evolved_message(self, topic: str, message: bytes):
        """Handle messages with evolved schemas"""
        
        # Extract schema ID from message
        schema_id = int.from_bytes(message[1:5], 'big')
        
        # Cache schema lookups
        if schema_id not in self.schema_cache:
            schema = self.sr_client.get_schema(schema_id)
            self.schema_cache[schema_id] = schema
        
        # Deserialize with specific schema version
        deserializer = AvroDeserializer(
            self.sr_client,
            self.schema_cache[schema_id].schema_str
        )
        
        return deserializer(message[5:], None)
```

#### Incremental Snapshots for Large Tables

```sql
-- Enable incremental snapshots in PostgreSQL
ALTER TABLE large_table REPLICA IDENTITY FULL;

-- Signal Debezium to start incremental snapshot
INSERT INTO debezium_signal (id, type, data) 
VALUES ('adhoc-1', 'execute-snapshot', 
  '{"data-collections": ["public.large_table"], "type": "incremental"}');
```

### ✅ DO: Implement Dead Letter Queue (DLQ) Handling

```python
# dlq_processor.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, current_timestamp
from delta import DeltaTable

def process_dlq_messages(spark: SparkSession):
    """Process failed messages from Kafka Connect DLQ"""
    
    # Read from DLQ topic
    dlq_df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:29092") \
        .option("subscribe", "connect-dlq") \
        .option("startingOffsets", "earliest") \
        .option("failOnDataLoss", "false") \
        .load()
    
    # Parse error details
    parsed_df = dlq_df.select(
        col("key").cast("string").alias("original_key"),
        col("value").cast("string").alias("original_value"),
        col("headers").alias("error_headers"),
        current_timestamp().alias("dlq_timestamp")
    )
    
    # Write to error investigation table
    query = parsed_df.writeStream \
        .format("iceberg") \
        .outputMode("append") \
        .option("checkpointLocation", "s3a://data-lake/checkpoints/dlq") \
        .option("fanout-enabled", "true") \
        .trigger(processingTime='5 minutes') \
        .toTable("iceberg.raw.kafka_dlq_events")
    
    return query
```

---

## 3. Efficient Storage Patterns with Iceberg

Apache Iceberg provides ACID transactions, time travel, and schema evolution at petabyte scale. These patterns maximize performance and minimize cost.

### ✅ DO: Optimize Partitioning Strategy

```python
# partitioning_strategy.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month, day, hour

def create_optimized_event_table(spark: SparkSession):
    """Create Iceberg table with hidden partitioning"""
    
    # Iceberg's hidden partitioning - no partition columns in data
    spark.sql("""
        CREATE TABLE IF NOT EXISTS iceberg.cleaned.events (
            event_id STRING,
            user_id BIGINT,
            event_type STRING,
            event_timestamp TIMESTAMP,
            properties MAP<STRING, STRING>,
            processing_time TIMESTAMP
        ) USING iceberg
        PARTITIONED BY (
            days(event_timestamp),  -- Daily partitions
            bucket(16, user_id),    -- Hash buckets for user queries
            truncate(event_type, 4) -- Truncate for event type prefix
        )
        TBLPROPERTIES (
            'write.distribution-mode'='hash',
            'write.metadata.compression-codec'='gzip',
            'write.parquet.compression-codec'='zstd',
            'write.parquet.dict-size-bytes'='134217728',
            'write.target-file-size-bytes'='268435456',  -- 256MB files
            'write.metadata.metrics.default'='counts',
            'write.metadata.metrics.column.user_id'='counts,lower_bound,upper_bound'
        )
    """)
```

### ✅ DO: Implement Efficient Merge Operations

```python
# iceberg_merge_operations.py
def merge_cdc_updates(spark: SparkSession, source_df, target_table: str):
    """Efficiently merge CDC updates using Iceberg's merge-on-read"""
    
    # Create temporary view for merge source
    source_df.createOrReplaceTempView("updates")
    
    # Iceberg MERGE with optimized conditions
    spark.sql(f"""
        MERGE INTO {target_table} AS target
        USING updates AS source
        ON target.id = source.id 
           AND target.updated_at < source.updated_at
        WHEN MATCHED AND source.op = 'DELETE' THEN
            DELETE
        WHEN MATCHED THEN
            UPDATE SET *
        WHEN NOT MATCHED AND source.op != 'DELETE' THEN
            INSERT *
    """)
    
    # Trigger compaction after merge
    spark.sql(f"""
        CALL iceberg.system.rewrite_data_files(
            table => '{target_table}',
            strategy => 'sort',
            sort_order => 'id,updated_at',
            options => map(
                'target-file-size-bytes', '134217728',
                'min-file-size-bytes', '67108864',
                'max-concurrent-file-group-rewrites', '20'
            )
        )
    """)
```

### ✅ DO: Implement Time Travel for Debugging

```python
# time_travel_queries.py
def analyze_data_changes(spark: SparkSession, table: str, hours_back: int = 24):
    """Compare current state with historical state"""
    
    # Get snapshot IDs
    snapshots = spark.sql(f"""
        SELECT snapshot_id, committed_at 
        FROM iceberg.{table}.snapshots 
        WHERE committed_at >= current_timestamp() - INTERVAL {hours_back} HOURS
        ORDER BY committed_at DESC
    """).collect()
    
    if len(snapshots) >= 2:
        current_snapshot = snapshots[0].snapshot_id
        previous_snapshot = snapshots[1].snapshot_id
        
        # Find changed records
        changes_df = spark.sql(f"""
            WITH current_data AS (
                SELECT * FROM iceberg.{table} VERSION AS OF {current_snapshot}
            ),
            previous_data AS (
                SELECT * FROM iceberg.{table} VERSION AS OF {previous_snapshot}
            )
            SELECT 
                COALESCE(c.id, p.id) as id,
                CASE 
                    WHEN p.id IS NULL THEN 'INSERT'
                    WHEN c.id IS NULL THEN 'DELETE'
                    ELSE 'UPDATE'
                END as change_type,
                c.* as current_record,
                p.* as previous_record
            FROM current_data c
            FULL OUTER JOIN previous_data p ON c.id = p.id
            WHERE c.* IS DISTINCT FROM p.*
        """)
        
        return changes_df
```

### Advanced Iceberg Maintenance

```python
# iceberg_maintenance.py
from datetime import datetime, timedelta
import concurrent.futures

class IcebergTableMaintainer:
    def __init__(self, spark: SparkSession):
        self.spark = spark
    
    def expire_snapshots(self, table: str, days_to_retain: int = 7):
        """Expire old snapshots to reduce metadata size"""
        
        cutoff_time = datetime.now() - timedelta(days=days_to_retain)
        
        self.spark.sql(f"""
            CALL iceberg.system.expire_snapshots(
                table => '{table}',
                older_than => TIMESTAMP '{cutoff_time.isoformat()}',
                retain_last => 3,
                stream_results => true
            )
        """)
    
    def remove_orphan_files(self, table: str):
        """Clean up orphaned data files"""
        
        self.spark.sql(f"""
            CALL iceberg.system.remove_orphan_files(
                table => '{table}',
                dry_run => false,
                max_concurrent_deletes => 10
            )
        """)
    
    def optimize_table_layout(self, table: str):
        """Full table optimization including sorting and z-ordering"""
        
        # Analyze table to determine optimal sort columns
        stats = self.spark.sql(f"""
            SELECT column_name, distinct_count, null_count
            FROM iceberg.{table}.column_stats
            WHERE distinct_count > 1
            ORDER BY distinct_count DESC
            LIMIT 3
        """).collect()
        
        sort_columns = [row.column_name for row in stats]
        
        # Rewrite with z-order for multi-dimensional clustering
        self.spark.sql(f"""
            CALL iceberg.system.rewrite_data_files(
                table => '{table}',
                strategy => 'sort',
                sort_order => 'zorder({",".join(sort_columns)})',
                options => map(
                    'compression-codec', 'zstd',
                    'compression-level', '3'
                )
            )
        """)
```

---

## 4. Stream Processing with Apache Flink 2.0

Flink 2.0 brings improved performance and native Iceberg integration for low-latency stream processing.

### ✅ DO: Use Flink's Native Iceberg Connector

```java
// FlinkIcebergProcessor.java
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.configuration.Configuration;

public class FlinkIcebergProcessor {
    
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        
        // Flink 2.0 improvements
        conf.setString("state.backend", "rocksdb");
        conf.setString("state.backend.incremental", "true");
        conf.setString("execution.checkpointing.interval", "60s");
        conf.setString("execution.checkpointing.mode", "EXACTLY_ONCE");
        
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment(conf);
        StreamTableEnvironment tableEnv = StreamTableEnvironment.create(env);
        
        // Register Iceberg catalog
        tableEnv.executeSql("""
            CREATE CATALOG iceberg_catalog WITH (
                'type' = 'iceberg',
                'catalog-type' = 'rest',
                'uri' = 'http://iceberg-rest:8181',
                'warehouse' = 's3a://data-lake/warehouse',
                's3.endpoint' = 'http://minio:9000',
                's3.access-key' = 'minioadmin',
                's3.secret-key' = 'minioadmin'
            )
        """);
        
        // Create Kafka source with schema registry
        tableEnv.executeSql("""
            CREATE TABLE kafka_events (
                event_id STRING,
                user_id BIGINT,
                event_type STRING,
                event_time TIMESTAMP(3),
                properties MAP<STRING, STRING>,
                WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
            ) WITH (
                'connector' = 'kafka',
                'topic' = 'events',
                'properties.bootstrap.servers' = 'kafka:29092',
                'properties.group.id' = 'flink-processor',
                'scan.startup.mode' = 'latest-offset',
                'format' = 'avro-confluent',
                'avro-confluent.url' = 'http://schema-registry:8081'
            )
        """);
        
        // Process and write to Iceberg with exactly-once semantics
        tableEnv.executeSql("""
            INSERT INTO iceberg_catalog.cleaned.events
            SELECT 
                event_id,
                user_id,
                event_type,
                event_time,
                properties,
                CURRENT_TIMESTAMP as processing_time
            FROM kafka_events
            WHERE event_type IS NOT NULL
        """);
    }
}
```

### ✅ DO: Implement Stateful Stream Processing

```python
# flink_stateful_processing.py
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment
from pyflink.table.udf import udf
from pyflink.table import DataTypes
import json

def create_session_aggregator():
    """Create Flink job for session windowing"""
    
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(4)
    
    # Enable adaptive batch scheduler for better resource usage
    env.get_config().set("jobmanager.scheduler", "adaptive")
    
    t_env = StreamTableEnvironment.create(env)
    
    # Define session detection UDF
    @udf(result_type=DataTypes.STRING())
    def detect_session_metrics(events_json):
        events = json.loads(events_json)
        return json.dumps({
            'session_duration': max(e['ts'] for e in events) - min(e['ts'] for e in events),
            'event_count': len(events),
            'unique_pages': len(set(e.get('page') for e in events))
        })
    
    t_env.create_temporary_function("detect_session_metrics", detect_session_metrics)
    
    # Session window aggregation
    t_env.execute_sql("""
        CREATE TABLE user_sessions AS
        SELECT 
            user_id,
            SESSION_START(event_time, INTERVAL '30' MINUTE) as session_start,
            SESSION_END(event_time, INTERVAL '30' MINUTE) as session_end,
            COUNT(*) as event_count,
            detect_session_metrics(
                JSON_ARRAYAGG(
                    JSON_OBJECT(
                        'ts' VALUE UNIX_TIMESTAMP(event_time),
                        'page' VALUE properties['page_url']
                    )
                )
            ) as session_metrics
        FROM kafka_events
        GROUP BY 
            user_id,
            SESSION(event_time, INTERVAL '30' MINUTE)
    """)
```

### Complex Event Processing (CEP) with Flink

```java
// FraudDetectionCEP.java
import org.apache.flink.cep.CEP;
import org.apache.flink.cep.PatternStream;
import org.apache.flink.cep.pattern.Pattern;
import org.apache.flink.cep.pattern.conditions.SimpleCondition;

public class FraudDetectionCEP {
    
    public static Pattern<Transaction, ?> createFraudPattern() {
        return Pattern.<Transaction>begin("first")
            .where(new SimpleCondition<Transaction>() {
                @Override
                public boolean filter(Transaction transaction) {
                    return transaction.getAmount() < 10;
                }
            })
            .followedBy("second")
            .where(new SimpleCondition<Transaction>() {
                @Override
                public boolean filter(Transaction transaction) {
                    return transaction.getAmount() > 1000;
                }
            })
            .within(Time.minutes(10));
    }
    
    public static void detectFraud(DataStream<Transaction> transactions) {
        Pattern<Transaction, ?> pattern = createFraudPattern();
        PatternStream<Transaction> patternStream = CEP.pattern(
            transactions.keyBy(Transaction::getUserId),
            pattern
        );
        
        DataStream<Alert> alerts = patternStream.process(
            new PatternProcessFunction<Transaction, Alert>() {
                @Override
                public void processMatch(
                    Map<String, List<Transaction>> match,
                    Context ctx,
                    Collector<Alert> out) {
                    
                    Transaction first = match.get("first").get(0);
                    Transaction second = match.get("second").get(0);
                    
                    out.collect(new Alert(
                        first.getUserId(),
                        "Suspicious activity: small transaction followed by large",
                        Alert.Severity.HIGH
                    ));
                }
            }
        );
        
        // Write alerts to Iceberg for investigation
        alerts.addSink(
            IcebergSink.<Alert>forRowData()
                .tableLoader(TableLoader.fromCatalog(...))
                .equalityFieldColumns(Arrays.asList("alert_id"))
                .overwrite(false)
                .build()
        );
    }
}
```

---

## 5. Batch Processing with Spark 4.0

Spark 4.0 introduces significant performance improvements and better integration with modern table formats.

### ✅ DO: Leverage Spark 4.0's Optimizations

```python
# spark4_optimizations.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, window, sum as spark_sum
import pyarrow.compute as pc

def create_optimized_spark_session():
    """Spark 4.0 with all optimizations enabled"""
    
    spark = SparkSession.builder \
        .appName("DataLakeBatchProcessor") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.sql.adaptive.localShuffleReader.enabled", "true") \
        .config("spark.sql.optimizer.runtime.bloomFilter.enabled", "true") \
        .config("spark.sql.optimizer.runtime.bloomFilter.creationSideThreshold", "10MB") \
        .config("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true") \
        .config("spark.sql.parquet.enableVectorizedReader", "true") \
        .config("spark.sql.columnVector.offheap.enabled", "true") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000") \
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "false") \
        .config("spark.sql.optimizer.collapseProjection", "true") \
        .config("spark.sql.optimizer.push.extraPredicates", "true") \
        .config("spark.sql.adaptive.optimizer.excludedRules", "") \
        .getOrCreate()
    
    # Enable GPU acceleration if available
    spark.conf.set("spark.rapids.sql.enabled", "true")
    spark.conf.set("spark.rapids.memory.pinnedPool.size", "2G")
    
    return spark
```

### ✅ DO: Implement Efficient Aggregation Pipelines

```python
# aggregation_pipeline.py
def process_daily_aggregates(spark: SparkSession, date: str):
    """Process daily aggregates with optimal performance"""
    
    # Read from Iceberg with pushdown filters
    events_df = spark.table("iceberg.cleaned.events") \
        .filter(col("event_timestamp").cast("date") == date) \
        .select("user_id", "event_type", "properties", "event_timestamp")
    
    # Cache if reused multiple times
    events_df.createOrReplaceTempView("daily_events")
    spark.sql("CACHE TABLE daily_events")
    
    # User activity aggregates with window functions
    user_stats = spark.sql("""
        WITH user_sessions AS (
            SELECT 
                user_id,
                event_timestamp,
                event_type,
                -- Identify session boundaries (30 min inactivity)
                SUM(CASE 
                    WHEN event_timestamp - LAG(event_timestamp) 
                        OVER (PARTITION BY user_id ORDER BY event_timestamp) 
                        > INTERVAL 30 MINUTES 
                    THEN 1 ELSE 0 
                END) OVER (PARTITION BY user_id ORDER BY event_timestamp) as session_id
            FROM daily_events
        ),
        session_stats AS (
            SELECT 
                user_id,
                session_id,
                MIN(event_timestamp) as session_start,
                MAX(event_timestamp) as session_end,
                COUNT(*) as event_count,
                COUNT(DISTINCT event_type) as unique_event_types,
                -- Use approx_percentile for large datasets
                APPROX_PERCENTILE(
                    UNIX_TIMESTAMP(event_timestamp) - 
                    LAG(UNIX_TIMESTAMP(event_timestamp)) 
                        OVER (PARTITION BY user_id, session_id ORDER BY event_timestamp),
                    0.5
                ) as median_time_between_events
            FROM user_sessions
            GROUP BY user_id, session_id
        )
        SELECT 
            user_id,
            COUNT(DISTINCT session_id) as session_count,
            SUM(event_count) as total_events,
            AVG(UNIX_TIMESTAMP(session_end) - UNIX_TIMESTAMP(session_start)) as avg_session_duration,
            SUM(CASE WHEN event_count = 1 THEN 1 ELSE 0 END) as bounce_sessions,
            PERCENTILE_APPROX(event_count, 0.5) as median_events_per_session
        FROM session_stats
        GROUP BY user_id
    """)
    
    # Write results with automatic file sizing
    user_stats.repartition(10) \
        .sortWithinPartitions("user_id") \
        .write \
        .mode("overwrite") \
        .option("write.target-file-size-bytes", "134217728") \
        .saveAsTable(f"iceberg.curated.user_daily_stats_{date.replace('-', '')}")
    
    # Uncache when done
    spark.sql("UNCACHE TABLE daily_events")
```

### Advanced Spark Patterns

#### Dynamic Partition Overwrite with Iceberg

```python
def upsert_partitioned_data(spark: SparkSession, updates_df, target_table: str):
    """Efficiently overwrite only affected partitions"""
    
    # Enable dynamic partition overwrite
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    
    # For Iceberg, use MERGE for more control
    updates_df.createOrReplaceTempView("updates")
    
    spark.sql(f"""
        MERGE INTO {target_table} AS target
        USING updates AS source
        ON target.id = source.id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)
```

#### Broadcast Joins for Dimension Tables

```python
from pyspark.sql.functions import broadcast

def enrich_with_dimensions(spark: SparkSession, fact_df):
    """Enrich facts with dimension data using broadcast joins"""
    
    # Load dimension tables (assumed to be small)
    users_dim = spark.table("iceberg.curated.dim_users")
    products_dim = spark.table("iceberg.curated.dim_products")
    
    # Explicitly broadcast dimensions
    enriched_df = fact_df \
        .join(broadcast(users_dim), "user_id", "left") \
        .join(broadcast(products_dim), "product_id", "left")
    
    # Alternative: Let Spark decide with hints
    enriched_df_auto = fact_df \
        .hint("broadcast", ["user_id"]) \
        .join(users_dim, "user_id", "left") \
        .hint("broadcast", ["product_id"]) \
        .join(products_dim, "product_id", "left")
    
    return enriched_df
```

---

## 6. SQL Transformations with dbt 1.9

dbt has become the standard for SQL-based transformations, bringing software engineering practices to analytics engineering.

### ✅ DO: Structure dbt Projects for Scale

```
dbt_project/
├── dbt_project.yml
├── profiles.yml
├── models/
│   ├── staging/              # Raw → Cleaned transformations
│   │   ├── _sources.yml      # Source definitions
│   │   ├── stg_events.sql
│   │   └── stg_users.sql
│   ├── intermediate/         # Business logic
│   │   ├── int_user_sessions.sql
│   │   └── int_daily_active_users.sql
│   ├── marts/               # Final business tables
│   │   ├── core/
│   │   │   ├── fct_user_engagement.sql
│   │   │   └── dim_users_scd2.sql
│   │   └── finance/
│   │       └── revenue_analytics.sql
│   └── metrics/            # dbt 1.9 semantic layer
│       └── metric_definitions.yml
├── macros/
│   ├── generate_schema_name.sql
│   └── incremental_predicates.sql
├── tests/
│   └── assert_positive_revenue.sql
└── analyses/
    └── adhoc_investigations.sql
```

### ✅ DO: Configure dbt for Iceberg/Spark

```yaml
# dbt_project.yml
name: 'data_lake_transformations'
version: '1.0.0'

profile: 'data_lake'

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

models:
  data_lake_transformations:
    # Global configurations
    +materialized: table
    +on_schema_change: "sync_all_columns"
    +persist_docs:
      relation: true
      columns: true
    
    staging:
      +materialized: view
      +tags: ['staging']
    
    intermediate:
      +materialized: incremental
      +on_schema_change: "append_new_columns"
      +incremental_strategy: merge
      +unique_key: ['id', 'updated_at']
      +merge_exclude_columns: ['inserted_at']
    
    marts:
      core:
        +materialized: incremental
        +partition_by:
          field: date_day
          data_type: date
          granularity: day
        +cluster_by: ['user_id']
        +incremental_strategy: insert_overwrite
        +partitions_to_replace: ["{{ var('start_date') }}"]
```

### dbt Profiles for Spark/Iceberg

```yaml
# profiles.yml
data_lake:
  outputs:
    dev:
      type: spark
      method: thrift
      host: spark-thrift-server
      port: 10000
      user: dbt_user
      schema: dbt_dev
      threads: 4
      
    prod:
      type: spark
      method: odbc
      driver: 'Simba Spark ODBC Driver'
      host: spark-prod.company.com
      port: 443
      token: "{{ env_var('DBT_SPARK_TOKEN') }}"
      http_path: '/sql/1.0/warehouses/prod'
      schema: analytics
      threads: 8
      
  target: dev
```

### ✅ DO: Implement Incremental Models Correctly

```sql
-- models/intermediate/int_user_sessions.sql
{{
    config(
        materialized='incremental',
        unique_key=['user_id', 'session_id'],
        on_schema_change='sync_all_columns',
        partition_by={
            'field': 'session_date',
            'data_type': 'date',
            'granularity': 'day'
        },
        incremental_strategy='merge',
        incremental_predicates=[
            "DBT_INTERNAL_DEST.session_date >= date_sub(current_date(), 3)"
        ],
        pre_hook="{{ log_model_start() }}",
        post_hook="ANALYZE TABLE {{ this }} COMPUTE STATISTICS"
    )
}}

WITH events AS (
    SELECT * FROM {{ ref('stg_events') }}
    
    {% if is_incremental() %}
        -- Look back 1 day to catch late-arriving events
        WHERE event_timestamp >= (
            SELECT DATE_SUB(MAX(session_date), 1) 
            FROM {{ this }}
        )
    {% endif %}
),

session_boundaries AS (
    SELECT 
        user_id,
        event_id,
        event_timestamp,
        event_type,
        properties,
        -- Session boundary detection
        CASE 
            WHEN LAG(event_timestamp) OVER (
                PARTITION BY user_id 
                ORDER BY event_timestamp
            ) IS NULL 
            OR event_timestamp - LAG(event_timestamp) OVER (
                PARTITION BY user_id 
                ORDER BY event_timestamp
            ) > INTERVAL 30 MINUTES 
            THEN 1 
            ELSE 0 
        END AS is_new_session
    FROM events
),

sessions_identified AS (
    SELECT 
        *,
        SUM(is_new_session) OVER (
            PARTITION BY user_id 
            ORDER BY event_timestamp 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS session_sequence
    FROM session_boundaries
)

SELECT 
    user_id,
    MD5(CONCAT(user_id, '::', session_sequence)) AS session_id,
    DATE(MIN(event_timestamp)) AS session_date,
    MIN(event_timestamp) AS session_start,
    MAX(event_timestamp) AS session_end,
    COUNT(*) AS event_count,
    COUNT(DISTINCT event_type) AS unique_event_types,
    
    -- Session metrics
    TIMESTAMPDIFF(
        SECOND,
        MIN(event_timestamp),
        MAX(event_timestamp)
    ) AS session_duration_seconds,
    
    -- Entry and exit pages
    FIRST_VALUE(properties['page_url']) OVER (
        PARTITION BY user_id, session_sequence 
        ORDER BY event_timestamp
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS entry_page,
    
    LAST_VALUE(properties['page_url']) OVER (
        PARTITION BY user_id, session_sequence 
        ORDER BY event_timestamp
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS exit_page,
    
    -- Aggregated properties as JSON
    TO_JSON(
        OBJECT_AGG(
            event_type,
            COUNT(*)
        )
    ) AS event_type_counts,
    
    CURRENT_TIMESTAMP() AS dbt_updated_at

FROM sessions_identified
GROUP BY 
    user_id,
    session_sequence
```

### Advanced dbt Patterns

#### Generic Tests with Custom Thresholds

```sql
-- tests/generic/test_anomaly_detection.sql
{% test anomaly_detection(model, column_name, 
                         metric='avg', 
                         lookback_days=7, 
                         stddev_threshold=3) %}

WITH historical_data AS (
    SELECT 
        DATE({{ column_name }}) as date_day,
        {{ metric }}({{ column_name }}) as daily_metric
    FROM {{ model }}
    WHERE {{ column_name }} >= CURRENT_DATE - INTERVAL {{ lookback_days }} DAY
    GROUP BY 1
),

stats AS (
    SELECT 
        AVG(daily_metric) as mean_value,
        STDDEV(daily_metric) as stddev_value
    FROM historical_data
    WHERE date_day < CURRENT_DATE
),

today_data AS (
    SELECT 
        {{ metric }}({{ column_name }}) as today_metric
    FROM {{ model }}
    WHERE DATE({{ column_name }}) = CURRENT_DATE
)

SELECT 
    today_metric,
    mean_value,
    stddev_value,
    ABS(today_metric - mean_value) / NULLIF(stddev_value, 0) as z_score
FROM today_data
CROSS JOIN stats
WHERE ABS(today_metric - mean_value) > {{ stddev_threshold }} * stddev_value

{% endtest %}
```

#### dbt Macros for Iceberg Operations

```sql
-- macros/iceberg_maintenance.sql
{% macro run_iceberg_maintenance(table_name, operation='expire_snapshots') %}
    {% set query %}
        {% if operation == 'expire_snapshots' %}
            CALL iceberg.system.expire_snapshots(
                table => '{{ table_name }}',
                older_than => TIMESTAMP '{{ (modules.datetime.datetime.now() - modules.datetime.timedelta(days=7)).isoformat() }}',
                retain_last => 3
            )
        {% elif operation == 'compact' %}
            CALL iceberg.system.rewrite_data_files(
                table => '{{ table_name }}',
                strategy => 'binpack',
                options => map(
                    'target-file-size-bytes', '134217728',
                    'min-file-size-bytes', '33554432'
                )
            )
        {% endif %}
    {% endset %}
    
    {% do run_query(query) %}
    {{ log("Maintenance operation " ~ operation ~ " completed for " ~ table_name, info=True) }}
{% endmacro %}

-- Usage in post-hook
{{ config(
    post_hook="{{ run_iceberg_maintenance(this, 'compact') }}"
) }}
```

#### Semantic Layer with dbt Metrics

```yaml
# models/metrics/metric_definitions.yml
version: 2

metrics:
  - name: daily_active_users
    label: Daily Active Users (DAU)
    model: ref('fct_user_engagement')
    description: "Count of unique users active on a given day"
    
    type: count_distinct
    sql: user_id
    
    timestamp: event_date
    time_grains: [day, week, month, quarter, year]
    
    dimensions:
      - platform
      - country
      - user_segment
    
    filters:
      - field: is_active
        operator: '='
        value: 'true'
    
    meta:
      owner: '@analytics-team'
      tier: 1
      
  - name: user_retention_rate
    label: User Retention Rate
    model: ref('fct_user_engagement')
    description: "Percentage of users who return after first use"
    
    type: derived
    sql: |
      CAST(COUNT(DISTINCT CASE 
        WHEN days_since_first_use > 0 THEN user_id 
      END) AS FLOAT) / NULLIF(COUNT(DISTINCT user_id), 0)
    
    timestamp: event_date
    time_grains: [day, week, month]
    
    dimensions:
      - acquisition_channel
      - user_segment
```

---

## 7. Local Analytics with DuckDB

DuckDB serves as the "SQLite for analytics," enabling rapid local analysis of cloud data.

### ✅ DO: Use DuckDB for Development and Testing

```python
# duckdb_local_analytics.py
import duckdb
import pyarrow.parquet as pq
from typing import Optional

class LocalDataLakeAnalyzer:
    def __init__(self, s3_endpoint: Optional[str] = None):
        self.conn = duckdb.connect(':memory:')
        
        # Configure S3 access
        if s3_endpoint:
            self.conn.execute(f"""
                SET s3_endpoint='{s3_endpoint}';
                SET s3_access_key_id='minioadmin';
                SET s3_secret_access_key='minioadmin';
                SET s3_use_ssl=false;
                SET s3_url_style='path';
            """)
        
        # Install and load extensions
        self.conn.execute("INSTALL iceberg; LOAD iceberg;")
        self.conn.execute("INSTALL httpfs; LOAD httpfs;")
        self.conn.execute("INSTALL aws; LOAD aws;")
        
    def analyze_iceberg_table(self, table_path: str):
        """Analyze Iceberg table directly from S3"""
        
        # Create view from Iceberg table
        self.conn.execute(f"""
            CREATE VIEW events AS 
            SELECT * FROM iceberg_scan('{table_path}')
        """)
        
        # Run analytical queries with DuckDB's columnar engine
        results = self.conn.execute("""
            WITH user_cohorts AS (
                SELECT 
                    user_id,
                    DATE_TRUNC('month', MIN(event_timestamp)) as cohort_month,
                    COUNT(DISTINCT DATE_TRUNC('day', event_timestamp)) as active_days
                FROM events
                GROUP BY user_id
            ),
            cohort_summary AS (
                SELECT 
                    cohort_month,
                    COUNT(DISTINCT user_id) as cohort_size,
                    AVG(active_days) as avg_active_days,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY active_days) as median_active_days,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY active_days) as p95_active_days
                FROM user_cohorts
                GROUP BY cohort_month
            )
            SELECT * FROM cohort_summary
            ORDER BY cohort_month DESC
        """).fetchdf()
        
        return results
    
    def export_sample_data(self, table_path: str, sample_size: int = 10000):
        """Export sample data for local development"""
        
        sample_df = self.conn.execute(f"""
            SELECT * FROM iceberg_scan('{table_path}')
            USING SAMPLE {sample_size}
        """).fetchdf()
        
        # Save as Parquet for fast local access
        sample_df.to_parquet('sample_data.parquet', compression='zstd')
        
        # Also create a DuckDB database file
        file_conn = duckdb.connect('sample_data.duckdb')
        file_conn.execute("CREATE TABLE events AS SELECT * FROM sample_df")
        file_conn.close()
```

### DuckDB for CI/CD Testing

```python
# test_transformations.py
import duckdb
import pytest
from pathlib import Path

class TestDataTransformations:
    @pytest.fixture
    def test_db(self):
        """Create test database with sample data"""
        conn = duckdb.connect(':memory:')
        
        # Load test data
        conn.execute("""
            CREATE TABLE raw_events AS 
            SELECT * FROM read_parquet('tests/fixtures/events.parquet')
        """)
        
        yield conn
        conn.close()
    
    def test_session_logic(self, test_db):
        """Test session identification logic using DuckDB"""
        
        # Run the transformation
        test_db.execute("""
            CREATE VIEW sessions AS
            WITH session_boundaries AS (
                SELECT *,
                    LAG(event_timestamp) OVER (
                        PARTITION BY user_id 
                        ORDER BY event_timestamp
                    ) as prev_timestamp
                FROM raw_events
            )
            SELECT 
                user_id,
                event_timestamp,
                CASE 
                    WHEN prev_timestamp IS NULL 
                        OR event_timestamp - prev_timestamp > INTERVAL '30 minutes'
                    THEN 1 ELSE 0 
                END as is_new_session
            FROM session_boundaries
        """)
        
        # Verify results
        result = test_db.execute("""
            SELECT COUNT(*) as session_count
            FROM sessions
            WHERE is_new_session = 1
        """).fetchone()
        
        assert result[0] > 0, "Should identify at least one session"
```

---

## 8. Data Warehouse Integration

Connecting your data lake to Snowflake or BigQuery for broader analytics consumption.

### ✅ DO: Implement Efficient Sync Patterns

#### Snowflake Integration via External Tables

```sql
-- snowflake_iceberg_integration.sql
-- Create storage integration (one-time setup by admin)
CREATE STORAGE INTEGRATION s3_data_lake
    TYPE = EXTERNAL_STAGE
    STORAGE_PROVIDER = 'S3'
    ENABLED = TRUE
    STORAGE_AWS_ROLE_ARN = 'arn:aws:iam::123456789:role/SnowflakeAccess'
    STORAGE_ALLOWED_LOCATIONS = ('s3://data-lake/');

-- Create external stage
CREATE OR REPLACE STAGE data_lake_stage
    STORAGE_INTEGRATION = s3_data_lake
    URL = 's3://data-lake/curated/'
    FILE_FORMAT = (TYPE = PARQUET COMPRESSION = SNAPPY);

-- Create Iceberg external table
CREATE OR REPLACE EXTERNAL TABLE user_engagement_iceberg
    USING TEMPLATE (
        SELECT OBJECT_CONSTRUCT(*)
        FROM TABLE(
            INFER_SCHEMA(
                LOCATION => '@data_lake_stage/user_engagement',
                FILE_FORMAT => 'PARQUET'
            )
        )
    )
    LOCATION = @data_lake_stage/user_engagement
    FILE_FORMAT = (TYPE = PARQUET)
    TABLE_FORMAT = 'ICEBERG'
    EXTERNAL_VOLUME = 'data_lake_volume';

-- Create automated refresh
CREATE OR REPLACE TASK refresh_external_tables
    WAREHOUSE = COMPUTE_WH
    SCHEDULE = 'USING CRON 0 */2 * * * UTC'  -- Every 2 hours
AS
    ALTER EXTERNAL TABLE user_engagement_iceberg REFRESH;
```

#### BigQuery Integration

```python
# bigquery_sync.py
from google.cloud import bigquery
from google.cloud import storage
import pyiceberg

class BigQueryDataLakeSync:
    def __init__(self, project_id: str, dataset_id: str):
        self.bq_client = bigquery.Client(project=project_id)
        self.storage_client = storage.Client()
        self.dataset_id = dataset_id
        
    def create_external_table(self, 
                            iceberg_table_path: str, 
                            bq_table_name: str):
        """Create BigQuery external table from Iceberg"""
        
        # Configure external table
        external_config = bigquery.ExternalConfig("PARQUET")
        external_config.source_uris = [
            f"{iceberg_table_path}/data/*.parquet"
        ]
        external_config.autodetect = True
        
        # Create table with partitioning
        table = bigquery.Table(f"{self.dataset_id}.{bq_table_name}")
        table.external_data_configuration = external_config
        
        # Set up clustering for performance
        table.clustering_fields = ["user_id", "event_date"]
        
        # Create the table
        table = self.bq_client.create_table(table)
        
        # Create materialized view for better performance
        query = f"""
        CREATE MATERIALIZED VIEW `{self.dataset_id}.{bq_table_name}_mv`
        PARTITION BY DATE(event_timestamp)
        CLUSTER BY user_id, event_type
        AS
        SELECT 
            *,
            DATE(event_timestamp) as event_date
        FROM `{self.dataset_id}.{bq_table_name}`
        WHERE event_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
        """
        
        self.bq_client.query(query).result()
    
    def sync_incremental_data(self, 
                            source_table: str, 
                            target_table: str,
                            watermark_column: str = "updated_at"):
        """Incrementally sync data to BigQuery"""
        
        # Get last sync timestamp
        query = f"""
        SELECT MAX({watermark_column}) as last_sync
        FROM `{self.dataset_id}.{target_table}`
        """
        
        last_sync = list(self.bq_client.query(query))[0].last_sync
        
        # Read incremental data from Iceberg
        # (Implementation depends on your Iceberg reader)
        incremental_data = read_iceberg_incremental(
            source_table, 
            watermark_column, 
            last_sync
        )
        
        # Load to BigQuery
        job_config = bigquery.LoadJobConfig(
            schema_update_options=[
                bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
            ],
            write_disposition="WRITE_APPEND",
            time_partitioning=bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field=watermark_column
            ),
            clustering_fields=["user_id"]
        )
        
        job = self.bq_client.load_table_from_dataframe(
            incremental_data, 
            f"{self.dataset_id}.{target_table}",
            job_config=job_config
        )
        
        job.result()  # Wait for completion
```

---

## 9. Security and Governance

Implementing comprehensive security across all layers of the data lake.

### ✅ DO: Implement Column-Level Encryption

```python
# encryption_handler.py
from cryptography.fernet import Fernet
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
import os

class DataLakeEncryption:
    def __init__(self):
        # Use AWS KMS or similar in production
        self.key = os.environ.get("ENCRYPTION_KEY").encode()
        self.cipher = Fernet(self.key)
    
    def encrypt_sensitive_columns(self, df, columns: list):
        """Encrypt specified columns before writing to data lake"""
        
        # Create UDF for encryption
        encrypt_udf = udf(
            lambda x: self.cipher.encrypt(str(x).encode()).decode() if x else None,
            StringType()
        )
        
        # Apply encryption
        for column in columns:
            df = df.withColumn(
                f"{column}_encrypted",
                encrypt_udf(col(column))
            ).drop(column)
        
        return df
    
    def create_encrypted_view(self, spark, table_name: str, 
                            encrypted_columns: list):
        """Create view that automatically decrypts data"""
        
        decrypt_udf = spark.udf.register(
            "decrypt_value",
            lambda x: self.cipher.decrypt(x.encode()).decode() if x else None,
            StringType()
        )
        
        # Build SELECT statement
        select_cols = []
        for col_name in spark.table(table_name).columns:
            if col_name.endswith("_encrypted"):
                original_name = col_name.replace("_encrypted", "")
                if original_name in encrypted_columns:
                    select_cols.append(
                        f"decrypt_value({col_name}) as {original_name}"
                    )
            else:
                select_cols.append(col_name)
        
        # Create secure view
        spark.sql(f"""
            CREATE OR REPLACE VIEW {table_name}_decrypted AS
            SELECT {', '.join(select_cols)}
            FROM {table_name}
        """)
```

### ✅ DO: Implement Row-Level Security with Iceberg

```python
# row_level_security.py
def apply_row_level_security(spark, table_name: str, user_context: dict):
    """Apply RLS based on user context"""
    
    # Get user's data access permissions
    user_id = user_context.get("user_id")
    user_role = user_context.get("role")
    allowed_regions = user_context.get("allowed_regions", [])
    
    # Build filter conditions based on role
    filters = []
    
    if user_role == "analyst":
        # Analysts can only see aggregated data
        filters.append("aggregation_level = 'daily'")
        
    if user_role == "regional_manager":
        # Regional managers see only their regions
        region_filter = " OR ".join([
            f"region = '{region}'" for region in allowed_regions
        ])
        filters.append(f"({region_filter})")
    
    if user_role not in ["admin", "data_engineer"]:
        # Non-admins cannot see PII
        filters.append("contains_pii = false")
    
    # Apply filters
    where_clause = " AND ".join(filters) if filters else "1=1"
    
    return spark.sql(f"""
        SELECT * FROM {table_name}
        WHERE {where_clause}
    """)
```

### Data Quality Monitoring

```python
# data_quality_monitor.py
from pyspark.sql import DataFrame
from typing import Dict, List
import great_expectations as gx

class DataQualityMonitor:
    def __init__(self, spark):
        self.spark = spark
        self.context = gx.get_context()
    
    def create_quality_suite(self, table_name: str):
        """Create Great Expectations suite for table"""
        
        # Add data source
        datasource = self.context.sources.add_spark(
            "data_lake_source"
        )
        
        # Define expectations
        suite = self.context.create_expectation_suite(
            expectation_suite_name=f"{table_name}_quality_suite"
        )
        
        # Common expectations
        expectations = [
            {
                "expectation_type": "expect_table_row_count_to_be_between",
                "kwargs": {"min_value": 1000, "max_value": 1000000000}
            },
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "user_id"}
            },
            {
                "expectation_type": "expect_column_values_to_be_unique",
                "kwargs": {"column": "event_id"}
            }
        ]
        
        for exp in expectations:
            suite.add_expectation(
                expectation_configuration=exp
            )
        
        return suite
    
    def run_quality_checks(self, df: DataFrame, suite_name: str) -> Dict:
        """Run quality checks on DataFrame"""
        
        # Convert to GX batch
        batch = self.context.get_batch(
            batch_request={
                "datasource_name": "data_lake_source",
                "data_asset_name": "runtime_data",
                "batch_spec_passthrough": {
                    "reader_method": "spark",
                    "reader_options": {"dataframe": df}
                }
            }
        )
        
        # Run validation
        results = self.context.run_validation_operator(
            "action_list_operator",
            assets_to_validate=[(batch, suite_name)]
        )
        
        # Log results to monitoring system
        self._log_results(results)
        
        return results
    
    def _log_results(self, results):
        """Log validation results for monitoring"""
        # Implementation depends on your monitoring system
        pass
```

---

## 10. Monitoring and Observability

Comprehensive monitoring across all data lake components.

### ✅ DO: Implement Unified Metrics Collection

```python
# monitoring_setup.py
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from functools import wraps
import time

# Metrics registry
registry = CollectorRegistry()

# Define metrics
kafka_messages_processed = Counter(
    'kafka_messages_processed_total',
    'Total Kafka messages processed',
    ['topic', 'status'],
    registry=registry
)

iceberg_write_duration = Histogram(
    'iceberg_write_duration_seconds',
    'Iceberg write operation duration',
    ['table', 'operation'],
    registry=registry
)

data_lake_storage_bytes = Gauge(
    'data_lake_storage_bytes',
    'Total storage used in data lake',
    ['zone', 'format'],
    registry=registry
)

def monitor_kafka_processing(topic: str):
    """Decorator to monitor Kafka message processing"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                kafka_messages_processed.labels(
                    topic=topic, 
                    status='success'
                ).inc()
                return result
            except Exception as e:
                kafka_messages_processed.labels(
                    topic=topic, 
                    status='error'
                ).inc()
                raise e
        return wrapper
    return decorator

def monitor_iceberg_operation(table: str, operation: str):
    """Monitor Iceberg table operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with iceberg_write_duration.labels(
                table=table,
                operation=operation
            ).time():
                return func(*args, **kwargs)
        return wrapper
    return decorator
```

### Data Lineage Tracking

```python
# lineage_tracker.py
from typing import Dict, List
import networkx as nx
from datetime import datetime

class DataLineageTracker:
    def __init__(self, metadata_store_uri: str):
        self.graph = nx.DiGraph()
        self.metadata_store = metadata_store_uri
    
    def track_transformation(self, 
                           job_name: str,
                           input_tables: List[str],
                           output_tables: List[str],
                           transformation_sql: str = None):
        """Track data transformation lineage"""
        
        job_node = f"job:{job_name}:{datetime.now().isoformat()}"
        
        # Add job node
        self.graph.add_node(job_node, {
            "type": "transformation",
            "timestamp": datetime.now().isoformat(),
            "sql": transformation_sql
        })
        
        # Connect inputs to job
        for input_table in input_tables:
            self.graph.add_edge(input_table, job_node, {
                "relationship": "input_to"
            })
        
        # Connect job to outputs
        for output_table in output_tables:
            self.graph.add_edge(job_node, output_table, {
                "relationship": "output_from"
            })
        
        # Persist to metadata store
        self._persist_lineage()
    
    def get_upstream_lineage(self, table_name: str, depth: int = 3):
        """Get upstream dependencies for a table"""
        
        ancestors = nx.ancestors(self.graph, table_name)
        subgraph = self.graph.subgraph(ancestors | {table_name})
        
        return {
            "table": table_name,
            "upstream_tables": list(ancestors),
            "lineage_graph": nx.node_link_data(subgraph)
        }
    
    def _persist_lineage(self):
        """Save lineage to persistent store"""
        # Implementation depends on your metadata store
        pass
```

### Cost Monitoring

```python
# cost_monitor.py
import boto3
from datetime import datetime, timedelta

class DataLakeCostMonitor:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.cloudwatch = boto3.client('cloudwatch')
        
    def calculate_storage_costs(self, bucket_name: str):
        """Calculate S3 storage costs by tier"""
        
        # Get storage metrics
        response = self.cloudwatch.get_metric_statistics(
            Namespace='AWS/S3',
            MetricName='BucketSizeBytes',
            Dimensions=[
                {'Name': 'BucketName', 'Value': bucket_name},
                {'Name': 'StorageType', 'Value': 'StandardStorage'}
            ],
            StartTime=datetime.now() - timedelta(days=1),
            EndTime=datetime.now(),
            Period=86400,
            Statistics=['Average']
        )
        
        # Calculate costs (simplified)
        storage_gb = response['Datapoints'][0]['Average'] / (1024**3)
        standard_cost = storage_gb * 0.023  # $/GB/month
        
        # Intelligent tiering analysis
        lifecycle_savings = self._calculate_lifecycle_savings(
            bucket_name, 
            storage_gb
        )
        
        return {
            "current_storage_gb": storage_gb,
            "monthly_cost": standard_cost,
            "potential_savings": lifecycle_savings
        }
    
    def _calculate_lifecycle_savings(self, bucket_name: str, 
                                   total_gb: float):
        """Calculate potential savings from lifecycle policies"""
        
        # Analyze access patterns
        # (Implementation depends on S3 Inventory)
        
        # Example calculation
        infrequent_access_gb = total_gb * 0.3  # 30% rarely accessed
        glacier_eligible_gb = total_gb * 0.1   # 10% archive-ready
        
        ia_savings = infrequent_access_gb * (0.023 - 0.0125)
        glacier_savings = glacier_eligible_gb * (0.023 - 0.004)
        
        return {
            "infrequent_access_savings": ia_savings,
            "glacier_savings": glacier_savings,
            "total_monthly_savings": ia_savings + glacier_savings
        }
```

---

## 11. Production Deployment Architecture

Deploying a production data lake requires careful orchestration of all components.

### ✅ DO: Use Infrastructure as Code

```hcl
# terraform/data_lake.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

# S3 Buckets with lifecycle policies
resource "aws_s3_bucket" "data_lake" {
  for_each = toset(["raw", "cleaned", "curated"])
  
  bucket = "company-data-lake-${each.key}"
  
  tags = {
    Environment = "production"
    Zone        = each.key
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "data_lake_lifecycle" {
  for_each = aws_s3_bucket.data_lake
  
  bucket = each.value.id
  
  rule {
    id     = "transition-to-ia"
    status = "Enabled"
    
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
    
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
    
    noncurrent_version_transition {
      noncurrent_days = 7
      storage_class   = "GLACIER"
    }
  }
}

# EKS Cluster for Spark/Flink
resource "aws_eks_cluster" "data_processing" {
  name     = "data-lake-processing"
  role_arn = aws_iam_role.eks_cluster.arn
  
  vpc_config {
    subnet_ids = aws_subnet.private[*].id
    
    endpoint_private_access = true
    endpoint_public_access  = false
  }
  
  enabled_cluster_log_types = [
    "api", "audit", "authenticator", "controllerManager", "scheduler"
  ]
}

# Kafka on Kubernetes
resource "kubernetes_namespace" "kafka" {
  metadata {
    name = "kafka"
  }
}

resource "helm_release" "strimzi_operator" {
  name       = "strimzi"
  repository = "https://strimzi.io/charts/"
  chart      = "strimzi-kafka-operator"
  namespace  = kubernetes_namespace.kafka.metadata[0].name
  
  set {
    name  = "replicas"
    value = "1"
  }
}
```

### Kubernetes Configurations

```yaml
# k8s/kafka-cluster.yaml
apiVersion: kafka.strimzi.io/v1beta2
kind: Kafka
metadata:
  name: data-lake-kafka
  namespace: kafka
spec:
  kafka:
    version: 3.8.0
    replicas: 3
    listeners:
      - name: plain
        port: 9092
        type: internal
        tls: false
      - name: tls
        port: 9093
        type: internal
        tls: true
        authentication:
          type: tls
    config:
      # Optimized for throughput
      num.network.threads: 8
      num.io.threads: 8
      socket.send.buffer.bytes: 1048576
      socket.receive.buffer.bytes: 1048576
      socket.request.max.bytes: 104857600
      
      # Log retention
      log.retention.hours: 72
      log.segment.bytes: 1073741824
      
      # Replication
      min.insync.replicas: 2
      default.replication.factor: 3
      
      # Compression
      compression.type: zstd
      
    storage:
      type: jbod
      volumes:
      - id: 0
        type: persistent-claim
        size: 500Gi
        class: gp3-high-iops
        
    metricsConfig:
      type: jmxPrometheusExporter
      valueFrom:
        configMapKeyRef:
          name: kafka-metrics
          key: kafka-metrics-config.yml
          
  zookeeper:
    # Note: Kafka 3.8 can run without Zookeeper in KRaft mode
    replicas: 0
    
  entityOperator:
    topicOperator:
      watchedNamespace: kafka
    userOperator:
      watchedNamespace: kafka
      
  kafkaExporter:
    topicRegex: ".*"
    groupRegex: ".*"
```

### Production Spark Configuration

```yaml
# k8s/spark-operator.yaml
apiVersion: sparkoperator.k8s.io/v1beta2
kind: SparkApplication
metadata:
  name: data-lake-processor
  namespace: spark
spec:
  type: Scala
  mode: cluster
  image: "company/spark-iceberg:4.0.0"
  imagePullPolicy: Always
  mainClass: com.company.DataLakeProcessor
  mainApplicationFile: "s3a://artifacts/data-lake-processor.jar"
  
  sparkVersion: "4.0.0"
  restartPolicy:
    type: OnFailure
    onFailureRetries: 3
    
  sparkConf:
    # Iceberg configuration
    spark.sql.catalog.iceberg: org.apache.iceberg.spark.SparkCatalog
    spark.sql.catalog.iceberg.type: rest
    spark.sql.catalog.iceberg.uri: http://iceberg-rest:8181
    
    # Performance optimizations
    spark.sql.adaptive.enabled: "true"
    spark.sql.adaptive.coalescePartitions.enabled: "true"
    spark.sql.adaptive.skewJoin.enabled: "true"
    spark.serializer: org.apache.spark.serializer.KryoSerializer
    spark.sql.execution.arrow.pyspark.enabled: "true"
    
    # S3 configuration
    spark.hadoop.fs.s3a.impl: org.apache.hadoop.fs.s3a.S3AFileSystem
    spark.hadoop.fs.s3a.aws.credentials.provider: com.amazonaws.auth.WebIdentityTokenCredentialsProvider
    
  driver:
    cores: 2
    memory: "4g"
    serviceAccount: spark-driver
    env:
      - name: AWS_REGION
        value: us-east-1
        
  executor:
    cores: 4
    instances: 10
    memory: "8g"
    memoryOverhead: "1g"
    
  dynamicAllocation:
    enabled: true
    initialExecutors: 5
    minExecutors: 2
    maxExecutors: 50
    
  monitoring:
    exposeDriverMetrics: true
    exposeExecutorMetrics: true
    prometheus:
      jmxExporterJar: "/prometheus/jmx_prometheus_javaagent-0.16.1.jar"
      port: 8090
```

---

## 12. Performance Optimization Strategies

### ✅ DO: Implement Smart Compaction

```python
# smart_compaction.py
from typing import Dict, List
import math

class SmartCompactionStrategy:
    def __init__(self, spark):
        self.spark = spark
        self.target_file_size = 256 * 1024 * 1024  # 256MB
        
    def analyze_table_for_compaction(self, table_name: str) -> Dict:
        """Analyze table to determine compaction needs"""
        
        file_stats = self.spark.sql(f"""
            SELECT 
                file_path,
                file_size_in_bytes,
                record_count,
                partition
            FROM {table_name}.files
        """).collect()
        
        # Group by partition
        partition_stats = {}
        for file in file_stats:
            partition = file.partition
            if partition not in partition_stats:
                partition_stats[partition] = {
                    'files': [],
                    'total_size': 0,
                    'file_count': 0,
                    'small_files': 0
                }
            
            partition_stats[partition]['files'].append({
                'path': file.file_path,
                'size': file.file_size_in_bytes
            })
            partition_stats[partition]['total_size'] += file.file_size_in_bytes
            partition_stats[partition]['file_count'] += 1
            
            if file.file_size_in_bytes < self.target_file_size * 0.75:
                partition_stats[partition]['small_files'] += 1
        
        # Identify partitions needing compaction
        compaction_candidates = []
        for partition, stats in partition_stats.items():
            small_file_ratio = stats['small_files'] / stats['file_count']
            
            if small_file_ratio > 0.5 or stats['file_count'] > 100:
                optimal_file_count = math.ceil(
                    stats['total_size'] / self.target_file_size
                )
                
                compaction_candidates.append({
                    'partition': partition,
                    'current_files': stats['file_count'],
                    'target_files': optimal_file_count,
                    'size_gb': stats['total_size'] / (1024**3),
                    'priority': small_file_ratio
                })
        
        return sorted(
            compaction_candidates, 
            key=lambda x: x['priority'], 
            reverse=True
        )
    
    def execute_compaction(self, table_name: str, 
                         partitions: List[str] = None):
        """Execute compaction on specified partitions"""
        
        if partitions:
            filter_expr = f"WHERE partition IN ({','.join(partitions)})"
        else:
            filter_expr = ""
            
        self.spark.sql(f"""
            CALL iceberg.system.rewrite_data_files(
                table => '{table_name}',
                strategy => 'binpack',
                sort_order => 'none',
                where => '{filter_expr}',
                options => map(
                    'target-file-size-bytes', '{self.target_file_size}',
                    'min-file-size-bytes', '{int(self.target_file_size * 0.75)}',
                    'max-file-group-size-bytes', '{self.target_file_size * 10}',
                    'partial-progress.enabled', 'true',
                    'max-concurrent-file-group-rewrites', '10'
                )
            )
        """)
```

### Query Optimization Patterns

```python
# query_optimizer.py
class QueryOptimizer:
    @staticmethod
    def optimize_join_order(spark, query: str):
        """Analyze and optimize join order"""
        
        # Enable cost-based optimization
        spark.conf.set("spark.sql.cbo.enabled", "true")
        spark.conf.set("spark.sql.cbo.joinReorder.enabled", "true")
        
        # Collect statistics for all tables in query
        tables = extract_tables_from_query(query)
        for table in tables:
            spark.sql(f"ANALYZE TABLE {table} COMPUTE STATISTICS")
            spark.sql(f"""
                ANALYZE TABLE {table} 
                COMPUTE STATISTICS FOR ALL COLUMNS
            """)
        
        # Create optimized plan
        return spark.sql(f"EXPLAIN COST {query}")
    
    @staticmethod 
    def add_bloom_filters(df, join_columns: List[str]):
        """Add bloom filters for join optimization"""
        
        for col in join_columns:
            df = df.repartition(200, col) \
                   .sortWithinPartitions(col)
        
        # Bloom filter will be automatically created by Spark 4.0
        return df
```

---

## 13. Disaster Recovery and High Availability

### ✅ DO: Implement Cross-Region Replication

```python
# disaster_recovery.py
import boto3
from concurrent.futures import ThreadPoolExecutor
import hashlib

class DataLakeReplication:
    def __init__(self, primary_region: str, dr_region: str):
        self.primary_s3 = boto3.client('s3', region_name=primary_region)
        self.dr_s3 = boto3.client('s3', region_name=dr_region)
        
    def setup_cross_region_replication(self, bucket_name: str):
        """Configure S3 cross-region replication"""
        
        # Create replication configuration
        replication_config = {
            'Role': 'arn:aws:iam::123456789:role/S3ReplicationRole',
            'Rules': [{
                'ID': 'ReplicateAll',
                'Status': 'Enabled',
                'Priority': 1,
                'Filter': {},
                'DeleteMarkerReplication': {'Status': 'Enabled'},
                'Destination': {
                    'Bucket': f'arn:aws:s3:::{bucket_name}-dr',
                    'ReplicationTime': {
                        'Status': 'Enabled',
                        'Time': {'Minutes': 15}
                    },
                    'Metrics': {
                        'Status': 'Enabled',
                        'EventThreshold': {'Minutes': 15}
                    },
                    'StorageClass': 'STANDARD_IA'
                }
            }]
        }
        
        self.primary_s3.put_bucket_replication(
            Bucket=bucket_name,
            ReplicationConfiguration=replication_config
        )
    
    def verify_replication_status(self, bucket_name: str):
        """Monitor replication lag and status"""
        
        # Get replication metrics
        cloudwatch = boto3.client('cloudwatch')
        
        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/S3',
            MetricName='ReplicationLatency',
            Dimensions=[
                {'Name': 'SourceBucket', 'Value': bucket_name},
                {'Name': 'DestinationBucket', 'Value': f'{bucket_name}-dr'}
            ],
            StartTime=datetime.now() - timedelta(hours=1),
            EndTime=datetime.now(),
            Period=300,
            Statistics=['Average', 'Maximum']
        )
        
        return {
            'average_latency_seconds': response['Datapoints'][-1]['Average'],
            'max_latency_seconds': response['Datapoints'][-1]['Maximum'],
            'is_healthy': response['Datapoints'][-1]['Maximum'] < 900  # 15 min
        }
```

### Iceberg Table Backup Strategy

```python
# iceberg_backup.py
class IcebergBackupManager:
    def __init__(self, spark, backup_location: str):
        self.spark = spark
        self.backup_location = backup_location
        
    def create_table_backup(self, table_name: str, 
                          backup_type: str = "incremental"):
        """Create Iceberg table backup"""
        
        if backup_type == "full":
            # Full backup - copy all data and metadata
            self.spark.sql(f"""
                CREATE TABLE IF NOT EXISTS {self.backup_location}.{table_name}
                USING iceberg
                AS SELECT * FROM {table_name}
            """)
            
        elif backup_type == "incremental":
            # Get last backup snapshot
            last_backup = self._get_last_backup_snapshot(table_name)
            
            # Copy only new snapshots
            self.spark.sql(f"""
                INSERT INTO {self.backup_location}.{table_name}
                SELECT * FROM {table_name}
                WHERE modified_time > '{last_backup}'
            """)
            
            # Also backup metadata
            self._backup_table_metadata(table_name)
    
    def restore_table(self, table_name: str, 
                     restore_point: str = None):
        """Restore table from backup"""
        
        if restore_point:
            # Point-in-time restore
            self.spark.sql(f"""
                CREATE OR REPLACE TABLE {table_name}
                USING iceberg
                AS SELECT * FROM {self.backup_location}.{table_name}
                VERSION AS OF '{restore_point}'
            """)
        else:
            # Latest restore
            self.spark.sql(f"""
                CREATE OR REPLACE TABLE {table_name}
                USING iceberg
                AS SELECT * FROM {self.backup_location}.{table_name}
            """)
```

---

## 14. Advanced Use Cases

### Real-Time ML Feature Store

```python
# feature_store_integration.py
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta

class DataLakeFeatureStore:
    def __init__(self, spark, iceberg_catalog: str):
        self.spark = spark
        self.catalog = iceberg_catalog
        
    def create_feature_table(self, 
                           feature_group: str,
                           features: Dict[str, str],
                           primary_keys: List[str],
                           timestamp_key: str = "feature_timestamp"):
        """Create a feature table with time-travel capabilities"""
        
        # Build schema
        schema_parts = []
        for name, dtype in features.items():
            schema_parts.append(f"{name} {dtype}")
        
        schema = ", ".join(schema_parts)
        
        # Create Iceberg table optimized for point lookups
        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {self.catalog}.features.{feature_group} (
                {schema},
                {timestamp_key} TIMESTAMP,
                created_at TIMESTAMP,
                is_deleted BOOLEAN DEFAULT false
            ) USING iceberg
            PARTITIONED BY (days({timestamp_key}))
            TBLPROPERTIES (
                'write.metadata.metrics.column.{'.'.join(primary_keys)}'='counts,lower_bound,upper_bound',
                'write.parquet.bloom-filter-enabled.column.{'.'.join(primary_keys)}'='true',
                'format-version'='2',
                'write.delete.mode'='merge-on-read'
            )
        """)
        
        # Create materialized view for latest features
        self.spark.sql(f"""
            CREATE MATERIALIZED VIEW {self.catalog}.features.{feature_group}_latest AS
            WITH ranked_features AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY {', '.join(primary_keys)}
                        ORDER BY {timestamp_key} DESC
                    ) as rn
                FROM {self.catalog}.features.{feature_group}
                WHERE NOT is_deleted
            )
            SELECT * EXCEPT(rn)
            FROM ranked_features
            WHERE rn = 1
        """)
    
    def compute_streaming_features(self, 
                                  source_topic: str,
                                  feature_group: str,
                                  feature_logic: str):
        """Compute features from Kafka stream"""
        
        # Read from Kafka
        stream_df = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "kafka:29092") \
            .option("subscribe", source_topic) \
            .option("startingOffsets", "latest") \
            .load()
        
        # Parse and compute features
        features_df = self.spark.sql(f"""
            WITH parsed_events AS (
                SELECT 
                    get_json_object(CAST(value AS STRING), '$.user_id') as user_id,
                    get_json_object(CAST(value AS STRING), '$.event_type') as event_type,
                    CAST(get_json_object(CAST(value AS STRING), '$.timestamp') AS TIMESTAMP) as event_timestamp,
                    CAST(value AS STRING) as raw_event
                FROM {stream_df}
            )
            {feature_logic}
        """)
        
        # Write to feature store with exactly-once semantics
        query = features_df.writeStream \
            .format("iceberg") \
            .outputMode("append") \
            .option("checkpointLocation", f"s3a://checkpoints/{feature_group}") \
            .option("commit.retry.num-retries", "3") \
            .option("commit.retry.min-wait-ms", "100") \
            .trigger(processingTime='10 seconds') \
            .toTable(f"{self.catalog}.features.{feature_group}")
        
        return query
    
    def get_point_in_time_features(self,
                                  entity_df: pd.DataFrame,
                                  feature_groups: List[str],
                                  timestamp_column: str):
        """Get point-in-time correct features for training"""
        
        # Convert to Spark DataFrame
        entity_spark_df = self.spark.createDataFrame(entity_df)
        entity_spark_df.createOrReplaceTempView("entities")
        
        # Build point-in-time join query
        join_parts = []
        for fg in feature_groups:
            join_parts.append(f"""
                LEFT JOIN (
                    SELECT * FROM {self.catalog}.features.{fg}
                    WHERE feature_timestamp <= entities.{timestamp_column}
                    QUALIFY ROW_NUMBER() OVER (
                        PARTITION BY {self._get_join_keys(fg)}
                        ORDER BY feature_timestamp DESC
                    ) = 1
                ) {fg} ON {self._build_join_condition('entities', fg)}
            """)
        
        query = f"""
            SELECT entities.*, {', '.join([f'{fg}.*' for fg in feature_groups])}
            FROM entities
            {' '.join(join_parts)}
        """
        
        return self.spark.sql(query).toPandas()
```

### Change Data Capture Analytics

```python
# cdc_analytics.py
class CDCAnalytics:
    def __init__(self, spark):
        self.spark = spark
        
    def create_scd_type2_dimension(self, 
                                 source_table: str,
                                 target_table: str,
                                 primary_key: str,
                                 tracked_columns: List[str]):
        """Build SCD Type 2 dimension from CDC events"""
        
        # Read CDC events
        cdc_df = self.spark.table(source_table)
        
        # Process SCD Type 2 logic
        self.spark.sql(f"""
            MERGE INTO {target_table} AS target
            USING (
                WITH cdc_events AS (
                    SELECT 
                        {primary_key},
                        {', '.join(tracked_columns)},
                        __op as operation,
                        __source_ts_ms as change_timestamp,
                        CAST(__source_ts_ms / 1000 AS TIMESTAMP) as valid_from,
                        CAST(NULL AS TIMESTAMP) as valid_to,
                        __op != 'DELETE' as is_current
                    FROM {source_table}
                ),
                
                -- Identify records that need to be closed
                records_to_close AS (
                    SELECT DISTINCT 
                        t.{primary_key},
                        t.valid_from,
                        c.valid_from as new_valid_to
                    FROM {target_table} t
                    JOIN cdc_events c ON t.{primary_key} = c.{primary_key}
                    WHERE t.is_current = true 
                    AND c.operation IN ('UPDATE', 'DELETE')
                )
                
                SELECT * FROM cdc_events
                
            ) AS source
            ON target.{primary_key} = source.{primary_key} 
               AND target.valid_from = source.valid_from
            
            -- Close old records
            WHEN MATCHED AND target.is_current = true 
                         AND source.operation IN ('UPDATE', 'DELETE') THEN
                UPDATE SET 
                    valid_to = source.valid_from,
                    is_current = false
                    
            -- Insert new versions
            WHEN NOT MATCHED AND source.operation != 'DELETE' THEN
                INSERT ({primary_key}, {', '.join(tracked_columns)}, 
                        valid_from, valid_to, is_current)
                VALUES (source.{primary_key}, {', '.join([f'source.{col}' for col in tracked_columns])},
                        source.valid_from, source.valid_to, source.is_current)
        """)
    
    def analyze_data_mutations(self, table: str, time_window: str = "1 day"):
        """Analyze patterns in data changes"""
        
        mutation_analysis = self.spark.sql(f"""
            WITH change_events AS (
                SELECT 
                    id,
                    __op as operation,
                    __source_ts_ms as change_timestamp,
                    LAG(__op) OVER (PARTITION BY id ORDER BY __source_ts_ms) as prev_op,
                    LAG(__source_ts_ms) OVER (PARTITION BY id ORDER BY __source_ts_ms) as prev_timestamp
                FROM {table}_cdc
                WHERE __source_ts_ms >= UNIX_TIMESTAMP(CURRENT_TIMESTAMP - INTERVAL {time_window}) * 1000
            ),
            
            mutation_patterns AS (
                SELECT 
                    DATE_TRUNC('hour', CAST(change_timestamp/1000 AS TIMESTAMP)) as hour,
                    operation,
                    COUNT(*) as operation_count,
                    COUNT(DISTINCT id) as unique_records,
                    AVG(change_timestamp - prev_timestamp) / 1000 as avg_seconds_between_changes,
                    
                    -- Detect rapid updates
                    SUM(CASE 
                        WHEN prev_op = 'UPDATE' 
                        AND operation = 'UPDATE' 
                        AND (change_timestamp - prev_timestamp) < 60000  -- Within 1 minute
                        THEN 1 ELSE 0 
                    END) as rapid_update_count
                    
                FROM change_events
                GROUP BY hour, operation
            )
            
            SELECT 
                hour,
                operation,
                operation_count,
                unique_records,
                avg_seconds_between_changes,
                rapid_update_count,
                rapid_update_count / NULLIF(operation_count, 0) as rapid_update_ratio
            FROM mutation_patterns
            ORDER BY hour DESC, operation
        """)
        
        return mutation_analysis
```

### Multi-Modal Data Processing

```python
# multimodal_processor.py
import io
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

class MultiModalDataProcessor:
    def __init__(self, spark):
        self.spark = spark
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        
    def process_product_images(self, 
                             image_table: str,
                             output_table: str):
        """Extract embeddings from product images"""
        
        # Register UDF for image processing
        @udf(returnType="array<float>")
        def extract_image_embedding(image_bytes):
            if not image_bytes:
                return None
                
            try:
                # Load image
                image = Image.open(io.BytesIO(image_bytes))
                
                # Process with CLIP
                inputs = self.clip_processor(
                    images=image, 
                    return_tensors="pt"
                )
                
                # Get embedding
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                    
                return image_features.squeeze().tolist()
                
            except Exception as e:
                # Log error and return None
                return None
        
        # Process images in batches
        self.spark.sql(f"""
            CREATE OR REPLACE TABLE {output_table} AS
            SELECT 
                product_id,
                image_url,
                image_bytes,
                extract_image_embedding(image_bytes) as image_embedding,
                CURRENT_TIMESTAMP() as processed_at
            FROM {image_table}
            WHERE image_bytes IS NOT NULL
        """)
        
        # Create vector index for similarity search
        self._create_vector_index(output_table, "image_embedding")
    
    def _create_vector_index(self, table: str, embedding_column: str):
        """Create ANN index for vector similarity search"""
        
        self.spark.sql(f"""
            ALTER TABLE {table}
            ADD CONSTRAINT vector_index
            USING INDEX (
                TYPE = 'HNSW',
                COLUMN = {embedding_column},
                M = 16,
                EF_CONSTRUCTION = 200,
                DISTANCE = 'cosine'
            )
        """)
```

---

## 15. Migration Strategies

### ✅ DO: Implement Gradual Migration from Legacy Systems

```python
# migration_orchestrator.py
from enum import Enum
from typing import Dict, List, Optional
import logging

class MigrationPhase(Enum):
    ASSESSMENT = "assessment"
    DUAL_WRITE = "dual_write"
    VALIDATION = "validation"
    CUTOVER = "cutover"
    DECOMMISSION = "decommission"

class LegacyMigrationOrchestrator:
    def __init__(self, spark, legacy_connection: Dict, 
                 target_catalog: str):
        self.spark = spark
        self.legacy = legacy_connection
        self.target = target_catalog
        self.logger = logging.getLogger(__name__)
        
    def assess_migration_scope(self, schema: str) -> Dict:
        """Analyze legacy system for migration planning"""
        
        # Connect to legacy database
        legacy_df = self.spark.read \
            .format("jdbc") \
            .option("url", self.legacy["url"]) \
            .option("driver", self.legacy["driver"]) \
            .option("user", self.legacy["user"]) \
            .option("password", self.legacy["password"]) \
            .option("dbtable", f"""
                (SELECT 
                    table_schema,
                    table_name,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                    n_live_tup as row_count
                 FROM information_schema.tables
                 JOIN pg_stat_user_tables USING (table_schema, table_name)
                 WHERE table_schema = '{schema}'
                ) as tables
            """) \
            .load()
        
        # Analyze migration complexity
        assessment = {
            "total_tables": legacy_df.count(),
            "total_size": legacy_df.agg({"size": "sum"}).collect()[0][0],
            "migration_batches": self._plan_migration_batches(legacy_df),
            "estimated_duration_hours": self._estimate_duration(legacy_df)
        }
        
        return assessment
    
    def setup_dual_write(self, table_mappings: Dict[str, str]):
        """Configure Debezium for dual-write phase"""
        
        debezium_config = {
            "name": f"legacy-migration-{self.legacy['database']}",
            "config": {
                "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
                "database.hostname": self.legacy["host"],
                "database.port": self.legacy["port"],
                "database.user": self.legacy["user"],
                "database.password": self.legacy["password"],
                "database.dbname": self.legacy["database"],
                "database.server.name": "legacy",
                "plugin.name": "pgoutput",
                
                # Table filtering
                "table.include.list": ",".join(table_mappings.keys()),
                
                # Transform legacy to new schema
                "transforms": "route,unwrap",
                "transforms.route.type": "org.apache.kafka.connect.transforms.RegexRouter",
                "transforms.route.regex": "legacy\\.(.*)",
                "transforms.route.replacement": "migration.$1",
                
                "transforms.unwrap.type": "io.debezium.transforms.ExtractNewRecordState",
                "transforms.unwrap.drop.tombstones": "false",
                
                # Initial snapshot
                "snapshot.mode": "initial",
                "snapshot.isolation.mode": "repeatable_read"
            }
        }
        
        # Deploy connector
        self._deploy_kafka_connector(debezium_config)
        
        # Create Flink jobs for transformation
        for legacy_table, target_table in table_mappings.items():
            self._create_transformation_job(legacy_table, target_table)
    
    def validate_migration(self, table: str, 
                         validation_queries: List[str]) -> Dict:
        """Validate data consistency between systems"""
        
        validation_results = {}
        
        # Row count validation
        legacy_count = self._get_legacy_count(table)
        target_count = self.spark.table(f"{self.target}.{table}").count()
        
        validation_results["row_count"] = {
            "legacy": legacy_count,
            "target": target_count,
            "match": legacy_count == target_count
        }
        
        # Custom validation queries
        for query in validation_queries:
            legacy_result = self._execute_legacy_query(query)
            target_result = self.spark.sql(
                query.replace("${table}", f"{self.target}.{table}")
            ).collect()
            
            validation_results[query] = {
                "match": legacy_result == target_result
            }
        
        # Data quality checks
        quality_check = self.spark.sql(f"""
            WITH quality_metrics AS (
                SELECT 
                    COUNT(*) as total_rows,
                    SUM(CASE WHEN primary_key IS NULL THEN 1 ELSE 0 END) as null_keys,
                    COUNT(DISTINCT primary_key) as unique_keys,
                    MAX(updated_at) as latest_update
                FROM {self.target}.{table}
            )
            SELECT 
                null_keys = 0 as no_null_keys,
                unique_keys = total_rows as all_unique,
                latest_update > CURRENT_TIMESTAMP - INTERVAL 1 HOUR as recently_updated
            FROM quality_metrics
        """).collect()[0]
        
        validation_results["quality"] = quality_check.asDict()
        
        return validation_results
```

---

## 16. Team Collaboration & Best Practices

### ✅ DO: Implement Data Contracts

```yaml
# data_contracts/user_events.yaml
version: 1.0
name: user_events
owner: platform-team
consumers:
  - analytics-team
  - ml-team
  
schema:
  type: record
  name: UserEvent
  namespace: com.company.events
  fields:
    - name: event_id
      type: string
      doc: "Unique identifier for the event"
      
    - name: user_id
      type: long
      doc: "User identifier"
      
    - name: event_type
      type: string
      doc: "Type of event"
      allowed_values:
        - page_view
        - click
        - purchase
        - signup
        
    - name: event_timestamp
      type: long
      doc: "Unix timestamp in milliseconds"
      
    - name: properties
      type: map
      values: string
      doc: "Additional event properties"
      
quality:
  completeness:
    - field: event_id
      requirement: NOT_NULL
    - field: user_id
      requirement: NOT_NULL
      
  timeliness:
    - field: event_timestamp
      max_delay: 5_MINUTES
      
  accuracy:
    - field: event_type
      validation: IN_ALLOWED_VALUES
      
sla:
  availability: 99.9
  latency_p99: 1000ms
  
versioning:
  strategy: BACKWARD_COMPATIBLE
  deprecation_notice: 30_DAYS
```

### Data Contract Enforcement

```python
# contract_enforcer.py
import yaml
from typing import Dict
import great_expectations as gx

class DataContractEnforcer:
    def __init__(self, contract_path: str):
        with open(contract_path, 'r') as f:
            self.contract = yaml.safe_load(f)
            
    def validate_schema(self, df) -> bool:
        """Validate DataFrame against contract schema"""
        
        expected_fields = {
            field['name']: field['type'] 
            for field in self.contract['schema']['fields']
        }
        
        # Check all required fields exist
        actual_fields = {f.name: str(f.dataType) for f in df.schema.fields}
        
        missing_fields = set(expected_fields.keys()) - set(actual_fields.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
            
        return True
    
    def create_quality_suite(self) -> gx.core.ExpectationSuite:
        """Generate GX suite from contract"""
        
        suite = gx.core.ExpectationSuite(
            expectation_suite_name=f"{self.contract['name']}_contract"
        )
        
        # Add completeness expectations
        for rule in self.contract['quality']['completeness']:
            if rule['requirement'] == 'NOT_NULL':
                suite.add_expectation(
                    gx.core.ExpectationConfiguration(
                        expectation_type="expect_column_values_to_not_be_null",
                        kwargs={"column": rule['field']}
                    )
                )
        
        # Add accuracy expectations  
        for rule in self.contract['quality']['accuracy']:
            if rule['validation'] == 'IN_ALLOWED_VALUES':
                field_def = next(
                    f for f in self.contract['schema']['fields'] 
                    if f['name'] == rule['field']
                )
                
                suite.add_expectation(
                    gx.core.ExpectationConfiguration(
                        expectation_type="expect_column_values_to_be_in_set",
                        kwargs={
                            "column": rule['field'],
                            "value_set": field_def.get('allowed_values', [])
                        }
                    )
                )
        
        return suite
```

### Documentation Generator

```python
# doc_generator.py
from typing import Dict, List
import markdown

class DataLakeDocGenerator:
    def __init__(self, spark, catalog: str):
        self.spark = spark
        self.catalog = catalog
        
    def generate_table_docs(self, output_dir: str):
        """Generate documentation for all tables"""
        
        tables = self.spark.sql(f"SHOW TABLES IN {self.catalog}").collect()
        
        for table in tables:
            table_name = table.tableName
            doc = self._generate_table_doc(table_name)
            
            with open(f"{output_dir}/{table_name}.md", 'w') as f:
                f.write(doc)
    
    def _generate_table_doc(self, table_name: str) -> str:
        """Generate markdown documentation for a table"""
        
        # Get table metadata
        schema = self.spark.table(f"{self.catalog}.{table_name}").schema
        stats = self.spark.sql(f"""
            SELECT 
                COUNT(*) as row_count,
                COUNT(DISTINCT partition) as partition_count,
                MIN(created_at) as earliest_record,
                MAX(created_at) as latest_record
            FROM {self.catalog}.{table_name}
        """).collect()[0]
        
        doc = f"""# {table_name}

## Overview
Table containing {stats['row_count']:,} rows across {stats['partition_count']} partitions.

**Data Range**: {stats['earliest_record']} to {stats['latest_record']}

## Schema

| Column | Type | Description |
|--------|------|-------------|
"""
        
        for field in schema.fields:
            doc += f"| {field.name} | {field.dataType} | {field.metadata.get('comment', '')} |\n"
        
        # Add lineage information
        lineage = self._get_lineage(table_name)
        if lineage:
            doc += f"\n## Data Lineage\n\n```mermaid\n{lineage}\n```\n"
        
        # Add sample queries
        doc += f"""
## Sample Queries

### Recent Data
```sql
SELECT *
FROM {self.catalog}.{table_name}
WHERE created_at >= CURRENT_DATE - INTERVAL 7 DAYS
LIMIT 100
```

### Aggregations
```sql
SELECT 
    DATE(created_at) as date,
    COUNT(*) as count
FROM {self.catalog}.{table_name}
GROUP BY DATE(created_at)
ORDER BY date DESC
```
---

## Conclusion

This guide provides a comprehensive foundation for building and operating modern data lakes in 2025. Key takeaways:

1. **Streaming-First Architecture**: Treat batch as a special case of streaming
2. **Table Format Choice Matters**: Iceberg for flexibility, Delta for Databricks
3. **Schema Evolution is Inevitable**: Plan for it from day one with Schema Registry
4. **Compaction is Critical**: Small files kill performance at scale
5. **dbt for SQL Transformations**: Brings software engineering to analytics
6. **Monitor Everything**: Cost, performance, quality, and lineage
7. **Security by Design**: Encryption, access control, and audit from the start

Remember that data lakes are living systems that evolve with your organization's needs. Start simple, measure everything, and iterate based on actual usage patterns rather than theoretical requirements.

For additional resources and updates, refer to:
- Apache Iceberg documentation: https://iceberg.apache.org/
- Apache Kafka documentation: https://kafka.apache.org/
- dbt documentation: https://docs.getdbt.com/
- Apache Spark documentation: https://spark.apache.org/

Happy data engineering!