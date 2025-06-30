#!/usr/bin/env bash

# Setup script for Data Lakes development environment
# Installs: Apache Kafka, Apache Spark, Snowflake connectors, streaming tools

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions
SPARK_VERSION="3.5.0"
KAFKA_VERSION="3.6.1"
SCALA_VERSION="2.12"
HADOOP_VERSION="3"

main() {
    show_banner "Data Lakes Development Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "java" "-version 2>&1 | head -n 1"
    show_tool_status "python3" "--version"
    show_tool_status "spark-shell" "--version 2>&1 | grep version"
    show_tool_status "kafka-topics.sh" "--version 2>&1 | head -n 1"
    echo
    
    # Update apt if needed
    update_apt_if_needed || exit 1
    
    # Install system dependencies
    log_step "Installing system dependencies"
    install_apt_package "build-essential" "Build Essential"
    install_apt_package "curl" "cURL"
    install_apt_package "wget" "Wget"
    install_apt_package "unzip" "Unzip"
    install_apt_package "git" "Git"
    install_apt_package "python3-dev" "Python 3 Dev"
    install_apt_package "python3-pip" "Python 3 pip"
    install_apt_package "python3-venv" "Python 3 venv"
    
    # Install Java (required for Spark and Kafka)
    log_step "Installing Java"
    if ! command_exists java; then
        if confirm "Install OpenJDK 11 (required for Spark/Kafka)?"; then
            install_apt_package "openjdk-11-jdk" "OpenJDK 11"
            
            # Set JAVA_HOME
            java_home_config='
# Java configuration
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$PATH:$JAVA_HOME/bin'
            
            if [[ -f "$HOME/.bashrc" ]]; then
                add_to_file_if_missing "$HOME/.bashrc" "$java_home_config" "Java configuration"
            fi
            
            export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
            export PATH=$PATH:$JAVA_HOME/bin
        else
            log_error "Java is required for Spark and Kafka"
            exit 1
        fi
    else
        log_info "Java is already installed"
        java -version 2>&1 | head -n 1
    fi
    
    # Install Apache Spark
    log_step "Installing Apache Spark"
    if ! command_exists spark-shell; then
        if confirm "Install Apache Spark $SPARK_VERSION?"; then
            install_spark
        fi
    else
        log_info "Spark is already installed"
        spark-shell --version 2>&1 | grep version || true
    fi
    
    # Install Apache Kafka
    log_step "Installing Apache Kafka"
    if ! command_exists kafka-topics.sh; then
        if confirm "Install Apache Kafka $KAFKA_VERSION?"; then
            install_kafka
        fi
    else
        log_info "Kafka is already installed"
    fi
    
    # Create virtual environment for Python tools
    log_step "Setting up Python Data Lakes Environment"
    log_info "Creating virtual environment for data lakes tools..."
    LAKES_VENV="$HOME/.data-lakes"
    if [[ ! -d "$LAKES_VENV" ]]; then
        python3 -m venv "$LAKES_VENV"
    fi
    
    # Activate virtual environment
    source "$LAKES_VENV/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install PySpark and related tools
    log_step "Installing Python Data Lakes Libraries"
    if confirm "Install PySpark and streaming libraries?"; then
        # Core libraries
        pip install pyspark==$SPARK_VERSION
        pip install kafka-python confluent-kafka
        pip install pyarrow pandas numpy
        
        # Snowflake connector
        pip install snowflake-connector-python[pandas]
        pip install snowflake-sqlalchemy
        
        # Delta Lake
        pip install delta-spark
        
        # Stream processing
        pip install faust-streaming  # Kafka Streams for Python
        pip install streamz  # Real-time stream processing
        
        log_success "Python data lakes libraries installed"
    fi
    
    # Install additional connectors
    if confirm "Install additional data connectors?"; then
        pip install pyhive[presto]  # Presto/Hive connector
        pip install pymongo  # MongoDB connector
        pip install cassandra-driver  # Cassandra connector
        pip install redis  # Redis connector
        pip install psycopg2-binary  # PostgreSQL connector
        pip install mysql-connector-python  # MySQL connector
        
        log_success "Data connectors installed"
    fi
    
    # Install monitoring and management tools
    log_step "Installing Monitoring Tools"
    if confirm "Install Kafka monitoring tools?"; then
        # Kafka Manager (now CMAK)
        install_cmak
        
        # Kafka UI
        if command_exists docker; then
            setup_kafka_ui
        else
            log_warning "Docker not found, skipping Kafka UI setup"
        fi
    fi
    
    # Install development tools
    if confirm "Install development and testing tools?"; then
        pip install pytest pytest-spark
        pip install jupyterlab notebook
        pip install black isort mypy
        pip install apache-airflow  # Workflow orchestration
        
        log_success "Development tools installed"
    fi
    
    # Create example projects
    log_step "Creating example projects"
    if confirm "Create data lakes example projects?"; then
        create_lakes_examples
    fi
    
    # Setup Spark configuration
    log_step "Configuring Spark"
    if confirm "Configure Spark for optimal performance?"; then
        configure_spark
    fi
    
    # Deactivate virtual environment
    deactivate
    
    # Setup shell aliases
    log_step "Setting up shell aliases"
    if confirm "Add data lakes aliases to shell?"; then
        setup_lakes_aliases
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "java" "-version 2>&1 | head -n 1"
    show_tool_status "spark-shell" "--version 2>&1 | grep version"
    show_tool_status "kafka-topics.sh" "--version 2>&1 | head -n 1"
    show_tool_status "$LAKES_VENV/bin/python" "--version"
    
    echo
    log_success "Data Lakes development environment is ready!"
    log_info "Useful commands:"
    echo -e "  ${CYAN}source ~/.data-lakes/bin/activate${RESET} - Activate environment"
    echo -e "  ${CYAN}spark-shell${RESET} - Start Spark shell"
    echo -e "  ${CYAN}pyspark${RESET} - Start PySpark shell"
    echo -e "  ${CYAN}kafka-start${RESET} - Start Kafka services"
    echo -e "  ${CYAN}kafka-stop${RESET} - Stop Kafka services"
    echo -e "  ${CYAN}jupyter lab${RESET} - Start Jupyter Lab"
}

install_spark() {
    log_info "Installing Apache Spark $SPARK_VERSION..."
    
    # Create directory
    sudo mkdir -p /opt/spark
    cd /tmp
    
    # Download Spark
    wget -q "https://dlcdn.apache.org/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz"
    
    # Extract
    sudo tar -xzf "spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" -C /opt/spark --strip-components=1
    rm "spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz"
    
    # Set permissions
    sudo chown -R $USER:$USER /opt/spark
    
    # Configure environment
    spark_config='
# Spark configuration
export SPARK_HOME=/opt/spark
export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=python3'
    
    if [[ -f "$HOME/.bashrc" ]]; then
        add_to_file_if_missing "$HOME/.bashrc" "$spark_config" "Spark configuration"
    fi
    
    # Source for current session
    export SPARK_HOME=/opt/spark
    export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
    export PYSPARK_PYTHON=python3
    export PYSPARK_DRIVER_PYTHON=python3
    
    cd "$SCRIPT_DIR"
    log_success "Apache Spark installed"
}

install_kafka() {
    log_info "Installing Apache Kafka $KAFKA_VERSION..."
    
    # Create directory
    sudo mkdir -p /opt/kafka
    cd /tmp
    
    # Download Kafka
    wget -q "https://downloads.apache.org/kafka/${KAFKA_VERSION}/kafka_${SCALA_VERSION}-${KAFKA_VERSION}.tgz"
    
    # Extract
    sudo tar -xzf "kafka_${SCALA_VERSION}-${KAFKA_VERSION}.tgz" -C /opt/kafka --strip-components=1
    rm "kafka_${SCALA_VERSION}-${KAFKA_VERSION}.tgz"
    
    # Set permissions
    sudo chown -R $USER:$USER /opt/kafka
    
    # Configure environment
    kafka_config='
# Kafka configuration
export KAFKA_HOME=/opt/kafka
export PATH=$PATH:$KAFKA_HOME/bin'
    
    if [[ -f "$HOME/.bashrc" ]]; then
        add_to_file_if_missing "$HOME/.bashrc" "$kafka_config" "Kafka configuration"
    fi
    
    # Source for current session
    export KAFKA_HOME=/opt/kafka
    export PATH=$PATH:$KAFKA_HOME/bin
    
    # Create data directories
    mkdir -p "$HOME/kafka-data"/{zookeeper,kafka-logs}
    
    # Create basic configuration
    cat > "$HOME/kafka-data/server.properties" << EOF
broker.id=0
listeners=PLAINTEXT://localhost:9092
log.dirs=$HOME/kafka-data/kafka-logs
num.partitions=1
offsets.topic.replication.factor=1
transaction.state.log.replication.factor=1
transaction.state.log.min.isr=1
log.retention.hours=168
log.segment.bytes=1073741824
zookeeper.connect=localhost:2181
EOF
    
    cd "$SCRIPT_DIR"
    log_success "Apache Kafka installed"
}

install_cmak() {
    log_info "Setting up Cluster Manager for Apache Kafka (CMAK)..."
    
    # Create directory for CMAK
    mkdir -p "$HOME/kafka-tools"
    
    # Create docker-compose file for CMAK
    cat > "$HOME/kafka-tools/docker-compose-cmak.yml" << 'EOF'
version: '3'
services:
  cmak:
    image: hlebalbau/kafka-manager:stable
    container_name: cmak
    ports:
      - "9000:9000"
    environment:
      ZK_HOSTS: "localhost:2181"
    network_mode: host
EOF
    
    log_success "CMAK configuration created"
    log_info "Start CMAK with: cd ~/kafka-tools && docker-compose -f docker-compose-cmak.yml up -d"
}

setup_kafka_ui() {
    log_info "Setting up Kafka UI..."
    
    cat > "$HOME/kafka-tools/docker-compose-kafka-ui.yml" << 'EOF'
version: '3'
services:
  kafka-ui:
    image: provectuslabs/kafka-ui:latest
    container_name: kafka-ui
    ports:
      - "8080:8080"
    environment:
      KAFKA_CLUSTERS_0_NAME: local
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: localhost:9092
      KAFKA_CLUSTERS_0_ZOOKEEPER: localhost:2181
    network_mode: host
EOF
    
    log_success "Kafka UI configuration created"
    log_info "Start Kafka UI with: cd ~/kafka-tools && docker-compose -f docker-compose-kafka-ui.yml up -d"
}

configure_spark() {
    log_info "Configuring Spark..."
    
    # Create Spark configuration
    mkdir -p "$SPARK_HOME/conf"
    
    cat > "$SPARK_HOME/conf/spark-defaults.conf" << EOF
# Spark configuration
spark.master                     local[*]
spark.driver.memory              2g
spark.executor.memory            2g
spark.sql.shuffle.partitions     200
spark.sql.adaptive.enabled       true
spark.sql.adaptive.coalescePartitions.enabled true

# Enable Delta Lake
spark.jars.packages              io.delta:delta-core_2.12:2.4.0
spark.sql.extensions             io.delta.sql.DeltaSparkSessionExtension
spark.sql.catalog.spark_catalog  org.apache.spark.sql.delta.catalog.DeltaCatalog

# Monitoring
spark.eventLog.enabled           true
spark.eventLog.dir               file:///tmp/spark-events
spark.history.fs.logDirectory    file:///tmp/spark-events
EOF
    
    # Create event log directory
    mkdir -p /tmp/spark-events
    
    log_success "Spark configured"
}

create_lakes_examples() {
    log_info "Creating data lakes example projects..."
    
    mkdir -p "$HOME/data-lakes-examples"/{spark,kafka,streaming,notebooks}
    
    # Spark batch processing example
    cat > "$HOME/data-lakes-examples/spark/batch_processing.py" << 'EOF'
#!/usr/bin/env python3
"""Spark batch processing example"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, count, window
from datetime import datetime, timedelta
import random

# Initialize Spark session
spark = SparkSession.builder \
    .appName("BatchProcessingExample") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Generate sample data
print("Generating sample sales data...")
data = []
for i in range(10000):
    data.append({
        'transaction_id': i,
        'timestamp': datetime.now() - timedelta(minutes=random.randint(0, 10080)),
        'customer_id': random.randint(1, 1000),
        'product_id': random.randint(1, 100),
        'quantity': random.randint(1, 10),
        'price': round(random.uniform(10, 1000), 2),
        'category': random.choice(['Electronics', 'Clothing', 'Food', 'Books'])
    })

# Create DataFrame
df = spark.createDataFrame(data)

# Show schema
print("\nDataFrame Schema:")
df.printSchema()

# Basic aggregations
print("\nSales by Category:")
category_sales = df.groupBy("category") \
    .agg(
        count("transaction_id").alias("transactions"),
        sum("price").alias("total_revenue"),
        avg("price").alias("avg_price")
    ) \
    .orderBy("total_revenue", ascending=False)

category_sales.show()

# Time-based analysis
print("\nHourly Sales:")
hourly_sales = df \
    .groupBy(window(col("timestamp"), "1 hour")) \
    .agg(
        count("transaction_id").alias("transactions"),
        sum("price").alias("revenue")
    ) \
    .orderBy("window")

hourly_sales.show(10)

# Save to Parquet
output_path = "sales_data.parquet"
df.write.mode("overwrite").parquet(output_path)
print(f"\nData saved to {output_path}")

# Save aggregated results
category_sales.write.mode("overwrite").csv("category_sales.csv", header=True)
print("Aggregated results saved to category_sales.csv")

spark.stop()
EOF
    
    # Kafka producer example
    cat > "$HOME/data-lakes-examples/kafka/producer.py" << 'EOF'
#!/usr/bin/env python3
"""Kafka producer example"""

from kafka import KafkaProducer
import json
import time
import random
from datetime import datetime

# Create producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Generate and send messages
print("Starting Kafka producer...")
try:
    for i in range(100):
        message = {
            'id': i,
            'timestamp': datetime.now().isoformat(),
            'temperature': round(random.uniform(20, 30), 2),
            'humidity': round(random.uniform(40, 80), 2),
            'sensor_id': f'sensor_{random.randint(1, 10)}'
        }
        
        producer.send('sensor-data', value=message)
        print(f"Sent: {message}")
        time.sleep(1)
        
except KeyboardInterrupt:
    print("\nStopping producer...")
finally:
    producer.close()
EOF
    
    # Kafka consumer example
    cat > "$HOME/data-lakes-examples/kafka/consumer.py" << 'EOF'
#!/usr/bin/env python3
"""Kafka consumer example"""

from kafka import KafkaConsumer
import json

# Create consumer
consumer = KafkaConsumer(
    'sensor-data',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='sensor-group',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

print("Starting Kafka consumer...")
try:
    for message in consumer:
        data = message.value
        print(f"Received: {data}")
        
        # Process message (e.g., check temperature threshold)
        if data['temperature'] > 28:
            print(f"⚠️  High temperature alert: {data['temperature']}°C")
            
except KeyboardInterrupt:
    print("\nStopping consumer...")
finally:
    consumer.close()
EOF
    
    # Spark Streaming example
    cat > "$HOME/data-lakes-examples/streaming/spark_streaming.py" << 'EOF'
#!/usr/bin/env python3
"""Spark Structured Streaming example"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, window, avg
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

# Initialize Spark
spark = SparkSession.builder \
    .appName("StreamingExample") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()

# Define schema
schema = StructType([
    StructField("id", StringType()),
    StructField("timestamp", StringType()),
    StructField("temperature", DoubleType()),
    StructField("humidity", DoubleType()),
    StructField("sensor_id", StringType())
])

# Read from Kafka
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "sensor-data") \
    .load()

# Parse JSON data
sensor_data = df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

# Calculate windowed averages
windowed_avg = sensor_data \
    .groupBy(
        window(col("timestamp"), "1 minute", "30 seconds"),
        col("sensor_id")
    ) \
    .agg(
        avg("temperature").alias("avg_temperature"),
        avg("humidity").alias("avg_humidity")
    )

# Write to console
query = windowed_avg \
    .writeStream \
    .outputMode("append") \
    .format("console") \
    .trigger(processingTime='10 seconds') \
    .start()

query.awaitTermination()
EOF
    
    # Create Jupyter notebook
    cat > "$HOME/data-lakes-examples/notebooks/data_lakes_tutorial.ipynb" << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data Lakes Tutorial\n",
    "This notebook demonstrates various data lakes operations using PySpark, Kafka, and Snowflake."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize Spark Session\n",
    "from pyspark.sql import SparkSession\n",
    "\n",
    "spark = SparkSession.builder \\\n",
    "    .appName(\"DataLakesTutorial\") \\\n",
    "    .config(\"spark.sql.adaptive.enabled\", \"true\") \\\n",
    "    .getOrCreate()\n",
    "\n",
    "print(f\"Spark version: {spark.version}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create sample DataFrame\n",
    "data = [\n",
    "    (1, \"Alice\", 25, \"Engineering\"),\n",
    "    (2, \"Bob\", 30, \"Marketing\"),\n",
    "    (3, \"Charlie\", 35, \"Engineering\"),\n",
    "    (4, \"Diana\", 28, \"HR\")\n",
    "]\n",
    "\n",
    "columns = [\"id\", \"name\", \"age\", \"department\"]\n",
    "df = spark.createDataFrame(data, columns)\n",
    "\n",
    "df.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# SQL operations\n",
    "df.createOrReplaceTempView(\"employees\")\n",
    "\n",
    "result = spark.sql(\"\"\"\n",
    "    SELECT department, \n",
    "           COUNT(*) as count,\n",
    "           AVG(age) as avg_age\n",
    "    FROM employees\n",
    "    GROUP BY department\n",
    "\"\"\")\n",
    "\n",
    "result.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF
    
    # Make scripts executable
    chmod +x "$HOME/data-lakes-examples/spark/"*.py
    chmod +x "$HOME/data-lakes-examples/kafka/"*.py
    chmod +x "$HOME/data-lakes-examples/streaming/"*.py
    
    log_success "Example projects created in ~/data-lakes-examples/"
}

setup_lakes_aliases() {
    log_info "Setting up data lakes aliases..."
    
    lakes_aliases='
# Data Lakes aliases
alias lakes-env="source ~/.data-lakes/bin/activate"
alias lakes-jupyter="source ~/.data-lakes/bin/activate && jupyter lab"

# Spark aliases
alias spark-ui="xdg-open http://localhost:4040"
alias spark-history="$SPARK_HOME/sbin/start-history-server.sh && xdg-open http://localhost:18080"
alias spark-stop-history="$SPARK_HOME/sbin/stop-history-server.sh"

# Kafka aliases
kafka-start() {
    echo "Starting Zookeeper..."
    $KAFKA_HOME/bin/zookeeper-server-start.sh -daemon $KAFKA_HOME/config/zookeeper.properties
    sleep 5
    echo "Starting Kafka..."
    $KAFKA_HOME/bin/kafka-server-start.sh -daemon $HOME/kafka-data/server.properties
    echo "Kafka services started"
}

kafka-stop() {
    echo "Stopping Kafka..."
    $KAFKA_HOME/bin/kafka-server-stop.sh
    sleep 5
    echo "Stopping Zookeeper..."
    $KAFKA_HOME/bin/zookeeper-server-stop.sh
    echo "Kafka services stopped"
}

kafka-topics() {
    $KAFKA_HOME/bin/kafka-topics.sh --bootstrap-server localhost:9092 "$@"
}

kafka-console-producer() {
    $KAFKA_HOME/bin/kafka-console-producer.sh --bootstrap-server localhost:9092 "$@"
}

kafka-console-consumer() {
    $KAFKA_HOME/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 "$@"
}

# Quick operations
create-topic() {
    if [[ -z "$1" ]]; then
        echo "Usage: create-topic <topic-name> [partitions] [replication]"
        return 1
    fi
    kafka-topics --create --topic "$1" --partitions "${2:-3}" --replication-factor "${3:-1}"
}

list-topics() {
    kafka-topics --list
}

# Spark submit helper
spark-run() {
    if [[ -z "$1" ]]; then
        echo "Usage: spark-run <python-file> [args...]"
        return 1
    fi
    spark-submit --master local[*] "$@"
}

# PySpark with packages
pyspark-with-packages() {
    pyspark --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,io.delta:delta-core_2.12:2.4.0
}'
    
    if [[ -f "$HOME/.bashrc" ]]; then
        add_to_file_if_missing "$HOME/.bashrc" "$lakes_aliases" "Data lakes aliases"
    fi
    
    if [[ -f "$HOME/.zshrc" ]]; then
        add_to_file_if_missing "$HOME/.zshrc" "$lakes_aliases" "Data lakes aliases"
    fi
    
    log_success "Data lakes aliases added to shell"
}

# Run main function
main "$@" 