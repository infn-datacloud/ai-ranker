import json

import numpy as np
import pandas as pd
from kafka import KafkaConsumer, TopicPartition


def load_dataset_from_kafka(kafka_server_url:str, topic:str, partition: int, offset:int):

    consumer = KafkaConsumer(
            #topic,
            bootstrap_servers=kafka_server_url,
            auto_offset_reset='earliest',
            #enable_auto_commit=False,
            value_deserializer=lambda x: json.loads(x.decode("utf-8")),  # deserializza il JSON
            consumer_timeout_ms=500
            )

    tp = TopicPartition(topic,partition)
    consumer.assign([tp])
    consumer.seek(tp, offset)
    l_data = [message.value for message in consumer]
    df = pd.DataFrame(l_data)
    return df

def filter_df(df: pd.DataFrame, columns_to_return: list) -> pd.DataFrame:
    return df[columns_to_return]

def preprocessing(df: pd.DataFrame, template_complex_types: list) -> pd.DataFrame:
    df["cpu_diff"] = (df["vcpus_quota"] - df["vcpus_usage"]) - df["vcpus_requ"]
    df["ram_diff"] = (df["ram_gb_quota"] - df["ram_gb_usage"]) - df["ram_gb_requ"]
    df["storage_diff"] = (df["storage_gb_quota"] - df["storage_gb_usage"]) - df["storage_gb_requ"]
    df["instances_diff"] = (df["n_instances_quota"] - df["n_instances_usage"]) - df["n_instances_requ"]
    df["volumes_diff"] = (df["n_volumes_quota"] - df["n_volumes_usage"]) - df["n_volumes_requ"]
    df["floatingips_diff"] = (df["floating_ips_quota"] - df["floating_ips_usage"])-df["floating_ips_requ"]
    df["gpu"] = df["gpus_requ"].astype(bool).astype(float)
    mapStatus={"CREATE_COMPLETED":0, "CREATE_FAILED":1}
    df["status"]=df["status"].map(mapStatus).astype(int)
    df["complexity"]= df["template_name"].isin(template_complex_types).astype(float)
    df["deployment_time"] = np.where(df["completed_time"] != 0.0, df["completed_time"], df["tot_failed_time"]/df["n_failures"])
    df["provider"]=df["provider_name"] + "-" + df["region_name"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    grouped = df.groupby(['provider', 'template_name'])
    df = df.copy() # <-- perchè qua devi fare la copia?
    df['failure_percentage'] = df.apply(lambda row: calculate_failure_percentage(grouped.get_group((row['provider'], row['template_name'])), row), axis=1)
    df['avg_success_time'] = df.apply(lambda row: calculate_avg_success_time(grouped.get_group((row['provider'], row['template_name'])), row), axis=1)
    df['avg_failure_time'] = df.apply(lambda row: calculate_avg_failure_time(grouped.get_group((row['provider'], row['template_name'])), row), axis=1)
    return df

def calculate_failure_percentage(group, row):
    mask = (group['timestamp'] <= row['timestamp']) & (group['timestamp'] > row['timestamp'] - pd.Timedelta(days=90))
    filtered_group = group[mask]
    return filtered_group['status'].mean() if not filtered_group.empty else None

def calculate_avg_success_time(group, row):
    mask = (group['timestamp'] <= row['timestamp']) & (group['timestamp'] > row['timestamp'] - pd.Timedelta(days=90)) & (group['status'] == 0)
    filtered_group = group[mask]
    return filtered_group['deployment_time'].mean() if not filtered_group.empty else 0.0

def calculate_avg_failure_time(group, row):
    mask = (group['timestamp'] <= row['timestamp']) & (group['timestamp'] > row['timestamp'] - pd.Timedelta(days=90)) & (group['status'] == 1)
    filtered_group = group[mask]
    return filtered_group['deployment_time'].mean() if not filtered_group.empty else 0.0



