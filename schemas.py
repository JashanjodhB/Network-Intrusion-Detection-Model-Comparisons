# schemas for database tables

#raw network traffic data from csvs
RAW_NETWORK_SQL= """
CREATE TABLE IF NOT EXISTS raw_network_traffic (
    id BIGINT,
    dur DOUBLE PRECISION,
    proto TEXT,
    service TEXT,
    state TEXT,
    spkts INT,
    dpkts INT,
    sbytes BIGINT,
    dbytes BIGINT,
    rate DOUBLE PRECISION,
    sttl INT,
    dttl INT,
    sload DOUBLE PRECISION,
    dload DOUBLE PRECISION,
    sloss INT,
    dloss INT,
    sinpkt DOUBLE PRECISION,
    dinpkt DOUBLE PRECISION,
    sjit DOUBLE PRECISION,
    djit DOUBLE PRECISION,
    swin INT,
    stcpb BIGINT,
    dtcpb BIGINT,
    dwin INT,
    tcprtt DOUBLE PRECISION,
    synack DOUBLE PRECISION,
    ackdat DOUBLE PRECISION,
    smean DOUBLE PRECISION,
    dmean DOUBLE PRECISION,
    trans_depth INT,
    response_body_len INT,
    ct_srv_src INT,
    ct_state_ttl INT,
    ct_dst_ltm INT,
    ct_src_dport_ltm INT,
    ct_dst_sport_ltm INT,
    ct_dst_src_ltm INT,
    is_ftp_login INT,
    ct_ftp_cmd INT,
    ct_flw_http_mthd INT,
    ct_src_ltm INT,
    ct_srv_dst INT,
    is_sm_ips_ports INT,
    attack_cat TEXT,
    label INT,
    dataset_split TEXT,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

#processed features table
PROCESSED_FEATURES_SQL= """
CREATE TABLE IF NOT EXISTS processed_features (
    feature_run_id UUID,
    record_id BIGINT,
    feature_name TEXT,
    feature_value DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

#training runs metadata table
TRAINING_RUNS_SQL= """
CREATE TABLE IF NOT EXISTS training_runs (
    run_id UUID PRIMARY KEY,
    model_name TEXT,
    algorithm TEXT,
    hyperparameters JSONB,
    train_rows INT,
    test_rows INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
#model metrics table
MODEL_METRICS_SQL= """
CREATE TABLE IF NOT EXISTS model_metrics (
    run_id UUID REFERENCES training_runs(run_id),
    metric_name TEXT,
    metric_value DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


#predictions table
PREDICTIONS_SQL= """
CREATE TABLE IF NOT EXISTS predictions (
    run_id UUID REFERENCES training_runs(run_id),
    record_id BIGINT,
    predicted_label INT,
    predicted_probability DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
#all table schemas
TABLES={
    "raw_network_traffic": RAW_NETWORK_SQL,
    "processed_features": PROCESSED_FEATURES_SQL,
    "training_runs": TRAINING_RUNS_SQL,
    "model_metrics": MODEL_METRICS_SQL,
    "predictions": PREDICTIONS_SQL,
}



#Constants
FEATURE_COLUMNS=[
    'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 
    'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 
    'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 
    'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 
    'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 
    'response_body_len', 'ct_srv_src', 'ct_state_ttl', 
    'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 
    'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 
    'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 
    'is_sm_ips_ports'
]
#categorical features
CATEGORICAL_FEATURES=['proto','service','state']
#numerical features
NUMERICAL_FEATURES=[col for col in FEATURE_COLUMNS if col not in CATEGORICAL_FEATURES]

TARGET_COLUMN='label'
ID_COLUMN='id'



#Model hyperparameters for pytorch and sklearn models
SKLEARN_PARAMS={
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42
}

PYTORCH_PARAMS={
    'hidden_dims': [256, 128, 64],
    'dropout': 0.3,
    'learning_rate': 0.001,
    'batch_size': 1024,
    'epochs': 50
}