from database import insert_data, fetch_data
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Fetch raw data from the database for a given dataset split
def fetch_raw_data(connection, split):
    print(f"Fetching raw data for split={split}")
    query="""
    SELECT * FROM raw_network_traffic 
    WHERE dataset_split=%s;
    """
    df=fetch_data(connection,query,(split,))

    #Droping extra and unnecessary columns
    cols_to_drop=['dataset_split', 'ingested_at', 'attack_cat']
    df=df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
    print(f"Fetched {len(df)} records for split={split}")
    return df

# Handle missing values in the dataframe
def handle_missing_values(df):
    print("Handling missing values")
    df=df.fillna(0)
    print("Handled missing values")
    return df


#Engineering new features based on existing ones for better model performance
def engineer_features(df):
    print("Engineering new features")

    #Byte ration, Packet ratio, TTL difference, Window size ratio, Mean packet size, Average source packet size, Average destination packet size, Load ratio
    if 'sbytes' in df.columns and 'dbytes' in df.columns:
        df['byte_ratio']=df['sbytes']/(df['dbytes']+ 1)
    
    if 'spkts' in df.columns and 'dpkts' in df.columns:
        df['pkt_ratio']=df['spkts']/(df['dpkts']+ 1)

    if 'sttl' in df.columns and 'dttl' in df.columns:
        df['ttl_diff']=abs(df['sttl']-df['dttl'])

    if 'swin' in df.columns and 'dwin' in df.columns:
        df['win_ratio']=df['swin']/(df['dwin']+ 1)

    if 'smean' in df.columns and 'dmean' in df.columns:
        df['mean_pkt_size']=(df['smean']/(df['dmean']+ 1))

    if 'sbytes' in df.columns and 'spkts' in df.columns:
        df['avg_src_pkt_size']=df['sbytes']/(df['spkts']+1)

    if 'dbytes' in df.columns and 'dpkts' in df.columns:
        df['avg_dst_pkt_size']=df['dbytes']/(df['dpkts']+1)

    if 'sload' in df.columns and 'dload' in df.columns:
        df['load_ratio']=df['sload']/(df['dload']+1)


    print("Engineered new features")
    return df



#Encoding the categorical features using OneHotEncoder
def encode_categorical_features(df, fit=True, encoder=None):
    print("Encoding categorical features")
    cat_col=['proto', 'state', 'service']
    label=df['label'] if 'label' in df.columns else None

    cat_col_in_df=[col for col in cat_col if col in df.columns]


    if fit:
        encoder=OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        cat_encoded=encoder.fit_transform(df[cat_col_in_df])
    else:
        cat_encoded=encoder.transform(df[cat_col_in_df])
    #Creating DataFrame for encoded categorical features
    cat_encoded_df=pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_col_in_df), index=df.index)
    #Combining encoded categorical features with the rest of the dataframe
    df_processed= df.drop(columns=cat_col_in_df+ (['label'] if label is not None else []))
    df_processed=pd.concat([df_processed.reset_index(drop=True), cat_encoded_df.reset_index(drop=True)], axis=1)
    #Adding label column back if it exists
    if label is not None:
        df_processed['label']=label.reset_index(drop=True)
    
    print("Encoded categorical features")
    return df_processed, encoder    


#Scaling numerical features using StandardScaler
def scale_numerical_features(df, fit=True, scaler=None):
    print("Scaling numerical features")
    if fit:
        scaler=StandardScaler()
        df_scaled=pd.DataFrame(scaler.fit_transform(df), columns=df.columns,index=df.index)
    else:
        df_scaled=pd.DataFrame(scaler.transform(df), columns=df.columns,index=df.index)
    print(f"Scaled features")
    return df_scaled, scaler

#ETL Pipeline
def process_pipeline(connection, split, fit=True, encoder=None, scaler=None):
    #Fetch raw data -> Handle missing values -> Engineer features -> Encode categorical features -> Scale numerical features
    print(f"Starting ETL pipeline for split={split}")
    df=fetch_raw_data(connection, split)
    df=handle_missing_values(df)
    df=engineer_features(df)
    df, encoder=encode_categorical_features(df, fit=fit, encoder=encoder)
    y=df['label']
    X=df.drop(columns=['label','id'])
    X, scaler=scale_numerical_features(X, fit=fit, scaler=scaler)

    #Returning processed features, labels, encoder and scaler
    print(f"Processed {len(X)} records for split={split}")
    return X, y, encoder, scaler

#Saving the processed features back to the database
def save_processed_features(connection, feature_run_id, record_ids, feature_matrix, feature_names):
    print(f"Saving processed features for run_id {feature_run_id}")
    records=[]
    for i, record_id in enumerate(record_ids):
        for j, feature_name in enumerate(feature_names):
            records.append({
                'record_id': record_id,
                'feature_run_id': feature_run_id,
                'feature_name': feature_name,
                'feature_value': feature_matrix[i,j]
            })

    chunk_size=1000
    for i in range(0,len(records),chunk_size):
        chunk=records[i:i+chunk_size]
        df_chunked=pd.DataFrame(chunk)
        insert_data(connection, df_chunked, 'processed_features')

    print(f"Saved {len(records)} processed features for run_id {feature_run_id}")

