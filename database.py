import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import os
load_dotenv()
import pandas as pd
import schemas 

#Dataset source:
#https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15/data

#Connect to local PostgreSQL database using environment variables
def connect_to_db():
    conn=None

    try:
        #Using environment variables for database connection parameters
        conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("USER"),
        password=os.getenv("PASSWORD"),
        host=os.getenv("HOST"),
        port=os.getenv("PORT")
        )
       
        print("Connected to the PostgreSQL database successfully")
        return conn
    

    except psycopg2.DatabaseError as e:
        raise Exception(f"Database connection error: {e}")


#Create tables that are defined in schemas.py
def create_table(connection, create_table_sql, table_name):
    try:
        with connection.cursor() as cursor:
            cursor.execute(create_table_sql)
        connection.commit()
        print(f"Table {table_name} created successfully")

    except Exception as e:
        #If table creation fails, rollback the transaction
        connection.rollback()
        raise Exception(f"Error creating table {table_name}: {e}")

#Clean dataframe column names
def clean_cols(df):
    df.columns= (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.strip("_")

    )
    
    return df



#Insert the data from pandas dataFrame into the table
def insert_data(connection, df, table_name):
    print(f"Inserting data into {table_name}")
    try:
        #Cleaning columns, replacing NaN with None for SQL compatibility
        df=clean_cols(df)
        df=df.where(pd.notnull(df), None)
        cols=list(df.columns)

        #Changing DataFrame to list of tuples for insertion
        values=df.to_numpy().tolist()
        query= sql.SQL("""
            INSERT INTO {table} ({fields})
            VALUES %s
            """).format(
                table=sql.Identifier(table_name),
                fields=sql.SQL(', ').join(map(sql.Identifier, cols))
                )
        #Using execute_values for efficient bulk insertion, page size set to 1000 for performance
        with connection.cursor() as cursor:
            execute_values(cursor, query, values, page_size=1000)

        connection.commit()
        print(f"{len(df)} inserted successfully into {table_name}")
    except Exception as e:
        #If insertion fails, rollback the transaction
        connection.rollback()
        raise Exception(f"An error occurred while inserting data: {e}")
        


#Fetch data from the database and return as pandas DataFrame
def fetch_data(connection, query, params=None):
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            rows=cursor.fetchall()
            colnames=[desc[0] for desc in cursor.description]
            df=pd.DataFrame(rows, columns=colnames)
            return df
    except Exception as e:
        raise Exception(f"An error occurred while fetching data: {e}")

#Close the database connection
def close_connection(connection):
    if connection:
        connection.close()
        print("Database connection closed.")


#Inser the raw data from csv files into the raw_network_traffic table in datbase
def insert_raw_data(connection,csv_path,split):
    df=pd.read_csv(csv_path)
    df['dataset_split']=split
    insert_data(connection,df,'raw_network_traffic')

#inserting a training run record and returning the run_id for further use
def insert_training_run(connection,run_id,model_name,algorithm,hyperparameters,train_rows,test_rows):
    df=pd.DataFrame([{
        'run_id':run_id,
        'model_name':model_name,
        'algorithm':algorithm,
        'hyperparameters':hyperparameters,
        'train_rows':train_rows,
        'test_rows':test_rows
    }])
    insert_data(connection,df,'training_runs')
    return run_id


#Inserting model metrics into model_metrics table for a given run_id
def insert_metrics(connection,run_id,metrics):
    records=[]
    for metric_name, metric_value in metrics.items():
        records.append({
            'run_id':run_id,
            'metric_name':metric_name,
            'metric_value':metric_value
        })
    df=pd.DataFrame(records)
    insert_data(connection,df,'model_metrics')


#Inserting predictions into predictions table for a given run_id
def insert_predictions(connection,run_id,predictions):
    records=[]
    for record_id, pred in predictions.items():
        records.append({
            'run_id':run_id,
            'record_id':record_id,
            'predicted_label':pred['predicted_label'],
            'predicted_probability':pred['predicted_probability']
        })
    df=pd.DataFrame(records)
    insert_data(connection,df,'predictions')


#Create all tables defined in schemas.TABLES
def create_tables(connection):
    try:
        for table_name, create_table_sql in schemas.TABLES.items():
            create_table(connection, create_table_sql, table_name)
        print("All tables created successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    connection=connect_to_db()

    #Create the tables layed out in schemas.py
    try:
        create_tables(connection)


        #Insert raw data from CSVs
        insert_raw_data(connection,"data/raw/UNSW_NB15_training-set.csv",'train')
        insert_raw_data(connection,"data/raw/UNSW_NB15_testing-set.csv",'test')

    finally:
        close_connection(connection)

if __name__ == "__main__":
    main()
