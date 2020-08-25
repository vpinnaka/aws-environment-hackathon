from __future__ import print_function

import base64

print('Loading function')

import boto3
from decimal import Decimal
import uuid
import json
import os



# Get the service resource.
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
def lambda_handler(event, context):
    co2_table_name = os.environ['co2_table_name']
    temp_table_name = os.environ['temp_table_name']
    dew_table_name = os.environ['dew_table_name']
    relhum_table_name = os.environ['relhum_table_name']

    co2_table = dynamodb.Table(co2_table_name)
    temp_table = dynamodb.Table(temp_table_name)
    dew_table = dynamodb.Table(dew_table_name)
    relhum_table = dynamodb.Table(relhum_table_name)
    
    sensor_map = {
        "co2":[],
        "temp":[],
        "dew":[],
        "relh":[],
    }
    sensor_table_map = {
        "co2":co2_table_name,
        "temp":temp_table_name,
        "dew":dew_table_name,
        "relh":relhum_table_name,
    }
    
    for record in event['Records']:
        payload = base64.b64decode(record["kinesis"]["data"])
        data = json.loads(payload)
        sensor_map[data.get('sensor_type')].append({
            'id': str(uuid.uuid4()),
            'anamoly': data.get('anamoly'),
            'name': data.get('name'),
            'timestamp': data.get('timestamp'),
            'value': Decimal(str(data.get('value')))
        })
        
    for sensor_type, values in sensor_map.items():
        table = sensor_table_map.get(sensor_type)
        with table.batch_writer() as batch:
            for value in values:
                batch.put_item(
                    Item=value
                )

    print('Successfully processed {} records.'.format(len(event['Records'])))

    
