import os
import sys
import logging

import boto3
import re
import time

from configparser import ConfigParser

from src.utils.time_helper import log_time_cost
from src.utils.log_helper import initialize_log

import typing
from functools import wraps
from typing import List

import awswrangler as wr

def get_configuration(dbserver: str = "athena"):
	conf = ConfigParser()
	config_file = os.path.dirname(__file__) + "/../config.ini"
	conf.read(config_file)
	return conf

# def read_sql_query_df(query: str, dbserver: str = "athena"):
# 	conf = get_configuration()
# 	res = wr.athena.read_sql_query(
# 		query,
# 		database = conf[dbserver]["databasename"],
# 		s3_output = "s3://tmg-aws-ml-shared/"
# 		s3_output = "s3://tmg-ml-service-prod-analyticsbucket-j4qy6ksrdaa4/raw_events"
# 	)
# 	return res

def get_meta_data(conf: str, dbserver: str = "athena"):
	athena_client = boto3.client(dbserver)
	res = athena_client.get_table_metadata(
		CatalogName = conf[dbserver]["catalogname"],
		DatabaseName = conf[dbserver]["databasename"],
		TableName = "raw_events"
	)
	return res


def get_query_result(conf: str, dbserver: str = "athena"):
	athena_client = boto3.client(dbserver)
	execution = athena_client.start_query_execution(
		QueryString = 'SELECT * FROM raw_events LIMIT 2;',
		QueryExecutionContext = {
			'Catalog': conf[dbserver]["catalogname"],
			'Database': conf[dbserver]["databasename"]
		},
		ResultConfiguration = {
			'OutputLocation': 's3://tmg-aws-ml-shared/',
		},
		WorkGroup = 'primary'
	)
	execution_id = execution['QueryExecutionId']
	state = 'RUNNING'

	max_execution = 120
	while max_execution > 0 and state in ['RUNNING', 'QUEUED']:
		max_execution = max_execution - 1
		response = athena_client.get_query_execution(QueryExecutionId = execution_id)

		if 'QueryExecution' in response and 'Status' in response['QueryExecution'] and 'State' in response['QueryExecution']['Status']:
			state = response['QueryExecution']['Status']['State']
			print(state)
			if state == 'FAILED':
				return False
			elif state == 'SUCCEEDED':
				return response
		time.sleep(1)

	return False


def cleanup_athena_results(region, database, bucket, output_path):
	session = boto3.Session()
	params = {
		'region': region,
		'database': database,
		'bucket': bucket,
		'path': output_path
	}
	s3 = session.resource('s3')
	my_bucket = s3.Bucket(params['bucket'])
	for item in my_bucket.objects.filter(Prefix = params['path']):
		item.delete()


def get_athena_s3_file(query_file, region, database, bucket, output_path):
	sql = open(query_file, 'r').read()
	params = {
		'region': region,
		'database': database,
		'bucket': bucket,
		'path': output_path,
		'query': sql
	}
	print(params)
	session = boto3.Session()
	s3_file = athena_to_s3(session, params)
	print(s3_file)
	print('sleep')
	time.sleep(1)  # wait for s3 file
	return s3_file


if __name__ != "main":
	initialize_log()
	conf = get_configuration()
	meta = get_meta_data(conf)
	res = get_query_result(conf)
	print(res)
