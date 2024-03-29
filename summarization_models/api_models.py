import os

import requests
import nlpcloud

from dotenv import load_dotenv
load_dotenv()

NLPCLOUD_API_KEY = os.getenv("NLPCLOUD_API_KEY")
MEANING_CLOUD_API_KEY = os.getenv("MEANING_CLOUD_API_KEY")

def nlpcloud_bart(data):

    client = nlpcloud.Client("bart-large-cnn", NLPCLOUD_API_KEY, False)

    results = []
    for instance in data.index:
        import pdb; pdb.set_trace()
        response = client.summarization(data.loc[instance]['dialogue'])
        summary = response['summary_text']
        results.append(summary)

    data['result_summary'] = results
    return data

def nlpcloud_pegasus(data):

    client = nlpcloud.Client("pegasus-xsum", NLPCLOUD_API_KEY, False)

    results = []
    for instance in data.index:
        import pdb; pdb.set_trace()
        response = client.summarization(data.loc[instance]['dialogue'])
        summary = response['summary_text']
        results.append(summary)

    data['result_summary'] = results
    return data

def meaning_cloud(data):

    meaning_cloud_url = 'http://api.meaningcloud.com/summarization-1.0'

    results = []
    for instance in data.index:

        files = {
            'key': (None, MEANING_CLOUD_API_KEY),
            'txt': (None, data.loc[instance]['dialogue']),
            'sentences': (None, str(2)),
        }

        response = requests.post(meaning_cloud_url, files=files)
        response = response.json()
        if response['status']['code'] == '0':
            summary = response['summary']
        elif response['status']['code'] == '212':
            summary = ''

        print(data.loc[instance]['summary'], summary)
        results.append(summary)

    data['result_summary'] = results
    return data
