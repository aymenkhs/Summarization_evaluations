import pandas as pd

def load_data(path='data/test.json'):
    test_data = pd.read_json(path)
    return test_data
