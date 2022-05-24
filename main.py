import pandas as pd
from summarization_models import api_models, developped_models

def main():
    test_data = pd.read_json('data/test.json')
    result_data = api_models.openAI(test_data)
    result_data.to_csv('openai_results/openai.csv', index=False)

if __name__ == '__main__':
    main()
