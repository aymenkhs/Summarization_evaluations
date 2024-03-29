import os

import pandas as pd

from summarization_models import api_models, developped_models
from summarization_metrics import evaluations

AVAILABLE_MODELS = ['pegasus', 'distill_pegasus', 'distill_bart_1_1', 'distill_bart_6_6', 'distill_bart_6_6_checkpoint-6000']

def load_data(path='data/test.json'):
    test_data = pd.read_json(path)
    return test_data

def execute_and_evaluate_models(data, models):

    if type(models) is str and models.lower() == 'all':
        models = AVAILABLE_MODELS
    else:
        models = [model.lower() for model in models if model.lower() in AVAILABLE_MODELS]

    results = {}
    for model in models:

        print(model)

        if '{}.csv'.format(model) in os.listdir('result_data'):
            result_data = pd.read_csv(os.path.join('result_data', '{}.csv'.format(model)))
        else:
            if model == 'pegasus':
                result_data = developped_models.pegasus(pd.DataFrame(data),
                    model_path='models/pegasus_75000_steps',
                    model_name='google/pegasus-large')
            elif model == 'distill_pegasus':
                result_data = developped_models.distilled_pegasus(pd.DataFrame(data),
                    model_path='models/distill_pegasus_30_epochs',
                    model_name='sshleifer/distill-pegasus-xsum-16-4')
            elif model == 'distill_bart_1_1':
                result_data = developped_models.distilled_bart(pd.DataFrame(data),
                    model_path='models/distill_bart_1_1',
                    model_name='sshleifer/distilbart-xsum-1-1')
            elif model == 'distill_bart_6_6':
                result_data = developped_models.distilled_bart(pd.DataFrame(data),
                    model_path='models/distill_bart_6_6',
                    model_name='sshleifer/distilbart-xsum-6-6')
            elif model == 'distill_bart_6_6_checkpoint-6000':
                result_data = developped_models.distilled_bart(pd.DataFrame(data),
                    model_path='models/distill_bart_6_6_checkpoint-6000',
                    model_name='sshleifer/distilbart-xsum-6-6')

            result_data.columns = ['id', 'referance', 'dialogue', 'prediction', 'execution_time']
            result_data.to_csv(os.path.join('result_data', '{}.csv'.format(model)), index=False)

        result_data = evaluations.compute_metrics(result_data, metrics='all')
        result_data.to_csv(os.path.join('result_data', '{}.csv'.format(model)), index=False)

        results[model] = result_data

    return results
