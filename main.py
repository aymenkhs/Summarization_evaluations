import utils
from summarization_models import api_models, developped_models

def main():
    test_data = utils.load_data()
    utils.execute_and_evaluate_models(test_data, models='all')

if __name__ == '__main__':
    main()
