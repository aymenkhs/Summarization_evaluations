import utils
from summarization_models import api_models, developped_models

def main():
    test_data = utils.load_data()
    developped_models.pegasus(test_data)


if __name__ == '__main__':
    main()
