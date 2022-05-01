import utils
from summarization_models import api_models

def main():
    test_data = utils.load_data()
    api_models.meaning_cloud(test_data)


if __name__ == '__main__':
    main()
