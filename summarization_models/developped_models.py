import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

def pegasus(data, model_path='models/pegasus_75000_steps', model_name='google/pegasus-large'):

    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_path)

    results = []
    for instance in data.index:
        source = data.loc[instance]['dialogue']
        tokens = tokenizer(source, truncation=True, padding="longest", return_tensors="pt").to(device)
        summary = model.generate(**tokens)
        summary = tokenizer.batch_decode(summary, skip_special_tokens=True)
        results += summary
        print(summary)
        import pdb; pdb.set_trace()

    data['result_summary'] = results
    return data

def distilled_pegasus(data, model_path):
    pass

def distilled_bart(data, model_path):
    pass

def transformers(data, model_path):
    pass
