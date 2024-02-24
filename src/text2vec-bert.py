import torch
from transformers import BertTokenizerFast, BertModel
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
# for local development 
# MODEL_PATH = './src/assets/bert'
# TOKENIZER_PATH = './src/assets/tokenizer'

#for container deployment
MODEL_PATH = './assets/bert'
TOKENIZER_PATH = './assets/tokenizer'

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def load_model(model_path):
    if os.listdir(model_path):
        model = BertModel.from_pretrained(model_path)
    else:
        model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True
                                  )
        model.save_pretrained(model_path)

    model.eval()

    return model.to(DEVICE)

def load_tokenizer(tokenizer_path):
    if os.listdir(tokenizer_path):
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    else:
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        tokenizer.save_pretrained(tokenizer_path)

    return tokenizer

def tokenize(queries, tokenizer):
    encoded_dict = tokenizer.batch_encode_plus(
                            queries,
                            add_special_tokens = True,
                            padding = 'max_length',
                            return_attention_mask = True,
                            return_tensors = 'pt',
                        )

    input_ids = list()
    attention_masks = list()

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0).to(DEVICE)
    attention_masks = torch.cat(attention_masks, dim=0).to(DEVICE)

    return input_ids, attention_masks

def vectorize(input_ids, attention_masks, model):
    output = list()

    for i in range(len(input_ids)):
        with torch.no_grad():
            outputs = model(input_ids[i].unsqueeze(0), attention_masks[i].unsqueeze(0))
            hidden_states = outputs[2]

        token_vecs = hidden_states[-2][0]

        sentence_embedding = torch.mean(token_vecs, dim=0)
        output.append(sentence_embedding.to('cpu').tolist())

    return output

@app.route("/", methods=['POST'])
@cross_origin()
def main():
    queries = request.get_json(force=True)['data']
    
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    
    input_ids, attention_masks = tokenize(queries, tokenizer)

    model = load_model(MODEL_PATH)

    output = vectorize(input_ids, attention_masks, model)

    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
