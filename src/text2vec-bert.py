import torch
from transformers import BertTokenizer, BertModel
from flask import Flask, request

app = Flask(__name__)

@app.route("/vectorize")
def vectorize():
    if not request.args:
        return "Provided invalid arguments"
    
    query = request.args["query"]
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_dict = tokenizer.encode_plus(
                            query,
                            add_special_tokens = True,
                            padding = 'max_length',
                            return_attention_mask = True,
                            return_tensors = 'pt',
                        )

    input_ids = []
    attention_masks = []

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True
                                  )
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids, attention_masks)
        hidden_states = outputs[2]

    token_vecs = hidden_states[-2][0]

    sentence_embedding = torch.mean(token_vecs, dim=0)

    return sentence_embedding.tolist()

if __name__ == "__main__":
    app.run(debug=True)
