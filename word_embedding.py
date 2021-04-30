import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_word_embedding(preprocessed_text):
    """
    Get the last 4 layers of 768 hidden states weights and average it to become a 768 vector  
    """
    marked_text = "[CLS] " + " ".join(preprocessed_text) + " [SEP]"

    # Tokenize our sentence with the BERT tokenizer.
    tokenized_text = tokenizer.tokenize(marked_text)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Mark each of the tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased',
                                    output_hidden_states = True, # Whether the model returns all hidden-states.
                                    )

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]

    token_embeddings = torch.stack(hidden_states, dim=0)

    token_embeddings = token_embeddings.reshape(-1, 768)

    sentence_embedding = torch.mean(token_embeddings, dim=0)

    return sentence_embedding

if __name__ == '__main__':
    preprocessed_text = ["hi","how","are","you"]
    word_embedding = get_word_embedding(preprocessed_text)
    # feature vector of 768 for each sentence
    assert 768 == list(word_embedding.size())[0]