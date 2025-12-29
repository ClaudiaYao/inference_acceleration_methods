import time
import torch
from transformers import AutoTokenizer, AutoModel

# 1. Load model & tokenizer
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).eval().cuda()

# 2. Example batch of sentences
text = """ A good story encourages us to turn the next page and read more. We want to find out what happens next and what the main characters do and what they say to each other. 
We may feel excited, sad, afraid, angry or really happy. This is because the experience of reading or listening to a story is much more likely to make us 'feel' that we are part 
of the story, too. Just like in our 'real' lives, we might love or hate different characters in the story. Perhaps we recognise ourselves or others in some of them. Perhaps we 
have similar problems. Because of this natural empathy with the characters, our brains process the reading of stories differently from the way we read factual information. 
Our brains don't always recognise the difference between an imagined situation and a real one so the characters become 'alive' to us. What they say and do is therefore more meaningful. 
This is why the words and structures that relate a story's events, descriptions and conversations are processed in this deeper way. In fact, cultures all around the world have always 
used storytelling to pass knowledge from one generation to another. Our ancestors understood very well that this was the best way to make sure our histories and information about 
how to relate to others and to our world was not only understood, but remembered too. (Notice that the word ‘history’ contains the word ‘story’ – More accurately, the word ‘story’ 
derives from ‘history’.) Encouraging your child to read or listen to stories should therefore help them to learn a second language in a way that is not only fun, but memorable. 
Let's take a quick look at learning vocabulary within a factual text or within a story. Imagine the readers are eight-year-olds interested in animals. In your opinion, are they more 
likely to remember AND want to continue reading the first or second text? """

texts = [item.strip() for item in text.split(".")][:16]  # adjust batch size here
inputs = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=32).to("cuda")

print("called tokenizer.")

# 3. Baseline PyTorch inference
with torch.no_grad():
    start = time.time()
    outputs = model(**inputs)
    torch.cuda.synchronize()
    end = time.time()
    print(f"PyTorch latency: {end-start:.4f} sec")

print("Output shape (PyTorch):", outputs.last_hidden_state.shape)


# Cast inputs to int32 for TensorRT
input_ids = inputs["input_ids"].to(torch.int32)
attention_mask = inputs["attention_mask"].to(torch.int32)

# 4: generate onnx file
torch.onnx.export(
    model.float(),
    (input_ids, attention_mask),
    "roberta.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits", "other"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size", 1: "seq_len"},
        "other":{0: "batch_size"}
    },
    opset_version=17
)

print("save to onnx file")