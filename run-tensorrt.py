import tensorrt as trt
import numpy as np
import cuda
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModel

ENGINE_PATH = "roberta.engine"

# 1: load and tokenize inputs
# Load model & tokenizer
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).eval().cuda()

# Example batch of sentences
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
print("finish processing input and tokenizer.")

# 2. Baseline PyTorch inference
print("**********Pytorch Result*************")
with torch.no_grad():
    start = time.time()
    outputs = model(**inputs)
    torch.cuda.synchronize()
    end = time.time()
    print(f"PyTorch latency -----> {end-start:.4f} sec/n")
print("Output shape (PyTorch):", outputs.last_hidden_state.shape)

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
output_logits = outputs.last_hidden_state
pytorch_base_tensor = outputs.last_hidden_state.cpu().numpy()
print("prepare for the inputs and outputs.")

# 3. onnx inference
import onnxruntime as ort
print("")
print("*******ONNX Result**********")
sess = ort.InferenceSession(
    "roberta.onnx",
    providers=["CUDAExecutionProvider"]
)

input_ids_np = inputs["input_ids"].to(torch.int32).cpu().numpy()
attention_mask_np = inputs["attention_mask"].to(torch.int32).cpu().numpy()

onnx_inputs = {"input_ids": input_ids_np, "attention_mask": attention_mask_np}
start = time.time()
onnx_outputs = sess.run(None, onnx_inputs)
end = time.time()
print(f"onnx latency -----> {end-start:.4f} sec")

# 4: compare pytorch baseline output and onnx output
import numpy as np
onnx_tensor = onnx_outputs[0]
diff = np.abs(pytorch_base_tensor - onnx_tensor)
diff_mean = diff.mean()
diff_max = diff.max()
diff_percent = (diff > 0.1).mean() * 100
print("pytorch baseline vs. onnx output:")
print("Mean absolute difference:", diff_mean)
print("Max absolute difference:", diff_max)
print("Percentage of values with diff > 0.1:", diff_percent, "%")


# 5: load TNT engine
print("")
print("*******TensorRT Result**********")
logger = trt.Logger(trt.Logger.WARNING)

with open(ENGINE_PATH, "rb") as f:
    engine_bytes = f.read()

runtime = trt.Runtime(logger)
trt_engine = runtime.deserialize_cuda_engine(engine_bytes)
assert trt_engine is not None, "Failed to load engine"

context = trt_engine.create_execution_context()
assert context is not None, "Failed to create execution context"

# Check context
if context is None:
    raise RuntimeError("Could not create TensorRT context")
print("TensorRT engine loaded")

# 6: prepare for tensorrt inference
# Convert to PyTorch tensor, specify input tensor shape and address
input_ids_t = input_ids.detach().to(device='cuda', dtype=torch.int32)
attention_mask_t = attention_mask.detach().to(device='cuda', dtype=torch.int32)
context.set_input_shape("input_ids", tuple(input_ids_t.shape))
context.set_input_shape("attention_mask", tuple(attention_mask_t.shape))
context.set_tensor_address("input_ids", int(input_ids_t.data_ptr()))
context.set_tensor_address("attention_mask", int(attention_mask_t.data_ptr()))

# Now query output shapes and specify output shape and address
logits_shape = context.get_tensor_shape("logits")
other_shape = context.get_tensor_shape("other")
print("output logits shape:", logits_shape)
print("output other shape:", other_shape)

# Outputs
logits_t = torch.empty(output_logits.shape, device='cuda', dtype=torch.float32)
context.set_tensor_address("logits", int(logits_t.data_ptr()))
other_shape = tuple([output_logits.shape[0], output_logits.shape[2]])
other_t = torch.empty(other_shape, device='cuda', dtype=torch.float32)
context.set_tensor_address("other", int(other_t.data_ptr()))

# 7. Execute tensorRT inference
stream = torch.cuda.Stream()
start=time.time()
with torch.cuda.stream(stream):
    context.execute_async_v3(
        stream_handle=stream.cuda_stream
    )
# Synchronize only when you need outputs
stream.synchronize()
end = time.time()
trt_time = end - start
print(f"tensorRT latency -----> {trt_time:.4f} sec")

# 8. compare pytorch baseline with tensorRT inference result
output_tensorrt_np = logits_t.cpu().numpy()
diff = np.abs(pytorch_base_tensor - output_tensorrt_np)
diff_mean = diff.mean()
diff_max = diff.max()
diff_percent = (diff > 0.1).mean() * 100

print("compare pytorch base tensor with tensorRT output:")
print("Mean absolute difference:", diff_mean)
print("Max absolute difference:", diff_max)
print("Percentage of values with diff > 0.1:", diff_percent, "%")