from transformers import GPTNeoXConfig
from transformers import GPTNeoXModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")
print(model)