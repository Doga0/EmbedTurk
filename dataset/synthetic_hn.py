# !pip install -q transformers torch
# !pip install -q flash-attn --no-build-isolation
# !pip install -q torchvision

# %%capture
# import os
# if "COLAB_" not in "".join(os.environ.keys()):
#     !pip install unsloth
# else:
#     !pip install --no-deps bitsandbytes accelerate xformers==0.0.29 peft trl triton
#     !pip install --no-deps cut_cross_entropy unsloth_zoo
#     !pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
#     !pip install --no-deps unsloth


from datasets import load_dataset
import json
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

class MineHN:
  def __init__(self, dataset_id, subset, split, temperature, top_p, model_id):
    self.dataset = load_dataset(dataset_id, subset, split=split)
    self.dataset = self.dataset.select(range(10))
    self.temperature = temperature
    self.top_p = top_p

    self.model, self.tokenizer = FastLanguageModel.from_pretrained(
      model_name = model_id,
      max_seq_length = 512,
      dtype = None,
      load_in_4bit = True,
    )

  def generate_response(self, prompt):
    FastLanguageModel.for_inference(self.model)

    self.tokenizer = get_chat_template(
      self.tokenizer,
      map_eos_token = True,
    )

    input_ids = self.tokenizer.apply_chat_template(
        prompt,
        tokenize = True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(self.model.device)

    outputs = self.model.generate(
        input_ids,
        temperature=self.temperature,
        top_p=self.top_p,
    )
    response = outputs[0][input_ids.shape[-1]:]

    return self.tokenizer.decode(response, skip_special_tokens=True)

  def generate_hn(self, anchor, positive, num_words):
    prompt = [
          {"role": "system", "content": f"""Veri kümesinde her satır bir "anchor" (sorgu), bir "positive" (doğru belge) içerir:
          - "anchor": {anchor} -- Bilgi getirme görevine göre belirlenmiş rastgele bir kullanıcı arama sorgusu olan bir dizi.
          - "positive": {positive} -- Kullanıcı sorgusuyla alakalı bir belge olan bir dizi.
          Senden istenen, bu verilere ek olarak şu iki çıktıyı üretmendir:
           - "hard_negative_document": Sorguyla alakalı gibi görünen ancak gerçekte yanlış veya alakasız olan bir belge.
          Lütfen aşağıdaki kurallara uyun:
           - Tüm belgeler en az {num_words} kelime uzunluğunda olmalıdır.
           - Belgeler Türkçe dilinde olmalıdır.
          Çıktınız yalnızca JSON nesnesi olmalıdır, açıklama yapmayın veya ekstra metin eklemeyin.
          """},
          {"role": "user", "content": """Verilen "anchor" ve "positive" belgelerine dayanarak, bir "hard negative" belge oluştur.
          "Anchor" ve "positive" belgeleri ile alakasız fakat sorguyla benzerlik gösterebilecek bir belgeyi yaz."""}
      ]

    text = self.generate_response(prompt)

    return text

  def create_json(self, num_words):
    data = []

    for row in self.dataset:
      anchor = row["anchor"]
      positive = row["positive"]

      negative = self.generate_hn(anchor, positive, num_words)

      data.append({
          "anchor": anchor,
          "positive": positive,
          "negative": negative
      })

    with open("dnm.json", "w", encoding="utf-8") as f:
      json.dumb(data, f)

    print("Json file succesfully created!")

dataset = "emrecan/all-nli-tr"
subset = "pair"
split= "dev"
temp = 0.4
top_p = 0.3
num_words = 50
model_id = "unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit"


hn = MineHN(dataset, subset, split, temp, top_p, model_id)
hn.create_json(num_words)