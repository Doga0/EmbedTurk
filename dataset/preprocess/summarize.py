import torch
import pandas as pd
import re
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

class ProcessDataset:
  def __init__(self, df, temperature, top_p, model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
    self.df = df
    self.temperature = temperature
    self.top_p = top_p
    self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    self.model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

  def generate_response(self, messages):

    input_ids = self.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(self.model.device)

    terminators = [
        self.tokenizer.eos_token_id,
        self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = self.model.generate(
        input_ids,
        pad_token_id=self.tokenizer.eos_token_id,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=self.temperature,
        top_p=self.top_p,
    )
    response = outputs[0][input_ids.shape[-1]:]

    return self.tokenizer.decode(response, skip_special_tokens=True)

  def process_product_reviews(self, reviews):
    messages = [
          {"role": "system", "content": """Bir yapay zeka asistanı olarak sana bir
          ürün hakkında kullanıcı yorumları vereceğim.
          Görevin, bu yorumları analiz ederek temel noktaları ve genel duyguları özetlemektir.
          Yorumları inceleyerek genel geri bildirimin kısa bir özetini yap, öne çıkan
          olumlu yönleri ve güçlü yanları vurgula, belirgin olumsuz yönleri ve zayıf
          noktaları belirt ve yorumlar arasında tekrar eden ortak temaları veya kalıpları
          tespit et. Tüm bu istenilenleri sadece 1-2 cümleyle Türkçe olarak anlat.
          Ürün dışında kalan gereksiz detaylara (hediye alma amacı, alıcı profili vb.) yer verme."""},
          {"role": "user", "content": reviews}
      ]

    text = self.generate_response(messages)

    return text

  def process_product_info(self, info):
    messages = [
          {"role": "system", "content": """Ürün bilgisini Türkçe olarak temizle ve yapılandır:
          1. Kullanıcı ve ürün için önem arz etmeyen bilgileri kaldır.
          2. Gereksiz satır boşluklarını sil ve formatı düzenle.
          3. Bilgileri daha anlaşılır hale getir ve yapılandırılmış bir formatta sun.
          4. Noktalama hatalarını ve yazım yanlışlarını düzelt.
          **Önemli:** Yanıtında yalnızca temizlenmiş ve yapılandırılmış ürün bilgisini sun. Hiçbir ek açıklama, yorum veya not ekleme."""},
          {"role": "user", "content": info}
      ]

    text = self.generate_response(messages)

    return text

  def clean(self, product):
    cleaned_text = re.sub(r'[^\w\s,.!?:;@#&%*()\-\'"€£₺]+', '', product)
    cleaned_text = re.sub(r"['\"]", "", cleaned_text)

    concatenated_text = " ".join([text.strip() for text in cleaned_text.split(",")])

    return concatenated_text

  def process_df(self):

    summaries = []
    cleaned_infos = []

    for index, row in self.df.iterrows():
      product_comments = row["comments"]
      product_info = row["info"]

      summarized_comments = "yok"

      if product_comments and len(product_comments) > 2:
        cleaned_com = self.clean(product_comments)
        summarized_comments = self.process_product_reviews(cleaned_com)

      if product_info:
        cleaned_info = self.clean(product_info)
        processed_info = self.process_product_info(cleaned_info)

      summaries.append(summarized_comments)
      cleaned_infos.append(processed_info)

    self.df["comments_summary"] = summaries
    self.df["cleaned_info"] = cleaned_infos

    return self.df

if __name__ == "__main__":
  temp = 0.4
  top_p = 0.3

  df = pd.read_csv("/content/drive/MyDrive/EmbedTurk/dataset/new_dataset3-56k.csv")

  process = ProcessDataset(df, temp, top_p)

  new_df = process.process_df()

  new_df.to_csv("/content/drive/MyDrive/EmbedTurk/dataset/new_dataset3-56k_cleaned.csv", index=False)
