import torch
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    GPT2Config, GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments,
    AutoTokenizer, pipeline
)
import os

# GPU kontrolü
print("GPU available:", torch.cuda.is_available())

# 1. Veri setini yükle
dataset = load_dataset("wikipedia", "20220301.tr", split="train")

# 2. Ham metni tek bir dosyaya yaz
os.makedirs("data", exist_ok=True)
with open("data/wiki_tr.txt", "w", encoding="utf-8") as f:
    for entry in dataset:
        f.write(entry["text"].replace("\n", " ") + "\n")

# 3. Tokenizer eğitimi
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["data/wiki_tr.txt"],
    vocab_size=30000,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)
tokenizer.save_model("tokenizer_tr")
hf_tokenizer = AutoTokenizer.from_pretrained("tokenizer_tr", bos_token="<s>", eos_token="</s>", pad_token="<pad>")

# 4. Model yapılandırması
config = GPT2Config(
    vocab_size=hf_tokenizer.vocab_size,
    n_positions=512,
    n_ctx=512,
    n_embd=256,
    n_layer=6,
    n_head=8,
    bos_token_id=hf_tokenizer.bos_token_id,
    eos_token_id=hf_tokenizer.eos_token_id
)
model = GPT2LMHeadModel(config)

# 5. Veri tokenizasyonu
def tokenize_fn(examples):
    return hf_tokenizer(examples["text"], truncation=True, max_length=512)

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=hf_tokenizer, mlm=False)

# 6. Eğitim ayarları
training_args = TrainingArguments(
    output_dir="tiny-transformer-tr-checkpoint",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    weight_decay=0.01,
    logging_steps=200,
    save_steps=500,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator
)

# 7. Eğitimi başlat
trainer.train()

# 8. Modeli kaydet
trainer.save_model("tiny-transformer-tr")

# 9. Örnek üretim
generator = pipeline("text-generation", model="./tiny-transformer-tr", tokenizer=hf_tokenizer)
print(generator("Merhaba, bugün hava nasıl?", max_length=50, do_sample=True))
