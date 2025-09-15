# tiny-transformer-tr

Türkçe küçük GPT modeli.

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım (Colab)

- Colab ortamında `train.py` dosyasını çalıştırın.

## Proje Adımları

1. `wikipedia` veri setini indirir.
2. Byte-Level BPE tokenizer eğitilir.
3. Küçük GPT2 modeli tanımlanır.
4. Trainer API ile model eğitilir.
5. Eğitilen model `tiny-transformer-tr` klasörüne kaydedilir.
6. Test üretim örnekleri konsola yazdırılır.

## Gereksinimler

- Python 3.8+
- GPU önerilir.
