# Analisis Sentimen Ulasan Aplikasi SimobiPlus dengan Pendekatan Semi-Supervised Learning

Proyek ini mengimplementasikan pemrosesan bahasa alami (Natural Language Processing) untuk melakukan analisis sentimen terhadap ulasan pengguna aplikasi SimobiPlus di Google Play Store. Pendekatan yang digunakan adalah kombinasi Supervised Learning dan Semi-Supervised Learning (Pseudo-Labeling) untuk mengatasi keterbatasan data berlabel manual. Proyek ini melatih dan membandingkan tiga arsitektur model: Support Vector Machine (SVM), Bidirectional Long Short-Term Memory (Bi-LSTM), dan IndoBERT.

## Struktur Proyek

Struktur direktori dari proyek ini adalah sebagai berikut:

```
.
|-- data/
|   |-- processed/
|   |   |-- final/
|   |   |   `-- dataset_final_10000.csv
|   |   |-- manual-labeling/
|   |   |   |-- data_siap_label_manual.csv
|   |   |   |-- data_siap_label_manual_1000.csv
|   |   |   `-- manual_labelled_data.csv
|   |   `-- unlabelled-data/
|   |       `-- sisa_data_unlabeled_9000.csv
|   `-- raw/
|       `-- simobi_reviews.csv
|-- model/
|   |-- model_indobert_final/
|   |   |-- config.json
|   |   |-- model.safetensors
|   |   |-- tokenizer.json
|   |   |-- tokenizer_config.json
|   |   `-- training_args.bin
|   `-- saved_teacher_model/
|       |-- config.json
|       |-- model.safetensors
|       |-- tokenizer.json
|       |-- tokenizer_config.json
|       `-- training_args.bin
|-- notebook/
|   |-- 01_scrapping_data.ipynb
|   |-- 02_preprocessing_and_data_splitting.ipynb
|   |-- 03_pseudo_labeling.ipynb
|   `-- 04_model_training_and_evaluation.ipynb
|-- requirements.txt
`-- README.md
```

## Alur Kerja Proyek

Alur kerja proyek terbagi menjadi empat tahap utama yang direpresentasikan oleh masing-masing notebook:

### 1. Scraping Data (01_scrapping_data.ipynb)
Mengambil ulasan aplikasi SimobiPlus (App ID: com.simas.mobile.SimobiPlus) langsung dari Google Play Store menggunakan library google-play-scraper. 
* Total ulasan mentah yang berhasil diambil sebanyak 10.000 ulasan.
* Hasil ekspor data mentah disimpan pada berkas data/raw/simobi_reviews.csv.

### 2. Preprocessing dan Pembagian Data (02_preprocessing_and_data_splitting.ipynb)
Melakukan pembersihan data mentah sebelum proses pelabelan dan pemodelan:
* Mengubah semua teks menjadi huruf kecil (case folding).
* Menghapus tautan URL, mention, hashtag, angka, dan tanda baca.
* Normalisasi kata tidak baku (slang) menggunakan kamus normalisasi bahasa Indonesia (slang_dict).
* Pembagian data awal untuk melatih model dasar (Teacher Model):
  - 1.000 ulasan diambil secara proporsional berdasarkan rating bintang (Rating 1-2: 400 sampel Negatif; Rating 3: 200 sampel Netral; Rating 4-5: 400 sampel Positif) untuk proses manual labeling menggunakan platform Label Studio (Heartex, 2020).
  - Sisa 9.000 ulasan disimpan sebagai data tidak berlabel (unlabeled data) di berkas data/processed/unlabelled-data/sisa_data_unlabeled_9000.csv.

### 3. Pseudo-Labeling (03_pseudo_labeling.ipynb)
Menerapkan pendekatan Semi-Supervised Learning dengan Pseudo-Labeling berdasarkan metodologi Lee (2013):
* Menggunakan 976 data hasil pelabelan manual dari Label Studio (data/processed/manual-labeling/manual_labelled_data.csv) untuk melatih Teacher Model menggunakan arsitektur IndoBERT (indobenchmark/indobert-base-p1) (Wilie dkk., 2020).
* Teacher Model digunakan untuk memprediksi label dari 9.000 data tidak berlabel.
* Hanya ulasan dengan probabilitas keyakinan (confidence score) prediksi model >= 90% (didapatkan sebanyak 7.789 ulasan) yang dijadikan sebagai data Pseudo-Labels (Lee, 2013).
* Menggabungkan 976 ulasan berlabel manual dengan 7.789 ulasan pseudo-labeled menjadi total 8.765 ulasan (dataset_final_10000.csv).

### 4. Pelatihan Model dan Evaluasi Akhir (04_model_training_and_evaluation.ipynb)
Melatih dan membandingkan performa tiga model klasifikasi sentimen pada dataset gabungan final.

## Distribusi Dataset Final

Distribusi kelas pada dataset final (8.765 ulasan) adalah sebagai berikut:
* Kelas Negatif (Negative): 4.812 ulasan
* Kelas Positif (Positive): 3.761 ulasan
* Kelas Netral (Neutral): 192 ulasan

## Eksperimen dan Hasil Evaluasi Model

Proyek ini menguji tiga skema pemodelan dengan pembagian dataset yang berbeda untuk mendapatkan performa terbaik:

### 1. SVM + TF-IDF (Skema 1)
* Pembagian Data: 80% Train, 20% Test (1.753 data uji)
* Representasi Teks: TF-IDF Vectorizer
* Akurasi Pengujian: 97.15%

Laporan Klasifikasi SVM:
```
              precision    recall  f1-score   support

    Negative       0.96      0.99      0.98       963
     Neutral       1.00      0.05      0.10        38
    Positive       0.99      0.99      0.99       752

    accuracy                           0.97      1753
   macro avg       0.98      0.68      0.69      1753
weighted avg       0.97      0.97      0.96      1753
```

### 2. Bi-LSTM + Embedding (Skema 2)
* Pembagian Data: 80% Train, 20% Test (1.753 data uji)
* Representasi Teks: Keras Tokenizer & Word Embedding
* Akurasi Pengujian: 97.03%

Laporan Klasifikasi Bi-LSTM:
```
              precision    recall  f1-score   support

    Negative       0.96      0.99      0.97       963
     Neutral       0.67      0.26      0.38        38
    Positive       0.99      0.99      0.99       752

    accuracy                           0.97      1753
   macro avg       0.87      0.75      0.78      1753
weighted avg       0.97      0.97      0.97      1753
```

### 3. IndoBERT Fine-Tuning (Skema 3)
* Pembagian Data: 70% Train, 30% Test (2.630 data uji)
* Arsitektur Model: indobenchmark/indobert-base-p1
* Akurasi Pengujian: 99.00%

Laporan Klasifikasi IndoBERT:
```
              precision    recall  f1-score   support

    Negative       0.99      0.99      0.99      1444
     Neutral       0.89      0.72      0.80        58
    Positive       0.99      1.00      1.00      1128

    accuracy                           0.99      2630
   macro avg       0.96      0.91      0.93      2630
weighted avg       0.99      0.99      0.99      2630
```

## Ringkasan Perbandingan Model

| Model | Akurasi Pengujian | F1-Score (Macro Avg) | F1-Score (Weighted Avg) |
|---|---|---|---|
| SVM + TF-IDF | 97.15% | 69% | 96% |
| Bi-LSTM | 97.03% | 78% | 97% |
| IndoBERT | 99.00% | 93% | 99% |

## Kesimpulan

* IndoBERT memberikan performa terbaik dengan akurasi pengujian mencapai 99.00% dan F1-Score (Macro Avg) sebesar 93%. Model ini sangat baik dalam mengenali pola kalimat berbahasa Indonesia karena didukung oleh arsitektur Transformer yang dilatih pada korpus bahasa Indonesia yang besar.
* Pada kelas minoritas (Netral), IndoBERT menunjukkan peningkatan performa yang sangat signifikan (F1-score 80%) dibandingkan dengan SVM (F1-score 10%) dan Bi-LSTM (F1-score 38%). Hal ini menunjukkan kekuatan pre-trained model dalam menangani ketidakseimbangan kelas (*class imbalance*) pada data sentimen ulasan.

## Contoh Inferensi Model (IndoBERT)

Untuk melakukan prediksi sentimen pada ulasan baru, Anda dapat menggunakan modul transformers dengan kode berikut:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./model/model_indobert_final"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

def prediksi_sentimen(teks):
    inputs = tokenizer(teks, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_label_id = torch.argmax(probs, dim=-1).item()
        
    labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return labels[pred_label_id], probs[0][pred_label_id].item()

# Contoh Penggunaan:
teks_ulasan = "Aplikasi ini sangat memudahkan transaksi saya sehari-hari."
hasil, skor = prediksi_sentimen(teks_ulasan)
print(f"Sentimen: {hasil} (Confidence: {skor * 100:.2f}%)")
```

## Author

* Muhammad Attan (https://github.com/attmhd)

## References

* Lee, D. H. (2013). Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks. Workshop on Challenges in Representation Learning, ICML.
* Heartex. (2020). Label Studio: Open Source Multi-type Data Labeling Tool. https://github.com/HumanSignal/label-studio.
* Wilie, B., et al. (2020). IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding. arXiv preprint arXiv:2009.01857.
