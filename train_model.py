# train_model.py

from mental_health_detector import AdvancedMentalHealthDetector
import pandas as pd
from datetime import datetime, timedelta
import random

# Data pelatihan yang diperluas (Bahasa Indonesia)
training_data = {
    'depresi': [
        "Saya merasa sangat sedih dan tidak ada harapan lagi dalam hidup",
        "Hidup saya tidak ada artinya, saya gagal terus",
        "Saya kesepian dan merasa tidak ada yang peduli dengan saya",
        "Tidak ada yang benar dalam hidup saya, semuanya salah",
        "Saya ingin menghilang saja dari dunia ini 😢",
        "Saya lelah hidup, rasanya tidak ada gunanya lagi",
        "Saya merasa tidak berharga dan tidak berguna",
        "Hidup saya hampa dan kosong",
        "Saya tidak punya motivasi untuk melakukan apapun",
        "Saya merasa gagal sebagai manusia",
        "Setiap hari rasanya berat untuk dijalani",
        "Saya tidak bisa merasakan kebahagiaan lagi",
        "Tidur seharian pun masih merasa lelah",
        "Saya menyesal dilahirkan ke dunia ini",
        "Tidak ada yang mengerti perasaan saya 😔",
        "Saya kehilangan minat pada semua hal",
        "Rasanya ingin tidur dan tidak bangun lagi",
        "Saya membenci diri saya sendiri",
        "Hidup terasa seperti beban yang berat",
        "Saya merasa terjebak dalam kegelapan",
        "Sedih banget hari ini",
        "Gak ada keinginan untuk ngapa-ngapain",
        "Putus asa dengan semua yang terjadi",
        "Rasanya hidup gak ada makna",
        "Mau nangis terus tapi air mata gak keluar"
    ],
    
    'kecemasan': [
        "Saya sangat cemas dengan ujian besok, jantung saya berdebar kencang",
        "Saya takut sekali kalau harus presentasi di depan banyak orang",
        "Pikiran saya tidak bisa tenang, selalu overthinking",
        "Rasanya ada yang buruk akan terjadi, saya sangat khawatir",
        "Napas saya sesak kalau memikirkan masa depan",
        "Saya selalu gelisah dan tidak bisa tidur karena cemas",
        "Tangan saya berkeringat dingin saat memikirkan deadline",
        "Saya panik kalau harus bertemu orang baru",
        "Pikiran negatif terus menghantui saya",
        "Saya takut gagal dan mengecewakan orang lain",
        "Deg-degan terus tanpa sebab yang jelas 😰",
        "Saya takut kehilangan kontrol",
        "Khawatir berlebihan dengan hal-hal kecil",
        "Sulit berkonsentrasi karena pikiran yang berputar-putar",
        "Takut sesuatu yang buruk terjadi pada keluarga",
        "Saya menghindari situasi sosial karena cemas",
        "Merasa seperti akan pingsan saat panik",
        "Selalu memikirkan skenario terburuk",
        "Cemas berlebihan tentang kesehatan",
        "Takut sendirian tapi juga takut keramaian",
        "Panik attack lagi datang",
        "Gak bisa fokus gara-gara cemas",
        "Kepikiran terus hal yang belum tentu terjadi",
        "Khawatir banget sama masa depan",
        "Jantung berdebar tiap mikirin tugas"
    ],
    
    'stress': [
        "Deadline kerjaan banyak banget, saya stress dan pusing",
        "Saya merasa tertekan dengan semua tanggung jawab ini",
        "Burned out rasanya, lelah mental dengan semua beban ini",
        "Tidak bisa tidur karena memikirkan pekerjaan",
        "Terlalu banyak yang harus dikerjakan, saya kewalahan",
        "Tekanan dari atasan membuat saya stress berat",
        "Saya lelah dengan semua tuntutan yang ada",
        "Beban hidup terlalu berat, saya tidak kuat",
        "Pusing memikirkan masalah yang menumpuk",
        "Saya merasa overwhelmed dengan semua ini",
        "Tugas kuliah menumpuk bikin stress 😫",
        "Konflik keluarga membuat saya tertekan",
        "Masalah keuangan bikin kepala mau pecah",
        "Saya tidak punya waktu untuk diri sendiri",
        "Ekspektasi orang lain terlalu tinggi",
        "Multitasking terus sampai kelelahan",
        "Saya merasa terjebak dalam rutinitas",
        "Tekanan untuk sukses sangat membebani",
        "Work-life balance berantakan total",
        "Saya butuh liburan dari semua ini",
        "Capek banget sama kerjaan yang numpuk",
        "Stress banget sama target yang gak realistis",
        "Kepala rasanya mau pecah mikirin deadline",
        "Burnout parah nih rasanya",
        "Tekanan kerja bikin gak bisa santai"
    ],
    
    'normal': [
        "Hari ini saya senang bisa bertemu teman-teman",
        "Saya bersyukur dengan apa yang saya miliki sekarang",
        "Weekend ini mau refreshing ke pantai sama keluarga",
        "Produktif banget hari ini, banyak yang bisa diselesaikan",
        "Feeling good dan semangat untuk besok",
        "Saya optimis dengan masa depan saya",
        "Hari ini menyenangkan, bisa quality time dengan keluarga",
        "Saya merasa damai dan tenang",
        "Bangga dengan pencapaian saya hari ini",
        "Life is good, saya bahagia 😊",
        "Mendapat kabar baik hari ini, senang sekali",
        "Olahraga pagi bikin mood jadi bagus",
        "Grateful untuk semua hal baik dalam hidup",
        "Excited untuk project baru yang akan dimulai",
        "Merasa diberkati dengan keluarga yang supportive",
        "Hari ini penuh dengan momen berharga",
        "Saya percaya diri dengan kemampuan saya",
        "Menikmati me-time dengan baca buku favorit",
        "Bersemangat menjalani hari-hari ke depan",
        "Merasa content dengan hidup saat ini",
        "Alhamdulillah hari ini lancar semua",
        "Happy banget bisa ketemu keluarga",
        "Semangat ngerjain hobi yang disuka",
        "Merasa tenang dan damai",
        "Syukur banyak hal positif hari ini"
    ]
}

# Persiapkan data untuk pelatihan
all_texts = []
all_labels = []

for condition, texts in training_data.items():
    all_texts.extend(texts)
    all_labels.extend([condition] * len(texts))

# Buat timestamp sintetis untuk demo analisis temporal
base_time = datetime.now() - timedelta(days=30)
timestamps = []
for i in range(len(all_texts)):
    # Tambahkan randomness pada timestamps
    hours_offset = random.randint(0, 720)  # Random dalam 30 hari
    timestamps.append(base_time + timedelta(hours=hours_offset))

# Inisialisasi dan latih model
print("🔧 Inisialisasi Advanced Mental Health Detector...")
detector = AdvancedMentalHealthDetector()

print("\n📚 Melatih model dengan fitur yang ditingkatkan...")
train_score, test_score = detector.train(all_texts, all_labels)

# Simpan model
print("\n💾 Menyimpan model yang telah dilatih...")
detector.save_model('mental_health_model_advanced.pkl')
print("✅ Model berhasil disimpan!")

# Uji model
print("\n🧪 Menguji model...")
test_cases = [
    "Saya merasa sangat cemas dan khawatir dengan masa depan 😟",
    "Hari ini mood saya bagus, produktif banget!",
    "Stress berat mikirin deadline yang mepet",
    "Sedih banget, rasanya gak ada yang peduli",
    "Alhamdulillah hari ini banyak hal baik yang terjadi",
    "Panik attack datang lagi, susah napas",
    "Burnout parah nih rasanya, capek banget",
    "Bersyukur banget bisa ngumpul sama keluarga"
]

for test_text in test_cases:
    result = detector.predict(test_text)
    print(f"\n📝 Teks: {test_text}")
    print(f"🔍 Kondisi: {result['condition']}")
    print(f"📊 Confidence: {result['confidence']:.2%}")
    print(f"⚠️ Level Risiko: {result['risk_level']}")
    print(f"💭 Sentimen: {result['sentiment']['compound']:.3f}")

# Pentingnya fitur
print("\n📈 10 Fitur Paling Penting:")
importance = detector.get_feature_importance()
if importance is not None:
    print(importance.head(10))

# Uji analisis riwayat chat
print("\n📅 Menguji Analisis Riwayat Chat...")
chat_history = [
    "Pagi ini merasa cemas banget",
    "Siang tadi stress mikirin kerjaan",
    "Malam ini lebih tenang setelah meditasi",
    "Besok ada presentasi, deg-degan"
]

history_result = detector.analyze_chat_history(chat_history, timestamps[:4])
print(f"Kondisi dominan: {history_result['summary']['dominant_condition']}")
print(f"Pesan risiko tinggi: {history_result['summary']['high_risk_messages']}")
print(f"Confidence rata-rata: {history_result['summary']['average_confidence']:.2%}")

print("\n✅ Pelatihan dan pengujian model selesai!")
print("🎉 Model siap digunakan dengan aplikasi Streamlit!")
print("Jalankan: streamlit run app.py")