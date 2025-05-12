# train_model.py

from mental_health_detector import AdvancedMentalHealthDetector
import pandas as pd
from datetime import datetime, timedelta
import random

# Extended training data
training_data = {
    'depression': [
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
        "Saya merasa terjebak dalam kegelapan"
    ],
    
    'anxiety': [
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
        "Takut sendirian tapi juga takut keramaian"
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
        "Saya butuh liburan dari semua ini"
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
        "Merasa content dengan hidup saat ini"
    ]
}

# Prepare data for training
all_texts = []
all_labels = []

for condition, texts in training_data.items():
    all_texts.extend(texts)
    all_labels.extend([condition] * len(texts))

# Create synthetic timestamps for temporal analysis demo
base_time = datetime.now() - timedelta(days=30)
timestamps = []
for i in range(len(all_texts)):
    # Add some randomness to timestamps
    hours_offset = random.randint(0, 720)  # Random within 30 days
    timestamps.append(base_time + timedelta(hours=hours_offset))

# Initialize and train model
print("🔧 Initializing Advanced Mental Health Detector...")
detector = AdvancedMentalHealthDetector()

print("\n📚 Training model with enhanced features...")
train_score, test_score = detector.train(all_texts, all_labels)

# Save model
print("\n💾 Saving trained model...")
detector.save_model('mental_health_model_advanced.pkl')
print("✅ Model saved successfully!")

# Test the model
print("\n🧪 Testing the model...")
test_cases = [
    "Saya merasa sangat cemas dan khawatir dengan masa depan 😟",
    "Hari ini mood saya bagus, produktif banget!",
    "Stress berat mikirin deadline yang mepet",
    "Sedih banget, rasanya gak ada yang peduli"
]

for test_text in test_cases:
    result = detector.predict(test_text)
    print(f"\n📝 Text: {test_text}")
    print(f"🔍 Condition: {result['condition']}")
    print(f"📊 Confidence: {result['confidence']:.2%}")
    print(f"⚠️ Risk Level: {result['risk_level']}")
    print(f"💭 Sentiment: {result['sentiment']['compound']:.3f}")

# Feature importance
print("\n📈 Top 10 Most Important Features:")
importance = detector.get_feature_importance()
if importance is not None:
    print(importance.head(10))

# Test chat history analysis
print("\n📅 Testing Chat History Analysis...")
chat_history = [
    "Pagi ini merasa cemas banget",
    "Siang tadi stress mikirin kerjaan",
    "Malam ini lebih tenang setelah meditasi",
    "Besok ada presentasi, deg-degan"
]

history_result = detector.analyze_chat_history(chat_history, timestamps[:4])
print(f"Dominant condition: {history_result['summary']['dominant_condition']}")
print(f"High risk messages: {history_result['summary']['high_risk_messages']}")
print(f"Average confidence: {history_result['summary']['average_confidence']:.2%}")