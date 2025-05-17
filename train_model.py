# train_model.py

from mental_health_detector import AdvancedMentalHealthDetector
import pandas as pd
from datetime import datetime, timedelta
import random
import numpy as np
import os

# =============================================
# DATA PELATIHAN YANG DITINGKATKAN (BAHASA INDONESIA)
# =============================================

# train_model.py (Perbaikan Syntax Error)

training_data = {
    'depresi': [
        # Kalimat panjang
        "Saya merasa tidak ada harapan lagi dalam hidup ini",
        "Setiap hari rasanya seperti beban yang tak tertahankan",
        "Tidur seharian pun masih merasa lelah luar biasa",
        "Aku benci diri sendiri dan semua kesalahan yang telah kulakukan",
        "Rasanya ingin menghilang saja dari dunia ini",
        "Tidak ada yang peduli dengan keberadaanku di sini",
        "Gua ngerasa kayak hidup gak ada tujuan lagi, bangun tidur cuma nunggu malem doang, ga ada semangat buat ngapa-ngapain 😔",
        "Tiap hari scroll TikTok liat orang happy-happy, tapi gua di sini ngerasa hampa banget, kayak ada yang kurang tapi ga tau kurang apa",
        "Pengen nangis tapi air mata udah kering, pengen cerita tapi takut dikira cari perhatian doang 😞",
        "Gua gabut banget, hidup kayak gak ada arti",
        "Capek jiwa, pengen lenyap aja dari dunia",
        "Tiap hari cuma ngebengong doang, ga semangat",
        "Rasanya semua salah gue, worthless banget",
        "Gue benci banget sama diri sendiri",
        "Mati aja kali ya, daripada begini terus",
        "Ga ada yang peduli, kayak gue nggak exist",
        "Dunia tuh kejam banget, ga adil",
        "Gue cuma beban buat orang lain",
        "Udh 3 hari gue cuma tidur doang, males bgt",

        # Kalimat pendek
        "Hidup ini hampa",
        "Aku lelah",
        "Gak ada artinya",
        "Mati saja",
        "Putus asa",
        "Sia-sia",
        "Gue worthless banget",
        "Mati aja deh",
        "Ga ada yang peduli",
        "Beban keluarga",
        "Lelah mental 😴",
        "Gabut akut"
    ],
    
    'kecemasan': [
        # Kalimat panjang
        "Jantungku berdebar kencang setiap kali memikirkan presentasi besok",
        "Aku terus membayangkan skenario terburuk yang mungkin terjadi",
        "Tangan berkeringat dingin dan napas sesak saat berada di keramaian",
        "Pikiran tidak bisa berhenti mengkhawatirkan masa depan",
        "Gua overthinking mulu sampe kepala cenat-cenut, tiap mau tidur otak masih kepikiran skenario terburuk ujian besok 🥺",
        "Jantung deg-degan nggak jelas tiap lewat depan gebetan, takut dia ngejudge penampilan gua yang acak-acakan ini",
        "Napas sesak tiap ingat tagihan kosan belum dibayar, padahal duit di rekening cuma cukup buat makan indomie sebulan 💸",
        "Jantung deg-degan terus gue kalo mikir ujian",
        "Gue overthinking mulu sampe gabisa tidur",
        "Takutt banget presentasi nanti ditertawain",
        "Napas sesak tiap mau ketemu orang baru",
        "Gue panik banget pas tau pacar ghosting",
        "Bad trip mulu akhir-akhir ini, galau terus",
        "Gue takut banget gagal di kampus",
        "Keringat dingin gue tiap mau meeting zoom",
        "Kayak ada yang mau celaka gitu rasanya",
        "Gue sering banget ngerinding tiba-tiba",

        # Kalimat pendek
        "Deg-degan terus",
        "Gelisah banget",
        "Aku takut",
        "Panik attack",
        "Overthinking",
        "Sesak napas",
        "Baper mulu",
        "Gue gagal terus",
        "Takut ditolak",
        "Galau midnight 🌃",
        "Drama ga jelas"
    ],
    
    'stress': [
        # Kalimat panjang
        "Beban kerja yang menumpak membuatku tidak bisa tidur semalaman",
        "Deadline bertabrakan dan atasan terus menambah tekanan",
        "Masalah keuangan dan keluarga membuat kepalaku mau pecah",
        "Kerjaan numpuk kayak gunung, dikatain bos terus, pulang malem masih harus urus adek yang sakit, rasanya mau teriak aja di tengah jalan 😤",
        "Dikejar deadline tugas kampus, pacar minta putus, motor kena tilang pula! Apa lagi yang mau terjadi hari ini? 🤯",
        "Tiap buka IG liat temen-temen pada sukses, gua di sini masih nganggur 2 tahun lulus kampus, malu tapi bingung harus mulai dari mana 😓",
        "Tugas numpuk banget, mau nangis aja",
        "Dikejar deadline mulu kayak robot",
        "Gue burnout parah sama kerjaan",
        "Masalah duit bikin kepala cenat-cenut",
        "Drama keluarga bikin mental down",
        "Gue kewalahan ngadepin semuanya sendirian",
        "Mager level dewa, gabisa ngapa-ngapain",
        "Dikejar tagihan mulu tiap bulan",
        "Gue bingung mau mulai dari mana",
        "Rasanya semua hal jadi berantakan",

        # Kalimat pendek
        "Burnout",
        "Beban berat",
        "Kewalahan",
        "Pusing tujuh keliling",
        "Tekanan kerja",
        "Gak kuat lagi",
        "Burnout level 100 🔥",
        "Mager akut 🛌",
        "Pusing tujuh keliling",
        "Dikejar zombie 🧟♂️",
        "Mau meledak! 💣",
        "Hidup berantakan"
    ],
    
    'normal': [
        # Kalimat panjang
        "Hari ini aku merasa bersyukur dengan semua nikmat yang diberikan Tuhan",
        "Quality time dengan keluarga membuat hatiku tenang dan bahagia",
        "Menyelesaikan pekerjaan tepat waktu memberiku kepuasan tersendiri",
        "Baru aja nongki sama squad sambil ngopi-ngopi cantik di kafe kekinian, seneng banget bisa ketawa-ketawa gini setelah minggu yang hectic 😆",
        "Walaupun kerjaan masih banyak, tapi gua bersyukur punya keluarga yang selalu support dan pacar yang ngerti kesibukan gua 💖",
        "Hari ini akhirnya berani coba terapi ke psikolog, ternyata nggak semenyeramkan yang dibayangin malah jadi lega bisa cerita banyak hal 🫂",
        "Alhamdulillah hari ini chill banget",
        "Baru nongki sama squad, seneng bgt",
        "Mood lagi bagus nih buat produktif",
        "Gue bersyukur punya temen-temen keren",
        "Lagi asik main game sama gebetan",
        "Baru selesai ngerjain tugas, lega bgt",
        "Weekend jalan-jalan ke mall, fun bgt",
        "Makan mie ayam enak bikin hati senang",
        "Baru dapet nilai bagus, syukur deh",
        "Nongkrong di kafe sambil dengerin musik",

        # Kalimat pendek dan netral
        "Halo",
        "Tes",
        "Apa kabar?",
        "Baik-baik saja",
        "Tidak apa-apa",
        "Oke",
        "Iya",
        "Tidak",
        "Terima kasih",
        "Sama-sama",
        "Laporan sudah siap",
        "Besok meeting jam 10",
        "Makan siang yuk",
        "Cuaca hari ini cerah",
        "Lagi di mana?",
        "Sampai jumpa besok",
        "Mood lagi bagus 🌈",
        "Healing ke pantai 🌊",
        "Gajian! 💰",
        "Lulus sidang 🎓",
        "Date seru 💑",
        "Me time nyaman 🧘♀️"
    ]
}

# =============================================
# PERSIAPAN DATA PELATIHAN
# =============================================

def prepare_training_data():
    """Mengkonversi data pelatihan ke format yang sesuai"""
    all_texts = []
    all_labels = []
    
    for condition, texts in training_data.items():
        all_texts.extend(texts)
        all_labels.extend([condition] * len(texts))
    
    return all_texts, all_labels

def generate_synthetic_timestamps(num_entries):
    """Membuat timestamp sintetis untuk simulasi riwayat chat"""
    base_time = datetime.now() - timedelta(days=30)
    timestamps = []
    
    for i in range(num_entries):
        # Tambahkan variasi waktu acak dalam 30 hari terakhir
        time_offset = random.randint(0, 30*24*60*60)  # Acak dalam 30 hari dalam detik
        new_time = base_time + timedelta(seconds=time_offset)
        timestamps.append(new_time)
    
    return timestamps

# =============================================
# PROSES PELATIHAN UTAMA
# =============================================

def main():
    print("🔄 Memulai proses pelatihan model...")
    
    # 1. Persiapan data
    texts, labels = prepare_training_data()
    timestamps = generate_synthetic_timestamps(len(texts))
    
    print(f"📊 Jumlah data pelatihan: {len(texts)} teks")
    print("🔢 Distribusi label:")
    print(pd.Series(labels).value_counts())
    
    # 2. Inisialisasi detector
    detector = AdvancedMentalHealthDetector()
    
    # 3. Pelatihan model dengan validasi
    print("\n🎓 Melatih model...")
    train_score, test_score = detector.train(texts, labels, validate=True)
    
    # 4. Menyimpan model
    print("\n💾 Menyimpan model...")
    detector.save_model('mental_health_model_advanced.pkl')
    
    # 5. Validasi tambahan
    print("\n🧪 Uji model dengan contoh kasus:")
    test_cases = [
        ("Halo", "normal"),
        ("Tes", "normal"),
        ("Aku benci hidup ini", "depresi"),
        ("Jantungku berdebar-debar terus", "kecemasan"),
        ("Beban kerja tak tertahankan", "stress"),
        ("Alhamdulillah hari ini menyenangkan", "normal")
    ]
    
    for text, expected in test_cases:
        result = detector.predict(text)
        print(f"\n📝 Teks: '{text}'")
        print(f"🏷️ Hasil: {result['condition']} (Harapan: {expected})")
        print(f"📊 Confidence: {result['confidence']:.1%}")
        print(f"⚠️ Level Risiko: {result['risk_level']}")
    
    # 6. Tampilkan fitur penting
    print("\n🔍 10 Fitur Paling Penting:")
    importance = detector.get_feature_importance()
    if importance is not None:
        print(importance.head(10))
    
    print("\n✅ Pelatihan selesai! Model siap digunakan.")

if __name__ == "__main__":
    # Cek dan buat folder model jika belum ada
    if not os.path.exists('models'):
        os.makedirs('models')
    
    main()