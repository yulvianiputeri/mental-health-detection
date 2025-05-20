# test_model.py
from mental_health_detector import AdvancedMentalHealthDetector
import pandas as pd

# Data pengujian - teks bahasa Indonesia untuk setiap kategori
test_data = {
    'depresi': [
        "Saya merasa tidak ada gunanya hidup lagi", 
        "Aku benci diri sendiri, mau mati aja rasanya",
        "Gue ga ada guna di dunia ini, pengen ngilang aja",
        "Tiap hari rasanya cape banget, hidup ga ada arti",
        "Semua orang lebih baik tanpa keberadaan aku"
    ],
    'kecemasan': [
        "Aku selalu khawatir tentang masa depan", 
        "Jantung deg-degan, keringat dingin setiap mikirin ujian",
        "Gue overthinking parah, gak bisa tidur tiap malam",
        "Selalu takut salah di depan orang, deg-degan terus",
        "Rasanya panik gak jelas, napas sesak tiba-tiba"
    ],
    'stress': [
        "Deadline bertumpuk, rasanya mau meledak kepala", 
        "Burnout parah, gak sanggup kerja lagi",
        "Tekanan kerjaan bikin stress, gak bisa santai",
        "Tugas kuliah numpuk, dosen marah-marah terus",
        "Beban hidup terlalu berat, kewalahan ngadepin semuanya"
    ],
    'normal': [
        "Hari ini saya merasa bahagia", 
        "Alhamdulillah projek selesai tepat waktu",
        "Lagi asik nonton film sambil makan popcorn",
        "Senang bisa kumpul sama teman-teman lama",
        "Biasa aja sih, cuma mau tanya kabar"
    ]
}

def test_model():
    print("\n🧪 PENGUJIAN MODEL DETEKSI KESEHATAN MENTAL")
    print("=" * 60)
    
    # Inisialisasi detektor
    detector = AdvancedMentalHealthDetector()
    
    try:
        # Muat model
        detector.load_model('mental_health_model_advanced.pkl')
        print("✅ Model berhasil dimuat\n")
    except Exception as e:
        print(f"❌ Error saat memuat model: {e}")
        print("\nMelatih model terlebih dahulu...")
        from train_model import main as train_main
        train_main()
        detector.load_model('mental_health_model_advanced.pkl')
    
    # Hasil akurasi
    results = {
        'kategori': [],
        'teks': [],
        'prediksi': [],
        'confidence': [],
        'benar': []
    }
    
    # Uji setiap kategori
    for category, texts in test_data.items():
        print(f"\n📊 Menguji kategori: {category.upper()}")
        print("-" * 40)
        
        for text in texts:
            prediction = detector.predict(text)
            
            # Tampilkan hasil
            predicted = prediction['condition']
            confidence = prediction['confidence']
            correct = predicted == category
            
            print(f"📝 Teks: '{text}'")
            print(f"🔍 Prediksi: {predicted.upper()} (Ekspektasi: {category.upper()})")
            print(f"📈 Confidence: {confidence:.2%}")
            
            # Tampilkan informasi debugging
            if 'debug_info' in prediction:
                debug = prediction['debug_info']
                print("\n🔬 Debug Info:")
                print(f"📊 Skor Kata Kunci: {debug['keyword_scores']}")
                print(f"🔑 Kata Kunci Terdeteksi: {debug['detected_keywords']}")
            
            if correct:
                print("✅ BENAR\n")
            else:
                print("❌ SALAH\n")
            
            # Tambahkan ke hasil
            results['kategori'].append(category)
            results['teks'].append(text)
            results['prediksi'].append(predicted)
            results['confidence'].append(confidence)
            results['benar'].append(correct)
    
    # Hitung akurasi
    results_df = pd.DataFrame(results)
    accuracy = results_df['benar'].mean() * 100
    
    print("\n📋 RINGKASAN HASIL")
    print("=" * 60)
    print(f"🎯 Akurasi Keseluruhan: {accuracy:.1f}%")
    
    # Akurasi per kategori
    print("\n🔍 Akurasi per Kategori:")
    for category in test_data.keys():
        category_df = results_df[results_df['kategori'] == category]
        category_acc = category_df['benar'].mean() * 100
        print(f"- {category.upper()}: {category_acc:.1f}%")
    
    # Confusion matrix
    print("\n🧮 Confusion Matrix:")
    cm = pd.crosstab(
        results_df['kategori'], 
        results_df['prediksi'],
        rownames=['Actual'],
        colnames=['Predicted']
    )
    print(cm)
    
    # Cek jenis kesalahan
    print("\n❌ Jenis Kesalahan:")
    errors = results_df[results_df['benar'] == False]
    if not errors.empty:
        for i, row in errors.iterrows():
            print(f"- Teks: '{row['teks']}'")
            print(f"  Actual: {row['kategori']}, Predicted: {row['prediksi']}, Confidence: {row['confidence']:.2f}")
    else:
        print("Tidak ada kesalahan! 🎉")

if __name__ == "__main__":
    test_model()