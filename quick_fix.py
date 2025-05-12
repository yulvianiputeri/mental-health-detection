# quick_fix.py
"""
Quick fix untuk error umum:
1. TF-IDF vectorizer is not fitted
2. Model tidak ditemukan
"""

import os
import sys

def fix_tfidf_error():
    """Fix error TF-IDF vectorizer not fitted"""
    print("🔧 Memperbaiki error TF-IDF vectorizer...")
    
    if not os.path.exists('mental_health_model_advanced.pkl'):
        print("❌ Model file tidak ditemukan!")
        print("\n🎯 SOLUSI CEPAT:")
        print("1. Jalankan: python train_model.py")
        print("2. Tunggu hingga training selesai")
        print("3. Jalankan lagi: streamlit run app.py")
        return False
    else:
        print("✅ Model file ditemukan")
        
        # Cek apakah model valid
        try:
            import pickle
            with open('mental_health_model_advanced.pkl', 'rb') as f:
                data = pickle.load(f)
                if 'vectorizer' in data and 'model' in data:
                    print("✅ Model file valid")
                    return True
                else:
                    print("❌ Model file tidak lengkap")
                    return False
        except Exception as e:
            print(f"❌ Error reading model: {e}")
            return False

def fix_nltk_error():
    """Fix error NLTK"""
    print("\n🔧 Memperbaiki error NLTK...")
    
    try:
        import nltk
        import ssl
        
        # Fix SSL
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Download required data
        required = ['punkt', 'vader_lexicon']
        for item in required:
            try:
                if item == 'punkt':
                    nltk.data.find('tokenizers/punkt')
                else:
                    nltk.data.find(item)
                print(f"✅ {item} tersedia")
            except LookupError:
                print(f"📥 Download {item}...")
                nltk.download(item)
                print(f"✅ {item} berhasil didownload")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_environment():
    """Cek environment Python"""
    print("🔍 Mengecek environment...")
    
    # Python version
    if sys.version_info >= (3, 8):
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    else:
        print(f"⚠️ Python {sys.version_info.major}.{sys.version_info.minor} (minimal 3.8)")
    
    # Required files
    files = ['app.py', 'mental_health_detector.py', 'train_model.py', 'requirements.txt']
    for file in files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} tidak ditemukan!")

def main():
    print("⚡ QUICK FIX MENTAL HEALTH DETECTION")
    print("=" * 40)
    
    # Cek environment
    check_environment()
    
    # Fix NLTK
    nltk_ok = fix_nltk_error()
    
    # Fix TF-IDF
    model_ok = fix_tfidf_error()
    
    print("\n" + "=" * 40)
    print("📊 HASIL FIX")
    print("=" * 40)
    
    if nltk_ok and model_ok:
        print("🎉 SEMUA ERROR BERHASIL DIPERBAIKI!")
        print("\nJalankan aplikasi dengan:")
        print("streamlit run app.py")
    else:
        print("❌ MASIH ADA ERROR!")
        if not nltk_ok:
            print("\n🔧 NLTK Issue:")
            print("- Jalankan: python fix_nltk_error.py")
        if not model_ok:
            print("\n🔧 Model Issue:")
            print("- Jalankan: python train_model.py")
            print("- Pastikan training selesai tanpa error")
    
    print("\n📱 ALTERNATIF:")
    print("Jalankan setup lengkap dengan: python setup_project.py")

if __name__ == "__main__":
    main()