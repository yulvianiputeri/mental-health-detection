# fix_nltk_error.py
"""
Script untuk memperbaiki error NLTK secara otomatis
Jalankan script ini jika mendapat error LookupError untuk vader_lexicon
"""

import nltk
import ssl
import os
import sys

def fix_ssl_certificate():
    """Fix SSL certificate issue di Windows/MacOS"""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
        print("✅ SSL certificate issue fixed")

def download_required_data():
    """Download semua data NLTK yang diperlukan"""
    required_packages = [
        'punkt',
        'vader_lexicon',
        'stopwords'
    ]
    
    print("🔽 Mulai download data NLTK...")
    
    for package in required_packages:
        try:
            # Cek apakah sudah ada
            if package == 'punkt':
                nltk.data.find('tokenizers/punkt')
                print(f"✅ {package} sudah tersedia")
            elif package == 'vader_lexicon':
                nltk.data.find('vader_lexicon')
                print(f"✅ {package} sudah tersedia")
            elif package == 'stopwords':
                nltk.data.find('corpora/stopwords')
                print(f"✅ {package} sudah tersedia")
        except LookupError:
            print(f"📥 Download {package}...")
            try:
                nltk.download(package)
                print(f"✅ {package} berhasil didownload")
            except Exception as e:
                print(f"❌ Error download {package}: {e}")

def verify_installation():
    """Verifikasi apakah data sudah terinstall dengan benar"""
    print("\n🔍 Memverifikasi instalasi...")
    
    # Test punkt
    try:
        from nltk.tokenize import sent_tokenize
        sent_tokenize("This is a test.")
        print("✅ punkt berfungsi dengan baik")
    except Exception as e:
        print(f"❌ punkt error: {e}")
    
    # Test vader_lexicon
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        analyzer.polarity_scores("This is a test.")
        print("✅ vader_lexicon berfungsi dengan baik")
    except Exception as e:
        print(f"❌ vader_lexicon error: {e}")

def check_python_environment():
    """Cek environment Python"""
    print(f"🐍 Python version: {sys.version}")
    print(f"📍 Python executable: {sys.executable}")
    print(f"📁 Current directory: {os.getcwd()}")
    
    # Cek virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️ Not in virtual environment")

def manual_instructions():
    """Berikan instruksi manual jika otomatis gagal"""
    print("\n" + "="*50)
    print("📋 INSTRUKSI MANUAL")
    print("="*50)
    print("Jika script ini gagal, coba langkah berikut:")
    print("\n1. Buka Python interpreter dan jalankan:")
    print("   python -c \"import nltk; nltk.download('vader_lexicon')\"")
    print("\n2. Atau jalankan di Python REPL:")
    print("   >>> import nltk")
    print("   >>> nltk.download('vader_lexicon')")
    print("   >>> nltk.download('punkt')")
    print("\n3. Restart aplikasi setelah download selesai")
    print("\n4. Pastikan Anda menggunakan virtual environment yang benar")

def main():
    print("🔧 NLTK Error Fixer")
    print("="*30)
    print("Script ini akan memperbaiki error NLTK yang umum terjadi.\n")
    
    # Cek environment
    check_python_environment()
    
    # Fix SSL
    fix_ssl_certificate()
    
    # Download data
    download_required_data()
    
    # Verify
    verify_installation()
    
    print("\n✅ Proses selesai!")
    print("Coba jalankan aplikasi lagi dengan: streamlit run app.py")
    
    # Instruksi manual jika masih bermasalah
    user_input = input("\nApakah masih ada error? (y/n): ").lower().strip()
    if user_input == 'y':
        manual_instructions()

if __name__ == "__main__":
    main()