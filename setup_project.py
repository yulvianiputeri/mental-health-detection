# setup_project.py
"""
Script setup otomatis untuk Mental Health Detection System
Mengatasi semua error umum dan mempersiapkan aplikasi untuk dijalankan
"""

import os
import sys
import subprocess
import pickle
from pathlib import Path

def check_python_version():
    """Cek versi Python"""
    print("🐍 Mengecek versi Python...")
    if sys.version_info >= (3, 8):
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} OK")
        return True
    else:
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} terlalu lama")
        print("   Minimal Python 3.8 diperlukan")
        return False

def check_virtual_environment():
    """Cek apakah menggunakan virtual environment"""
    print("\n🔧 Mengecek virtual environment...")
    
    # Cek apakah di virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment aktif")
        return True
    else:
        print("⚠️ Tidak menggunakan virtual environment")
        print("   Disarankan menggunakan virtual environment")
        return False

def install_requirements():
    """Install requirements.txt"""
    print("\n📦 Menginstall dependencies...")
    
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt tidak ditemukan!")
        return False
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies berhasil diinstall")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def setup_nltk():
    """Setup NLTK data"""
    print("\n📚 Mengsetup NLTK...")
    
    try:
        import nltk
        import ssl
        
        # Handle SSL issue
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Download required NLTK data
        required_data = ['punkt', 'vader_lexicon', 'stopwords']
        
        for data in required_data:
            try:
                # Check if already downloaded
                if data == 'punkt':
                    nltk.data.find('tokenizers/punkt')
                elif data == 'vader_lexicon':
                    nltk.data.find('vader_lexicon')
                elif data == 'stopwords':
                    nltk.data.find('corpora/stopwords')
                
                print(f"✅ {data} sudah tersedia")
            except LookupError:
                print(f"📥 Download {data}...")
                nltk.download(data, quiet=True)
                print(f"✅ {data} berhasil didownload")
        
        return True
    except Exception as e:
        print(f"❌ Error setting up NLTK: {e}")
        return False

def check_model_file():
    """Cek apakah model sudah tersedia"""
    print("\n🤖 Mengecek model file...")
    
    model_file = 'mental_health_model_advanced.pkl'
    
    if os.path.exists(model_file):
        print("✅ Model file ditemukan")
        
        # Cek apakah model valid
        try:
            with open(model_file, 'rb') as f:
                pickle.load(f)
            print("✅ Model file valid")
            return True
        except Exception as e:
            print(f"❌ Model file corrupt: {e}")
            return False
    else:
        print("❌ Model file tidak ditemukan")
        return False

def train_model():
    """Train model jika belum ada"""
    print("\n🎓 Training model...")
    
    try:
        # Import dan jalankan training
        from train_model import main as train_main
        train_main()
        print("✅ Model berhasil dilatih")
        return True
    except Exception as e:
        print(f"❌ Error training model: {e}")
        print("Coba jalankan manual: python train_model.py")
        return False

def verify_installation():
    """Verifikasi instalasi lengkap"""
    print("\n🔍 Verifikasi instalasi...")
    
    try:
        # Test import modules
        from mental_health_detector import AdvancedMentalHealthDetector
        print("✅ mental_health_detector dapat diimport")
        
        # Test create detector
        detector = AdvancedMentalHealthDetector()
        print("✅ AdvancedMentalHealthDetector dapat diinisialisasi")
        
        # Test load model
        if os.path.exists('mental_health_model_advanced.pkl'):
            detector.load_model('mental_health_model_advanced.pkl')
            print("✅ Model dapat dimuat")
            
            # Test prediction
            test_text = "Saya merasa senang hari ini"
            result = detector.predict(test_text)
            print("✅ Prediksi berfungsi")
            print(f"   Test prediction: {result['condition']} ({result['confidence']:.1%})")
        
        return True
    except Exception as e:
        print(f"❌ Error verifikasi: {e}")
        return False

def main():
    """Fungsi utama setup"""
    print("🧠 SETUP MENTAL HEALTH DETECTION SYSTEM")
    print("=" * 50)
    
    checks = []
    
    # 1. Cek Python version
    checks.append(check_python_version())
    
    # 2. Cek virtual environment
    check_virtual_environment()  # Warning tapi tidak critical
    
    # 3. Install requirements
    checks.append(install_requirements())
    
    # 4. Setup NLTK
    checks.append(setup_nltk())
    
    # 5. Cek model file
    model_exists = check_model_file()
    
    # 6. Train model jika belum ada
    if not model_exists:
        print("\n⚠️ Model belum tersedia. Memulai training...")
        answer = input("Training memerlukan waktu. Lanjutkan? (y/n): ").lower().strip()
        if answer == 'y':
            checks.append(train_model())
        else:
            print("❌ Training dibatalkan")
            checks.append(False)
    else:
        checks.append(True)
    
    # 7. Verifikasi final
    if all(checks):
        checks.append(verify_installation())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY SETUP")
    print("=" * 50)
    
    if all(checks):
        print("🎉 SETUP BERHASIL!")
        print("\nAplikasi siap digunakan.")
        print("Jalankan: streamlit run app.py")
    else:
        print("❌ SETUP GAGAL!")
        print("\nBeberapa komponen bermasalah.")
        print("Silakan cek error di atas dan jalankan ulang setup.")
        
        # Berikan instruksi manual
        print("\n📋 INSTRUKSI MANUAL:")
        print("1. Pastikan Python 3.8+ terinstall")
        print("2. Buat virtual environment: python -m venv venv")
        print("3. Aktifkan venv: venv\\Scripts\\activate (Windows) atau source venv/bin/activate (Linux/Mac)")
        print("4. Install requirements: pip install -r requirements.txt")
        print("5. Download NLTK data: python download_nltk_data.py")
        print("6. Train model: python train_model.py")
        print("7. Jalankan app: streamlit run app.py")

if __name__ == "__main__":
    main()
