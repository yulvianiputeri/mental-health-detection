# validate_setup.py
"""
Script untuk memvalidasi setup sistem deteksi kesehatan mental
Memeriksa apakah semua dependensi dan file sudah benar
"""

import importlib
import pkg_resources
import os
import sys
from pathlib import Path

def check_python_version():
    """Cek versi Python"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ diperlukan!")
        return False
    else:
        print("✅ Versi Python sudah sesuai")
        return True

def check_requirements():
    """Cek apakah semua package sudah terinstall"""
    required_packages = [
        'streamlit>=1.28.0',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'scikit-learn>=1.3.0',
        'nltk>=3.8.0',
        'plotly>=5.17.0'
    ]
    
    print("\n📦 Mengecek package requirements...")
    all_installed = True
    
    for package in required_packages:
        try:
            pkg_name = package.split('>=')[0]
            version_required = package.split('>=')[1] if '>=' in package else None
            
            # Coba import
            if pkg_name == 'scikit-learn':
                import sklearn
                installed_version = sklearn.__version__
            else:
                module = importlib.import_module(pkg_name)
                installed_version = getattr(module, '__version__', 'Unknown')
            
            print(f"✅ {pkg_name}: {installed_version}")
            
        except ImportError:
            print(f"❌ {pkg_name} belum terinstall!")
            all_installed = False
        except Exception as e:
            print(f"⚠️ Error checking {pkg_name}: {e}")
    
    return all_installed

def check_nltk_data():
    """Cek apakah NLTK data sudah terdownload"""
    print("\n📚 Mengecek NLTK data...")
    
    try:
        import nltk
        
        # Check punkt
        try:
            nltk.data.find('tokenizers/punkt')
            print("✅ NLTK punkt tokenizer tersedia")
        except LookupError:
            print("❌ NLTK punkt belum terdownload!")
            print("   Jalankan: python -c \"import nltk; nltk.download('punkt')\"")
            return False
        
        # Check vader_lexicon
        try:
            nltk.data.find('vader_lexicon')
            print("✅ NLTK vader_lexicon tersedia")
        except LookupError:
            print("❌ NLTK vader_lexicon belum terdownload!")
            print("   Jalankan: python -c \"import nltk; nltk.download('vader_lexicon')\"")
            return False
        
        return True
        
    except ImportError:
        print("❌ NLTK belum terinstall!")
        return False

def check_files():
    """Cek apakah file-file penting ada"""
    print("\n📁 Mengecek file struktur...")
    
    required_files = [
        'app.py',
        'mental_health_detector.py',
        'train_model.py',
        'requirements.txt'
    ]
    
    all_present = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} tersedia")
        else:
            print(f"❌ {file} tidak ditemukan!")
            all_present = False
    
    # Check if model exists
    if os.path.exists('mental_health_model_advanced.pkl'):
        print("✅ Model terlatih ditemukan")
    else:
        print("⚠️ Model belum dilatih - jalankan train_model.py terlebih dahulu")
    
    return all_present

def validate_imports():
    """Validasi impor Python untuk memastikan tidak ada error sintaks"""
    print("\n🔍 Validasi impor Python...")
    
    files_to_check = [
        'mental_health_detector.py',
        'train_model.py'
    ]
    
    all_valid = True
    
    for file in files_to_check:
        if os.path.exists(file):
            try:
                # Baca file dan compile
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    compile(content, file, 'exec')
                print(f"✅ {file} valid")
            except SyntaxError as e:
                print(f"❌ {file} memiliki syntax error: {e}")
                all_valid = False
            except Exception as e:
                print(f"⚠️ Error checking {file}: {e}")
        else:
            print(f"❌ {file} tidak ditemukan!")
            all_valid = False
    
    return all_valid

def run_basic_test():
    """Jalankan test dasar untuk memastikan sistem bisa berjalan"""
    print("\n🧪 Menjalankan test dasar...")
    
    try:
        # Test import mental_health_detector
        from mental_health_detector import AdvancedMentalHealthDetector
        print("✅ AdvancedMentalHealthDetector berhasil diimpor")
        
        # Test inisialisasi
        detector = AdvancedMentalHealthDetector()
        print("✅ AdvancedMentalHealthDetector berhasil diinisialisasi")
        
        # Test preprocessing
        processed, original = detector.preprocess_text("Ini adalah test 😊")
        print("✅ Preprocessing text berfungsi")
        
        # Test feature extraction
        features = detector.extract_advanced_features("Test text dengan emoji 😊")
        print("✅ Feature extraction berfungsi")
        
        return True
        
    except Exception as e:
        print(f"❌ Error dalam test: {e}")
        return False

def main():
    """Fungsi utama validasi"""
    print("🔧 VALIDASI SETUP SISTEM DETEKSI KESEHATAN MENTAL")
    print("=" * 50)
    
    results = {
        'python_version': check_python_version(),
        'requirements': check_requirements(),
        'nltk_data': check_nltk_data(),
        'files': check_files(),
        'imports': validate_imports(),
        'basic_test': run_basic_test()
    }
    
    print("\n" + "=" * 50)
    print("📊 RINGKASAN VALIDASI")
    print("=" * 50)
    
    for check, result in results.items():
        status = "✅ BERHASIL" if result else "❌ GAGAL"
        print(f"{check.replace('_', ' ').title()}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 SEMUA VALIDASI BERHASIL!")
        print("Sistem siap digunakan. Jalankan:")
        print("1. python train_model.py (jika model belum dilatih)")
        print("2. streamlit run app.py")
    else:
        print("\n⚠️ ADA MASALAH YANG PERLU DIPERBAIKI!")
        print("Silakan selesaikan masalah di atas sebelum melanjutkan.")
    
    return all_passed

if __name__ == "__main__":
    main()