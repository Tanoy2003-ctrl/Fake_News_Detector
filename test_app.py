#!/usr/bin/env python3
"""
Test script to verify the Streamlit app is working
"""

import requests
import time

def test_streamlit_app():
    """Test if the Streamlit app is running and accessible"""
    url = "http://localhost:8501"
    
    print("🔍 Testing Fake News Detector Streamlit App...")
    print(f"📡 Checking URL: {url}")
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print("✅ App is running successfully!")
            print(f"📊 Status Code: {response.status_code}")
            print(f"🌐 App URL: {url}")
            print("\n🎯 Next Steps:")
            print("1. Open your web browser")
            print(f"2. Navigate to: {url}")
            print("3. Click 'Load & Train Model' in the sidebar")
            print("4. Start analyzing news articles!")
            return True
        else:
            print(f"⚠️ App responded with status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to the app. Is it running?")
        print("💡 Try running: python run_app.py")
        return False
    except requests.exceptions.Timeout:
        print("⏰ Connection timed out. The app might be starting up...")
        return False
    except Exception as e:
        print(f"❌ Error testing app: {e}")
        return False

def check_requirements():
    """Check if required packages are installed"""
    print("\n📦 Checking requirements...")
    
    required_packages = [
        'streamlit',
        'plotly', 
        'pandas',
        'sklearn',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("💡 Install with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages are installed!")
        return True

def main():
    """Main test function"""
    print("🔍 FAKE NEWS DETECTOR - APP TEST")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Test app
    print("\n" + "=" * 50)
    if test_streamlit_app():
        print("\n🎉 Everything looks good! Your app is ready to use.")
    else:
        print("\n🔧 Troubleshooting:")
        print("1. Make sure the app is running: python run_app.py")
        print("2. Check if port 8501 is available")
        print("3. Verify all requirements are installed")

if __name__ == "__main__":
    main()