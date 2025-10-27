#!/usr/bin/env python3
"""
Setup script to create .env file with API credentials
"""

import os
import secrets

def create_env_file():
    """Create .env file with template credentials"""
    
    env_content = """# T2D Insulin Prediction Tool - Environment Variables
# Fill in your actual API credentials below

# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017/

# Flask Session Secret Key (auto-generated)
SECRET_KEY={secret_key}

# Google OAuth Configuration
# Get these from: https://console.cloud.google.com/apis/credentials
GOOGLE_CLIENT_ID=your-google-client-id-here
GOOGLE_CLIENT_SECRET=your-google-client-secret-here

# Email Configuration (for OTP)
# Enable 2-Step Verification and generate App Password at: https://myaccount.google.com/apppasswords
EMAIL_ADDRESS=your-gmail-address@gmail.com
EMAIL_PASSWORD=your-gmail-app-password

# Twilio Configuration (for SMS OTP)
# Get these from: https://www.twilio.com/console
TWILIO_ACCOUNT_SID=your-twilio-account-sid
TWILIO_AUTH_TOKEN=your-twilio-auth-token
TWILIO_PHONE_NUMBER=your-twilio-phone-number
DEFAULT_COUNTRY_CODE=+91

# Development Settings
DEV_FAKE_OTP=false
""".format(secret_key=secrets.token_hex(32))
    
    # Create .env file in backend directory
    env_path = "backend/.env"
    
    if os.path.exists(env_path):
        print(f"⚠️  .env file already exists at {env_path}")
        response = input("Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            print("❌ Setup cancelled")
            return
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print(f"✅ Created .env file at {env_path}")
    print("\n📋 Next steps:")
    print("1. Edit backend/.env file")
    print("2. Fill in your actual API credentials:")
    print("   - Google OAuth: https://console.cloud.google.com/apis/credentials")
    print("   - Twilio: https://www.twilio.com/console")
    print("   - Gmail App Password: https://myaccount.google.com/apppasswords")
    print("3. Restart your servers")

def show_credential_locations():
    """Show where to find API credentials"""
    
    print("\n🔍 Where to find your API credentials:")
    print("=" * 50)
    
    print("\n1. Google OAuth Credentials:")
    print("   📍 Go to: https://console.cloud.google.com/apis/credentials")
    print("   📝 Create OAuth 2.0 Client ID")
    print("   🔗 Authorized redirect URIs:")
    print("      - http://localhost:5001/api/auth/google/callback")
    print("      - http://localhost:5173/api/auth/google/callback")
    
    print("\n2. Twilio Credentials:")
    print("   📍 Go to: https://www.twilio.com/console")
    print("   📝 Find: Account SID, Auth Token, Phone Number")
    
    print("\n3. Gmail App Password:")
    print("   📍 Go to: https://myaccount.google.com/apppasswords")
    print("   📝 Enable 2-Step Verification first")
    print("   📝 Generate App Password for 'Mail'")
    
    print("\n4. MongoDB:")
    print("   📍 Local: mongodb://localhost:27017/")
    print("   📍 Atlas: Get connection string from MongoDB Atlas")

if __name__ == "__main__":
    print("🔧 Setting up API credentials...")
    create_env_file()
    show_credential_locations()

