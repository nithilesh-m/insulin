#!/usr/bin/env python3
"""
MongoDB Setup Script for T2D Insulin Prediction Tool

This script helps you set up and test MongoDB connection for the application.
"""

import sys
import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

def test_mongodb_connection(uri="mongodb://localhost:27017/"):
    """Test MongoDB connection"""
    try:
        print("Testing MongoDB connection...")
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        
        # Test the connection
        client.admin.command('ping')
        print("✅ MongoDB connection successful!")
        
        # Test database creation
        db = client['protein_prediction_db']
        collection = db['users']
        
        # Test a simple operation
        collection.find_one()
        print("✅ Database and collection access successful!")
        
        client.close()
        return True
        
    except ConnectionFailure:
        print("❌ Failed to connect to MongoDB")
        print("   Make sure MongoDB is running on your system")
        return False
    except ServerSelectionTimeoutError:
        print("❌ MongoDB server selection timeout")
        print("   Check if MongoDB is running and accessible")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

def create_sample_user(uri="mongodb://localhost:27017/"):
    """Create a sample user for testing"""
    try:
        client = MongoClient(uri)
        db = client['protein_prediction_db']
        collection = db['users']
        
        # Check if sample user already exists
        existing_user = collection.find_one({'username': 'testuser'})
        if existing_user:
            print("ℹ️  Sample user 'testuser' already exists")
            return True
        
        # Create sample user (password will be hashed by the app)
        sample_user = {
            'username': 'testuser',
            'password': 'testpass123',  # This will be hashed by bcrypt in the app
            'created_at': None,  # Will be set by the app
            'last_login': None
        }
        
        result = collection.insert_one(sample_user)
        print("✅ Sample user created successfully!")
        print("   Username: testuser")
        print("   Password: testpass123")
        return True
        
    except Exception as e:
        print(f"❌ Error creating sample user: {str(e)}")
        return False
    finally:
        client.close()

def main():
    print("=" * 50)
    print("MongoDB Setup for T2D Insulin Prediction Tool")
    print("=" * 50)
    
    # Test connection
    if not test_mongodb_connection():
        print("\n🔧 Troubleshooting steps:")
        print("1. Install MongoDB Community Server")
        print("2. Start MongoDB service:")
        print("   - Windows: MongoDB should start automatically")
        print("   - macOS: brew services start mongodb-community")
        print("   - Linux: sudo systemctl start mongod")
        print("3. Verify MongoDB is running: mongosh")
        print("4. Run this script again")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("MongoDB Setup Complete!")
    print("=" * 50)
    print("\n📋 Next steps:")
    print("1. Start the backend server: cd backend && python app.py")
    print("2. Start the frontend: npm run dev")
    print("3. Open http://localhost:5173 in your browser")
    print("4. Register a new account or use the sample user")
    
    # Ask if user wants to create a sample user
    response = input("\n❓ Would you like to create a sample user for testing? (y/n): ")
    if response.lower() in ['y', 'yes']:
        if create_sample_user():
            print("\n🎉 Setup complete! You can now run the application.")
        else:
            print("\n⚠️  Sample user creation failed, but you can still register manually.")

if __name__ == "__main__":
    main()
