# Google OAuth Setup Guide

## Prerequisites
1. A Google Cloud Platform account
2. Access to Google Cloud Console

## Setup Steps

### 1. Create a Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Note down your project ID

### 2. Enable Google+ API
1. In the Google Cloud Console, go to "APIs & Services" > "Library"
2. Search for "Google+ API" and enable it
3. Also enable "Google OAuth2 API"

### 3. Create OAuth 2.0 Credentials
1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth 2.0 Client IDs"
3. Choose "Web application" as the application type
4. Set the name (e.g., "T2D Insulin Predictor")
5. Add authorized redirect URIs:
   - `http://localhost:5001/api/auth/google/callback`
   - `http://127.0.0.1:5001/api/auth/google/callback`
6. Click "Create"
7. Copy the Client ID and Client Secret

### 4. Set Environment Variables
Create a `.env` file in the backend directory with:

```env
GOOGLE_CLIENT_ID=your_client_id_here
GOOGLE_CLIENT_SECRET=your_client_secret_here
SECRET_KEY=your_secret_key_here
MONGO_URI=mongodb://localhost:27017/
```

### 5. Test the Setup
1. Start the backend server: `python combined_server.py`
2. Start the frontend: `npm run dev`
3. Navigate to `http://localhost:5173`
4. Click "Continue with Google" to test the OAuth flow

## Troubleshooting

### Common Issues:
1. **"OAuth not configured" error**: Make sure environment variables are set correctly
2. **"Invalid redirect URI" error**: Check that the redirect URI in Google Console matches exactly
3. **"Access blocked" error**: Make sure the OAuth consent screen is configured properly

### OAuth Consent Screen Setup:
1. Go to "APIs & Services" > "OAuth consent screen"
2. Choose "External" user type
3. Fill in the required fields:
   - App name: "T2D Insulin Predictor"
   - User support email: your email
   - Developer contact: your email
4. Add scopes: `openid`, `email`, `profile`
5. Add test users if needed

## Security Notes
- Never commit the `.env` file to version control
- Use environment variables in production
- Regularly rotate your OAuth credentials
- Monitor OAuth usage in Google Cloud Console
