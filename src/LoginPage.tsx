import React, { useState } from 'react';
import { Eye, EyeOff, User, Lock, LogIn, UserPlus, AlertCircle, CheckCircle } from 'lucide-react';

interface LoginPageProps {
  onLogin: (user: { id: string; username: string }) => void;
}

interface AuthResponse {
  success: boolean;
  message: string;
  user?: {
    id: string;
    username: string;
  };
  error?: string;
}

const LoginPage: React.FC<LoginPageProps> = ({ onLogin }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [selectedAuthMethod, setSelectedAuthMethod] = useState<'default' | 'google' | 'mobile'>('default');
  // Email OTP flow
  const [email, setEmail] = useState('');
  const [emailOtp, setEmailOtp] = useState('');
  const [emailStep, setEmailStep] = useState<'collect' | 'otp' | 'setPassword'>('collect');
  // Mobile OTP flow
  const [mobile, setMobile] = useState('');
  const [mobileOtp, setMobileOtp] = useState('');
  const [mobileStep, setMobileStep] = useState<'collect' | 'otp' | 'setPassword'>('collect');
  // New password for OTP signup
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

  // Password strength helper
  const passwordStrength = (pwd: string) => {
    let score = 0;
    if (pwd.length >= 8) score++;
    if (/[A-Z]/.test(pwd)) score++;
    if (/[a-z]/.test(pwd)) score++;
    if (/\d/.test(pwd)) score++;
    if (/[^A-Za-z0-9]/.test(pwd)) score++;
    const percent = Math.min(100, (score / 5) * 100);
    const label = score <= 2 ? 'Weak' : score === 3 ? 'Fair' : score === 4 ? 'Good' : 'Strong';
    const color = score <= 2 ? 'bg-red-500' : score === 3 ? 'bg-yellow-500' : score === 4 ? 'bg-green-500' : 'bg-emerald-600';
    return { score, percent, label, color };
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const endpoint = isLogin ? '/api/login' : '/api/register';
      const response = await fetch(`http://localhost:5001${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ username, password }),
      });

      const data: AuthResponse = await response.json();

      if (data.success) {
        setSuccess(data.message);
        if (isLogin && data.user) {
          // Small delay to show success message
          setTimeout(() => {
            onLogin(data.user!);
          }, 1000);
        } else if (!isLogin) {
          // Switch to login mode after successful registration
          setTimeout(() => {
            setIsLogin(true);
            setSuccess(null);
          }, 2000);
        }
      } else {
        setError(data.error || 'An error occurred');
      }
    } catch (err) {
      console.error('Auth error:', err);
      setError('Unable to connect to the server. Please ensure the backend is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const toggleMode = () => {
    setIsLogin(!isLogin);
    setError(null);
    setSuccess(null);
    setUsername('');
    setPassword('');
    // reset OTP flows
    setSelectedAuthMethod('default');
    setEmail('');
    setEmailOtp('');
    setEmailStep('collect');
    setMobile('');
    setMobileOtp('');
    setMobileStep('collect');
    setNewPassword('');
    setConfirmPassword('');
  };

  // Change selected auth method and reset OTP flows
  const handleSelectAuth = (method: 'default' | 'google' | 'mobile') => {
    const next = selectedAuthMethod === method ? 'default' : method;
    setSelectedAuthMethod(next);
    setError(null);
    setSuccess(null);
    setEmail('');
    setEmailOtp('');
    setEmailStep('collect');
    setMobile('');
    setMobileOtp('');
    setMobileStep('collect');
    setNewPassword('');
    setConfirmPassword('');
  };

  // Google OAuth handler - Simple redirect approach
  const handleGoogleAuth = () => {
    setIsLoading(true);
    setError(null);

    // Direct redirect to Google OAuth endpoint
    window.location.href = 'http://localhost:5001/api/auth/google';
  };

  // Email OTP handlers
  const requestEmailOtp = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const res = await fetch('http://localhost:5001/api/request-otp-email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ email }),
      });
      const data = await res.json();
      if (data.success) {
        setSuccess('OTP sent to email');
        setEmailStep('otp');
      } else {
        setError(data.error || 'Failed to send OTP');
      }
    } catch (e) {
      setError('Unable to request OTP');
    } finally {
      setIsLoading(false);
    }
  };

  const verifyEmailOtp = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const res = await fetch('http://localhost:5001/api/verify-otp-email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ email, otp: emailOtp }),
      });
      const data = await res.json();
      if (data.success) {
        setSuccess('OTP verified');
        setEmailStep('setPassword');
      } else {
        setError(data.error || 'Invalid OTP');
      }
    } catch (e) {
      setError('Unable to verify OTP');
    } finally {
      setIsLoading(false);
    }
  };

  const setEmailPassword = async () => {
    if (newPassword !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }
    try {
      setIsLoading(true);
      setError(null);
      const res = await fetch('http://localhost:5001/api/set-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ password: newPassword }),
      });
      const data: AuthResponse = await res.json();
      if (data.success && data.user) {
        setSuccess('Account created');
        onLogin(data.user);
      } else {
        setError(data.error || 'Failed to set password');
      }
    } catch (e) {
      setError('Unable to set password');
    } finally {
      setIsLoading(false);
    }
  };

  // Mobile OTP handlers
  const requestMobileOtp = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const res = await fetch('http://localhost:5001/api/request-otp-mobile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ mobile, username }),
      });
      const data = await res.json();
      if (data.success) {
        setSuccess('OTP sent to mobile');
        setMobileStep('otp');
      } else {
        setError(data.error || 'Failed to send OTP');
      }
    } catch (e) {
      setError('Unable to request OTP');
    } finally {
      setIsLoading(false);
    }
  };

  const verifyMobileOtp = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const res = await fetch('http://localhost:5001/api/verify-otp-mobile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ mobile, otp: mobileOtp }),
      });
      const data = await res.json();
      if (data.success) {
        setSuccess('OTP verified');
        setMobileStep('setPassword');
      } else {
        setError(data.error || 'Invalid OTP');
      }
    } catch (e) {
      setError('Unable to verify OTP');
    } finally {
      setIsLoading(false);
    }
  };

  const setMobilePassword = async () => {
    if (newPassword !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }
    try {
      setIsLoading(true);
      setError(null);
      const res = await fetch('http://localhost:5001/api/set-password-mobile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ password: newPassword }),
      });
      const data: AuthResponse = await res.json();
      if (data.success && data.user) {
        setSuccess('Account created');
        onLogin(data.user);
      } else {
        setError(data.error || 'Failed to set password');
      }
    } catch (e) {
      setError('Unable to set password');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center bg-gradient-to-br from-blue-900 via-indigo-900 to-purple-900 relative overflow-hidden">
      {/* Background particles effect */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-20 left-20 w-2 h-2 bg-white rounded-full opacity-20 animate-pulse"></div>
        <div className="absolute top-40 right-32 w-1 h-1 bg-blue-300 rounded-full opacity-30 animate-pulse"></div>
        <div className="absolute bottom-32 left-16 w-2 h-2 bg-purple-300 rounded-full opacity-25 animate-pulse"></div>
        <div className="absolute top-1/3 left-1/4 w-1 h-1 bg-white rounded-full opacity-40 animate-pulse"></div>
        <div className="absolute bottom-20 right-20 w-2 h-2 bg-blue-200 rounded-full opacity-20 animate-pulse"></div>
        <div className="absolute top-2/3 right-1/4 w-1 h-1 bg-purple-200 rounded-full opacity-30 animate-pulse"></div>
      </div>

          {/* Logo/Avatar */}
      <div className="relative z-10 flex flex-col items-center mb-8">
        <div className="w-20 h-20 rounded-full bg-gradient-to-r from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg mb-4">
          <span className="text-white text-3xl font-bold">T2D</span>
            </div>
        <span className="text-2xl font-bold text-white">Insulin Predictor</span>
        <p className="text-blue-200 text-sm mt-2">
          {isLogin ? 'Sign in to access the protein prediction tool' : 'Create a new account to get started'}
            </p>
          </div>

      {/* Three Authentication Blocks */}
      <div className="relative z-10 w-full max-w-6xl px-4 mb-12">
        <div className="relative flex items-center justify-center h-96 overflow-hidden">
          {/* Liquid background effect */}
          <div className="absolute inset-0 bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-green-500/10 rounded-full blur-3xl animate-pulse"></div>
          {/* Guest Authentication Block */}
          <div 
            className={`absolute bg-white/95 backdrop-blur-md rounded-2xl shadow-lg border-2 cursor-pointer transform hover:scale-105 w-80 ${
              selectedAuthMethod === 'default' 
                ? 'border-blue-500 shadow-2xl z-20' 
                : 'border-gray-200 hover:border-blue-300'
            }`}
            onClick={() => handleSelectAuth('default')}
            style={{
              transition: 'all 1.2s cubic-bezier(0.175, 0.885, 0.32, 1.275)',
              transform: selectedAuthMethod === 'default' 
                ? 'translateX(0) scale(1.1) rotateY(0deg)' 
                : selectedAuthMethod === 'google'
                ? 'translateX(-400px) scale(0.9) rotateY(-10deg)'
                : selectedAuthMethod === 'mobile'
                ? 'translateX(-400px) scale(0.9) rotateY(-10deg)'
                : 'translateX(-400px) scale(0.9) rotateY(-10deg)',
              filter: selectedAuthMethod === 'default' 
                ? 'brightness(1) saturate(1.1)' 
                : 'brightness(0.8) saturate(0.7)',
              zIndex: selectedAuthMethod === 'default' ? 30 : 10
            }}
          >
            <div className="p-8 text-center">
              <div className={`w-16 h-16 rounded-full mx-auto mb-4 flex items-center justify-center transition-all duration-300 ${
                selectedAuthMethod === 'default' 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-100 text-gray-600'
              }`}>
                <User className="w-8 h-8" />
              </div>
              <h3 className={`text-xl font-bold mb-2 transition-colors duration-300 ${
                selectedAuthMethod === 'default' ? 'text-blue-600' : 'text-gray-800'
              }`}>
                Guest Login
              </h3>
              <p className="text-gray-600 text-sm mb-4">
                Sign in with username and password
              </p>
            {selectedAuthMethod === 'default' && (
                <div className="space-y-4 animate-fadeIn">
            <div className="relative">
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                      className="w-full pl-10 pr-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-600 focus:border-transparent transition-all duration-300 bg-white/90 hover:bg-white shadow-sm"
                placeholder="Username"
                disabled={isLoading}
                      onClick={(e) => e.stopPropagation()}
              />
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <User className="h-5 w-5 text-gray-400" />
              </div>
            </div>
            <div className="relative">
              <input
                type={showPassword ? 'text' : 'password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                      className="w-full pl-10 pr-10 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-600 focus:border-transparent transition-all duration-300 bg-white/90 hover:bg-white shadow-sm"
                placeholder="Password"
                disabled={isLoading}
                      onClick={(e) => e.stopPropagation()}
              />
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Lock className="h-5 w-5 text-gray-400" />
              </div>
              <button
                type="button"
                className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-700 transition-colors duration-300"
                      onClick={(e) => {
                        e.stopPropagation();
                        setShowPassword((prev) => !prev);
                      }}
              >
                {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
              </button>
            </div>
                  
                  {/* Password Strength Meter for Registration */}
                  {!isLogin && password && (
                    <div className="space-y-1">
                      <div className="h-2 w-full bg-gray-200 rounded-full overflow-hidden">
                        <div 
                          className={`h-2 transition-all duration-300 ${passwordStrength(password).color}`} 
                          style={{ width: `${passwordStrength(password).percent}%` }}
                        ></div>
                      </div>
                      <div className="text-xs text-gray-600">
                        Password strength: <span className="font-medium">{passwordStrength(password).label}</span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Google Authentication Block */}
          <div 
            className={`absolute bg-white/95 backdrop-blur-md rounded-2xl shadow-lg border-2 cursor-pointer transform hover:scale-105 w-80 ${
              selectedAuthMethod === 'google' 
                ? 'border-red-500 shadow-2xl z-20' 
                : 'border-gray-200 hover:border-red-300'
            }`}
            onClick={() => handleSelectAuth('google')}
            style={{
              transition: 'all 1.2s cubic-bezier(0.175, 0.885, 0.32, 1.275)',
              transform: selectedAuthMethod === 'google' 
                ? 'translateX(0) scale(1.1) rotateY(0deg)' 
                : selectedAuthMethod === 'default'
                ? 'translateX(-400px) scale(0.9) rotateY(-10deg)'
                : selectedAuthMethod === 'mobile'
                ? 'translateX(400px) scale(0.9) rotateY(10deg)'
                : 'translateX(-400px) scale(0.9) rotateY(-10deg)',
              filter: selectedAuthMethod === 'google' 
                ? 'brightness(1) saturate(1.1)' 
                : 'brightness(0.8) saturate(0.7)',
              zIndex: selectedAuthMethod === 'google' ? 30 : 10
            }}
          >
            <div className="p-8 text-center">
              <div className={`w-16 h-16 rounded-full mx-auto mb-4 flex items-center justify-center transition-all duration-300 ${
                selectedAuthMethod === 'google' 
                  ? 'bg-red-500 text-white' 
                  : 'bg-gray-100 text-gray-600'
              }`}>
                <svg className="w-8 h-8" viewBox="0 0 24 24">
                  <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                  <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                  <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                  <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                </svg>
              </div>
              <h3 className={`text-xl font-bold mb-2 transition-colors duration-300 ${
                selectedAuthMethod === 'google' ? 'text-red-600' : 'text-gray-800'
              }`}>
                Google Sign In
              </h3>
              <p className="text-gray-600 text-sm">
                Continue with your Google account
              </p>
            </div>
          </div>

          {/* Mobile Authentication Block */}
          <div 
            className={`absolute bg-white/95 backdrop-blur-md rounded-2xl shadow-lg border-2 cursor-pointer transform hover:scale-105 w-80 ${
              selectedAuthMethod === 'mobile' 
                ? 'border-green-500 shadow-2xl z-20' 
                : 'border-gray-200 hover:border-green-300'
            }`}
                onClick={() => handleSelectAuth('mobile')}
            style={{
              transition: 'all 1.2s cubic-bezier(0.175, 0.885, 0.32, 1.275)',
              transform: selectedAuthMethod === 'mobile' 
                ? 'translateX(0) scale(1.1) rotateY(0deg)' 
                : selectedAuthMethod === 'default'
                ? 'translateX(400px) scale(0.9) rotateY(10deg)'
                : selectedAuthMethod === 'google'
                ? 'translateX(400px) scale(0.9) rotateY(10deg)'
                : 'translateX(400px) scale(0.9) rotateY(10deg)',
              filter: selectedAuthMethod === 'mobile' 
                ? 'brightness(1) saturate(1.1)' 
                : 'brightness(0.8) saturate(0.7)',
              zIndex: selectedAuthMethod === 'mobile' ? 30 : 10
            }}
          >
            <div className="p-8 text-center">
              <div className={`w-16 h-16 rounded-full mx-auto mb-4 flex items-center justify-center transition-all duration-300 ${
                selectedAuthMethod === 'mobile' 
                  ? 'bg-green-500 text-white' 
                  : 'bg-gray-100 text-gray-600'
              }`}>
                <svg className="w-8 h-8" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                  <rect width="16" height="20" x="4" y="2" rx="2"/>
                  <path d="M8 6h8M9 18h6"/>
                </svg>
              </div>
              <h3 className={`text-xl font-bold mb-2 transition-colors duration-300 ${
                selectedAuthMethod === 'mobile' ? 'text-green-600' : 'text-gray-800'
              }`}>
                Mobile OTP
              </h3>
              <p className="text-gray-600 text-sm">
                Sign in with mobile number and OTP
              </p>
            {selectedAuthMethod === 'mobile' && (
                <div className="space-y-4 animate-fadeIn mt-4">
                {mobileStep === 'collect' && (
                    <div className="space-y-3">
                    <input
                      type="text"
                      value={username}
                      onChange={(e) => setUsername(e.target.value)}
                      placeholder="Choose a username"
                        className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-transparent bg-white/90 transition-all duration-300 hover:bg-white shadow-sm"
                      disabled={isLoading}
                        onClick={(e) => e.stopPropagation()}
                    />
                      <div className="flex gap-2">
                      <input
                        type="tel"
                        value={mobile}
                        onChange={(e) => setMobile(e.target.value)}
                          placeholder="Mobile number"
                          className="flex-1 px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-transparent bg-white/90 transition-all duration-300 hover:bg-white shadow-sm"
                        disabled={isLoading}
                          onClick={(e) => e.stopPropagation()}
                      />
                      <button 
                        type="button" 
                          onClick={(e) => {
                            e.stopPropagation();
                            requestMobileOtp();
                          }}
                        disabled={!mobile || !username || isLoading}
                          className="px-4 py-3 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-xl font-medium disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed hover:from-green-600 hover:to-green-700 transition-all duration-300 text-sm"
                      >
                        Send OTP
                      </button>
                    </div>
                  </div>
                )}
                {mobileStep === 'otp' && (
                    <div className="flex gap-2">
                    <input
                      type="text"
                      value={mobileOtp}
                      onChange={(e) => setMobileOtp(e.target.value)}
                      placeholder="Enter OTP"
                        className="flex-1 px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-green-500 bg-white/80 backdrop-blur-sm transition-all duration-300 hover:border-gray-400"
                      disabled={isLoading}
                        onClick={(e) => e.stopPropagation()}
                    />
                    <button 
                      type="button" 
                        onClick={(e) => {
                          e.stopPropagation();
                          verifyMobileOtp();
                        }}
                      disabled={!mobileOtp || isLoading}
                        className="px-4 py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl font-medium disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed hover:from-blue-600 hover:to-blue-700 transition-all duration-300 text-sm"
                    >
                      Verify
                    </button>
                  </div>
                )}
                {mobileStep === 'setPassword' && (
                    <div className="space-y-3">
                    <input
                      type="password"
                      value={newPassword}
                      onChange={(e) => setNewPassword(e.target.value)}
                      placeholder="Create password"
                        className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-green-500 bg-white/80 backdrop-blur-sm transition-all duration-300 hover:border-gray-400"
                        onClick={(e) => e.stopPropagation()}
                    />
                    {(() => { const s = passwordStrength(newPassword); return (
                      <div className="space-y-1">
                        <div className="h-2 w-full bg-gray-200 rounded-full overflow-hidden">
                          <div className={`${s.color} h-2`} style={{ width: `${s.percent}%` }}></div>
                        </div>
                        <div className="text-xs text-gray-600">Password strength: {s.label}</div>
                      </div>
                    ); })()}
                    <input
                      type="password"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      placeholder="Confirm password"
                        className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-green-500 bg-white/80 backdrop-blur-sm transition-all duration-300 hover:border-gray-400"
                        onClick={(e) => e.stopPropagation()}
                    />
                    <button 
                      type="button" 
                        onClick={(e) => {
                          e.stopPropagation();
                          setMobilePassword();
                        }}
                      disabled={!newPassword || newPassword !== confirmPassword || isLoading}
                        className="w-full py-3 bg-gradient-to-r from-green-600 to-green-700 text-white rounded-xl font-medium disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed hover:from-green-700 hover:to-green-800 transition-all duration-300 text-sm"
                    >
                      Set Password & Continue
                    </button>
                  </div>
                )}
              </div>
            )}
            </div>
          </div>
        </div>
      </div>

      {/* Error and Success Messages */}
      {(error || success) && (
        <div className="relative z-10 w-full max-w-md px-4 mb-6">
            {error && (
              <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-xl animate-fadeIn">
                <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
                <p className="text-red-700 text-sm">{error}</p>
              </div>
            )}
            {success && (
              <div className="flex items-center gap-3 p-4 bg-green-50 border border-green-200 rounded-xl animate-fadeIn">
                <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0" />
                <p className="text-green-700 text-sm">{success}</p>
            </div>
          )}
              </div>
            )}

      {/* Bottom Controls - Detached from blocks */}
      <div className="relative z-10 w-full max-w-md px-4">
        <div className="bg-white/95 backdrop-blur-md rounded-2xl shadow-lg p-6 border border-gray-100">
          {/* Sign In/Up Button */}
          <form onSubmit={handleSubmit}>
            {selectedAuthMethod === 'default' && (
              <button
                type="submit"
                disabled={isLoading || !username.trim() || !password.trim()}
                className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed text-white font-semibold py-4 px-6 rounded-xl transition-all duration-300 flex items-center justify-center gap-3 shadow-lg hover:shadow-xl transform hover:scale-[1.02] disabled:transform-none text-lg"
              >
                {isLoading ? (
                  <>
                    <div className="w-6 h-6 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    {isLogin ? 'Signing In...' : 'Creating Account...'}
                  </>
                ) : (
                  <>
                    {isLogin ? <LogIn className="w-6 h-6" /> : <UserPlus className="w-6 h-6" />}
                    {isLogin ? 'Sign In' : 'Create Account'}
                  </>
                )}
              </button>
            )}
            
            {(selectedAuthMethod === 'google' || selectedAuthMethod === 'mobile') && (
              <button
                type="button"
                onClick={selectedAuthMethod === 'google' ? handleGoogleAuth : undefined}
                disabled={isLoading}
                className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed text-white font-semibold py-4 px-6 rounded-xl transition-all duration-300 flex items-center justify-center gap-3 shadow-lg hover:shadow-xl transform hover:scale-[1.02] disabled:transform-none text-lg"
              >
                {isLoading ? (
                  <>
                    <div className="w-6 h-6 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    {selectedAuthMethod === 'google' ? 'Connecting to Google...' : 'Processing...'}
                  </>
                ) : (
                  <>
                    {selectedAuthMethod === 'google' ? (
                      <>
                        <svg className="w-6 h-6" viewBox="0 0 24 24">
                          <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                          <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                          <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                          <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                        </svg>
                        Continue with Google
                      </>
                    ) : (
                      <>
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                          <rect width="16" height="20" x="4" y="2" rx="2"/>
                          <path d="M8 6h8M9 18h6"/>
                        </svg>
                        Continue with Mobile
                      </>
                    )}
                  </>
                )}
              </button>
            )}
          </form>

          {/* Toggle Mode */}
          <div className="mt-6 text-center">
            <p className="text-gray-600">
              {isLogin ? "Don't have an account? " : "Already have an account? "}
              <button
                type="button"
                onClick={toggleMode}
                className="text-blue-600 hover:text-blue-700 font-medium transition-colors"
                disabled={isLoading}
              >
                {isLogin ? 'Sign up' : 'Sign in'}
              </button>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;




