import React, { useState, useEffect } from 'react';
import { Send, AlertCircle, CheckCircle, Loader, Activity, LogOut, User, Moon, Sun } from 'lucide-react';
import LoginPage from './LoginPage';

interface User {
  id: string;
  username: string;
}

interface PredictionResult {
  prediction: string;
  confidence: number;
  probabilities: Record<string, number>;
}

interface GeneratedSequence {
  sequence: string;
  average_probability: number;
  levenshtein: number;
  hamming: number;
  cosine: number;
  pearson: number;
}


interface ApiResponse {
  success: boolean;
  result: PredictionResult;
  processed_sequence: string;
  error?: string;
}

interface SequenceApiResponse {
  success: boolean;
  input_sequence: string;
  generated_sequences: GeneratedSequence[];
  error?: string;
}

interface SmilesApiResponse {
  success: boolean;
  input_sequence: string;
  smiles: string;
  error?: string;
}

function App() {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [sequence, setSequence] = useState('');
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [generatorSequence, setGeneratorSequence] = useState('');
  const [sequenceGenerationLoading, setSequenceGenerationLoading] = useState(false);
  const [generatedSequences, setGeneratedSequences] = useState<GeneratedSequence[]>([]);
  const [sequenceGenerationError, setSequenceGenerationError] = useState<string | null>(null);
  const [smilesSequence, setSmilesSequence] = useState('');
  const [smilesLoading, setSmilesLoading] = useState(false);
  const [generatedSmiles, setGeneratedSmiles] = useState<string>('');
  const [smilesError, setSmilesError] = useState<string | null>(null);

  // Check authentication status on component mount
  useEffect(() => {
    checkAuthStatus();
    // Check for Google OAuth callback
    checkGoogleAuthCallback();
  }, []);

  const checkAuthStatus = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/check-auth', {
        credentials: 'include',
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.authenticated && data.user) {
          setUser(data.user);
        }
      }
    } catch (err) {
      console.error('Auth check failed:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const checkGoogleAuthCallback = () => {
    const urlParams = new URLSearchParams(window.location.search);
    const authStatus = urlParams.get('auth');
    const username = urlParams.get('user');
    
    if (authStatus === 'success' && username) {
      // Google OAuth was successful
      setUser({ id: username, username: username });
      
      // Clean up URL
      window.history.replaceState({}, document.title, window.location.pathname);
    } else if (authStatus === 'error') {
      const error = urlParams.get('error');
      console.error('Google OAuth error:', error);
      setError(`Google sign-in failed: ${error}`);
      // Clean up URL
      window.history.replaceState({}, document.title, window.location.pathname);
    }
  };

  const handleLogin = (userData: User) => {
    setUser(userData);
  };

  const handleLogout = async () => {
    try {
      await fetch('http://localhost:5001/api/logout', {
        method: 'POST',
        credentials: 'include',
      });
      setUser(null);
      setSequence('');
      setResult(null);
      setError(null);
    } catch (err) {
      console.error('Logout error:', err);
    }
  };

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };

  const validateSequence = (seq: string): boolean => {
    const validAminoAcids = /^[ACDEFGHIKLMNPQRSTVWY]+$/i;
    return validAminoAcids.test(seq.trim());
  };

  const handlePredict = async () => {
    const trimmedSequence = sequence.trim().toUpperCase();
    
    // Validation
    if (!trimmedSequence) {
      setError('Please enter a protein sequence');
      return;
    }
    
    if (trimmedSequence.length < 10) {
      setError('Sequence too short. Please provide at least 10 amino acids.');
      return;
    }
    
    if (!validateSequence(trimmedSequence)) {
      setError('Invalid amino acid sequence. Please use single-letter amino acid codes (A-Z, excluding B, J, O, U, X, Z).');
      return;
    }

    setPredictionLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:5001/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ sequence: trimmedSequence }),
      });

      if (response.status === 401) {
        setUser(null);
        setError('Session expired. Please log in again.');
        return;
      }

      const data: ApiResponse = await response.json();

      if (data.success && data.result) {
        setResult(data.result);
      } else {
        setError(data.error || 'Prediction failed');
      }
    } catch (err) {
      console.error('Prediction error:', err);
      setError('Unable to connect to the server. Please ensure the Flask backend is running.');
    } finally {
      setPredictionLoading(false);
    }
  };

  const handleGenerateSequences = async () => {
    const trimmedSequence = generatorSequence.trim().toUpperCase();
    
    // Validation
    if (!trimmedSequence) {
      setSequenceGenerationError('Please enter a protein sequence');
      return;
    }
    
    if (trimmedSequence.length < 10) {
      setSequenceGenerationError('Sequence too short. Please provide at least 10 amino acids.');
      return;
    }
    
    if (!validateSequence(trimmedSequence)) {
      setSequenceGenerationError('Invalid amino acid sequence. Please use single-letter amino acid codes (A-Z, excluding B, J, O, U, X, Z).');
      return;
    }

    setSequenceGenerationLoading(true);
    setSequenceGenerationError(null);
    setGeneratedSequences([]);

    try {
      const response = await fetch('http://localhost:5001/generate-sequences', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ sequence: trimmedSequence }),
      });

      if (response.status === 401) {
        setUser(null);
        setSequenceGenerationError('Session expired. Please log in again.');
        return;
      }

      const data: SequenceApiResponse = await response.json();

      if (data.success && data.generated_sequences) {
        setGeneratedSequences(data.generated_sequences);
      } else {
        setSequenceGenerationError(data.error || 'Sequence generation failed');
      }
    } catch (err) {
      console.error('Sequence generation error:', err);
      setSequenceGenerationError('Unable to connect to the server. Please ensure the Flask backend is running.');
    } finally {
      setSequenceGenerationLoading(false);
    }
  };

  const handleGenerateSmiles = async () => {
    const trimmedSequence = smilesSequence.trim().toUpperCase();
    
    // Validation
    if (!trimmedSequence) {
      setSmilesError('Please enter a protein sequence');
      return;
    }
    
    if (trimmedSequence.length < 10) {
      setSmilesError('Sequence too short. Please provide at least 10 amino acids.');
      return;
    }
    
    if (!validateSequence(trimmedSequence)) {
      setSmilesError('Invalid amino acid sequence. Please use single-letter amino acid codes (A-Z, excluding B, J, O, U, X, Z).');
      return;
    }

    setSmilesLoading(true);
    setSmilesError(null);
    setGeneratedSmiles('');

    try {
      const response = await fetch('http://localhost:5001/generate-smiles', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ sequence: trimmedSequence }),
      });

      if (response.status === 401) {
        setUser(null);
        setSmilesError('Session expired. Please log in again.');
        return;
      }

      const data: SmilesApiResponse = await response.json();

      if (data.success && data.smiles) {
        setGeneratedSmiles(data.smiles);
      } else {
        setSmilesError(data.error || 'SMILES generation failed');
      }
    } catch (err) {
      console.error('SMILES generation error:', err);
      setSmilesError('Unable to connect to the server. Please ensure the Flask backend is running.');
    } finally {
      setSmilesLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handlePredict();
    }
  };

  const getSeverityColor = (prediction: string): string => {
    const severityColors: Record<string, string> = {
      'Pathogenic': 'text-red-500',
      'Likely_pathogenic': 'text-orange-500',
      'Uncertain_significance': 'text-yellow-500',
      'Likely_benign': 'text-blue-500',
      'Benign': 'text-green-500',
    };
    return severityColors[prediction] || 'text-gray-500';
  };

  const formatPrediction = (prediction: string): string => {
    return prediction.replace(/_/g, ' ');
  };

  // Show loading screen while checking authentication
  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-900 via-indigo-900 to-purple-900 flex items-center justify-center animate-fadeIn">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-white border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-white text-lg">Loading...</p>
        </div>
      </div>
    );
  }

  // Show login page if not authenticated
  if (!user) {
    return (
      <div className="animate-fadeIn">
        <LoginPage onLogin={handleLogin} />
      </div>
    );
  }

  // Main application
  return (
    <div className={`min-h-screen relative overflow-hidden transition-all duration-500 ease-in-out ${isDarkMode ? 'bg-gradient-to-br from-slate-900 via-blue-950 to-indigo-950' : 'bg-gradient-to-br from-blue-900 via-indigo-900 to-purple-900'}`}>
      {/* Background particles effect */}
      <div className="absolute inset-0 overflow-hidden">
        <div className={`absolute top-20 left-20 w-2 h-2 rounded-full opacity-20 animate-pulse transition-colors duration-500 ${isDarkMode ? 'bg-blue-400' : 'bg-white'}`}></div>
        <div className={`absolute top-40 right-32 w-1 h-1 rounded-full opacity-30 animate-pulse transition-colors duration-500 ${isDarkMode ? 'bg-purple-400' : 'bg-blue-300'}`}></div>
        <div className={`absolute bottom-32 left-16 w-2 h-2 rounded-full opacity-25 animate-pulse transition-colors duration-500 ${isDarkMode ? 'bg-teal-400' : 'bg-purple-300'}`}></div>
        <div className={`absolute top-1/3 left-1/4 w-1 h-1 rounded-full opacity-40 animate-pulse transition-colors duration-500 ${isDarkMode ? 'bg-indigo-400' : 'bg-white'}`}></div>
        <div className={`absolute bottom-20 right-20 w-2 h-2 rounded-full opacity-20 animate-pulse transition-colors duration-500 ${isDarkMode ? 'bg-cyan-400' : 'bg-blue-200'}`}></div>
        <div className={`absolute top-2/3 right-1/4 w-1 h-1 rounded-full opacity-30 animate-pulse transition-colors duration-500 ${isDarkMode ? 'bg-violet-400' : 'bg-purple-200'}`}></div>
      </div>

      <div className="relative z-10 min-h-screen flex flex-col items-center justify-center p-4">
        {/* Header with user info, theme toggle, and logout */}
        <div className="absolute top-4 right-4 flex items-center gap-4">
          <div className={`flex items-center gap-2 backdrop-blur-sm rounded-full px-4 py-2 border transition-all duration-300 ${isDarkMode ? 'bg-slate-800/60 border-blue-600 shadow-lg' : 'bg-white/10 border-white/20'}`}>
            <User className={`w-4 h-4 transition-colors duration-300 ${isDarkMode ? 'text-blue-300' : 'text-blue-300'}`} />
            <span className={`text-sm font-medium transition-colors duration-300 ${isDarkMode ? 'text-blue-100' : 'text-white'}`}>{user.username}</span>
          </div>
          <button
            onClick={toggleTheme}
            className={`rounded-full p-2 transition-all duration-300 hover:scale-110 ${isDarkMode ? 'bg-indigo-700 hover:bg-indigo-600 text-indigo-200 shadow-lg' : 'bg-white/10 hover:bg-white/20 text-blue-300'}`}
            title={isDarkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
          >
            {isDarkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
          <button
            onClick={handleLogout}
            className="bg-red-500 hover:bg-red-600 text-white rounded-full p-2 transition-all duration-300 hover:scale-110 shadow-lg"
            title="Logout"
          >
            <LogOut className="w-4 h-4" />
          </button>
        </div>

        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-6">
            <div className={`backdrop-blur-sm rounded-full p-4 mr-4 border transition-all duration-300 ${isDarkMode ? 'bg-indigo-800/60 border-indigo-600 shadow-lg' : 'bg-white/10 border-white/20'}`}>
              <Activity className={`w-10 h-10 transition-colors duration-300 ${isDarkMode ? 'text-indigo-300' : 'text-blue-300'}`} />
            </div>
            <h1 className={`text-5xl md:text-6xl font-bold tracking-tight transition-colors duration-300 ${isDarkMode ? 'text-blue-100' : 'text-white'}`}>
              T2D Insulin prediction
            </h1>
          </div>
          <p className={`text-xl md:text-2xl font-light max-w-3xl mx-auto transition-colors duration-300 ${isDarkMode ? 'text-blue-200' : 'text-blue-200'}`}>
            Advanced AI-powered protein sequence pathogenicity classifier
          </p>
        </div>

        {/* Main Card */}
        <div className="w-full max-w-7xl">
          {/* Classifier Module */}
          <div className={`p-8 rounded-xl mb-12 transition-all duration-300 ${isDarkMode ? 'bg-slate-800/90 border border-indigo-600' : 'bg-white/95 border border-white/10'}`}>
            <label htmlFor="sequence" className={`block text-xl font-semibold mb-4 transition-colors duration-300 ${isDarkMode ? 'text-blue-100' : 'text-gray-800'}`}>Protein Sequence</label>
            <textarea
              id="sequence"
              value={sequence}
              onChange={(e) => setSequence(e.target.value)}
              placeholder="Enter your protein sequence here..."
              className={`w-full h-36 px-6 py-4 border-2 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-base font-mono leading-relaxed transition-all duration-300 hover:border-gray-300 ${isDarkMode ? 'bg-slate-700 border-indigo-600 text-blue-100 placeholder-blue-300 focus:ring-blue-400' : 'border-gray-200 text-gray-800 placeholder-gray-600'}`}
              disabled={predictionLoading}
            />
            <div className="mt-4 flex flex-col md:flex-row md:items-center md:gap-6 gap-3">
              <button
                onClick={handlePredict}
                disabled={predictionLoading || !sequence.trim()}
                className="w-full md:w-max bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 disabled:from-gray-400 disabled:to-gray-500 text-white font-semibold py-4 px-8 rounded-xl transition-all duration-300 flex items-center justify-center gap-3 shadow-lg hover:shadow-xl transform hover:scale-[1.02] disabled:transform-none disabled:cursor-not-allowed text-lg"
              >
                {predictionLoading ? (<><Loader className="w-6 h-6 animate-spin" />Analyzing...</>) : (<><Send className="w-6 h-6" />Predict</>)}
              </button>
              {error && <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-xl animate-fadeIn"><AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" /><p className="text-red-700">{error}</p></div>}
            </div>
            {result && (
                <div className="space-y-6 animate-fadeIn">
                  <div className="flex items-center gap-3 p-5 bg-green-50 border border-green-200 rounded-xl">
                    <CheckCircle className="w-6 h-6 text-green-500 flex-shrink-0" />
                    <div>
                      <p className="text-green-800 font-semibold text-lg">Prediction Complete</p>
                      <p className="text-green-600 text-sm">Analysis finished successfully</p>
                    </div>
                  </div>

                  <div className={`rounded-xl p-8 space-y-6 transition-all duration-300 ${isDarkMode ? 'bg-slate-700/50 border border-indigo-600' : 'bg-gray-50'}`}>
                    <div className="grid md:grid-cols-2 gap-8">
                      <div className="space-y-3">
                        <h3 className={`text-xl font-bold transition-colors duration-300 ${isDarkMode ? 'text-blue-100' : 'text-gray-800'}`}>Prediction</h3>
                        <p className={`text-3xl font-bold ${getSeverityColor(result.prediction)}`}>
                          {formatPrediction(result.prediction)}
                        </p>
                        <p className={`text-lg transition-colors duration-300 ${isDarkMode ? 'text-blue-200' : 'text-gray-600'}`}>
                          Confidence: <span className="font-semibold">{(result.confidence * 100).toFixed(1)}%</span>
                        </p>
                      </div>

                      <div className="space-y-3">
                        <h3 className={`text-xl font-bold transition-colors duration-300 ${isDarkMode ? 'text-blue-100' : 'text-gray-800'}`}>All Probabilities</h3>
                        <div className="space-y-3">
                          {Object.entries(result.probabilities)
                            .sort(([,a], [,b]) => b - a)
                            .map(([className, probability]) => (
                              <div key={className} className="flex justify-between items-center py-1">
                                <span className={`font-medium transition-colors duration-300 ${isDarkMode ? 'text-blue-200' : 'text-gray-700'}`}>
                                  {formatPrediction(className)}
                                </span>
                                <span className={`font-semibold transition-colors duration-300 ${isDarkMode ? 'text-blue-100' : 'text-gray-800'}`}>
                                  {(probability * 100).toFixed(1)}%
                                </span>
                              </div>
                            ))}
                        </div>
                      </div>
                    </div>

                    <div className={`pt-6 border-t transition-colors duration-300 ${isDarkMode ? 'border-indigo-600' : 'border-gray-200'}`}>
                      <h3 className={`text-lg font-semibold mb-4 transition-colors duration-300 ${isDarkMode ? 'text-blue-200' : 'text-gray-700'}`}>Probability Distribution</h3>
                      <div className="space-y-3">
                        {Object.entries(result.probabilities)
                          .sort(([,a], [,b]) => b - a)
                          .map(([className, probability]) => (
                            <div key={className} className="space-y-2">
                              <div className="flex justify-between items-center">
                                <span className={`text-sm font-medium transition-colors duration-300 ${isDarkMode ? 'text-blue-300' : 'text-gray-600'}`}>
                                  {formatPrediction(className)}
                                </span>
                                <span className={`text-sm font-semibold transition-colors duration-300 ${isDarkMode ? 'text-blue-200' : 'text-gray-700'}`}>
                                  {(probability * 100).toFixed(1)}%
                                </span>
                              </div>
                              <div className={`w-full rounded-full h-2 transition-colors duration-300 ${isDarkMode ? 'bg-slate-600' : 'bg-gray-200'}`}>
                                <div
                                  className="bg-gradient-to-r from-blue-500 to-indigo-600 h-2 rounded-full transition-all duration-700 ease-out"
                                  style={{ width: `${probability * 100}%` }}
                                ></div>
                              </div>
                            </div>
                          ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}
          </div>
          {/* Sequence Generator Module */}
          <div className={`p-8 rounded-xl mb-12 transition-all duration-300 ${isDarkMode ? 'bg-purple-950/80 border border-purple-600' : 'bg-purple-50/90 border border-purple-100'}`}>
            <label htmlFor="generatorSequence" className={`block text-xl font-semibold mb-4 transition-colors duration-300 ${isDarkMode ? 'text-purple-100' : 'text-purple-800'}`}>Input Protein Sequence for Generation</label>
            <textarea
              id="generatorSequence"
              value={generatorSequence}
              onChange={(e) => setGeneratorSequence(e.target.value)}
              placeholder="Enter protein sequence to generate new variants..."
              className={`w-full h-32 px-6 py-4 border-2 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none text-base font-mono leading-relaxed transition-all duration-300 hover:border-gray-300 ${isDarkMode ? 'bg-slate-700 border-purple-600 text-purple-100 placeholder-purple-300 focus:ring-purple-400' : 'border-gray-200 text-gray-800 placeholder-gray-600'}`}
              disabled={sequenceGenerationLoading}
            />
            <div className="mt-4 flex flex-col md:flex-row md:items-center md:gap-6 gap-3">
              <button
                onClick={handleGenerateSequences}
                disabled={sequenceGenerationLoading || !generatorSequence.trim()}
                className="w-full md:w-max bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700 disabled:from-gray-400 disabled:to-gray-500 text-white font-semibold py-4 px-8 rounded-xl transition-all duration-300 flex items-center justify-center gap-3 shadow-lg hover:shadow-xl transform hover:scale-[1.02] disabled:transform-none disabled:cursor-not-allowed text-lg"
              >
                {sequenceGenerationLoading ? (<><Loader className="w-6 h-6 animate-spin" />Generating...</>) : (<><Activity className="w-6 h-6" />Generate Sequences</>)}
              </button>
              {sequenceGenerationError && <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-xl animate-fadeIn"><AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" /><p className="text-red-700">{sequenceGenerationError}</p></div>}
            </div>
            {generatedSequences.length > 0 && (
                <div className="space-y-6 animate-fadeIn">
                  <div className="flex items-center gap-3 p-5 bg-purple-50 border border-purple-200 rounded-xl">
                    <CheckCircle className="w-6 h-6 text-purple-500 flex-shrink-0" />
                    <div>
                      <p className="text-purple-800 font-semibold text-lg">Sequence Generation Complete</p>
                      <p className="text-purple-600 text-sm">Generated {generatedSequences.length} sequences with similarity analysis</p>
                    </div>
                  </div>

                  <div className={`rounded-xl p-8 space-y-6 transition-all duration-300 ${isDarkMode ? 'bg-slate-700/50 border border-purple-600' : 'bg-purple-50'}`}>
                    <div className="text-center">
                      <h3 className={`text-2xl font-bold mb-2 transition-colors duration-300 ${isDarkMode ? 'text-purple-100' : 'text-purple-800'}`}>
                        Generated Protein Sequences
                      </h3>
                      <p className={`text-lg transition-colors duration-300 ${isDarkMode ? 'text-purple-200' : 'text-purple-600'}`}>
                        Top 5 sequences ranked by average similarity probability
                      </p>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      {generatedSequences.map((seq, index) => (
                        <div 
                          key={index}
                          className={`p-6 rounded-xl border-2 transition-all duration-300 hover:shadow-lg ${
                            isDarkMode 
                              ? 'bg-slate-800/50 border-purple-600 hover:border-purple-500' 
                              : 'bg-white border-purple-200 hover:border-purple-300'
                          }`}
                        >
                          <div className="space-y-4">
                            {/* Header with ranking */}
                            <div className="flex items-center gap-3">
                              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm ${
                                index === 0 ? 'bg-yellow-500' : 
                                index === 1 ? 'bg-gray-400' : 
                                index === 2 ? 'bg-orange-500' : 'bg-blue-500'
                              }`}>
                                {index + 1}
                              </div>
                              <h4 className={`text-lg font-semibold transition-colors duration-300 ${isDarkMode ? 'text-purple-100' : 'text-purple-800'}`}>
                                Sequence #{index + 1}
                              </h4>
                            </div>
                            
                            {/* Sequence display with better contrast */}
                            <div className={`font-mono text-sm p-4 rounded-lg transition-colors duration-300 ${
                              isDarkMode 
                                ? 'bg-slate-900 text-green-300 border border-slate-600' 
                                : 'bg-gray-900 text-green-300 border border-gray-700'
                            }`}>
                              {seq.sequence}
                            </div>
                            
                            {/* Average probability */}
                            <div className={`text-center p-4 rounded-lg ${isDarkMode ? 'bg-purple-900/50' : 'bg-purple-100'}`}>
                              <div className={`text-2xl font-bold transition-colors duration-300 ${isDarkMode ? 'text-purple-200' : 'text-purple-800'}`}>
                                {(seq.average_probability * 100).toFixed(1)}%
                              </div>
                              <div className={`text-sm transition-colors duration-300 ${isDarkMode ? 'text-purple-300' : 'text-purple-600'}`}>
                                Average Probability
                              </div>
                            </div>
                            
                            {/* Individual metrics */}
                            <div className="grid grid-cols-2 gap-3 text-xs">
                              <div className={`p-3 rounded-lg text-center ${isDarkMode ? 'bg-slate-600' : 'bg-gray-100'}`}>
                                <div className={`font-semibold transition-colors duration-300 ${isDarkMode ? 'text-blue-200' : 'text-gray-700'}`}>
                                  {(seq.levenshtein * 100).toFixed(1)}%
                                </div>
                                <div className={`transition-colors duration-300 ${isDarkMode ? 'text-blue-300' : 'text-gray-500'}`}>
                                  Levenshtein
                                </div>
                              </div>
                              <div className={`p-3 rounded-lg text-center ${isDarkMode ? 'bg-slate-600' : 'bg-gray-100'}`}>
                                <div className={`font-semibold transition-colors duration-300 ${isDarkMode ? 'text-green-200' : 'text-gray-700'}`}>
                                  {(seq.hamming * 100).toFixed(1)}%
                                </div>
                                <div className={`transition-colors duration-300 ${isDarkMode ? 'text-green-300' : 'text-gray-500'}`}>
                                  Hamming
                                </div>
                              </div>
                              <div className={`p-3 rounded-lg text-center ${isDarkMode ? 'bg-slate-600' : 'bg-gray-100'}`}>
                                <div className={`font-semibold transition-colors duration-300 ${isDarkMode ? 'text-orange-200' : 'text-gray-700'}`}>
                                  {(seq.cosine * 100).toFixed(1)}%
                                </div>
                                <div className={`transition-colors duration-300 ${isDarkMode ? 'text-orange-300' : 'text-gray-500'}`}>
                                  Cosine
                                </div>
                              </div>
                              <div className={`p-3 rounded-lg text-center ${isDarkMode ? 'bg-slate-600' : 'bg-gray-100'}`}>
                                <div className={`font-semibold transition-colors duration-300 ${isDarkMode ? 'text-pink-200' : 'text-gray-700'}`}>
                                  {(seq.pearson * 100).toFixed(1)}%
                                </div>
                                <div className={`transition-colors duration-300 ${isDarkMode ? 'text-pink-300' : 'text-gray-500'}`}>
                                  Pearson
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
          </div>
          {/* SMILES Generator Module */}
          <div className={`p-8 rounded-xl mb-8 transition-all duration-300 ${isDarkMode ? 'bg-green-950/90 border border-green-600' : 'bg-green-50/90 border border-green-200'}`}>
            <label htmlFor="smilesSequence" className={`block text-xl font-semibold mb-4 transition-colors duration-300 ${isDarkMode ? 'text-green-100' : 'text-green-800'}`}>Protein Sequence (for SMILES)</label>
            <textarea
              id="smilesSequence"
              value={smilesSequence}
              onChange={(e) => setSmilesSequence(e.target.value)}
              placeholder="Enter protein sequence to generate SMILES..."
              className={`w-full h-32 px-6 py-4 border-2 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-transparent resize-none text-base font-mono leading-relaxed transition-all duration-300 hover:border-gray-300 ${isDarkMode ? 'bg-slate-700 border-green-600 text-green-100 placeholder-green-300 focus:ring-green-400' : 'border-gray-200 text-gray-800 placeholder-gray-600'}`}
              disabled={smilesLoading}
            />
            <div className="mt-4 flex flex-col md:flex-row md:items-center md:gap-6 gap-3">
              <button
                onClick={handleGenerateSmiles}
                disabled={smilesLoading || !smilesSequence.trim()}
                className="w-full md:w-max bg-gradient-to-r from-green-500 to-teal-600 hover:from-green-600 hover:to-teal-700 disabled:from-gray-400 disabled:to-gray-500 text-white font-semibold py-4 px-8 rounded-xl transition-all duration-300 flex items-center justify-center gap-3 shadow-lg hover:shadow-xl transform hover:scale-[1.02] disabled:transform-none disabled:cursor-not-allowed text-lg"
              >
                {smilesLoading ? (<><Loader className="w-6 h-6 animate-spin" />Generating...</>) : (<><Activity className="w-6 h-6" />Generate SMILES</>)}
              </button>
              {smilesError && <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-xl animate-fadeIn"><AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" /><p className="text-red-700">{smilesError}</p></div>}
            </div>
            {generatedSmiles && (
                  <div className="space-y-4 animate-fadeIn">
                    <div className="flex items-center gap-3 p-4 bg-green-50 border border-green-200 rounded-xl">
                      <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0" />
                      <div>
                        <p className="text-green-800 font-semibold text-lg">SMILES Generation Complete</p>
                        <p className="text-green-600 text-sm">Generated SMILES structure from protein sequence</p>
                      </div>
                    </div>

                    <div className={`p-6 rounded-xl border-2 transition-all duration-300 ${
                      isDarkMode 
                        ? 'bg-slate-800/50 border-green-600' 
                        : 'bg-white border-green-200'
                    }`}>
                      <h4 className={`text-lg font-semibold mb-3 transition-colors duration-300 ${isDarkMode ? 'text-green-100' : 'text-green-800'}`}>
                        Generated SMILES Structure
                      </h4>
                      <div className={`font-mono text-sm p-4 rounded-lg transition-colors duration-300 ${
                        isDarkMode 
                          ? 'bg-slate-900 text-green-300 border border-slate-600' 
                          : 'bg-gray-900 text-green-300 border border-gray-700'
                      }`}>
                        {generatedSmiles}
                      </div>
                    </div>
                  </div>
                )}
          </div>
        </div>

        {/* Footer */}
        <div className="mt-12 text-center">
          <p className={`text-lg transition-colors duration-300 ${isDarkMode ? 'text-blue-200' : 'text-blue-200'}`}>
            Powered by advanced machine learning algorithms
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;