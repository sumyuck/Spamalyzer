import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

// Load trained model weights
let modelData: any = null;

async function loadModelWeights() {
  if (!modelData) {
    try {
      // Try to read from the same directory as the function
      const modelPath = new URL('./model_weights.json', import.meta.url).pathname;
      const modelFile = await Deno.readTextFile(modelPath);
      modelData = JSON.parse(modelFile);
      console.log('Model loaded successfully:', {
        models: modelData.model_types,
        features: modelData.dataset_info.n_features,
        svm_accuracy: modelData.svm.test_accuracy
      });
    } catch (error) {
      console.error('Failed to load model weights:', error);
      throw new Error('Model weights file not found. Please ensure model_weights.json is in the function directory.');
    }
  }
  return modelData;
}

// Extract word frequency features from email text
function extractWordFrequencies(emailText: string, featureNames: string[]): number[] {
  const lowerText = emailText.toLowerCase();
  // Remove special characters and split into words
  const words = lowerText.replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(w => w.length > 0);
  
  // Count word frequencies
  const wordCounts: { [key: string]: number } = {};
  words.forEach(word => {
    wordCounts[word] = (wordCounts[word] || 0) + 1;
  });
  
  // Create feature vector matching training data
  const features: number[] = [];
  featureNames.forEach(feature => {
    // Feature names in the dataset are the actual words
    features.push(wordCounts[feature.toLowerCase()] || 0);
  });
  
  return features;
}

// Scale features using MinMaxScaler parameters
function scaleFeatures(features: number[], scaler: any): number[] {
  return features.map((value, i) => {
    const min = scaler.data_min[i];
    const scale = scaler.scale[i];
    if (scale === 0) return 0; // Handle zero scale
    return (value - min) * scale;
  });
}

// Calculate Euclidean distance between two vectors
function euclideanDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += Math.pow(a[i] - b[i], 2);
  }
  return Math.sqrt(sum);
}

// KNN prediction
function predictKNN(scaledFeatures: number[], knnData: any): { prediction: number; confidence: number } {
  const k = knnData.n_neighbors;
  const trainingData = knnData.training_data;
  const trainingLabels = knnData.training_labels;
  
  // Calculate distances to all training samples
  const distances: { distance: number; label: number }[] = [];
  for (let i = 0; i < trainingData.length; i++) {
    const dist = euclideanDistance(scaledFeatures, trainingData[i]);
    distances.push({ distance: dist, label: trainingLabels[i] });
  }
  
  // Sort by distance and get k nearest neighbors
  distances.sort((a, b) => a.distance - b.distance);
  const kNearest = distances.slice(0, k);
  
  // Vote for the class
  const votes = kNearest.reduce((acc, neighbor) => {
    acc[neighbor.label] = (acc[neighbor.label] || 0) + 1;
    return acc;
  }, {} as { [key: number]: number });
  
  const prediction = votes[1] > (votes[0] || 0) ? 1 : 0;
  const confidence = (Math.max(votes[0] || 0, votes[1] || 0) / k) * 100;
  
  return { prediction, confidence };
}

// SVM prediction (simplified linear kernel)
function predictSVM(scaledFeatures: number[], svmData: any): { prediction: number; confidence: number } {
  const supportVectors = svmData.support_vectors;
  const dualCoef = svmData.dual_coef[0];
  const intercept = svmData.intercept;
  
  // Calculate decision function: sum(alpha_i * y_i * K(x_i, x)) + b
  let decisionValue = intercept;
  for (let i = 0; i < supportVectors.length; i++) {
    // Linear kernel: K(x_i, x) = x_i · x (dot product)
    let dotProduct = 0;
    for (let j = 0; j < scaledFeatures.length; j++) {
      dotProduct += supportVectors[i][j] * scaledFeatures[j];
    }
    decisionValue += dualCoef[i] * dotProduct;
  }
  
  // Prediction: sign(decision_value)
  const prediction = decisionValue >= 0 ? 1 : 0;
  
  // Convert decision value to confidence (0-100%)
  // Using sigmoid-like transformation
  const confidence = Math.min(Math.max(Math.abs(decisionValue) * 10, 50), 99);
  
  return { prediction, confidence };
}

// Enhanced spam indicators for explanation context
const SPAM_INDICATORS = {
  phishing: [
    'verify', 'confirm identity', 'unusual activity', 'suspended', 'limited', 
    'restore access', 'account number', 'security team', 'verify within',
    'temporarily limited', 'confirm your identity', 'last transaction'
  ],
  urgency: [
    'urgent', 'immediate', 'expires', '24 hours', 'limited time', 'act now', 
    'today only', 'within 24 hours', 'must act', 'expires soon'
  ],
  financial: [
    'cash', 'prize', 'winner', 'free', 'claim', 'won', 'guaranteed', 
    'bonus', 'discount', 'cheap', 'loan', 'credit'
  ],
  suspicious: [
    'click here', 'call now', 'txt', 'text', 'reply', 'mobile', 
    'congratulations', 'selected', 'offer expires'
  ]
};

function detectIndicators(emailText: string) {
  const lowerText = emailText.toLowerCase();
  const detected = {
    phishing: [] as string[],
    urgency: [] as string[],
    financial: [] as string[],
    suspicious: [] as string[]
  };

  for (const [category, indicators] of Object.entries(SPAM_INDICATORS)) {
    for (const indicator of indicators) {
      if (lowerText.includes(indicator.toLowerCase())) {
        detected[category as keyof typeof detected].push(indicator);
      }
    }
  }

  return detected;
}

// Main ML analysis using trained KNN and SVM models
async function analyzeEmailWithML(emailText: string) {
  const model = await loadModelWeights();
  
  // Extract word frequency features
  const features = extractWordFrequencies(emailText, model.feature_names);
  
  // Scale features
  const scaledFeatures = scaleFeatures(features, model.scaler);
  
  // Get predictions from both models
  const knnResult = predictKNN(scaledFeatures, model.knn);
  const svmResult = predictSVM(scaledFeatures, model.svm);
  
  // Use SVM as primary model (typically more accurate for this task)
  const isSpam = svmResult.prediction === 1;
  const confidence = svmResult.confidence;
  
  // Detect indicators for explanation context
  const indicators = detectIndicators(emailText);
  const allSuspiciousWords = [
    ...indicators.phishing,
    ...indicators.urgency,
    ...indicators.financial,
    ...indicators.suspicious
  ];
  
  console.log('ML Analysis:', {
    knn: { prediction: knnResult.prediction, confidence: knnResult.confidence },
    svm: { prediction: svmResult.prediction, confidence: svmResult.confidence },
    final: { isSpam, confidence }
  });
  
  return {
    isSpam,
    confidence,
    suspiciousWords: allSuspiciousWords,
    indicators,
    models: {
      knn: knnResult,
      svm: svmResult
    }
  };
}

// Generate explanation using Gemini LLM
async function generateExplanationWithGemini(
  emailText: string,
  isSpam: boolean,
  confidence: number,
  indicators: {
    phishing: string[];
    urgency: string[];
    financial: string[];
    suspicious: string[];
  }
) {
  const LOVABLE_API_KEY = Deno.env.get('LOVABLE_API_KEY');
  
  if (!LOVABLE_API_KEY) {
    console.warn('LOVABLE_API_KEY not found, using fallback explanation');
    return generateFallbackExplanation(isSpam, indicators);
  }

  const indicatorSummary = [
    indicators.phishing.length > 0 ? `- Phishing tactics: ${indicators.phishing.slice(0, 3).join(', ')}` : '',
    indicators.urgency.length > 0 ? `- Urgency language: ${indicators.urgency.slice(0, 3).join(', ')}` : '',
    indicators.financial.length > 0 ? `- Financial terms: ${indicators.financial.slice(0, 3).join(', ')}` : '',
    indicators.suspicious.length > 0 ? `- Suspicious patterns: ${indicators.suspicious.slice(0, 3).join(', ')}` : ''
  ].filter(line => line).join('\n');

  const prompt = `You are an expert email security analyst. Analyze this email and provide a clear, professional explanation.

Email Text:
"${emailText}"

ML Model Prediction: ${isSpam ? 'SPAM/PHISHING' : 'SAFE'} (${confidence}% confidence)

Detected Indicators:
${indicatorSummary || 'No significant indicators detected'}

Provide a 2-3 sentence explanation that:
1. States whether this is spam/phishing or legitimate
2. Explains the key red flags or why it's safe
3. Uses clear, user-friendly language

Keep it concise and actionable.`;

  try {
    const response = await fetch('https://ai.gateway.lovable.dev/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${LOVABLE_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'google/gemini-2.5-flash',
        messages: [
          { role: 'system', content: 'You are an expert email security analyst. Provide clear, concise explanations about email safety.' },
          { role: 'user', content: prompt }
        ],
        max_tokens: 300
      })
    });

    if (!response.ok) {
      console.error('Gemini API error:', response.status);
      return generateFallbackExplanation(isSpam, indicators);
    }

    const data = await response.json();
    return data.choices[0].message.content;
  } catch (error) {
    console.error('Error calling Gemini API:', error);
    return generateFallbackExplanation(isSpam, indicators);
  }
}

// Fallback explanation if Gemini is unavailable
function generateFallbackExplanation(
  isSpam: boolean,
  indicators: {
    phishing: string[];
    urgency: string[];
    financial: string[];
    suspicious: string[];
  }
) {
  if (isSpam) {
    const reasons = [];
    if (indicators.phishing.length > 0) {
      reasons.push(`phishing tactics detected (${indicators.phishing.slice(0, 2).join(', ')})`);
    }
    if (indicators.urgency.length > 0) {
      reasons.push(`urgency pressure (${indicators.urgency.slice(0, 2).join(', ')})`);
    }
    if (indicators.financial.length > 0) {
      reasons.push(`financial lures (${indicators.financial.slice(0, 2).join(', ')})`);
    }
    if (indicators.suspicious.length > 0) {
      reasons.push(`suspicious patterns (${indicators.suspicious.slice(0, 2).join(', ')})`);
    }
    
    return `This email appears to be SPAM/PHISHING. Key indicators: ${reasons.join(', ')}. Do not click links or provide personal information.`;
  } else {
    return `This email appears to be SAFE. No significant spam or phishing indicators were detected. The message structure and content suggest legitimate communication.`;
  }
}

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { emailText } = await req.json();

    if (!emailText || typeof emailText !== 'string') {
      return new Response(
        JSON.stringify({ error: 'Invalid email text provided' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    console.log('Analyzing email with length:', emailText.length);

    // Run ML analysis with trained KNN and SVM models
    const mlAnalysis = await analyzeEmailWithML(emailText);
    console.log('ML Analysis complete:', { isSpam: mlAnalysis.isSpam, confidence: mlAnalysis.confidence });

    // Generate AI explanation using Gemini
    const explanation = await generateExplanationWithGemini(
      emailText,
      mlAnalysis.isSpam,
      mlAnalysis.confidence,
      mlAnalysis.indicators
    );
    console.log('AI explanation generated');

    const result = {
      isSpam: mlAnalysis.isSpam,
      confidence: mlAnalysis.confidence,
      suspiciousWords: mlAnalysis.suspiciousWords,
      explanation
    };

    return new Response(
      JSON.stringify(result),
      { status: 200, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );

  } catch (error) {
    console.error('Unexpected error in analyze-email:', error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : 'Unknown error' }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});
