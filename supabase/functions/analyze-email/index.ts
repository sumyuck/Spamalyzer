import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

// Enhanced spam indicators categorized by threat type
const SPAM_INDICATORS = {
  phishing: [
    'verify', 'confirm identity', 'unusual activity', 'suspended', 'limited', 
    'restore access', 'account number', 'security team', 'verify within',
    'temporarily limited', 'confirm your identity', 'last transaction',
    'account will remain', 'restore full access'
  ],
  urgency: [
    'urgent', 'immediate', 'expires', '24 hours', 'limited time', 'act now', 
    'today only', 'within 24 hours', 'do not', 'must act', 'expires soon',
    'time sensitive', 'respond immediately'
  ],
  financial: [
    'cash', 'prize', 'winner', 'free', 'claim', 'won', 'guaranteed', 
    '£', '$', '€', 'bonus', 'discount', 'cheap', 'loan', 'credit'
  ],
  suspicious: [
    'click here', 'call now', 'txt', 'text', 'reply', 'mobile', 
    'congratulations', 'selected', 'offer expires', '$$$', '!!!'
  ]
};

// Detect indicators by category
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

// ML-inspired feature extraction and scoring
function analyzeEmailWithML(emailText: string) {
  const lowerText = emailText.toLowerCase();
  const words = lowerText.split(/\s+/);
  
  // Detect indicators
  const indicators = detectIndicators(emailText);
  const allSuspiciousWords = [
    ...indicators.phishing,
    ...indicators.urgency,
    ...indicators.financial,
    ...indicators.suspicious
  ];
  
  // Calculate ML-inspired score
  let score = 0;
  
  // Phishing indicators (highest weight - most dangerous)
  score += indicators.phishing.length * 35;
  
  // Urgency indicators (high weight - pressure tactics)
  score += indicators.urgency.length * 25;
  
  // Financial indicators (medium-high weight)
  score += indicators.financial.length * 20;
  
  // General suspicious patterns (medium weight)
  score += indicators.suspicious.length * 15;
  
  // Multiple exclamation marks
  const exclamationCount = (emailText.match(/!/g) || []).length;
  if (exclamationCount >= 3) score += 20;
  else if (exclamationCount >= 2) score += 10;
  
  // All caps words (shouting)
  const capsWords = words.filter(word => 
    word.length > 2 && word === word.toUpperCase() && /[A-Z]/.test(word)
  );
  if (capsWords.length >= 3) score += 20;
  else if (capsWords.length >= 1) score += 10;
  
  // Money symbols
  if (lowerText.includes('£') || lowerText.includes('$') || lowerText.includes('€')) {
    score += 15;
  }
  
  // Phone numbers (UK format)
  if (/\b0[0-9]{10}\b/.test(emailText)) score += 15;
  
  // Suspicious URLs
  if (lowerText.includes('http') || lowerText.includes('www.') || lowerText.includes('.com')) {
    score += 20;
  }
  
  // Normalize to confidence percentage
  const confidence = Math.min(score, 100);
  const isSpam = confidence >= 50;
  
  return {
    isSpam,
    confidence,
    suspiciousWords: allSuspiciousWords,
    indicators
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

    // Run ML-inspired analysis
    const mlAnalysis = analyzeEmailWithML(emailText);
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
