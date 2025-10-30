import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

// Calculate spam score based on detected indicators
function calculateSpamScore(indicators: {
  phishing: string[];
  urgency: string[];
  financial: string[];
  suspicious: string[];
}): { isSpam: boolean; confidence: number } {
  // Weight each category
  const weights = {
    phishing: 3.0,
    urgency: 2.0,
    financial: 2.5,
    suspicious: 1.5
  };
  
  // Calculate weighted score
  let score = 0;
  score += indicators.phishing.length * weights.phishing;
  score += indicators.urgency.length * weights.urgency;
  score += indicators.financial.length * weights.financial;
  score += indicators.suspicious.length * weights.suspicious;
  
  // Normalize to 0-100 scale
  const maxPossibleScore = 30;
  const normalizedScore = Math.min((score / maxPossibleScore) * 100, 100);
  
  // Determine if spam (threshold: 30%)
  const isSpam = normalizedScore > 30;
  const confidence = isSpam ? normalizedScore : (100 - normalizedScore);
  
  return { isSpam, confidence: Math.round(confidence) };
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

// AI-powered analysis using indicator detection
function analyzeEmail(emailText: string) {
  const indicators = detectIndicators(emailText);
  const { isSpam, confidence } = calculateSpamScore(indicators);
  
  const allSuspiciousWords = [
    ...indicators.phishing,
    ...indicators.urgency,
    ...indicators.financial,
    ...indicators.suspicious
  ];
  
  console.log('Analysis:', {
    isSpam,
    confidence,
    indicators: {
      phishing: indicators.phishing.length,
      urgency: indicators.urgency.length,
      financial: indicators.financial.length,
      suspicious: indicators.suspicious.length
    }
  });
  
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

    // Run AI-powered analysis
    const analysis = analyzeEmail(emailText);
    console.log('Analysis complete:', { isSpam: analysis.isSpam, confidence: analysis.confidence });

    // Generate AI explanation using Gemini
    const explanation = await generateExplanationWithGemini(
      emailText,
      analysis.isSpam,
      analysis.confidence,
      analysis.indicators
    );
    console.log('AI explanation generated');

    const result = {
      isSpam: analysis.isSpam,
      confidence: analysis.confidence,
      suspiciousWords: analysis.suspiciousWords,
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
