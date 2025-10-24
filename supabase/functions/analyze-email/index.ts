import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

// Common spam indicator words (used for suspicious word detection)
const SPAM_INDICATORS = [
  'free', 'win', 'winner', 'cash', 'prize', 'claim', 'urgent', 'act now',
  'limited time', 'click here', 'call now', 'congratulations', 'selected',
  'bonus', 'guaranteed', 'offer expires', '$$$', '!!!', 'txt', 'text',
  'reply', 'mobile', 'award', 'won', 'discount', 'cheap', 'loan',
];

// Generate explanation using Lovable AI (Gemini)
async function generateExplanationWithAI(
  emailText: string,
  isSpam: boolean,
  confidence: number,
  suspiciousWords: string[],
  indicators: {
    exclamationCount: number;
    capsWords: string[];
    hasMoney: boolean;
    hasPhoneNumber: boolean;
    hasURL: boolean;
  }
): Promise<string> {
  const LOVABLE_API_KEY = Deno.env.get('LOVABLE_API_KEY');
  
  if (!LOVABLE_API_KEY) {
    console.error('LOVABLE_API_KEY not found, using fallback explanation');
    return generateFallbackExplanation(isSpam, suspiciousWords, indicators);
  }

  try {
    const systemPrompt = `You are an expert email spam analyst. Your job is to explain spam detection results in clear, professional language that end-users can understand. Be concise but thorough.`;
    
    const userPrompt = `Analyze this email spam detection result and provide a clear explanation:

Email Text: "${emailText.substring(0, 500)}${emailText.length > 500 ? '...' : ''}"

Detection Results:
- Classification: ${isSpam ? 'SPAM' : 'SAFE'}
- Confidence: ${confidence}%
- Suspicious Words Found: ${suspiciousWords.length > 0 ? suspiciousWords.join(', ') : 'None'}
- Exclamation Marks: ${indicators.exclamationCount}
- All-Caps Words: ${indicators.capsWords.length}
- Contains Money Symbols: ${indicators.hasMoney ? 'Yes' : 'No'}
- Contains Phone Numbers: ${indicators.hasPhoneNumber ? 'Yes' : 'No'}
- Contains URLs: ${indicators.hasURL ? 'Yes' : 'No'}

Provide a 2-3 sentence explanation of why this email is classified as ${isSpam ? 'SPAM' : 'SAFE'}. Focus on the most important indicators and explain what they mean in simple terms.`;

    const response = await fetch('https://ai.gateway.lovable.dev/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${LOVABLE_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'google/gemini-2.5-flash',
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt }
        ],
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Lovable AI API error:', response.status, errorText);
      return generateFallbackExplanation(isSpam, suspiciousWords, indicators);
    }

    const data = await response.json();
    const explanation = data.choices?.[0]?.message?.content;

    if (!explanation) {
      console.error('No explanation in AI response');
      return generateFallbackExplanation(isSpam, suspiciousWords, indicators);
    }

    console.log('AI-generated explanation successfully');
    return explanation;

  } catch (error) {
    console.error('Error generating AI explanation:', error);
    return generateFallbackExplanation(isSpam, suspiciousWords, indicators);
  }
}

// Fallback explanation generator (original rule-based logic)
function generateFallbackExplanation(
  isSpam: boolean,
  suspiciousWords: string[],
  indicators: {
    exclamationCount: number;
    capsWords: string[];
    hasMoney: boolean;
    hasPhoneNumber: boolean;
    hasURL: boolean;
  }
): string {
  let explanation = '';
  if (isSpam) {
    explanation = `This email appears to be SPAM. `;
    const reasons = [];
    
    if (suspiciousWords.length > 0) {
      reasons.push(`contains ${suspiciousWords.length} spam indicator words (${suspiciousWords.slice(0, 3).join(', ')}${suspiciousWords.length > 3 ? '...' : ''})`);
    }
    if (indicators.exclamationCount >= 2) {
      reasons.push(`uses excessive exclamation marks (${indicators.exclamationCount})`);
    }
    if (indicators.capsWords.length >= 1) {
      reasons.push(`contains ${indicators.capsWords.length} all-caps words`);
    }
    if (indicators.hasMoney) {
      reasons.push('contains money-related symbols');
    }
    if (indicators.hasPhoneNumber) {
      reasons.push('includes phone numbers');
    }
    if (indicators.hasURL) {
      reasons.push('contains suspicious URLs');
    }
    
    explanation += 'Key indicators: ' + reasons.join(', ') + '.';
  } else {
    explanation = `This email appears to be SAFE. `;
    if (suspiciousWords.length > 0) {
      explanation += `While it contains some common words (${suspiciousWords.slice(0, 2).join(', ')}), the overall content and structure suggest it's a legitimate message.`;
    } else {
      explanation += 'No significant spam indicators were detected. The message appears to be genuine communication.';
    }
  }
  
  return explanation;
}

// Rule-based spam detection (ML-inspired scoring algorithm)
async function analyzeEmailRuleBased(emailText: string) {
  const lowerText = emailText.toLowerCase();
  const words = lowerText.split(/\s+/);
  
  // Find suspicious words
  const suspiciousWords: string[] = [];
  for (const indicator of SPAM_INDICATORS) {
    if (lowerText.includes(indicator.toLowerCase())) {
      suspiciousWords.push(indicator);
    }
  }
  
  // Calculate spam score based on indicators
  let score = 0;
  
  // Check for multiple exclamation marks
  const exclamationCount = (emailText.match(/!/g) || []).length;
  if (exclamationCount >= 3) score += 25;
  else if (exclamationCount >= 2) score += 15;
  
  // Check for all caps words
  const capsWords = words.filter(word => 
    word.length > 2 && word === word.toUpperCase() && /[A-Z]/.test(word)
  );
  if (capsWords.length >= 3) score += 25;
  else if (capsWords.length >= 1) score += 15;
  
  // Check for money symbols
  const hasMoney = lowerText.includes('£') || lowerText.includes('$') || lowerText.includes('€');
  if (hasMoney) score += 20;
  
  // Check for phone numbers (UK format)
  const hasPhoneNumber = /\b0[0-9]{10}\b/.test(emailText);
  if (hasPhoneNumber) score += 20;
  
  // Check for suspicious URLs
  const hasURL = lowerText.includes('http') || lowerText.includes('www.') || lowerText.includes('.com');
  if (hasURL) score += 15;
  
  // Suspicious words contribution (most important indicator)
  score += Math.min(suspiciousWords.length * 20, 60);
  
  // Determine if spam
  const confidence = Math.min(score, 100);
  const isSpam = confidence >= 50;
  
  // Generate explanation using AI
  const indicators = {
    exclamationCount,
    capsWords,
    hasMoney,
    hasPhoneNumber,
    hasURL
  };
  
  const explanation = await generateExplanationWithAI(
    emailText,
    isSpam,
    confidence,
    suspiciousWords,
    indicators
  );
  
  return {
    isSpam,
    confidence,
    suspiciousWords,
    explanation
  };
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

    // Use ML-inspired rule-based analysis with AI-generated explanations
    const analysis = await analyzeEmailRuleBased(emailText);

    console.log('Analysis complete:', { isSpam: analysis.isSpam, confidence: analysis.confidence });

    return new Response(
      JSON.stringify(analysis),
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
