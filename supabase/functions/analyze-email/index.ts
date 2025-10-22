import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

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

    const LOVABLE_API_KEY = Deno.env.get('LOVABLE_API_KEY');
    if (!LOVABLE_API_KEY) {
      console.error('LOVABLE_API_KEY not configured');
      return new Response(
        JSON.stringify({ error: 'AI service not configured' }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    console.log('Analyzing email with length:', emailText.length);

    // Call Lovable AI to analyze the email
    const aiResponse = await fetch('https://ai.gateway.lovable.dev/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${LOVABLE_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'google/gemini-2.5-flash',
        messages: [
          {
            role: 'system',
            content: `You are an expert email spam detection AI. Analyze emails and provide:
1. Whether it's spam (true/false)
2. Confidence score (0-100)
3. List of suspicious words/phrases found
4. Clear explanation of why it's spam or safe

Common spam indicators:
- Urgency tactics ("act now", "limited time")
- Money-related ("win", "prize", "free money", "$$$")
- Suspicious links or calls to action
- Poor grammar and excessive punctuation
- Requests for personal information
- Too-good-to-be-true offers
- ALL CAPS words
- Excessive exclamation marks

Respond in this exact JSON format:
{
  "isSpam": boolean,
  "confidence": number (0-100),
  "suspiciousWords": ["word1", "word2"],
  "explanation": "detailed explanation of the analysis"
}`
          },
          {
            role: 'user',
            content: `Analyze this email for spam:\n\n${emailText}`
          }
        ],
        temperature: 0.3,
      }),
    });

    if (!aiResponse.ok) {
      const status = aiResponse.status;
      console.error('AI Gateway error:', status, await aiResponse.text());
      
      if (status === 429) {
        return new Response(
          JSON.stringify({ error: 'Rate limit exceeded. Please try again later.' }),
          { status: 429, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }
      
      if (status === 402) {
        return new Response(
          JSON.stringify({ error: 'AI credits depleted. Please add credits.' }),
          { status: 402, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }

      return new Response(
        JSON.stringify({ error: 'AI analysis failed' }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const aiData = await aiResponse.json();
    console.log('AI response received');

    // Extract the analysis from the AI response
    const aiMessage = aiData.choices?.[0]?.message?.content;
    if (!aiMessage) {
      console.error('No AI message in response');
      return new Response(
        JSON.stringify({ error: 'Invalid AI response' }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Parse the JSON response from AI
    let analysis;
    try {
      // Extract JSON from markdown code blocks if present
      const jsonMatch = aiMessage.match(/```json\s*([\s\S]*?)\s*```/) || 
                       aiMessage.match(/```\s*([\s\S]*?)\s*```/);
      const jsonString = jsonMatch ? jsonMatch[1] : aiMessage;
      analysis = JSON.parse(jsonString.trim());
    } catch (parseError) {
      console.error('Failed to parse AI response:', parseError);
      console.log('AI message:', aiMessage);
      return new Response(
        JSON.stringify({ error: 'Failed to parse analysis results' }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Validate the analysis structure
    if (
      typeof analysis.isSpam !== 'boolean' ||
      typeof analysis.confidence !== 'number' ||
      !Array.isArray(analysis.suspiciousWords) ||
      typeof analysis.explanation !== 'string'
    ) {
      console.error('Invalid analysis structure:', analysis);
      return new Response(
        JSON.stringify({ error: 'Invalid analysis format' }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

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
