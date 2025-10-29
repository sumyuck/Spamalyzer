import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Loader2, Mail, Shield } from "lucide-react";
import { AnalysisResults } from "@/components/AnalysisResults";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";

interface AnalysisResult {
  isSpam: boolean;
  confidence: number;
  suspiciousWords: string[];
  explanation: string;
}

const Index = () => {
  const [emailText, setEmailText] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const analyzeEmail = async () => {
    if (!emailText.trim()) {
      toast.error("Please enter email text to analyze");
      return;
    }

    setIsAnalyzing(true);
    setResult(null);

    try {
      const { data, error } = await supabase.functions.invoke('analyze-email', {
        body: { emailText }
      });

      if (error) {
        console.error('Analysis error:', error);
        if (error.message?.includes('429')) {
          toast.error("Rate limit exceeded. Please try again later.");
        } else if (error.message?.includes('402')) {
          toast.error("AI credits depleted. Please add credits to continue.");
        } else {
          toast.error("Failed to analyze email. Please try again.");
        }
        return;
      }

      setResult(data);
    } catch (err) {
      console.error('Unexpected error:', err);
      toast.error("An unexpected error occurred");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5">
      {/* Header */}
      <header className="border-b border-border/40 backdrop-blur-sm bg-card/50">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-gradient-to-br from-primary to-primary-glow">
              <Shield className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-primary-glow bg-clip-text text-transparent">
                Spamalyzer
              </h1>
              <p className="text-sm text-muted-foreground">AI-Powered Email Spam Detection</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12 max-w-6xl">
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Input Section */}
          <div className="space-y-6">
            <div className="space-y-2">
              <h2 className="text-3xl font-bold">Analyze Your Email</h2>
              <p className="text-muted-foreground">
                Paste any email content below and let our AI determine if it's spam or legitimate.
              </p>
            </div>

            <Card className="p-6 shadow-lg border-border/50">
              <div className="space-y-4">
                <div className="space-y-2">
                  <label htmlFor="email-input" className="text-sm font-medium flex items-center gap-2">
                    <Mail className="h-4 w-4 text-primary" />
                    Email Content
                  </label>
                  <Textarea
                    id="email-input"
                    placeholder="Paste your email text here..."
                    value={emailText}
                    onChange={(e) => setEmailText(e.target.value)}
                    className="min-h-[300px] resize-none font-mono text-sm"
                  />
                </div>

                <Button
                  onClick={analyzeEmail}
                  disabled={isAnalyzing || !emailText.trim()}
                  variant="gradient"
                  size="lg"
                  className="w-full"
                >
                  {isAnalyzing ? (
                    <>
                      <Loader2 className="h-5 w-5 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Shield className="h-5 w-5" />
                      Analyze Email
                    </>
                  )}
                </Button>
              </div>
            </Card>

            {/* Example */}
            <Card className="p-4 border-dashed bg-muted/30">
              <p className="text-sm text-muted-foreground mb-2">
                <strong>Example:</strong> Try analyzing this spam email:
              </p>
              <p className="text-xs font-mono bg-background p-3 rounded-md border">
                "Subject: Urgent Account Verification Required\n\nDear Valued Customer,\n\nWe have detected unusual activity on your account. To restore full access, please verify your identity by clicking the link below within 24 hours. Failure to confirm may result in permanent account suspension.\n\nVerify Now: http://secure-bank-verify.com/confirm\n\nThank you,\nSecurity Team"
              </p>
            </Card>
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            <div className="space-y-2">
              <h2 className="text-3xl font-bold">Analysis Results</h2>
              <p className="text-muted-foreground">
                Detailed insights and confidence scoring powered by AI.
              </p>
            </div>

            {result ? (
              <AnalysisResults result={result} emailText={emailText} />
            ) : (
              <Card className="p-12 text-center border-dashed">
                <div className="inline-flex p-4 rounded-full bg-muted/50 mb-4">
                  <Shield className="h-12 w-12 text-muted-foreground" />
                </div>
                <h3 className="text-lg font-semibold mb-2">No Analysis Yet</h3>
                <p className="text-sm text-muted-foreground">
                  Enter email content and click "Analyze Email" to see results here.
                </p>
              </Card>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border/40 mt-20 py-6">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>Powered by AI • Protecting your inbox from spam</p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
