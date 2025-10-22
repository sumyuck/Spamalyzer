import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { AlertTriangle, CheckCircle2, AlertCircle } from "lucide-react";

interface AnalysisResult {
  isSpam: boolean;
  confidence: number;
  suspiciousWords: string[];
  explanation: string;
}

interface AnalysisResultsProps {
  result: AnalysisResult;
  emailText: string;
}

export const AnalysisResults = ({ result, emailText }: AnalysisResultsProps) => {
  const { isSpam, confidence, suspiciousWords, explanation } = result;

  // Highlight suspicious words in the email text
  const highlightText = () => {
    let highlightedText = emailText;
    suspiciousWords.forEach((word) => {
      const regex = new RegExp(`\\b${word}\\b`, 'gi');
      highlightedText = highlightedText.replace(
        regex,
        `<mark class="bg-destructive/20 text-destructive font-semibold px-1 rounded">${word}</mark>`
      );
    });
    return highlightedText;
  };

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      {/* Verdict Card */}
      <Card className={`p-6 border-2 ${
        isSpam 
          ? 'border-destructive/50 bg-destructive/5' 
          : 'border-success/50 bg-success/5'
      }`}>
        <div className="flex items-start gap-4">
          <div className={`p-3 rounded-full ${
            isSpam ? 'bg-destructive/10' : 'bg-success/10'
          }`}>
            {isSpam ? (
              <AlertTriangle className="h-8 w-8 text-destructive" />
            ) : (
              <CheckCircle2 className="h-8 w-8 text-success" />
            )}
          </div>
          <div className="flex-1">
            <h3 className="text-2xl font-bold mb-2">
              {isSpam ? 'Spam Detected' : 'Email is Safe'}
            </h3>
            <p className="text-muted-foreground">
              {isSpam 
                ? 'This email shows characteristics of spam or phishing.'
                : 'This email appears to be legitimate and safe.'}
            </p>
          </div>
          <Badge 
            variant={isSpam ? "destructive" : "default"}
            className={isSpam ? '' : 'bg-success hover:bg-success/90'}
          >
            {isSpam ? 'SPAM' : 'SAFE'}
          </Badge>
        </div>
      </Card>

      {/* Confidence Score */}
      <Card className="p-6">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="text-lg font-semibold flex items-center gap-2">
              <AlertCircle className="h-5 w-5 text-primary" />
              Confidence Score
            </h4>
            <span className="text-2xl font-bold">{Math.round(confidence)}%</span>
          </div>
          <Progress value={confidence} className="h-3" />
          <p className="text-sm text-muted-foreground">
            {confidence > 90 && 'Very high confidence in this assessment.'}
            {confidence > 70 && confidence <= 90 && 'High confidence in this assessment.'}
            {confidence > 50 && confidence <= 70 && 'Moderate confidence - review carefully.'}
            {confidence <= 50 && 'Low confidence - manual review recommended.'}
          </p>
        </div>
      </Card>

      {/* Suspicious Words */}
      {suspiciousWords.length > 0 && (
        <Card className="p-6">
          <h4 className="text-lg font-semibold mb-4">Suspicious Keywords Detected</h4>
          <div className="flex flex-wrap gap-2 mb-4">
            {suspiciousWords.map((word, index) => (
              <Badge key={index} variant="outline" className="border-destructive/50 text-destructive">
                {word}
              </Badge>
            ))}
          </div>
          <div className="mt-4 p-4 bg-muted/30 rounded-lg border border-border/50">
            <p className="text-sm font-mono leading-relaxed" 
               dangerouslySetInnerHTML={{ __html: highlightText() }}
            />
          </div>
        </Card>
      )}

      {/* AI Explanation */}
      <Card className="p-6">
        <h4 className="text-lg font-semibold mb-4">AI Analysis Explanation</h4>
        <div className="prose prose-sm max-w-none">
          <p className="text-muted-foreground leading-relaxed whitespace-pre-line">
            {explanation}
          </p>
        </div>
      </Card>

      {/* Statistics */}
      <Card className="p-6 bg-muted/30">
        <h4 className="text-lg font-semibold mb-4">Email Statistics</h4>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold text-primary">{emailText.length}</div>
            <div className="text-xs text-muted-foreground">Characters</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-primary">{emailText.split(/\s+/).length}</div>
            <div className="text-xs text-muted-foreground">Words</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-primary">{suspiciousWords.length}</div>
            <div className="text-xs text-muted-foreground">Red Flags</div>
          </div>
        </div>
      </Card>
    </div>
  );
};
