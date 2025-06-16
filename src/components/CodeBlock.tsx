import { useEffect, useState } from 'react';

interface CodeBlockProps {
  children: React.ReactNode;
  className?: string;
}

export default function CodeBlock({ children, className }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const copyToClipboard = async () => {
    const codeElement = document.querySelector(`[data-code-block] code`);
    if (codeElement) {
      const text = codeElement.textContent || '';
      try {
        await navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch (err) {
        console.error('Failed to copy code:', err);
      }
    }
  };

  if (!mounted) {
    return (
      <div className="relative group">
        <pre className={className}>
          <code>{children}</code>
        </pre>
      </div>
    );
  }

  return (
    <div className="relative group" data-code-block>
      <button
        onClick={copyToClipboard}
        className="absolute top-2 right-2 z-10 opacity-0 group-hover:opacity-100 transition-opacity duration-200 bg-light-text/10 dark:bg-dark-text/10 hover:bg-light-text/20 dark:hover:bg-dark-text/20 rounded px-2 py-1 text-xs font-medium text-light-text dark:text-dark-text border border-light-text/20 dark:border-dark-text/20 h-6 flex items-center justify-center min-w-fit"
        aria-label="Copy code to clipboard"
      >
        {copied ? (
          <span className="flex items-center gap-1 whitespace-nowrap">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="flex-shrink-0">
              <polyline points="20,6 9,17 4,12"></polyline>
            </svg>
            Copied
          </span>
        ) : (
          <span className="flex items-center gap-1 whitespace-nowrap">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="flex-shrink-0">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2 2v1"></path>
            </svg>
            Copy
          </span>
        )}
      </button>
      <pre className={className}>
        <code>{children}</code>
      </pre>
    </div>
  );
}