import { useEffect, useState } from 'react';

export default function ThemeToggle() {
  const [theme, setTheme] = useState<string>('light');
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') || 'light';
    const defaultTheme = savedTheme || 'light';
    
    // If no theme was saved, save 'light' as default
    if (!savedTheme) {
      localStorage.setItem('theme', 'light');
    }
    setTheme(savedTheme);
    setMounted(true);

    // Apply the theme to the document
    if (defaultTheme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, []);

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    localStorage.setItem('theme', newTheme);
    
    if (newTheme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  };

  if (!mounted) {
    return (
      <div className="w-12 h-6 bg-gray-300 rounded-full opacity-50">
        <div className="w-4 h-4 bg-white rounded-full m-1"></div>
      </div>
    );
  }

  return (
    <button
      onClick={toggleTheme}
      className="relative w-12 h-6 rounded-full transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-light-text/50 dark:focus:ring-dark-text/50 focus:ring-offset-light-bg dark:focus:ring-offset-dark-bg"
      style={{
        backgroundColor: theme === 'dark' ? '#f0eee6' : '#97938c'
      }}
      aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
    >
      <div
        className="absolute top-1 left-1 w-4 h-4 rounded-full transition-transform duration-300"
        style={{
          backgroundColor: theme === 'dark' ? '#1f1e1d' : '#f0eee6',
          transform: theme === 'dark' ? 'translateX(24px)' : 'translateX(0)'
        }}
      />
    </button>
  );
}