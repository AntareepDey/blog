/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        light: {
          text: '#97938c',
          bg: '#f0eee6',
        },
        dark: {
          text: '#f0eee6',
          bg: '#1f1e1d',
        }
      },
      fontFamily: {
        'eb-garamond': ['"EB Garamond"', 'serif'],
        'newsreader': ['Newsreader', 'serif'],
      },
      typography: {
        DEFAULT: {
          css: {
            maxWidth: 'none',
            color: '#97938c',
            lineHeight: '1.75',
            fontSize: '1.125rem',
            'h1, h2, h3, h4, h5, h6': {
              fontFamily: '"EB Garamond", serif',
              fontWeight: '700',
            },
            'p': {
              marginTop: '1.5rem',
              marginBottom: '1.5rem',
            },
          }
        },
        dark: {
          css: {
            color: '#f0eee6',
            'h1, h2, h3, h4, h5, h6': {
              color: '#f0eee6',
            },
            'strong': {
              color: '#f0eee6',
            },
            'blockquote': {
              color: '#f0eee6',
              borderLeftColor: '#f0eee6',
            },
          }
        }
      }
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}