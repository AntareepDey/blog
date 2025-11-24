/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}','./src/styles/**/*.css'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        light: {
          text: '#7e7b74',
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
            // color: '#97938c',
            lineHeight: '1.5',
            fontSize: '1.125rem',
            'h1, h2, h3, h4, h5, h6': {
              fontFamily: '"EB Garamond", serif',
              fontWeight: '700',
            },
            'p': {
               color: '#97938c !important', // Ensure paragraph text color matches the light mode text color
              marginTop: '1.0rem',
              marginBottom: '1.0rem',
            },
          }
        },
        dark: {
          css: {
            color: '#f0eee6',
            'h1, h2, h3, h4, h5, h6': {
              color: '#f0eee6',
            },
            'p': {
              color: '#f0eee6 !important', // Add this line to match heading color in dark mode
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