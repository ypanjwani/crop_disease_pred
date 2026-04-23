/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['"Syne"', 'sans-serif'],
        body:    ['"DM Sans"', 'sans-serif'],
        mono:    ['"JetBrains Mono"', 'monospace'],
      },
      colors: {
        forest: {
          50:  '#f0fdf4',
          100: '#dcfce7',
          200: '#bbf7d0',
          300: '#86efac',
          400: '#4ade80',
          500: '#22c55e',
          600: '#16a34a',
          700: '#15803d',
          800: '#166534',
          900: '#14532d',
          950: '#052e16',
        },
        soil: {
          50:  '#fdf8f0',
          100: '#faefd8',
          200: '#f4dab0',
          300: '#ecc07d',
          400: '#e39f4a',
          500: '#d98229',
          600: '#c46820',
          700: '#a24f1e',
          800: '#844020',
          900: '#6b351f',
        },
      },
      animation: {
        'fade-up':    'fadeUp 0.5s ease-out forwards',
        'fade-in':    'fadeIn 0.4s ease-out forwards',
        'slide-in':   'slideIn 0.4s ease-out forwards',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'scan':       'scan 2s ease-in-out infinite',
      },
      keyframes: {
        fadeUp:  { from: { opacity: 0, transform: 'translateY(20px)' }, to: { opacity: 1, transform: 'translateY(0)' } },
        fadeIn:  { from: { opacity: 0 }, to: { opacity: 1 } },
        slideIn: { from: { opacity: 0, transform: 'translateX(-20px)' }, to: { opacity: 1, transform: 'translateX(0)' } },
        scan:    { '0%,100%': { transform: 'translateY(-100%)' }, '50%': { transform: 'translateY(100%)' } },
      },
    },
  },
  plugins: [],
}
