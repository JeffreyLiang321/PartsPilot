/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['var(--font-geist-sans)', 'system-ui', 'sans-serif'],
        mono: ['var(--font-geist-mono)', 'monospace'],
      },
      colors: {
        bg: {
          primary:   '#0A0A0F',
          secondary: '#111118',
          elevated:  '#18181F',
          border:    '#242430',
        },
        amber: {
          400: '#FBBF24',
          500: '#F59E0B',
          600: '#D97706',
        },
        text: {
          primary:   '#F4F4F5',
          secondary: '#A1A1AA',
          muted:     '#52525B',
        }
      },
      animation: {
        'fade-in':    'fadeIn 0.3s ease forwards',
        'slide-in':   'slideIn 0.25s ease forwards',
        'pulse-dot':  'pulseDot 1.4s ease-in-out infinite',
      },
      keyframes: {
        fadeIn:   { from: { opacity: 0 }, to: { opacity: 1 } },
        slideIn:  { from: { opacity: 0, transform: 'translateY(6px)' }, to: { opacity: 1, transform: 'translateY(0)' } },
        pulseDot: {
          '0%, 80%, 100%': { transform: 'scale(0.6)', opacity: 0.4 },
          '40%':            { transform: 'scale(1)',   opacity: 1 },
        },
      },
    },
  },
  plugins: [],
}
