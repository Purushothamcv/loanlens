# LoanLens Frontend

A modern React frontend for the LoanLens AI loan prediction system.

## Features

- 🎨 Modern, responsive UI with Tailwind CSS
- ✨ Smooth animations with Framer Motion
- 📱 Mobile-first design
- 🚀 Fast development with Vite
- 🎯 Multi-step form with validation
- 📊 Beautiful result visualization
- 🔗 Seamless API integration

## Tech Stack

- **React 18** - UI library
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **Framer Motion** - Animation library
- **React Router** - Client-side routing
- **Axios** - HTTP client
- **React Hot Toast** - Toast notifications
- **Lucide React** - Icon library

## Getting Started

### Prerequisites

- Node.js 16 or higher
- npm or yarn

### Installation

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open your browser and visit `http://localhost:3000`

### Building for Production

```bash
npm run build
```

## Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── Navbar.jsx
│   ├── Footer.jsx
│   └── LoadingSpinner.jsx
├── pages/               # Page components
│   ├── HomePage.jsx
│   └── PredictionPage.jsx
├── services/            # API services
│   └── api.js
├── App.jsx              # Main app component
├── main.jsx            # Entry point
└── index.css           # Global styles
```

## API Integration

The frontend is configured to work with the FastAPI backend running on `http://localhost:8000`. The Vite proxy automatically forwards `/api` requests to the backend server.

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## Environment Variables

Create a `.env` file in the frontend directory for environment-specific configurations:

```env
VITE_API_BASE_URL=http://localhost:8000
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request
