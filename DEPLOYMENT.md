# Deployment Guide

## Clean Production Setup

### Backend (Render/Railway)

1. **Environment Variables:**
   ```
   PORT=8000
   HOST=0.0.0.0
   CORS_ORIGINS=https://your-frontend-domain.vercel.app
   ENVIRONMENT=production
   DEBUG=false
   ```

2. **Build Command:** `pip install -r requirements.txt`
3. **Start Command:** `python api.py`

### Frontend (Vercel/Netlify)

1. **Environment Variables:**
   ```
   VITE_API_BASE_URL=https://your-backend-url.onrender.com
   VITE_ENVIRONMENT=production
   ```

2. **Build Command:** `npm run build`
3. **Output Directory:** `dist`

## Files to Remove Before Deployment

### Server Directory Cleanup:
- `index.html` (unnecessary - frontend handles UI)
- `test_api.py` (development only)
- Any `.csv` files if not needed for model training

### Root Level Cleanup:
- `RUN_COMMANDS.sh` (development only)
- Any temporary files or logs

## Deployment Steps

### Backend on Render
1. Connect your GitHub repository
2. Select `server` as root directory  
3. Set environment variables
4. Deploy

### Frontend on Vercel
1. Import GitHub repository
2. Set framework preset to "Vite"
3. Set root directory to `frontend`
4. Set environment variables
5. Deploy

## Clean Folder Structure

```
loanlens/
├── README.md
├── DEPLOYMENT.md
├── .gitignore
├── server/                     # Backend (FastAPI)
│   ├── .env.example
│   ├── config.py
│   ├── api.py
│   ├── main.ipynb             # Model training
│   ├── loan_model.pkl         # Trained model
│   └── requirements.txt
└── frontend/                   # Frontend (React)
    ├── .env.example
    ├── package.json
    ├── vite.config.js
    ├── index.html
    ├── src/
    │   ├── main.jsx
    │   ├── App.jsx
    │   ├── index.css
    │   ├── config/
    │   │   └── constants.js
    │   ├── components/
    │   │   ├── Navbar.jsx
    │   │   ├── Footer.jsx
    │   │   └── LoadingSpinner.jsx
    │   ├── pages/
    │   │   ├── HomePage.jsx
    │   │   └── PredictionPage.jsx
    │   └── services/
    │       └── api.js
    └── public/
        └── vite.svg
```

## Health Checks
- Backend: `https://your-backend-url/health`
- Frontend: Check if the app loads and can communicate with backend

## Monitoring
- Check logs for any errors
- Monitor API response times
- Verify CORS configuration is working

## Production Checklist
- [ ] Remove development files
- [ ] Set environment variables
- [ ] Test API endpoints
- [ ] Verify CORS configuration
- [ ] Check model file exists
- [ ] Test frontend-backend communication
