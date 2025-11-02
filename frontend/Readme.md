# 🧠 Aether-AI Frontend

> Production-ready Next.js frontend for the Aether-AI AutoML platform

![Next.js](https://img.shields.io/badge/Next.js-14-black)
![TypeScript](https://img.shields.io/badge/TypeScript-5.5-blue)
![Tailwind](https://img.shields.io/badge/Tailwind-3.4-38bdf8)
![React Query](https://img.shields.io/badge/React%20Query-5-ff4154)

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ 
- npm 9+
- Backend API running on `http://localhost:8000`

### Installation

```bash
# Install dependencies
npm install

# Copy environment variables
cp .env.example .env.local

# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 📁 Project Structure

```
src/
├── app/              # Next.js 14 App Router
│   ├── auth/         # Authentication pages
│   ├── dashboard/    # Main dashboard
│   ├── projects/     # Project management
│   ├── datasets/     # Dataset management
│   ├── training/     # Training interface
│   ├── models/       # Model registry
│   └── deploy/       # Deployment
├── components/
│   ├── ui/           # Reusable UI components
│   ├── layout/       # Layout components
│   ├── features/     # Feature-specific components
│   └── shared/       # Shared utilities
├── lib/
│   ├── api/          # API client & endpoints
│   ├── utils/        # Utility functions
│   └── validators/   # Zod schemas
├── hooks/            # Custom React hooks
├── stores/           # Zustand stores
├── types/            # TypeScript types
├── services/         # Business logic
└── config/           # Configuration
```

## 🛠️ Tech Stack

### Core

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript 5.5
- **Styling**: Tailwind CSS 3.4
- **UI Components**: Custom (shadcn-inspired)

### State Management

- **Server State**: React Query 5
- **Client State**: Zustand 4
- **Forms**: React Hook Form + Zod

### Data & API

- **HTTP Client**: Axios
- **WebSockets**: Socket.IO Client
- **Validation**: Zod

### UI & Visualization

- **Charts**: Recharts
- **Icons**: Lucide React
- **Animations**: Framer Motion
- **File Upload**: React Dropzone

## 📦 Available Scripts

```bash
# Development
npm run dev              # Start dev server
npm run build            # Build for production
npm run start            # Start production server

# Code Quality
npm run lint             # Lint code
npm run lint:fix         # Fix linting issues
npm run type-check       # Type checking
npm run format           # Format code
npm run format:check     # Check formatting

# Testing
npm run test             # Run unit tests
npm run test:watch       # Watch mode
npm run test:coverage    # Coverage report
npm run e2e              # Run E2E tests
npm run e2e:ui           # E2E with UI
```

## 🎨 Key Features

### ✅ Built Pages

1. **Landing Page** - Marketing & product showcase
2. **Authentication** - Login, signup, password reset
3. **Dashboard** - Overview with key metrics
4. **Projects** - Create, manage, and organize projects
5. **Datasets** - Upload, preview, and analyze datasets
6. **Training Studio** - Configure and monitor training jobs
7. **Inference Studio** - Run Trained models 
8. **Model Registry** - Manage trained models
9. **Deployment** - Deploy models with one click

### 🔥 Core Capabilities

- **Real-time Updates**: WebSocket integration for live training metrics
- **File Upload**: Drag-and-drop with progress tracking
- **Data Visualization**: Interactive charts for metrics
- **Responsive Design**: Mobile-first approach
- **Dark Mode**: Theme switching support
- **Type-safe API**: End-to-end TypeScript
- **Form Validation**: Zod schemas with React Hook Form
- **Error Handling**: Comprehensive error boundaries
- **Loading States**: Skeleton screens and spinners
- **Toast Notifications**: User feedback system

## 🔧 Configuration

### Environment Variables

Create a `.env.local` file:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
NEXT_PUBLIC_APP_NAME=Aether AI
NEXT_PUBLIC_MAX_FILE_SIZE=524288000
```

### API Integration

The API client is configured in `src/lib/api/client.ts`:

```typescript
import { apiClient } from '@/lib/api/client';

// Example usage
const projects = await apiClient.get('/projects');
```

### WebSocket Connection

```typescript
import { useWebSocket } from '@/hooks/use-websocket';

const { data, isConnected } = useWebSocket(`/training/${jobId}`);
```

## 🎯 Development Workflow

### Adding a New Page

1. Create route in `src/app/[route]/page.tsx`
2. Add types in `src/types/`
3. Create API functions in `src/lib/api/`
4. Build components in `src/components/features/`
5. Add validation in `src/lib/validators/`

### Creating a Component

```typescript
// src/components/ui/button.tsx
import { cva, type VariantProps } from 'class-variance-authority';

const buttonVariants = cva(
  'inline-flex items-center justify-center rounded-md',
  {
    variants: {
      variant: {
        default: 'bg-primary text-primary-foreground',
        outline: 'border border-input bg-background',
      },
      size: {
        default: 'h-10 px-4 py-2',
        sm: 'h-9 px-3',
        lg: 'h-11 px-8',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export function Button({ variant, size, className, ...props }: ButtonProps) {
  return (
    <button
      className={buttonVariants({ variant, size, className })}
      {...props}
    />
  );
}
```

## 🧪 Testing

### Unit Tests

```bash
npm run test
```

### E2E Tests

```bash
npm run e2e
```

## 🚢 Deployment

### Build

```bash
npm run build
```

### Deploy to Vercel

```bash
vercel deploy
```

### Environment Variables

Set these in your deployment platform:
- `NEXT_PUBLIC_API_URL`
- `NEXT_PUBLIC_WS_URL`
- `NEXT_PUBLIC_APP_NAME`

## 📚 Documentation

- [Project Structure](./STRUCTURE.md) - Detailed structure guide
- [API Documentation](./docs/API.md) - API integration guide
- [Components](./docs/COMPONENTS.md) - Component library
- [Development](./docs/DEVELOPMENT.md) - Development setup

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Run tests and linting
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🔗 Links

- [Backend Repository](#)
- [Documentation](#)
- [Demo](#)

---

**Built with ❤️ by the Aether AI Team**
