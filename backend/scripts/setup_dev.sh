#!/bin/bash

# ModelForge-CV Development Setup Script
# This script sets up the development environment

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║              🧠 MODELFORGE-CV SETUP 🧠                     ║"
echo "║                                                            ║"
echo "║         Setting up development environment...             ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# ============================================================================
# Step 1: Check Prerequisites
# ============================================================================
log_info "Checking prerequisites..."

# Check Python version
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    log_success "Python $PYTHON_VERSION found"
    
    # Check if version is 3.9+
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
        log_error "Python 3.9+ is required. Found $PYTHON_VERSION"
        exit 1
    fi
else
    log_error "Python 3 not found. Please install Python 3.9 or higher."
    exit 1
fi

# Check pip
if command_exists pip3; then
    log_success "pip found"
else
    log_error "pip not found. Please install pip."
    exit 1
fi

# Check Git
if command_exists git; then
    log_success "Git found"
else
    log_warning "Git not found. Version control features may not work."
fi

# Check CUDA (optional)
if command_exists nvidia-smi; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    log_success "CUDA $CUDA_VERSION found - GPU training enabled"
    GPU_AVAILABLE=true
else
    log_warning "CUDA not found - CPU only mode"
    GPU_AVAILABLE=false
fi

# Check Docker (optional)
if command_exists docker; then
    log_success "Docker found"
    DOCKER_AVAILABLE=true
else
    log_warning "Docker not found - containerized deployment not available"
    DOCKER_AVAILABLE=false
fi

echo ""

# ============================================================================
# Step 2: Create Virtual Environment
# ============================================================================
log_info "Creating Python virtual environment..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    log_success "Virtual environment created"
else
    log_warning "Virtual environment already exists"
fi

# Activate virtual environment
log_info "Activating virtual environment..."
source venv/bin/activate
log_success "Virtual environment activated"

echo ""

# ============================================================================
# Step 3: Upgrade pip and install build tools
# ============================================================================
log_info "Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel
log_success "Build tools installed"

echo ""

# ============================================================================
# Step 4: Install PyTorch with appropriate CUDA version
# ============================================================================
log_info "Installing PyTorch..."

if [ "$GPU_AVAILABLE" = true ]; then
    log_info "Installing PyTorch with CUDA 11.8 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    log_info "Installing PyTorch (CPU only)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

log_success "PyTorch installed"

echo ""

# ============================================================================
# Step 5: Install dependencies
# ============================================================================
log_info "Installing project dependencies..."
pip install -r requirements.txt
log_success "Dependencies installed"

log_info "Installing development dependencies..."
pip install -r requirements-dev.txt
log_success "Development dependencies installed"

echo ""

# ============================================================================
# Step 6: Install package in editable mode
# ============================================================================
log_info "Installing ModelForge-CV in editable mode..."
pip install -e .
log_success "Package installed"

echo ""

# ============================================================================
# Step 7: Setup environment variables
# ============================================================================
log_info "Setting up environment variables..."

if [ ! -f ".env" ]; then
    cp .env.example .env
    log_success ".env file created from template"
    log_warning "Please update .env with your configuration"
else
    log_warning ".env file already exists"
fi

echo ""

# ============================================================================
# Step 8: Create necessary directories
# ============================================================================
log_info "Creating workspace directories..."

mkdir -p workspaces
mkdir -p logs
mkdir -p uploads
mkdir -p temp

log_success "Directories created"

echo ""

# ============================================================================
# Step 9: Setup database (if Docker is available)
# ============================================================================
if [ "$DOCKER_AVAILABLE" = true ]; then
    log_info "Do you want to start PostgreSQL and Redis with Docker? (y/n)"
    read -r START_SERVICES
    
    if [ "$START_SERVICES" = "y" ] || [ "$START_SERVICES" = "Y" ]; then
        log_info "Starting database services..."
        docker-compose up -d postgres redis
        
        # Wait for services to be ready
        log_info "Waiting for services to start..."
        sleep 5
        
        log_success "Database services started"
        log_info "PostgreSQL: localhost:5432"
        log_info "Redis: localhost:6379"
    fi
fi

echo ""

# ============================================================================
# Step 10: Run database migrations
# ============================================================================
log_info "Running database migrations..."

if command_exists alembic; then
    # Check if database is reachable
    if python3 -c "import psycopg2; psycopg2.connect('postgresql://modelforge:modelforge123@localhost:5432/modelforge')" 2>/dev/null; then
        alembic upgrade head
        log_success "Database migrations completed"
    else
        log_warning "Database not reachable. Skipping migrations."
        log_info "Run 'alembic upgrade head' manually after starting the database"
    fi
else
    log_warning "Alembic not found. Skipping migrations."
fi

echo ""

# ============================================================================
# Step 11: Run tests
# ============================================================================
log_info "Do you want to run tests? (y/n)"
read -r RUN_TESTS

if [ "$RUN_TESTS" = "y" ] || [ "$RUN_TESTS" = "Y" ]; then
    log_info "Running tests..."
    pytest tests/ -v
    log_success "Tests completed"
fi

echo ""

# ============================================================================
# Step 12: Setup pre-commit hooks (optional)
# ============================================================================
if command_exists pre-commit; then
    log_info "Setting up pre-commit hooks..."
    pre-commit install
    log_success "Pre-commit hooks installed"
fi

echo ""

# ============================================================================
# Setup Complete
# ============================================================================
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║              ✅ SETUP COMPLETE! ✅                          ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

log_success "ModelForge-CV development environment is ready!"
echo ""

log_info "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   ${GREEN}source venv/bin/activate${NC}"
echo ""
echo "2. Update .env with your configuration"
echo ""
echo "3. Start the backend server:"
echo "   ${GREEN}uvicorn backend.api.main:app --reload${NC}"
echo ""
echo "4. In another terminal, start the Celery worker:"
echo "   ${GREEN}celery -A backend.workers.celery_app worker --loglevel=info${NC}"
echo ""
echo "5. (Optional) Start all services with Docker:"
echo "   ${GREEN}docker-compose up -d${NC}"
echo ""
echo "6. Access the API documentation:"
echo "   ${BLUE}http://localhost:8000/docs${NC}"
echo ""

if [ "$GPU_AVAILABLE" = true ]; then
    log_success "GPU detected! You can use GPU acceleration for training."
else
    log_warning "No GPU detected. Training will use CPU (slower)."
fi

echo ""
log_info "Happy coding! 🚀"
echo ""