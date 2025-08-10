# dev_scripts.ps1 - Windows Compatible Version
# Windows PowerShell development helper scripts for Business RAG System

# Function to build and start development environment
function Start-DevEnvironment {
    Write-Host "Starting Business RAG Development Environment" -ForegroundColor Green
    
    # Check if Docker is running
    try {
        docker --version | Out-Null
        Write-Host "Docker is available" -ForegroundColor Blue
    }
    catch {
        Write-Host "ERROR: Docker is not running. Please start Docker Desktop." -ForegroundColor Red
        return
    }
    
    # Build the development container
    Write-Host "Building Docker image..." -ForegroundColor Blue
    docker-compose build rag-dev
    
    if ($LASTEXITCODE -eq 0) {
        # Start the development container
        Write-Host "Starting development container..." -ForegroundColor Blue
        docker-compose up -d rag-dev
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Development environment ready!" -ForegroundColor Green
            Write-Host "Use: Enter-DevContainer to access the container" -ForegroundColor Yellow
        }
        else {
            Write-Host "Failed to start container" -ForegroundColor Red
        }
    }
    else {
        Write-Host "Failed to build Docker image" -ForegroundColor Red
    }
}

# Function to run interactive development session
function Enter-DevContainer {
    Write-Host "Entering development container..." -ForegroundColor Green
    docker-compose exec rag-dev bash
}

# Function to run tests
function Run-Tests {
    Write-Host "Running tests..." -ForegroundColor Green
    docker-compose exec rag-dev python -m pytest tests/ -v
}

# Function to run document reader test
function Test-DocumentReader {
    Write-Host "Testing document reader..." -ForegroundColor Green
    #docker-compose exec rag-dev python test_reader.py
    docker-compose exec rag-dev python tests/test_reader.py
}

# Function to check what documents are available
function Show-Documents {
    Write-Host "Checking available documents..." -ForegroundColor Blue
    
    if (Test-Path "documents") {
        $files = Get-ChildItem "documents" -Include "*.pdf", "*.docx", "*.txt" -Recurse
        if ($files.Count -gt 0) {
            Write-Host "Found $($files.Count) document(s):" -ForegroundColor Green
            foreach ($file in $files) {
                $sizeKB = [math]::Round($file.Length / 1KB, 1)
                Write-Host "  - $($file.Name) ($sizeKB KB)" -ForegroundColor White
            }
        }
        else {
            Write-Host "No supported documents found in documents folder" -ForegroundColor Yellow
            Write-Host "Please add PDF, DOCX, or TXT files to the documents folder" -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "Documents folder not found. Creating it now..." -ForegroundColor Yellow
        New-Item -ItemType Directory -Path "documents" -Force
        Write-Host "Documents folder created. Please add some test files." -ForegroundColor Green
    }
}

# Function to stop all services
function Stop-DevEnvironment {
    Write-Host "Stopping development environment..." -ForegroundColor Red
    docker-compose down
}

# Function to clean up everything
function Clean-Environment {
    Write-Host "Cleaning up Docker resources..." -ForegroundColor Red
    Write-Host "This will remove containers, networks, and unused images" -ForegroundColor Yellow
    $confirm = Read-Host "Are you sure? (y/N)"
    if ($confirm -eq 'y' -or $confirm -eq 'Y') {
        docker-compose down -v
        docker system prune -f
        Write-Host "Cleanup complete!" -ForegroundColor Green
    }
    else {
        Write-Host "Cleanup cancelled" -ForegroundColor Yellow
    }
}

# Function to show logs
function Show-Logs {
    param([string]$Service = "rag-dev")
    Write-Host "Showing logs for $Service..." -ForegroundColor Blue
    docker-compose logs -f $Service
}

# Function to install local dependencies (for IDE support)
function Install-LocalDeps {
    Write-Host "Installing local dependencies for IDE support..." -ForegroundColor Green
    
    # Check if uv is available
    try {
        uv --version | Out-Null
        Write-Host "uv is available" -ForegroundColor Blue
    }
    catch {
        Write-Host "ERROR: uv is not installed. Please install uv first." -ForegroundColor Red
        Write-Host "Install with: irm https://astral.sh/uv/install.ps1 | iex" -ForegroundColor Yellow
        return
    }
    
    # Install dependencies
    uv sync
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Local dependencies installed!" -ForegroundColor Green
        Write-Host "You can now use your IDE with full code completion" -ForegroundColor Yellow
        Write-Host "Python interpreter location: .venv\Scripts\python.exe" -ForegroundColor Blue
    }
    else {
        Write-Host "Failed to install dependencies" -ForegroundColor Red
    }
}

# Function to check project status
function Check-Status {
    Write-Host "Business RAG System - Project Status" -ForegroundColor Cyan
    Write-Host "====================================" -ForegroundColor Cyan
    
    # Check Docker
    try {
        docker --version | Out-Null
        Write-Host "[OK] Docker is available" -ForegroundColor Green
    }
    catch {
        Write-Host "[ERROR] Docker is not available" -ForegroundColor Red
    }
    
    # Check uv
    try {
        uv --version | Out-Null
        Write-Host "[OK] uv is available" -ForegroundColor Green
    }
    catch {
        Write-Host "[ERROR] uv is not available" -ForegroundColor Red
    }
    
    # Check project files
    $requiredFiles = @(
        "pyproject.toml",
        "Dockerfile", 
        "docker-compose.yml",
        "src\rag\simple_reader.py",
        "test_reader.py"
    )
    
    foreach ($file in $requiredFiles) {
        if (Test-Path $file) {
            Write-Host "[OK] $file exists" -ForegroundColor Green
        }
        else {
            Write-Host "[MISSING] $file" -ForegroundColor Red
        }
    }
    
    # Check documents
    Show-Documents
    
    # Check containers
    Write-Host "`nDocker containers:" -ForegroundColor Blue
    docker-compose ps
}

# Show help
function Show-Help {
    Write-Host "Business RAG Development Commands:" -ForegroundColor Cyan
    Write-Host "=================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Setup & Status:" -ForegroundColor Yellow
    Write-Host "  Check-Status           - Check project setup status" -ForegroundColor White
    Write-Host "  Install-LocalDeps      - Install deps locally for IDE" -ForegroundColor White
    Write-Host "  Show-Documents         - Show available documents" -ForegroundColor White
    Write-Host ""
    Write-Host "Development:" -ForegroundColor Yellow  
    Write-Host "  Start-DevEnvironment   - Build and start dev environment" -ForegroundColor White
    Write-Host "  Enter-DevContainer     - Enter the development container" -ForegroundColor White
    Write-Host "  Test-DocumentReader    - Test the document reader" -ForegroundColor White
    Write-Host "  Run-Tests             - Run all tests" -ForegroundColor White
    Write-Host ""
    Write-Host "Monitoring:" -ForegroundColor Yellow
    Write-Host "  Show-Logs             - Show container logs" -ForegroundColor White
    Write-Host ""
    Write-Host "Cleanup:" -ForegroundColor Yellow
    Write-Host "  Stop-DevEnvironment   - Stop all services" -ForegroundColor White
    Write-Host "  Clean-Environment     - Clean up Docker resources" -ForegroundColor White
    Write-Host ""
    Write-Host "Help:" -ForegroundColor Yellow
    Write-Host "  Show-Help             - Show this help" -ForegroundColor White
    Write-Host ""
    Write-Host "Quick Start:" -ForegroundColor Green
    Write-Host "  1. Check-Status" -ForegroundColor White
    Write-Host "  2. Add documents to documents/ folder" -ForegroundColor White  
    Write-Host "  3. Start-DevEnvironment" -ForegroundColor White
    Write-Host "  4. Test-DocumentReader" -ForegroundColor White
}

# Auto-show help when script is loaded
Write-Host "Business RAG System - Development Scripts Loaded" -ForegroundColor Green
Write-Host "Run 'Show-Help' to see available commands" -ForegroundColor Yellow