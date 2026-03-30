# Contributing to OCR Vehicle Plate Detector

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- A clear title and description
- Steps to reproduce the issue
- Expected vs actual behavior
- Screenshots if applicable
- Your environment details (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please create an issue with:
- A clear title and description
- The motivation for the enhancement
- Example use cases
- Possible implementation approach

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following the code style guidelines
3. **Add tests** if you're adding functionality
4. **Update documentation** if needed
5. **Ensure tests pass** by running `pytest`
6. **Submit a pull request** with a clear description

## Development Setup

1. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/OCR_Vehicle_Plate_Detector.git
cd OCR_Vehicle_Plate_Detector
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies including dev tools:
```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Code Style Guidelines

- Follow PEP 8 style guide for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise
- Use type hints where appropriate

### Code Formatting

Format your code with Black:
```bash
black src/ tests/ scripts/
```

Check code style with Flake8:
```bash
flake8 src/ tests/ scripts/
```

Run type checking with MyPy:
```bash
mypy src/
```

## Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Aim for high test coverage (>80%)

Run tests:
```bash
pytest tests/
pytest --cov=src/ocr_plate_detector tests/
```

## Commit Messages

- Use clear and meaningful commit messages
- Start with a verb (Add, Fix, Update, Remove, etc.)
- Keep the first line under 72 characters
- Add detailed description if needed

Examples:
```
Add plate detection function using YOLOv8
Fix image preprocessing bug in resize function
Update documentation for training script
```

## Project Structure

Please maintain the existing project structure:
- Source code in `src/ocr_plate_detector/`
- Tests in `tests/`
- Scripts in `scripts/`
- Documentation in `docs/`

## Questions?

Feel free to open an issue for any questions or clarifications!
