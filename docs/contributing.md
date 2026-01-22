# Contributing to PyIndexNum

We welcome contributions from the community! This document outlines the process for contributing to PyIndexNum.

## Development Environment Setup

PyIndexNum uses `uv` for dependency management. To set up your development environment:

```bash
# Clone the repository
git clone https://github.com/paluigi/PyIndexNum.git
cd PyIndexNum

# Install dependencies with uv
uv sync --dev

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

## Code Quality

We maintain high code quality through:

- **Type annotations**: All functions should have complete type hints
- **Documentation**: Comprehensive docstrings following NumPy style
- **Testing**: Pytest for unit tests with good coverage
- **Linting**: Follow PEP 8 style guidelines

## Pull Request Guidelines

### Structure

When submitting a pull request, please:

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make focused commits** with clear messages:
   ```
   feat: add new multilateral index method

   - Implement GEKS-Törnqvist index
   - Add comprehensive tests
   - Update documentation
   ```

3. **Test thoroughly**:
   ```bash
   # Run all tests
   uv run pytest

   # Run with coverage
   uv run pytest --cov=pyindexnum --cov-report=html
   ```

4. **Update documentation** if adding new features

### PR Template

Please use this structure for your PR description:

```
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Code refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Documentation builds successfully

## Checklist
- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Type annotations complete
- [ ] No breaking changes without discussion
```

### Areas for Contribution

- **New index methods**: Implement additional bilateral or multilateral indices
- **Performance optimizations**: Improve computational efficiency
- **Documentation**: Enhance examples, tutorials, or API docs
- **Testing**: Add more comprehensive test cases
- **Bug fixes**: Identify and fix issues
- **Data validation**: Improve input validation and error handling

## Code Style

- Follow PEP 8 with 88 character line length
- Use descriptive variable names
- Add type hints to all function parameters and return values
- Write comprehensive docstrings for all public functions

## Testing

- Write unit tests for all new functions
- Aim for >90% test coverage
- Test edge cases and error conditions
- Use parametrized tests for multiple scenarios

## Documentation

- Update API documentation for new functions
- Add examples for new features
- Keep README and docs in sync

## Questions?

If you have questions about contributing, feel free to:

- Open an issue for discussion
- Start a discussion in the GitHub discussions tab
- Contact the maintainers

Thank you for contributing to PyIndexNum! 🎉
