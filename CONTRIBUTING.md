# Contributing to QuantumOX

Thank you for your interest in contributing to **QuantumOX**, an advanced Tic-Tac-Toe engine that implements the **UTTTI (Universal Tic-Tac-Toe Interface)** protocol.

This document outlines the process and standards for contributing to the project.

---

## Development Guidelines

### Project Structure

```
QuantumOX/
├── src/                # Engine source code
│   ├── board.*         # Board logic
│   ├── engine.*        # Core AI logic
│   ├── search.*        # Search and heuristics
│   ├── options.*       # Engine configuration
│   ├── utils.*         # Helper utilities
│   └── main.cpp        # Entry point
|
├── scripts/            # Helper scripts
|   └── natives.sh      # Getting native properties
|
├── .gitignore          # A basic gitignore
├── README.md           # Main documentation
├── CONTRIBUTING.md     # Contribution guide
├── CODE-OF-CONDUCT.md  # Code of Conduct
└── LICENSE             # GPL-3.0 license
```

---

## How to Contribute

### 1. Fork the repository

Click the **Fork** button on GitHub and clone your fork locally:

```bash
git clone https://github.com/<your-username>/QuantumOX.git
cd QuantumOX
```

### 2. Create a new branch

Always make your changes in a new branch:

```bash
git checkout -b feature/my-change
```

### 3. Follow the coding style

* Use consistent indentation (4 spaces).
* Keep functions modular and readable.
* Add comments for complex or non-obvious logic.
* Avoid hardcoded values; prefer constants or configuration options.

For C++ contributions:

* Format with `clang-format` before committing: `clang-format -i src/*.cpp src/*.h`
* Compile with all warnings enabled: `-Wall -Wextra -pedantic`.

### 4. Test your changes

Ensure that your modification does not break any functionality. Run the engine and verify with sample UTTTI commands such as:

```bash
python main.py  # or ./QuantumOX if compiled
```

### 5. Commit your work

Use descriptive commit messages:

```bash
git add .
git commit -m "feat: add principal variation merging for negamax and minimax"
```

### 6. Push and create a Pull Request

```bash
git push origin feature/my-change
```

Then, open a Pull Request (PR) on GitHub. Describe what you changed, why you changed it, and how it improves the project.

---

## Reporting Issues

If you find a bug or have a feature request, please open an issue at:
[https://github.com/Karuso1/QuantumOX/issues](https://github.com/Karuso1/QuantumOX/issues)

When reporting an issue, include:

* Steps to reproduce.
* Expected vs. actual behavior.
* Environment details (OS, Python/C++ version, etc.).

---

## Additional Notes

* Read the README.md to understand UTTTI command flow.
* Check existing issues and pull requests before starting new work.
* Keep performance in mind; QuantumOX emphasizes efficient computation.

---

### License Notice

By contributing to QuantumOX, you agree that your contributions will be licensed under the same **GPL-3.0 License** as the main project.
