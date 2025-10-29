# Contributing to QuantumOX

Thank you for your interest in contributing to **QuantumOX**, an advanced Tic-Tac-Toe engine written in **C++** that implements the **UTTTI (Universal Tic-Tac-Toe Interface)** protocol.

This document outlines the standards and workflow for contributing to the project.

---

## Project Structure

```
QuantumOX/
├── src/                # Engine source code
│   ├── board.*         # Board logic
│   ├── engine.*        # Core AI logic
│   ├── search.*        # Search and heuristics
│   ├── options.*       # Engine configuration
│   ├── utils.*         # Helper utilities
│   └── main.cpp        # Entry point
│
├── scripts/            # Helper scripts
│   └── natives.sh      # Getting native properties
│
├── .gitignore          # Basic gitignore rules
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

Always make your changes in a separate branch:

```bash
git checkout -b feature/my-change
```

### 3. Follow the coding style

* Use consistent indentation (4 spaces).
* Keep functions modular and readable.
* Add comments for non-trivial logic.
* Avoid hardcoded values; prefer constants in `constants.h`.

For formatting and compilation:

* Use `clang-format` before committing: `clang-format -i src/*.cpp src/*.h`
* Compile with all warnings enabled: `-Wall -Wextra -pedantic`

### 4. Test your changes

Compile the project and run tests manually:

```bash
make clean && make
./QuantumOX
```

Try common UTTTI commands like `uttti`, `isready`, and `go depth 3` to ensure correct behavior.

### 5. Commit your work

Use clear and descriptive commit messages:

```bash
git add .
git commit -m "feat: improve move evaluation heuristic in search.cpp"
```

### 6. Push and create a Pull Request

Push your branch to your fork and open a PR:

```bash
git push origin feature/my-change
```

Then open a Pull Request on GitHub. Explain what was changed, why, and how it improves the project.

---

## Reporting Issues

If you discover bugs or have feature suggestions, please [open an issue](https://github.com/Karuso1/QuantumOX/issues).

Include:

* Steps to reproduce
* Expected vs. actual behavior
* Your environment (OS, compiler, build method)

---

## Additional Notes

* Review the README.md to understand the UTTTI command flow.
* Check existing issues and PRs before starting new work.
* Keep performance and simplicity in mind.

---

### License Notice

By contributing to QuantumOX, you agree that your contributions will be licensed under the same **GPL-3.0 License** as the main project.
