<div align="center">

![QuantumOX][quantumox128-logo]

   <h3>QuantumOX</h3>

A tic tac toe engine supporting UTTTI. <br>
[Report bug][issue-link]
·
[Feature request][feature-link]

[![Build][build-badge]][build-link]
[![License][license-badge]][license-link] <br>
[![Release][release-badge]][release-link]

</div>

## Overview

QuantumOX is a high-performance, GPLv3-licensed Tic-Tac-Toe engine implemented in **C++**. It implements the **UTTTI (Universal Tic Tac Toe Interface)** — a protocol inspired by UCI — and is designed to handle a wide variety of board geometries (2D and 3D) and rule variants at scale.

Built with attention to search performance and extensibility, QuantumOX integrates advanced search techniques (iterative deepening, alpha-beta pruning, quiescence, transposition tables, and selective extensions) together with pragmatic engineering choices that make it fast and robust across small boards (3x3) and larger multi-dimensional variants (3x3x3, 4x4x4, up to 15x15 grids).

> **Note:** QuantumOX is a command-line / engine component only — it does **not** include a graphical user interface. A GUI that supports UTTTI is under active development by **[the author][author-link]** and will be published separately.

## Key features

* UTTTI-compatible engine protocol for programmatic control and integration.
* Highly optimized search with iterative deepening, quiescence search, and selective extensions.
* Configurable hash table (transposition table) and multi-threaded search support.
* Advanced move ordering heuristics (killer, history, TT, PV guidance) and per-iteration diagnostics in the info output.
* Support for multiple grid sizes and dimensions (2D and 3D variants).
* Portable C++ codebase with a Makefile for Unix-like systems.

## Files in this repository

This distribution includes the following top-level items:

* [README][readme-link] — this document.

* [LICENSE][license-link] — the project's GNU GPLv3 license.

* [src][src-link] — source tree containing the engine implementation and a [Makefile][makefile-link].

## Building

QuantumOX supports both 32-bit and 64-bit targets. The build system is intentionally minimal and portable; it uses a Makefile located in `src/`.

To build on Unix-like systems (Linux, macOS with developer tools), run:

```sh
cd src
make -j build
```

Run `make help` inside `src/` for additional build targets and cross-platform hints. The default targets are tuned for common Intel/AMD CPUs; adjust `CXXFLAGS` in the Makefile if you need compiler-specific tuning.

## Contributing

Contributions, bug reports, and feature requests are welcome. Please follow the repository [Contributing Guide](CONTRIBUTING.md) for the contribution process and coding guidelines.

## License & Redistribution

QuantumOX is distributed under the terms of the **GNU General Public License v3** ([LICENSE][license-link]). You are free to use, study, modify, and redistribute the engine — provided that any distributed binaries are accompanied by the corresponding source or a clear pointer to it, and derivative works are licensed under the GPLv3 as required.

## Support & Contact

For updates, releases, and community discussion, check the repository and the links at the top of this README. If you need to reach the project maintainer directly, see **[the author][author-link]** on GitHub.

<!-- Badges -->

[build-badge]: https://img.shields.io/github/actions/workflow/status/Karuso1/QuantumOX/build.yml?branch=master&style=for-the-badge&label=quantumox&logo=github
[license-badge]: https://img.shields.io/github/license/Karuso1/QuantumOX?style=for-the-badge&label=license&color=success
[release-badge]: https://img.shields.io/github/v/release/Karuso1/QuantumOX?style=for-the-badge&label=official%20release

<!-- Logos -->

[quantumox128-logo]: https://raw.githubusercontent.com/Karuso1/assets/refs/heads/main/qox.png

<!-- Links -->

[author-link]: https://github.com/Karuso1
[build-link]: https://github.com/Karuso1/QuantumOX/actions/workflows/build.yml
[issue-link]: https://github.com/Karuso1/QuantumOX/issues/new?assignees=&labels=&template=BUG-REPORT.yml
[feature-link]: https://github.com/Karuso1/QuantumOX/issues/new?assignees=&labels=&template=FEATURE-REQUEST.yml
[license-link]: https://github.com/Karuso1/QuantumOX/blob/master/LICENSE
[makefile-link]: https://github.com/Karuso1/QuantumOX/blob/master/src/Makefile
[readme-link]: https://github.com/Karuso1/QuantumOX/blob/master/README.md
[release-link]: https://github.com/Karuso1/QuantumOX/releases/latest
[src-link]: https://github.com/Karuso1/QuantumOX/tree/master/src
