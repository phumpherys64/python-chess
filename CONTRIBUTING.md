# Contributing

Thanks for your interest in contributing! This project welcomes issues and pull requests.

## Getting Started

- Clone the repo and install dependencies:
  - `pip install -r requirements.txt`
- Ensure Stockfish is installed and reachable via `STOCKFISH` or your PATH.
- (Optional) Create `.chessrc.json` with your engine/TB/Lichess defaults.

## Development Workflow

- Create a feature branch from `main`:
  - `git switch -c feature/<short-title>`
- Make focused changes with clear commit messages.
- Run your changes locally; for GUI features, sanity-check the app opens.
- If you add commands or flags, update `README.md`.

## Pull Requests

- Keep PRs small and scoped. Explain motivation, changes, and testing.
- If you add new files or options, include brief docs/examples.
- Automated tests are not yet set up; manual validation steps are appreciated.

## Code Style

- Favor clear, minimal code with explicit names.
- Keep changes consistent with the project’s structure and patterns.
- Avoid adding heavy dependencies unless necessary.

## Reporting Issues

- Include steps to reproduce, expected vs. actual behavior, and environment info.
- For feature requests, describe the use case and desired UX.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

