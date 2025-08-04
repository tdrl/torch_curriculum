---
applyTo: '**.py'
---
Prefer single-quote strings for consistency with existing code, except where double quotes are necessary (e.g., for JSON keys or when the string contains a single quote). This helps maintain a uniform style across the codebase.
- Example: Use `'example'` instead of `"example"` unless the string contains a single quote, like `"It's a test"`.

TODOs should be formatted as `# TODO(username): description` to clearly indicate the author of the TODO and provide context. This format helps in tracking and managing tasks effectively.