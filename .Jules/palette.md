## 2025-01-24 - [Keyboard Accessibility & ARIA Support]
**Learning:** Vanilla JS projects benefit from using `setAttribute` for ARIA roles and labels to ensure maximum browser compatibility, rather than using the `element.role` or `element.ariaLabel` properties.
**Action:** Always use `setAttribute('role', ...)` and `setAttribute('aria-label', ...)` in vanilla JS environments.
