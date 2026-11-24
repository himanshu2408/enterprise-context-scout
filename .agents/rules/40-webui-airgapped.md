---
trigger: glob
description: "Strict TypeScript, zero external CDN/Google Fonts egress, strict CSP, production OIDC session flow, and WCAG 2.2 AA rules."
globs: "**/web-ui/**"
---
# Air-Gapped React Workbench (`web-ui/`) Rules

1. **Zero External Egress:** Never reference `fonts.googleapis.com`, `fonts.gstatic.com`,
   or external CDNs. Bundle `.woff2` fonts locally and enforce strict `CSP`.
2. **Production Auth Flow:** Hydrate identity on boot via `GET /api/v1/auth/me`. Render
   `RoleSwitcher` and `/api/v1/auth/session` strictly when `runtime_mode == "development"`.
3. **Dev Server Safety:** Bind Vite dev server to `127.0.0.1`
   (`allowedHosts: ["localhost", "127.0.0.1"]`), never `0.0.0.0` + `allowedHosts: true`.
4. **Vendor-Neutral Handoff & Strict TS:** Maintain `"strict": true` and zero `any`.
   Never submit `custom_patch` diffs for server-side host execution.
