---
version: alpha
name: LA Spot
description: >-
  Two separate design systems live in this repository. The LA Spot app
  (Blazor, wwwroot/app.css) and the Nexus Dev Days page (a standalone published
  page). Tokens are namespaced app-* and nexus-*; the two are never mixed.
  Dark-theme values carry a -dark suffix because this format has no theme mode.
colors:
  # Alias of app-accent-strong, present because the format expects a primary colour.
  primary: "#2F4A3C"
  # Ground and surfaces
  app-bg: "#F3EFE3"
  app-bg-dark: "#12161A"
  app-bg-wash: "#E7E3D4"
  app-bg-wash-dark: "#171D23"
  app-surface: "#C0C3B9"
  app-surface-dark: "#1E252B"
  app-surface-alt: "#D3D6CD"
  app-surface-alt-dark: "#262F36"
  app-well: "#D7DAD1"
  app-well-dark: "#151A1F"
  app-field: "#F0ECE0"
  app-field-dark: "#161B20"
  # Borders
  app-border: "#7D8476"
  app-border-dark: "#3E4A54"
  app-border-strong: "#565D51"
  app-border-strong-dark: "#64727D"
  # Text
  app-text: "#1F2937"
  app-text-dark: "#F4F6F5"
  app-text-muted: "#24323F"
  app-text-muted-dark: "#C2CAD1"
  app-text-faint: "#28342D"
  app-text-faint-dark: "#AAB4BC"
  # Sage accent (fill, interactive fill, hover, text-as-accent, wash, on-colour)
  app-accent: "#769382"
  app-accent-dark: "#8BB09B"
  app-accent-strong: "#2F4A3C"
  app-accent-strong-dark: "#9EC2AE"
  app-accent-hover: "#243A2E"
  app-accent-hover-dark: "#B2D2C0"
  app-accent-ink: "#223429"
  app-accent-ink-dark: "#A7C8B6"
  app-accent-soft: "#DDE4DC"
  app-accent-soft-dark: "#24352C"
  app-on-accent: "#F6F8F5"
  app-on-accent-dark: "#0B0F13"
  # Coral (accent-2: inline code and chat bubbles only)
  app-accent-2: "#D98880"
  app-accent-2-dark: "#EAA096"
  app-accent-2-ink: "#531E1A"
  app-accent-2-ink-dark: "#F7BDB4"
  app-accent-2-soft: "#F8E3DF"
  app-accent-2-soft-dark: "#3C2824"
  app-on-accent-2: "#16202B"
  app-on-accent-2-dark: "#12161A"
  # Amber (watch, moderate risk)
  app-warn: "#C08A3E"
  app-warn-dark: "#E0B070"
  app-on-warn: "#231702"
  app-on-warn-dark: "#12161A"
  app-warn-ink: "#412B06"
  app-warn-ink-dark: "#F2D4A8"
  app-warn-soft: "#F6EDDC"
  app-warn-soft-dark: "#382D1C"
  # Alert (high risk, destructive actions, the medical disclaimer)
  app-alert: "#A8443B"
  app-alert-dark: "#E07A6E"
  app-alert-ink: "#531E1A"
  app-alert-ink-dark: "#F7BDB4"
  app-alert-soft: "#F8E3DF"
  app-alert-soft-dark: "#3C2824"
  app-on-alert: "#FFFFFF"
  app-on-alert-dark: "#12161A"
  # Positive
  app-positive: "#113725"
  app-positive-dark: "#8FD9B0"
  # Disabled
  app-disabled-bg: "#9FA79F"
  app-disabled-bg-dark: "#2E3840"
  app-disabled-text: "#2E3833"
  app-disabled-text-dark: "#9AA6AF"
  # Navigation
  app-nav-bg: "#2A3B31"
  app-nav-bg-dark: "#171D22"
  app-nav-text: "#EDF1EB"
  app-nav-text-dark: "#EDF1EF"
  app-nav-active-bg: "#DDE4DC"
  app-nav-active-bg-dark: "#8BB09B"
  app-nav-active-text: "#22322A"
  app-nav-active-text-dark: "#0B0F13"
  # Body diagram
  app-body-fill: "#E4E6DD"
  app-body-fill-dark: "#3B454D"
  app-body-fill-back: "#CFD2C8"
  app-body-fill-back-dark: "#2C353C"

  # Nexus Dev Days page: three colours. Ground and ink swap between themes;
  # the accent does not.
  nexus-ground: "#E6EADB"
  nexus-ground-dark: "#1E3A2B"
  nexus-ink: "#1E3A2B"
  nexus-ink-dark: "#DCE5D0"
  nexus-accent: "#FFC72C"
  nexus-on-accent: "#1E3A2B"
  nexus-figure-line: "#F1F0E2"
typography:
  # App: Manrope for headings, Work Sans for everything else.
  app-page-title-lg: { fontFamily: Manrope, fontSize: 30px, fontWeight: 800, lineHeight: 1.2 }
  app-page-title-md: { fontFamily: Manrope, fontSize: 26px, fontWeight: 800, lineHeight: 1.2 }
  app-panel-title: { fontFamily: Manrope, fontSize: 17px, fontWeight: 700, lineHeight: 1.2 }
  app-subtitle: { fontFamily: Work Sans, fontSize: 15px, fontWeight: 400, lineHeight: 1.5 }
  app-body: { fontFamily: Work Sans, fontSize: 1rem, fontWeight: 400, lineHeight: 1.5 }
  app-detail: { fontFamily: Work Sans, fontSize: 14px, fontWeight: 400, lineHeight: 1.5 }
  app-caption: { fontFamily: Work Sans, fontSize: 13px, fontWeight: 400, lineHeight: 1.5 }
  app-eyebrow: { fontFamily: Work Sans, fontSize: 12.5px, fontWeight: 600, lineHeight: 1.5, letterSpacing: 0.04em }
  app-badge: { fontFamily: Work Sans, fontSize: 11.5px, fontWeight: 700, lineHeight: 1.5 }
  app-button: { fontFamily: Work Sans, fontSize: 1rem, fontWeight: 600, lineHeight: 1.5 }
  # Nexus page: Bodoni Moda for display, DM Mono for everything else.
  # Fluid sizes are written at their maximum; the clamp() ranges are in the Typography section.
  nexus-h1: { fontFamily: Bodoni Moda, fontSize: 7.2rem, fontWeight: 500, lineHeight: 0.94, letterSpacing: -0.025em }
  nexus-h2: { fontFamily: Bodoni Moda, fontSize: 3.6rem, fontWeight: 500, lineHeight: 1.02, letterSpacing: -0.015em }
  nexus-h3: { fontFamily: Bodoni Moda, fontSize: 1.75rem, fontWeight: 500, lineHeight: 1.1 }
  nexus-letter: { fontFamily: Bodoni Moda, fontSize: 8rem, fontWeight: 400, lineHeight: 0.8 }
  nexus-figure: { fontFamily: Bodoni Moda, fontSize: 2.2rem, fontWeight: 500, lineHeight: 1 }
  nexus-lede: { fontFamily: DM Mono, fontSize: 1rem, fontWeight: 400, lineHeight: 1.75 }
  nexus-body: { fontFamily: DM Mono, fontSize: 0.9rem, fontWeight: 400, lineHeight: 1.7 }
  nexus-label: { fontFamily: DM Mono, fontSize: 0.72rem, fontWeight: 400, lineHeight: 1.7, letterSpacing: 0.14em }
  nexus-button: { fontFamily: DM Mono, fontSize: 0.9rem, fontWeight: 500, lineHeight: 1.7, letterSpacing: 0.1em }
rounded:
  app-control: 9px
  app-tile: 12px
  app-card: 15px
  app-pill: 999px
  nexus-square: 0px
spacing:
  app-gap-sm: 12px
  app-gap-md: 16px
  app-gutter-phone: 16px
  app-gutter-desktop: 40px
  app-card-padding-y: 18px
  app-card-padding-x: 20px
  app-nav-width: 168px
  nexus-gutter-min: 16px
  nexus-gutter-max: 40px
  nexus-rail: 9.5rem
  nexus-page-max: 76rem
components:
  # ── App (light, then -dark) ──
  app-page:
    backgroundColor: "{colors.app-bg}"
    textColor: "{colors.app-text}"
    typography: "{typography.app-body}"
  app-page-dark:
    backgroundColor: "{colors.app-bg-dark}"
    textColor: "{colors.app-text-dark}"
  app-page-wash:
    backgroundColor: "{colors.app-bg-wash}"
  app-page-wash-dark:
    backgroundColor: "{colors.app-bg-wash-dark}"
  app-page-header-title:
    textColor: "{colors.app-accent-ink}"
    typography: "{typography.app-page-title-lg}"
  app-page-header-title-dark:
    textColor: "{colors.app-accent-ink-dark}"
  app-page-header-subtitle:
    textColor: "{colors.app-text-muted}"
    typography: "{typography.app-subtitle}"
  app-page-header-subtitle-dark:
    textColor: "{colors.app-text-muted-dark}"
  app-page-header-eyebrow:
    textColor: "{colors.app-text-faint}"
    typography: "{typography.app-eyebrow}"
  app-page-header-eyebrow-dark:
    textColor: "{colors.app-text-faint-dark}"
  app-button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.app-on-accent}"
    typography: "{typography.app-button}"
    rounded: 0.375rem
    padding: 0.375rem 0.75rem
  app-button-primary-dark:
    backgroundColor: "{colors.app-accent-strong-dark}"
    textColor: "{colors.app-on-accent-dark}"
  app-button-primary-hover:
    backgroundColor: "{colors.app-accent-hover}"
    textColor: "{colors.app-on-accent}"
  app-button-primary-hover-dark:
    backgroundColor: "{colors.app-accent-hover-dark}"
    textColor: "{colors.app-on-accent-dark}"
  app-button-primary-disabled:
    backgroundColor: "{colors.app-disabled-bg}"
    textColor: "{colors.app-disabled-text}"
  app-button-primary-disabled-dark:
    backgroundColor: "{colors.app-disabled-bg-dark}"
    textColor: "{colors.app-disabled-text-dark}"
  app-button-outline:
    backgroundColor: "{colors.app-bg}"
    textColor: "{colors.app-accent-ink}"
    typography: "{typography.app-button}"
    rounded: 0.375rem
    padding: 0.375rem 0.75rem
  app-button-outline-dark:
    backgroundColor: "{colors.app-bg-dark}"
    textColor: "{colors.app-accent-ink-dark}"
  app-button-outline-hover:
    backgroundColor: "{colors.app-accent-strong}"
    textColor: "{colors.app-on-accent}"
  app-button-outline-hover-dark:
    backgroundColor: "{colors.app-accent-strong-dark}"
    textColor: "{colors.app-on-accent-dark}"
  app-button-danger-hover:
    backgroundColor: "{colors.app-alert}"
    textColor: "{colors.app-on-alert}"
  app-button-danger-hover-dark:
    backgroundColor: "{colors.app-alert-dark}"
    textColor: "{colors.app-on-alert-dark}"
  app-card-panel:
    backgroundColor: "{colors.app-surface}"
    textColor: "{colors.app-text}"
    rounded: "{rounded.app-card}"
    padding: 18px 20px
  app-card-panel-dark:
    backgroundColor: "{colors.app-surface-dark}"
    textColor: "{colors.app-text-dark}"
  app-field:
    backgroundColor: "{colors.app-field}"
    textColor: "{colors.app-text}"
    rounded: 0.375rem
  app-field-dark:
    backgroundColor: "{colors.app-field-dark}"
    textColor: "{colors.app-text-dark}"
  app-nav:
    backgroundColor: "{colors.app-nav-bg}"
    textColor: "{colors.app-nav-text}"
    width: 168px
  app-nav-dark:
    backgroundColor: "{colors.app-nav-bg-dark}"
    textColor: "{colors.app-nav-text-dark}"
  app-nav-link-active:
    backgroundColor: "{colors.app-nav-active-bg}"
    textColor: "{colors.app-nav-active-text}"
    rounded: "{rounded.app-control}"
    padding: 8px 10px
  app-nav-link-active-dark:
    backgroundColor: "{colors.app-nav-active-bg-dark}"
    textColor: "{colors.app-nav-active-text-dark}"
  app-nav-primary-mark-active:
    backgroundColor: "{colors.app-warn}"
    textColor: "{colors.app-on-warn}"
  app-nav-primary-mark-active-dark:
    backgroundColor: "{colors.app-warn-dark}"
    textColor: "{colors.app-on-warn-dark}"
  app-status-banner:
    backgroundColor: "{colors.app-accent-soft}"
    textColor: "{colors.app-accent-ink}"
    rounded: "{rounded.app-control}"
    padding: 12px 14px
  app-status-banner-dark:
    backgroundColor: "{colors.app-accent-soft-dark}"
    textColor: "{colors.app-accent-ink-dark}"
  app-status-banner-problem:
    backgroundColor: "{colors.app-warn-soft}"
    textColor: "{colors.app-warn-ink}"
    rounded: "{rounded.app-control}"
    padding: 12px 14px
  app-status-banner-problem-dark:
    backgroundColor: "{colors.app-warn-soft-dark}"
    textColor: "{colors.app-warn-ink-dark}"
  app-risk-badge-low:
    backgroundColor: "{colors.app-well}"
    textColor: "{colors.app-accent-ink}"
    typography: "{typography.app-badge}"
    rounded: "{rounded.app-pill}"
    padding: 2px 10px
  app-risk-badge-low-dark:
    backgroundColor: "{colors.app-well-dark}"
    textColor: "{colors.app-accent-ink-dark}"
  app-risk-badge-moderate:
    backgroundColor: "{colors.app-warn-soft}"
    textColor: "{colors.app-warn-ink}"
    typography: "{typography.app-badge}"
    rounded: "{rounded.app-pill}"
    padding: 2px 10px
  app-risk-badge-moderate-dark:
    backgroundColor: "{colors.app-warn-soft-dark}"
    textColor: "{colors.app-warn-ink-dark}"
  app-risk-badge-high:
    backgroundColor: "{colors.app-alert-soft}"
    textColor: "{colors.app-alert-ink}"
    typography: "{typography.app-badge}"
    rounded: "{rounded.app-pill}"
    padding: 2px 10px
  app-risk-badge-high-dark:
    backgroundColor: "{colors.app-alert-soft-dark}"
    textColor: "{colors.app-alert-ink-dark}"
  app-disclaimer:
    backgroundColor: "{colors.app-alert-soft}"
    textColor: "{colors.app-alert-ink}"
    rounded: "{rounded.app-card}"
    padding: 14px 16px
  app-disclaimer-dark:
    backgroundColor: "{colors.app-alert-soft-dark}"
    textColor: "{colors.app-alert-ink-dark}"
  app-chat-message-user:
    backgroundColor: "{colors.app-accent-2-soft}"
    textColor: "{colors.app-text}"
  app-chat-message-user-dark:
    backgroundColor: "{colors.app-accent-2-soft-dark}"
    textColor: "{colors.app-text-dark}"
  app-chat-close-hover:
    backgroundColor: "{colors.app-surface-alt}"
  app-chat-close-hover-dark:
    backgroundColor: "{colors.app-surface-alt-dark}"
  app-trend-improved:
    textColor: "{colors.app-positive}"
  app-trend-improved-dark:
    textColor: "{colors.app-positive-dark}"
  app-inline-code:
    textColor: "{colors.app-accent-2-ink}"
  app-inline-code-dark:
    textColor: "{colors.app-accent-2-ink-dark}"
  app-body-map-silhouette:
    backgroundColor: "{colors.app-body-fill}"
  app-body-map-silhouette-dark:
    backgroundColor: "{colors.app-body-fill-dark}"
  app-body-map-silhouette-back:
    backgroundColor: "{colors.app-body-fill-back}"
  app-body-map-silhouette-back-dark:
    backgroundColor: "{colors.app-body-fill-back-dark}"
  # ── Nexus page (ground and ink swap in dark; the accent does not) ──
  nexus-page:
    backgroundColor: "{colors.nexus-ground}"
    textColor: "{colors.nexus-ink}"
    typography: "{typography.nexus-body}"
  nexus-page-dark:
    backgroundColor: "{colors.nexus-ground-dark}"
    textColor: "{colors.nexus-ink-dark}"
  nexus-button:
    backgroundColor: "{colors.nexus-accent}"
    textColor: "{colors.nexus-on-accent}"
    typography: "{typography.nexus-button}"
    rounded: "{rounded.nexus-square}"
    padding: 0.85rem 1.2rem
  nexus-button-hover:
    backgroundColor: "{colors.nexus-ink}"
    textColor: "{colors.nexus-ground}"
  nexus-button-hover-dark:
    backgroundColor: "{colors.nexus-ink-dark}"
    textColor: "{colors.nexus-ground-dark}"
  nexus-highlight:
    backgroundColor: "{colors.nexus-accent}"
    textColor: "{colors.nexus-on-accent}"
    padding: 0 0.14em
  nexus-band-active:
    backgroundColor: "{colors.nexus-accent}"
    textColor: "{colors.nexus-on-accent}"
    padding: 1.25rem 1.25rem 1.5rem
  nexus-needle:
    backgroundColor: "{colors.nexus-accent}"
    width: 9px
---

## Overview

This file describes **two design systems that share a repository and a brand green but are deliberately kept apart**. Tokens are namespaced `app-*` and `nexus-*`. Do not mix them: no Manrope or Work Sans on the Nexus page, no Bodoni Moda or marigold in the app.

The values here were read from the code. When code and this file disagree, the code wins and this file is out of date.

### App: LA Spot

The working product: a Blazor Server app for photographing a skin spot, getting a preliminary ABCDE read, and tracking spots over time. The feeling is calm, private and clinical-warm: sage and forest greens on a bone ground, with amber for "watch" and a brick-red alert for high risk and the medical disclaimer. It has a light and a dark theme, both declared once per token with `light-dark()`, and text is held to WCAG AAA (7:1) against both the page and the cards.

Source of truth: `MelanomaDetection/MelanomaDetection.Web/wwwroot/app.css`. Its own header states the rule this file inherits: *every colour comes from the token block; page and component stylesheets must not introduce new hex values.*

### Nexus page: Nexus Dev Days project page

A single-page project presentation for the Nexus Dev Days event. The feeling is a clinical monograph: calm, exact and quietly serious, with a fashion-magazine didone against a technical mono. It is built on a strict **three-colour palette** (ground, ink, one sharp accent) and structured with 1px rules instead of cards.

Source: the published page at `https://claude.ai/artifact/Ag3gQzDNgojUTFiDmTb1G8`. **Its source file is not in this repository**; save it here before treating this section as checkable against code.

## Colors

### App

Ten colours are fixed by the brand (background, surface, both accents, both text tones, per theme). The rest are derived. The light-mode sage accent is a mid-tone that cannot carry text at AAA, so each accent splits into a **fill** (`accent`), an **interactive fill that carries text** (`accent-strong`), and an **ink** for when the accent has to be text (`accent-ink`). Light value first, then dark:

- **Ground and surfaces:** bg `#F3EFE3` / `#12161A` · bg-wash `#E7E3D4` / `#171D23` · surface `#C0C3B9` / `#1E252B` · surface-alt `#D3D6CD` / `#262F36` · well `#D7DAD1` / `#151A1F` · field `#F0ECE0` / `#161B20`
- **Borders:** border `#7D8476` / `#3E4A54` · border-strong `#565D51` / `#64727D`
- **Text:** text `#1F2937` / `#F4F6F5` · text-muted `#24323F` / `#C2CAD1` · text-faint `#28342D` / `#AAB4BC`
- **Sage accent:** accent `#769382` / `#8BB09B` · accent-strong `#2F4A3C` / `#9EC2AE` · accent-hover `#243A2E` / `#B2D2C0` · accent-ink `#223429` / `#A7C8B6` · accent-soft `#DDE4DC` / `#24352C` · on-accent `#F6F8F5` / `#0B0F13`
- **Coral:** accent-2 `#D98880` / `#EAA096` · accent-2-ink `#531E1A` / `#F7BDB4` · accent-2-soft `#F8E3DF` / `#3C2824` · on-accent-2 `#16202B` / `#12161A`
- **Amber:** warn `#C08A3E` / `#E0B070` · on-warn `#231702` / `#12161A` · warn-ink `#412B06` / `#F2D4A8` · warn-soft `#F6EDDC` / `#382D1C`
- **Alert:** alert `#A8443B` / `#E07A6E` · alert-ink `#531E1A` / `#F7BDB4` · alert-soft `#F8E3DF` / `#3C2824` · on-alert `#FFFFFF` / `#12161A`
- **Positive:** positive `#113725` / `#8FD9B0`
- **Disabled:** disabled-bg `#9FA79F` / `#2E3840` · disabled-text `#2E3833` / `#9AA6AF`
- **Navigation:** nav-bg `#2A3B31` / `#171D22` · nav-text `#EDF1EB` / `#EDF1EF` · nav-active-bg `#DDE4DC` / `#8BB09B` · nav-active-text `#22322A` / `#0B0F13`
- **Body diagram:** body-fill `#E4E6DD` / `#3B454D` · body-fill-back `#CFD2C8` / `#2C353C`

`primary` is an alias of `app-accent-strong`, added because the format expects a primary colour; it belongs to the app, and the Nexus page's equivalent is `nexus-ink`.

**Unused today:** `accent-2` (the coral fill) and `on-accent-2` have no consumers in the stylesheets. Only `accent-2-ink` (inline `code`) and `accent-2-soft` (the chat bubble) are used. `accent`, `border` and `border-strong` are mostly edge colours, which this format has no component property for, so the linter reports them as unreferenced.

Not in the YAML because they are translucent, not hex: `border-subtle`, `nav-hover-bg`, `overlay-soft`, `scrim`, `on-accent-wash`, the skeleton sheen, and the shadows and focus ring under Elevation.

### Nexus page

Three colours. Every tint is mixed from them at runtime, so the system has no other token hex values.

- **Ground** (base): `#E6EADB` lichen paper in light, `#1E3A2B` forest in dark.
- **Ink** (structural, for type and every 1px rule): `#1E3A2B` in light, `#DCE5D0` in dark.
- **Accent** (the one sharp colour): marigold `#FFC72C` in both themes. It is a **fill**, never text on the light ground (about 1.3:1); text on it uses `nexus-on-accent` `#1E3A2B`.
- **Figure line** `#F1F0E2`: the constant off-white used for marks drawn over the specimen image, which does not change with theme.
- **Derived tints** via `color-mix()`: secondary text is 72% ink over ground, rules are 30%, and the background lamp glow is 9% ink over transparent.

Colours inside the specimen illustration (skin, lesion, the C-row swatches) are artwork, not tokens.

## Typography

### App

Manrope for headings, Work Sans for body. Headings are 800 (page titles) or 700 (panel titles) and coloured `accent-ink`; links are `accent-ink`; inline `code` is `accent-2-ink`. Body text inherits Bootstrap's 1rem / 1.5, and headings its 1.2 line height. Sizes in use run from 11px to 34px, clustering at 12.5 to 14.5px for secondary text. Weight 600 is the button weight.

| Token | Face | Size | Weight |
|---|---|---|---|
| `app-page-title-lg` / `-md` | Manrope | 30px / 26px | 800 |
| `app-panel-title` | Manrope | 17px | 700 |
| `app-subtitle` | Work Sans | 15px | 400 |
| `app-body` | Work Sans | 1rem | 400 |
| `app-detail` / `app-caption` | Work Sans | 14px / 13px | 400 |
| `app-eyebrow` | Work Sans, uppercase, 0.04em | 12.5px | 600 |
| `app-badge` | Work Sans | 11.5px | 700 |

### Nexus page

**Bodoni Moda** for display, used at large sizes only because its hairlines are fragile small, with italics for emphasis. **DM Mono** for all body text, labels and data, at short measures (46 to 60 characters). Both load from Google Fonts. Banned here: Inter, Roboto, Arial, system sans and Space Grotesk.

| Token | Face | Size (fluid range) | Notes |
|---|---|---|---|
| `nexus-h1` | Bodoni Moda 500 | `clamp(3.1rem, 8.2vw, 7.2rem)` | line 0.94, tracking -0.025em; "the spot" is italic on a marigold highlight |
| `nexus-h2` | Bodoni Moda 500 | `clamp(2rem, 4.6vw, 3.6rem)` | line 1.02, tracking -0.015em |
| `nexus-h3` | Bodoni Moda 500 | 1.75rem | the paired sub-heads use italic 400 at 1.9rem |
| `nexus-letter` | Bodoni Moda italic 400 | `clamp(4.5rem, 10vw, 8rem)` | the ABCDE ledger letters |
| `nexus-figure` | Bodoni Moda 500 | 2.2rem | tabular figures, for the score readout |
| `nexus-lede` | DM Mono 400 | 1rem | line 1.75, 46ch |
| `nexus-body` | DM Mono 400 | 0.9rem | line 1.7 |
| `nexus-label` | DM Mono, uppercase | 0.72rem | tracking 0.14em |
| `nexus-button` | DM Mono 500, uppercase | 0.9rem | tracking 0.1em |

## Layout

### App

Mobile-first. On phones the navigation is a bottom bar and content has 16px side gutters (`20px 16px 48px`). From **860px** the navigation becomes a **168px** sidebar and content padding grows to `36px 40px 56px`. Other breakpoints in use: 520, 560, 620, 640, 700, 760 and 780px. Cards use `18px 20px` padding; header rows use a 16px gap and action groups a 12px gap.

### Nexus page

One reading column beside a **9.5rem sticky margin rail**, inside a **76rem** maximum width with `clamp(16px, 4vw, 40px)` side gutters. Sections are full-width blocks separated by a 1px ink rule with small registration crosses at both ends; section padding is `clamp(2.25rem, 5vw, 4rem)` above and `clamp(3rem, 6vw, 5rem)` below. The hero is two columns (`1.08fr / 0.92fr`). At 900px the rail and hero collapse to one column, and at 640px the ledger, bands and paired columns stack. Left-aligned throughout; nothing is centred.

## Elevation & Depth

### App

Depth comes from tonal surfaces and three shadows, light then dark:

- `--shadow-card`: `0 1px 2px` at rgba(31, 41, 55, 0.12) / rgba(0, 0, 0, 0.4)
- `--shadow-raised`: `0 4px 12px` at rgba(31, 41, 55, 0.18) / rgba(0, 0, 0, 0.5)
- `--shadow-lifted`: `0 6px 16px` at rgba(31, 41, 55, 0.22) / rgba(0, 0, 0, 0.55)

Focus is a single **3px amber ring** everywhere: rgba(146, 96, 20, 0.85) in light, rgba(224, 176, 112, 0.7) in dark. Pages sit on a faint dot grid with a slow drifting wash (`bg-wash`), and content fades up on navigation; both stop under `prefers-reduced-motion`.

### Nexus page

**No shadows and no glassmorphism.** Structure is 1px solid ink rules. Atmosphere is a paper-grain noise overlay (opacity 0.16 light, 0.24 dark, multiplied) and a soft radial lamp glow at the top right. The only thing that "lifts" is the marigold accent. Motion is CSS only: a staggered rise-in on load, an iris opening on the specimen field and the outline tracing itself. Everything rests fully visible, and all of it turns off under `prefers-reduced-motion`.

## Shapes

### App

Rounded and soft. Cards use **15px** (`--radius-card`), controls **9px** (`--radius-control`), badges and chips a full pill (**999px**), avatars and dots **50%** (this format only allows px, rem and em, so circles are described here rather than tokenised). A **12px** radius also appears inline in 16 places and a **10px** one in 8; neither is a token. Buttons and form fields currently fall back to **Bootstrap's 0.375rem** because `app.css` does not override it. That is an inconsistency with the 9px control radius, not a decision.

### Nexus page

**Square.** Every radius is 0. The single exception is the circular dermatoscope field (a 50% radius, which this format cannot express), which is round because the instrument's view is. Registration crosses mark section corners.

## Components

### App

Every component below has a `-dark` twin in the YAML that uses the dark-theme tokens.

- **Primary button:** `accent-strong` fill, `on-accent` text, weight 600; hover and active go to `accent-hover`; disabled uses `disabled-bg` and `disabled-text` at full opacity so the label stays readable.
- **Outline button:** transparent, `accent-ink` text, 1.5px `accent-strong` border; fills on hover.
- **Card panel:** `surface` fill, 1px `border-subtle`, 15px radius, `shadow-card`, `18px 20px` padding.
- **Form field:** `field` fill, 1.5px `border-strong`; focus swaps the border to `accent-strong` and adds the amber ring.
- **Navigation:** `nav-bg` ground with `nav-text`; the active item is `nav-active-bg` with `nav-active-text` at a 9px radius.
- **Risk badge:** pill with low `well` / `accent-ink`, moderate `warn-soft` / `warn-ink`, high `alert-soft` / `alert-ink`.
- **Mascot and logo:** the Mela dog is a CSS **mask** filled with a theme colour (`nav-active-bg` in the sidebar, `accent-strong` on the sign-in page), never a tile. The LA SPOT logo ships as `la-spot-logo-on-dark.png` and `la-spot-logo-on-light.png` and switches by theme through `--brand-logo`, because its white outline disappears on the cream ground.

### Nexus page

- **Button:** marigold fill, forest text, 1px ink border, uppercase mono; hover inverts to ink on ground.
- **Highlight:** `<mark>` as a full marigold block behind the words "the spot".
- **Ledger row:** a ruled row with a giant italic Bodoni letter, a title and description, and a small line glyph; no box around it.
- **Score scale:** a ruler split at the real thresholds (35 and 65) with a marigold needle; the active band fills marigold with forest text.
- **Dermatoscope field:** a circular canvas with a 1px ink border and an offset outline ring, the page's one round object (no token: the format has no way to express a 50% radius).
- **Notice:** a marigold square marker, a top rule and secondary text; the medical disclaimer appears in the hero and the footer.

## Do's and Don'ts

### App

- **Do** take every colour from a token; adding a hex value in a component stylesheet breaks the rule the token block exists for.
- **Do** pair each fill with its `on-` colour, and use `accent-ink` when the accent has to be text.
- **Do** reserve alert red for high risk, destructive actions and the medical disclaimer; sage means calm and amber means watch.
- **Do** keep text at AAA against the page and the cards, in both themes.
- **Don't** put text on `accent` in light mode (`#769382`); it is a mid-tone and fails AAA either way.
- **Don't** let Bootstrap defaults through: its `#0d6efd` blue on disabled buttons and its pink `code` colour were both bugs here.
- **Don't** build on `accent-2` or `on-accent-2` until something needs them; they are defined but unused.

### Nexus page

- **Do** hold to three colours and use the accent as a fill with `nexus-on-accent` text.
- **Do** structure with 1px rules, keep corners square, and leave everything left-aligned.
- **Do** carry a real detail only this subject has: the 35 and 65 thresholds, the recheck cadences and the 6 mm diameter.
- **Do** label the specimen as synthetic, and animate only from a visible resting state.
- **Don't** add cards, rounded corners, shadows, glassmorphism or gradients on the accent.
- **Don't** use Inter, Roboto, Arial, system sans, Space Grotesk, teal, muted corporate blue or a neon purple-to-blue gradient.
- **Don't** invent facts: no accuracy percentages, no event dates and no team names that the project does not state.
