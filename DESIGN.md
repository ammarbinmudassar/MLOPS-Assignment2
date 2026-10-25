---
name: "VisionGuard"
description: "A simple, readable coursework interface for uncertain image-origin estimates."
colors:
  ink: "#172a3a"
  muted: "#4d6070"
  accent: "#2056b8"
  accent-hover: "#173f88"
  line: "#d5dfe8"
  surface: "#ffffff"
  background: "#f4f7fa"
  soft-blue: "#edf3fc"
  warning: "#715016"
  warning-bg: "#fff8e9"
  error: "#962d34"
typography:
  body:
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif"
    fontSize: "1rem"
    lineHeight: 1.6
  headline:
    fontSize: "2rem"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "-.025em"
  title:
    fontSize: "1.15rem"
    fontWeight: 700
    lineHeight: 1.4
  label:
    fontSize: "1rem"
    fontWeight: 650
  small:
    fontSize: ".875rem"
    lineHeight: 1.6
  prediction:
    fontSize: "2.5rem"
    fontWeight: 700
    lineHeight: 1.2
  confidence:
    fontSize: "1.65rem"
    fontWeight: 650
    lineHeight: 1.6
rounded:
  panel: "12px"
  field: "8px"
  action: "7px"
  file-button: "6px"
spacing:
  compact: ".6rem"
  action-gap: ".75rem"
  base: "1rem"
  medium: "1.25rem"
  section: "1.5rem"
  panel: "2rem"
  page: "3rem"
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.surface}"
    rounded: "{rounded.action}"
    padding: ".65rem 1.2rem"
    typography: "{typography.label}"
  button-primary-hover:
    backgroundColor: "{colors.accent-hover}"
    textColor: "{colors.surface}"
  button-secondary:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.accent}"
    rounded: "{rounded.action}"
    padding: ".65rem 1.2rem"
  button-secondary-hover:
    backgroundColor: "{colors.soft-blue}"
  panel:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.panel}"
    padding: "2rem"
  file-field:
    backgroundColor: "#f8faff"
    rounded: "{rounded.field}"
    padding: "1.25rem"
  navigation:
    textColor: "{colors.muted}"
    padding: ".3rem 0"
  caveat:
    backgroundColor: "{colors.warning-bg}"
    textColor: "{colors.warning}"
    rounded: "{rounded.action}"
    padding: "1rem"
---

# Design System: VisionGuard

## Overview

**Creative North Star: "VisionGuard coursework utility"**

VisionGuard uses a restrained coursework utility world: white content on a cool gray surround, dark blue-gray text, and blue actions. The interface keeps the image check, estimate, recovery path, and saved metadata easy to scan.

Flat panels and fine borders provide structure. Native controls and concise explanations suit the confirmed direct HTML/CSS implementation; uncertainty stays visible beside the result.

**Key Characteristics:**
- Flat white panels and fine rules.
- Blue actions with explicit keyboard focus.
- Readable native forms and a horizontally scrollable history table.
- Visible model uncertainty and provenance.

## Colors

Cool neutrals hold the content; restrained blue identifies interactive and result emphasis.

### Primary

- **Action Blue** (`accent`): primary actions, links, current navigation, focus outlines, and prediction emphasis.
- **Deep Action Blue** (`accent-hover`): hovered links and primary actions.
- **Soft Blue** (`soft-blue`): native file-selector button and secondary-action hover.

### Secondary

- **Warm Warning** (`warning`, `warning-bg`): the model uncertainty caveat.
- **Error Red** (`error`): friendly-error headings.

### Neutral

- **Blue-gray Ink** (`ink`): body text and headings.
- **Muted Slate** (`muted`): explanations, field help, metadata labels, and footer.
- **Fine Rule** (`line`): panels, navigation surround, table divisions, and footer separator.
- **White Surface** (`surface`): panels and header.
- **Cool Surround** (`background`): page background.

**The Action Blue Rule.** Use blue for actions, current navigation, and the prediction value; use warning and error colors for their corresponding messages.

## Typography

**Body Font:** native system sans stack in the body token. There is no loaded font asset or separate display family. Current headings inherit that stack; this is an observation of the build, not a display-font prescription.

### Hierarchy

The frontmatter records the page headline, section title, body, label, small text, prediction, and confidence roles. The headline becomes (1.75rem) at the narrow breakpoint. Native file input and table text use (.9rem), and file-input text becomes (.8rem) on narrow screens. Result labels and metadata use (.9rem). Brand text uses (1.2rem), weight (750), and tracking (-.02em). Footer text uses (.85rem).

Paragraphs stop at (68ch); introductory copy stops at (58ch). Confidence and history cells use tabular numerals. Labels remain sentence case.

## Layout

The header and footer are centered within (70rem). Main content uses `min(100% - 3rem, 67rem)` with margins (3rem auto 4rem) and minimum height (55vh). Upload uses a flexible form column and a (17rem) aside separated by (3rem). Result and error panels stop at (48rem). Panel padding uses the panel spacing token.

At (760px) and below, upload becomes one column and the explanatory aside becomes two columns. At (480px) and below, side gutters become (1rem), main margins become (2rem auto 3rem), panel padding becomes (1.25rem), and aside sections stack. Result metadata changes from an (8rem) label column to one column; long values wrap anywhere. Actions wrap with the action-gap token.

History preserves a minimum table width of (46rem) inside a focusable horizontal scroll region. Filenames truncate at (23rem), with full names in link titles. The footer wraps naturally.

## Elevation & Depth

No box shadows, gradients, or raised layers are implemented. White panels, the cool surround, fine borders, and subdued table header/hover fills establish separation. Focus is a (3px) blue outline offset by (4px).

**The Flat Surface Rule.** Separate surfaces with background tone and fine borders; the shipped system uses no box shadows.

## Shapes

Panels, table wrappers, and empty states use the panel radius. File-field surrounds use the field radius; actions and caveats use the action radius. The native file-selector button uses the file-button radius. Borders are (1px); the file-field border is dashed, while the active navigation underline is (2px). There are no pictogram or illustration assets.

## Components

### Buttons

Compact sentence-case actions use an inline-flex centered label, minimum height (2.8rem), and the primary or secondary tokens. Primary hover darkens its fill and border; secondary hover adds Soft Blue. Background and border transitions take (150ms ease-out); reduced-motion preference removes transitions. Disabled submission uses a muted fill (`#647896`) and wait cursor. Submission changes the text to “Checking image…” and sets the form busy; a polite live status explains the wait. Returning through browser history restores the ready state.

### Cards / Containers

White, bordered panels organize the upload, result, error, and empty-history content. Result places prediction and confidence side by side, then the warning caveat, provenance details, retention note, and wrapping actions. Error uses an error-colored heading, alert message, and recovery action.

### Inputs / Fields

The native required file input sits inside a lightly tinted dashed surround. It retains an explicit label and connected type/size help. The file-selector button has a Soft Blue fill, blue text, fine border, and darker pale-blue hover. Global keyboard focus applies to inputs and links as well as actions.

### Navigation

The white header pairs a compact wordmark with Upload and History links. Muted links become blue on hover; the current page gains blue text, a blue bottom border, and weight (650). At the narrow breakpoint, navigation gap becomes (1rem) and text becomes (.9rem). A skip link appears when focused.

### History table

A caption labels the white table. Muted, left-aligned headers sit on a pale fill; rows use fine separators and a subtle hover tint. Numeric values align consistently through tabular figures. The empty state uses a bordered white panel and an upload action.

### Uncertainty caveat

A warm tinted, bordered block carries the estimate limitation in ordinary paragraph text. Its warning tone is independent of the AI/Real label: neither verdict is presented as a guaranteed truth.

## Do's and Don'ts

### Do:

- Do keep the uncertainty caveat visible with the prediction and confidence.
- Do preserve visible focus, the skip link, and semantic form and table labels.
- Do let long metadata wrap and let history scroll within its own container.
- Do use the established panel, button, and spacing tokens for extensions.

### Don't:

- Don't use confidence as proof of image origin.
- Don't hide history columns to fit a narrow viewport.
- Don't add decorative illustration to this scoped coursework interface.
