# Modern Frontend Design Guidelines for an Insurance Claims Processing Agent

## **1. Theme Overview**
- **Light & Dark Themes:** Support seamless switching with well-chosen background, accent, and surface palettes.
- **Vibrant Accent Colors:** Use bold, optimistic blues, greens, or purples for primary CTAs. Consider gradients and subtle animated transitions to add energy.
- **Accessible Contrast:** Maintain WCAG-compliant color contrast and ensure key elements (text, buttons, alerts) are readable.

---
## **2. Layout**
- **Bento Grids:** Use modular, card-based layouts for dashboards and detail views.
- **Layering & Dimensionality:** Add subtle shadows, overlays, and card elevation for depth.
- **Tall Cards (Mobile):** Vertically oriented cards enhance mobile storytelling and efficiency.
- **Hero Section:** Large, welcoming hero banner with bold text and simple illustration or AI-generated background.

---
## **3. Typography**
- **Large, Clear Headings:** Prioritize headline clarity (e.g. Inter, Open Sans, or IBM Plex).
- **Readable Body Text:** Use 16px+ for paragraph text; increase line height for readability.
- **Bold Highlighting:** Use font weight and color to emphasize actions or notifications.

---
## **4. Components**
- **Primary Button:** Rounded corners, subtle shadow, clear label, loading spinner.
- **Secondary Button:** Outlined style, muted accent color.
- **Input Fields:** Clean, borderless with focus states and clear error messaging.
- **Cards/Modules:** Shadow, border radius, hover elevation effect.
- **Stepper/List:** Clean, numbered progress indicator for workflow.
- **Alert/Notification:** Color-coded, dismissible, with icons for error/success/info.
- **File Upload:** Drag-and-drop area, preview thumbnails, progress bar.

---
## **5. Navigation**
- **Side Navigation (Desktop):** Fixed sidebar with icons & expandable menu.
- **Bottom Navbar (Mobile):** 3-5 icons, text labels, smooth sliding highlight.
- **Breadcrumbs:** Used in multi-step claim workflows for progress context.

---
## **6. Animations & Microinteractions**
- **Progressive Blur:** Use blur for loading/transitions.
- **Smooth Transitions:** All navigation/interaction animated with cubic bezier timing.
- **Feedback Effects:** Button clicks ripple, card hovers elevate.
- **Success Animation:** Animated checkmark on successful claim submission.

---
## **7. Content & Copy**
- **Human-Centered Language:** Use welcoming, plain-language copy. Avoid jargon.
- **Status Messaging:** Clear, positive, empathetic updates ("We’ve received your claim!", "Review in progress.", "Approval sent – payment is processing.")

---
## **8. AI Visuals & Imagery**
- **Custom Illustrations:** AI-generated graphics for hero section, empty states, or stepper backgrounds.
- **Photo Previews:** Thumbnails for uploaded docs/images.
- **Animated Icons:** For steps, alerts, loaders—adds warmth and modernity.

---
## **9. Accessibility**
- **Keyboard Navigation:** TAB accessible throughout, visible focus states.
- **ARIA Labels:** Used on all major elements (forms, nav, alerts).
- **Screen Reader Support:** Semantic HTML, skip to main content.

---
## **10. Documentation & Tokens**
- **Design Tokens:** Use CSS variables or a tokens system for color, spacing, typography.
- **Component Documentation:** Document props, states, and variants. Use Storybook for live previewing.

---
## **Sample CSS Variables**
```css
:root {
  --color-bg-light: #f8f9fb;
  --color-bg-dark: #18191c;
  --color-primary: #384cff;
  --color-success: #2dcc70;
  --color-error: #ff4976;
  --color-info: #f6c343;
  --color-gradient: linear-gradient(90deg, #384cff 0%, #56CAFF 100%);
  --radius-lg: 14px;
  --shadow-md: 0 6px 24px 0 rgba(56,76,255,0.08);
  --font-headline: 'Inter', sans-serif;
}
```

---
## **Recommended Design Tools & Libraries**
- **Figma:** For prototyping themes, components, and layouts.
- **React + Styled Components or TailwindCSS:** For efficient, themeable UI builds.
- **Storybook:** For documenting, demoing, and testing components.
- **Carbon Design System / Material UI:** As a solid component baseline.

---
## **Visual References**
- Bento grid dashboard (dashboard with cards/modules, clear metrics and actions)
- Animated status checkmark (success submission feedback)
- Large hero section with gradient, illustration, user greeting, action button
- Mobile bottom navbar with animated sliding accent

---
## **Summary**
This theme system aims for modularity, clarity, and empathy—where insurance UI feels genuinely friendly, efficient, and trustworthy. Designs balance modern trends (layering, gradients, animated feedback) with great usability, accessibility, and a clean, professional appearance.
