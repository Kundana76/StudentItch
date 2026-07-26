# StudentItch

**Every startup begins with one problem.**

An AI-curated bank of 60+ real-world engineering and hackathon problem statements for student final-year projects — tagged by domain, difficulty, and a **Build Score** that tells you whether it's actually finishable in a semester. No more staring at a blank final-year-project doc.

🔗 **Live site:** [studentitch-11nr.onrender.com](https://studentitch-11nr.onrender.com/)

---

## Features

- 🔮 **The Idea Core** — a cinematic Three.js hero scene: a glowing crystal core, real-time bloom, and a scroll-driven particle explosion
- 🪐 **Category Planets** — 16 glowing category buttons that fly you straight to filtered problems
- 🃏 **Swipeable Deck** — a Tinder-style card stack of the top 10 highest Build Score problems (drag to save/skip)
- 🔍 **AI Orb Search** — a filterable, searchable browse grid across all 60+ problems (category, difficulty, minimum Build Score)
- 🔖 **Bookmarks** — save problems locally and revisit them anytime
- 🤖 **Itch Bot** — an AI chat assistant that recommends problems based on your skill level and time budget
- 📄 **AI Project Brief Generator** — one click turns any problem into a scoped MVP plan, architecture outline, and milestones

## Tech

Single self-contained `index.html` — no build step, no framework.

- **Three.js** — the Idea Core 3D scene + bloom post-processing
- **GSAP + ScrollTrigger** — scroll-driven scene choreography
- **Lenis** — smooth scrolling
- Vanilla JS, HTML, CSS — everything else

## Running locally

Just open `index.html` in a browser, or serve it with anything static:

```bash
python3 -m http.server 8000
```

## ⚠️ A note on the AI features

The **Generate Project Brief** and **Itch Bot** features call the Anthropic API (`api.anthropic.com/v1/messages`) directly from the browser with no API key attached. This works inside certain sandboxed dev environments that proxy the request, but **will not work as-is on a public host** like Render — those calls will fail gracefully and show a fallback message. To make them functional here, you'd need a small backend/proxy holding your own Anthropic API key.

## License

Add a license of your choice (MIT is a common default for open student projects).
