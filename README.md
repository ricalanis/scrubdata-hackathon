---
title: Hackaton Small
emoji: 🏔️
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: 6.16.0
app_file: app.py
pinned: false
license: mit
---

# hackaton-small

Entry for the **Build Small Hackathon** (Gradio · Hugging Face).

> Small models, big adventure. Total model params ≤ 32B, built on Gradio,
> hosted as a Hugging Face Space under the `build-small-hackathon` org.

## Status

Scaffold. Idea + track not yet locked. See project memory for hackathon
guidelines (tracks, constraints, bonus quests, prizes).

## Develop

```bash
uv sync          # install deps
uv run app.py    # launch the Gradio app locally
```

## Deploy

Create a Space under `build-small-hackathon`, then push this repo to its
git remote. The YAML header above configures the Space.

## Submission checklist

- [ ] Model ≤ 32B params (≤ 4B unlocks Tiny Titan)
- [ ] Gradio app on a HF Space under `build-small-hackathon`
- [ ] Short demo video
- [ ] Social-media post
- [ ] (bonus) fine-tune published, llama.cpp runtime, custom `gr.Server` UI,
      agent trace shared, blog/field-notes post
