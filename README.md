# Antareep's Blog

This is the repository that powers my personal blog, where I occasionally share my thoughts and experiences on various topics. It's built with a focus on a clean reading experience, performance, and modern web technologies.

>This website achieves a **perfect 100 Lighthouse score** 🥳 across Performance, Accessibility, Best Practices, and SEO, ensuring a fast and pleasant experience for all readers.

## Features

- **Modern Stack**: Built with Astro,and Tailwind CSS
- **Responsive Design**: Optimized for all devices from mobile to desktop
- **MDX Support**: Write blog posts in Markdown with React component support
- **Math Rendering**: LaTeX math expressions with KaTeX
- **Code Highlighting**: Syntax highlighting based on Shika
- **Reading Time**: Automatic reading time calculation
- **SEO Optimized**: Meta tags, Open Graph, and Twitter Card support
- **Performance**: Optimized images, lazy loading, and fast page transitions
- **Automatic Footer Update** : The footer is automatically updated based on the year

## Quick Start

1. **Clone or download this template**
   ```bash
   git clone <your-repo-url>
   cd antareep-blog
   ```

2. **Install dependencies**
   ```bash
   bun install
   ```

3. **Start development server**
   ```bash
   bun run dev
   ```

4. **Open your browser**
   Navigate to `http://localhost:4321`
  



## Project Structure

```
/
├── public/                      # All Images to be stored here
├── src/
│   ├── assets/                  # Optimized images and media  
│   ├── components/              # Reusable UI components
│   │   ├── BackButton.astro
│   │   ├── BlogPostCard.astro
│   │   ├── CodeBlock.tsx
│   │   ├── Footer.astro
│   │   ├── Header.astro
│   │   ├── ImageWithCaption.astro
│   │   ├── ProfileSection.astro
│   │   └── ThemeToggle.tsx
│   ├── content/                # Content collections
│   │   ├── blog/               # Blog posts (MDX files)
│   │   └── config.ts           # Content collection configuration
│   ├── layouts/                # Page layouts
│   │   ├── BlogPostLayout.astro
│   │   └── MainLayout.astro
│   ├── pages/                  # File-based routing
│   │   ├── [slug].astro        # Dynamic blog post pages
│   │   ├── index.astro         # Homepage
│   │   └── 404.astro           # Error Page
│   ├── styles/
│   │   └── global.css         # Global styles and theme colors
│   └── utils/
│       └── readingTime.ts     # Utility function that calculates reading time
├── astro.config.mjs           # Astro configuration
├── tailwind.config.mjs
├── packedge.json       
└── bun.lock
```

## Usage :

#### Creating new Posts :
Create new `.md` or `.mdx` files inside `src/content/blog/`:

```markdown
---
title: "Your Post Title"
date: "2025-01-15"
category: "CATEGORY"
excerpt: "Brief description of your post for previews and SEO."
draft: false     # Set to true to hide from production
---

<Your blog content goes here.> 

Following markdown formatting options are supported:

- **Markdown formatting**
- `inline code`
- [Links](https://example.com)
- ![Image Description](/linktoimg.jpg)
- Math equations (inline): $E = mc^2$   
- Math equations (block):
$$
E = mc^2
$$
- For References use : [^1] beside the text and `[^1]:www.reference.com` at the end of the markdown file.

## Headings

### Subheadings

> Blockquotes

```javascript
// Code blocks with syntax highlighting
console.log("Hello, world!");```
```

### Credits :
If using this code as a template for your own Blog Site please provide necessary Credits.
