# Antareep's Blog Template
This is my personal blog site, where I occasionally share my thoughts, writings, and findings on various topics. It's built with a focus on a clean reading experience, performance, and modern web technologies.

>This website achieves a **perfect 100 Lighthouse score** 🥳 across Performance, Accessibility, Best Practices, and SEO, ensuring a fast and pleasant experience for all users.



## Features

- **Modern Stack**: Built with Astro 5,and Tailwind CSS
- **Responsive Design**: Optimized for all devices from mobile to desktop
- **MDX Support**: Write blog posts in Markdown with React component support
- **Math Rendering**: LaTeX math expressions with KaTeX
- **Code Highlighting**: Syntax highlighting based on Shika
- **Reading Time**: Automatic reading time calculation
- **SEO Optimized**: Meta tags, Open Graph, and Twitter Card support
- **Performance**: Optimized images, lazy loading, and fast page transitions
- **Accessibility**: WCAG compliant with proper ARIA labels and keyboard navigation

## Quick Start

1. **Clone or download this template**
   ```bash
   git clone <your-repo-url>
   cd antareep-blog
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```

4. **Open your browser**
   Navigate to `http://localhost:4321`
  



## Project Structure

```
/
├── public/
│   └── favicon.svg              # Site favicon
├── src/
│   ├── assets/                  # Optimized images and media
│   │   └── images/             # Store your images here
│   ├── components/             # Reusable UI components
│   │   ├── BackButton.astro
│   │   ├── BlogPostCard.astro
│   │   ├── Footer.astro
│   │   ├── Header.astro
│   │   ├── ImageWithCaption.astro
│   │   ├── ProfileSection.astro
│   │   └── ThemeToggle.tsx
│   ├── content/                # Content collections
│   │   ├── blog/              # Your blog posts (MDX files)
│   │   └── config.ts          # Content collection configuration
│   ├── layouts/               # Page layouts
│   │   ├── BlogPostLayout.astro
│   │   └── MainLayout.astro
│   ├── pages/                 # File-based routing
│   │   ├── blog/
│   │   │   └── [slug].astro   # Dynamic blog post pages
│   │   └── index.astro        # Homepage
│   ├── styles/
│   │   └── global.css         # Global styles and theme colors
│   └── utils/
│       └── readingTime.ts     # Utility functions
├── astro.config.mjs           # Astro configuration
├── tailwind.config.mjs        # Tailwind CSS configuration
└── package.json
```

## Usage :

#### Creating new Posts :
Create new `.md` or `.mdx` files in `src/content/blog/`:

```markdown
---
title: "Your Post Title"
date: "2025-01-15"
category: "CATEGORY"
excerpt: "Brief description of your post for previews and SEO."
featured: false  # Set to true for featured posts
draft: false     # Set to true to hide from production
---

Your blog content goes here. You can use:

- **Markdown formatting**
- `inline code`
- [Links](https://example.com)
- Images
- Math equations: $E = mc^2$

## Headings

### Subheadings

> Blockquotes

```javascript
// Code blocks with syntax highlighting
console.log("Hello, world!");```
```

**Store images in `src/assets/images/`**


### Happy blogging! 🎉
