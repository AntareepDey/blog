import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import tailwind from '@astrojs/tailwind';
import mdx from '@astrojs/mdx';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import sitemap from '@astrojs/sitemap';
import compress from 'astro-compress';

export default defineConfig({
  site: 'https://antareepdey.github.io',
  base: '/blog',
  format: 'file',
  output: 'static',
  devToolbar: {
    enabled: false
  },
  integrations: [
    react(),
    tailwind({
      applyBaseStyles: false,
    }),
    compress({
      gzip: false,
      brotli: true
    }),
    mdx(),
    sitemap()
  ],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      themes: {
        light: 'material-theme-lighter',
        dark: 'github-dark-dimmed',
      },
      defaultColor: "light",
      wrap: true
    }
  }
});
