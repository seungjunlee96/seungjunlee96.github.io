# Seungjun Lee's Personal Website

This is the personal website and blog of Seungjun Lee, hosted on GitHub Pages using Jekyll and the Minimal Mistakes theme.

## Site Structure

```
├── _config.yml          # Jekyll configuration
├── _data/               # Site data files
│   ├── navigation.yml   # Navigation menu
│   └── ui-text.yml      # UI text translations
├── _pages/              # Static pages
│   └── 404.md           # 404 error page
├── _posts/              # Blog posts
│   └── about/           # About page (single post)
└── assets/              # Images and media
    └── images/
        └── about/
            └── profile.jpg
```

## Features

- Clean, modern design using Minimal Mistakes theme
- Responsive layout
- Search functionality
- Author profile
- Social links (GitHub, LinkedIn, Google Scholar)
- Publication showcase
- Skills and experience display

## Local Development

### Prerequisites

- Ruby 2.7 or higher
- Jekyll 4.x
- Bundler

### Setup

1. Install dependencies:
```bash
bundle install
```

2. Run the Jekyll server:
```bash
bundle exec jekyll serve
```

3. Open your browser and navigate to `http://localhost:4000`

### Building

To build the site for production:
```bash
bundle exec jekyll build
```

The generated site will be in the `_site` directory.

## Updating Citation Counts

The site includes a script to automatically update citation counts from Google Scholar:

```bash
python update_citations.py
```

**Note:** This script may require adjustments if Google Scholar's HTML structure changes.

## Deployment

The site is automatically deployed to GitHub Pages when changes are pushed to the main branch. GitHub Pages uses the Jekyll build process to generate the static site.

## Customization

### Adding Posts

Create new markdown files in the `_posts/` directory with the naming convention: `YYYY-MM-DD-title.md`

### Modifying Navigation

Edit `_data/navigation.yml` to add or remove navigation links.

### Updating About Page

Edit `_posts/about/2023-10-04-about.md` to update your biography, publications, and experience.

## Theme

This site uses the [Minimal Mistakes](https://github.com/mmistakes/minimal-mistakes) Jekyll theme by Michael Rose.

## License

This site's content is © Seungjun Lee. All rights reserved.

