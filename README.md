# Seungjun Lee's Personal Website

This is the personal website of Seungjun Lee, hosted on GitHub Pages using Jekyll and the Minimal Mistakes theme. The site is a single-page portfolio showcasing research, publications, and professional experience.

## Site Structure

```
├── _config.yml          # Jekyll configuration
├── _data/               # Site data files
│   ├── navigation.yml   # Navigation menu (empty for single-page site)
│   └── ui-text.yml      # UI text translations
├── _pages/              # Static pages
│   └── 404.md           # 404 error page
├── index.md           # Main single-page content (Markdown)
└── assets/              # Images and media
    └── images/
        └── about/
            └── profile.jpg
```

## Features

- Clean, modern design using Minimal Mistakes theme
- Single-page layout
- Responsive design
- Author profile with social links (GitHub, LinkedIn, Google Scholar)
- Publication showcase
- Professional experience and awards

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

The site includes an automated citation updater using the Semantic Scholar API. Citation counts are automatically updated daily via GitHub Actions.

### Automatic Updates

Citation counts are automatically updated every day at 00:00 UTC (09:00 KST) via GitHub Actions. No manual intervention is required.

### Manual Updates

To manually update citation counts:

```bash
python update_citations.py
```

The script:
- Uses Semantic Scholar API to fetch citation counts
- Supports DOI-based lookup (most accurate)
- Falls back to title-based search for papers without DOI
- Updates citation counts in `index.md`
- Handles all 5 publications automatically

### How It Works

1. The script queries Semantic Scholar API for each publication
2. Uses DOI when available (more accurate)
3. Falls back to title search for arXiv papers
4. Updates citation counts in the format: `📊 X citations`
5. GitHub Actions runs daily to keep counts up-to-date

## Deployment

The site is automatically deployed to GitHub Pages when changes are pushed to the main branch. GitHub Pages uses the Jekyll build process to generate the static site.

## Customization

### Updating Content

Edit `index.md` to update your biography, publications, and experience. The site uses a single-page layout, so all content is in this file.

### Modifying Navigation

Edit `_data/navigation.yml` to add navigation links if needed (currently empty for single-page site).

## Theme

This site uses the [Minimal Mistakes](https://github.com/mmistakes/minimal-mistakes) Jekyll theme by Michael Rose.

## License

This site's content is © Seungjun Lee. All rights reserved.

