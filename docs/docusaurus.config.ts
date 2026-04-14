import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkCodeRegion from 'remark-code-region';

const config: Config = {
  title: 'Treant',
  tagline: 'High-performance, lock-free Monte Carlo Tree Search for Rust',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://mcts.dev',
  baseUrl: '/',

  onBrokenLinks: 'throw',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          remarkPlugins: [[remarkCodeRegion, { rootDir: '..' }]],
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  plugins: ['./plugins/wasm-plugin.js'],

  themeConfig: {
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Treant',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          to: '/playground',
          label: 'Playground',
          position: 'left',
        },
        {
          href: 'https://docs.rs/treant',
          label: 'API',
          position: 'left',
        },
        {
          href: 'https://github.com/patricker/treant',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Learn',
          items: [
            {
              label: 'Docs',
              to: '/docs/intro',
            },
            {
              label: 'Playground',
              to: '/playground',
            },
          ],
        },
        {
          title: 'Reference',
          items: [
            {
              label: 'API (docs.rs)',
              href: 'https://docs.rs/treant',
            },
            {
              label: 'Crates.io',
              href: 'https://crates.io/crates/treant',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/patricker/treant',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Treant Contributors. MIT License.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['rust', 'toml'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
