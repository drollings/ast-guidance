# Job Copilot — Chromium Extension

Local-only human-in-the-loop job application copilot. **NEVER auto-fills forms.**

## How to Install

### 1. Load the Extension Unpacked

1. Open `chrome://extensions` in Chromium
2. Enable **Developer mode** (toggle in top-right)
3. Click **Load unpacked**
4. Select the `extension/` directory from this repository

### 2. Register the Native Messaging Host

After loading the extension, copy the extension ID from `chrome://extensions`, then run:

```bash
cargo run -p job-copilot-daemon -- install-native-messaging \
  --chrome ~/.config/google-chrome/NativeMessagingHosts/ \
  --extension-id <YOUR_EXTENSION_ID>
```

### 3. Start the Daemon

```bash
cargo run -p job-copilot-daemon -- serve --profile ~/.config/job-copilot/profile.toml
```

## Usage

1. Open a job application form (e.g., Greenhouse, Lever, Workday)
2. The extension automatically scans form fields and sends them to the daemon
3. The side panel shows suggested values with confidence scores
4. Use hotkeys to paste values:
   - **Alt+V** — Paste the active field value and advance to the next field
   - **Alt+Shift+V** — Paste and step back to the previous field
   - **Alt+N** — Skip the current field without pasting
5. Edit values in the side panel before applying
6. Click **Apply** or **Skip** buttons in the side panel

## Architecture

- **Content script** (`content-script.js`): Scans DOM for form fields, handles paste actions
- **Background service worker** (`background.js`): Routes messages to/from the native host
- **Side panel** (`side-panel.html/js/css`): UI for reviewing and editing suggested values
- **Native host**: Rust daemon (`job-copilot-daemon`) running locally

## Privacy

- All processing happens locally — no data leaves your machine
- The daemon communicates only with the local LLM endpoint
- Form values are never sent to third-party servers
- The audit log stores only hashed values, never raw PII
