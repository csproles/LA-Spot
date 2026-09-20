# Melanoma Detection

## What it does

A mobile-friendly skin-check app: photograph a lesion, tag its body location and
any symptoms, and get a preliminary ABCDE-based risk read in under a minute.
Track each spot over time: re-check it later and the app measures how it has
changed, suggests when to look again, and lets you compare past checks side by
side.

The image pipeline mirrors the ABCDE rule dermatologists use for a quick visual
melanoma screen:

- **A**symmetry — mirror-overlap analysis around the lesion's mass centroid
- **B**order irregularity — contour circularity
- **C**olor variation — measured against *this photo's own* sampled skin tone,
  with specific "dangerous color" detection (pink/red, blue-gray, white, black)
- **D**iameter — a real millimeter measurement, calibrated using the known
  average width of vellus (fine body) hair visible in the photo as a physical
  reference scale (falls back to "not measurable" if no hair is detected —
  never a fabricated number)
- **E**volving — scored by comparing a spot's photo with its previous one:
  shape change, color change and, only when both photos carry the hair-based
  size calibration, growth. A spot's first photo shows "First photo of this spot"
  rather than a made-up number (a single photo can't show change over time)

A–D each get a 0–10 score, combined into one 0–100 risk indicator. Evolving is
scored separately and kept out of that number, so a spot's first check stays
comparable with its later ones. The pipeline
also removes the dermoscope vignette, denoises, removes hair, and segments the
lesion in LAB color space; the app shows every intermediate image plus four
annotated visualizations (one per scored ABCDE criterion) explaining what each
score is based on. An optional AI-generated plain-language explanation (OpenAI,
safety-constrained to never state a diagnosis) is available on demand. It is
grounded in excerpts of the National Cancer Institute booklet *What You Need To
Know About Melanoma* (`MelanomaDetection.Python/knowledge/`): the model only
sees passages for the features that were flagged, must cite and exactly quote
one for every claim, and its reply is checked in code (quotes, numbers,
diagnosis wording, unflagged features). If it can't produce a reply that passes
after one retry, an explanation built directly from the passages is shown
instead. The risk score itself never comes from the model.

There are two parts:

| Component | Tech | Port |
|---|---|---|
| `MelanomaDetection.Web` | Blazor Server (.NET 10) | `https://localhost:7001` |
| `MelanomaDetection.Python` | Flask + OpenCV | `http://localhost:5002` |

The Blazor app is the UI; it calls the Flask API to do the actual image
processing.

## Setup (quick)

Prerequisites: [.NET 10 SDK](https://dotnet.microsoft.com/download) and Python 3.

**1. Start the Python API:**

```powershell
cd MelanomaDetection/MelanomaDetection.Python
python -m venv venv
.\venv\Scripts\Activate.ps1    # Windows PowerShell
# venv\Scripts\activate.bat    # Windows Command Prompt
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
python main.py
```

If PowerShell blocks the activation script with an execution-policy error, run
`Set-ExecutionPolicy -Scope Process RemoteSigned` first (session-only, doesn't
change any system setting).

An OpenAI API key is only required for the optional "Get AI Explanation"
button — copy `.env.example` (repo root) to `.env` and fill in
`OPENAI_API_KEY` if you want that feature; everything else works without it.

Leave this running — it serves the API on `http://localhost:5002`. Confirm it's
up:

```bash
curl http://localhost:5002/health
# {"status":"healthy"}
```

**2. Start the Blazor app** (in a second terminal):

```bash
cd MelanomaDetection/MelanomaDetection.Web
dotnet run --launch-profile https
```

Open `https://localhost:7001` in a browser. Both processes need to stay running
— the Blazor app calls the Flask API over HTTP for every image analysis.

**3. Set up sign-in** (once):

Every page except sign-in, privacy and terms needs a signed-in person, and
Google is the only way to sign in — there are no passwords to manage. Google
sign-in is free.

1. In [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
   (the same project as the Maps key is fine) go to **APIs & Services →
   Credentials → Create credentials → OAuth client ID**, type **Web
   application**.
2. Under **Authorised redirect URIs** add one entry per origin you run the app
   on, each ending in `/signin-google`:
   - `https://localhost:7001/signin-google` (the `https` launch profile)
   - `http://localhost:5218/signin-google` (the `http` launch profile)
   - `http://localhost:7001/signin-google` (Docker Compose)
3. If the OAuth consent screen is in **Testing**, add each Google account that
   should be able to sign in under **Test users** (limit 100). Switching it to
   **Production** lets anyone sign in; with only the `openid`, `email` and
   `profile` scopes this app requests, Google does not require app
   verification, but it does require the **Privacy policy** and **Terms**
   links — those live at `/privacy` and `/terms` and need the bracketed
   placeholders filled in first.
4. Store the client ID and secret outside source control:

   ```bash
   cd MelanomaDetection/MelanomaDetection.Web
   dotnet user-secrets set "Authentication:Google:ClientId" "xxxx.apps.googleusercontent.com"
   dotnet user-secrets set "Authentication:Google:ClientSecret" "xxxx"
   ```

   For Docker, put the same two values in `.env` as
   `Authentication__Google__ClientId` / `Authentication__Google__ClientSecret`
   (see `.env.example`).

**Demo mode.** The sign-in page also offers **Try the demo**: a throw-away
account that starts with no spots, so a walkthrough always begins at the
onboarding flow without touching anyone's real data. Ending the demo (or
signing out) erases that account; abandoned demos are swept after 8 hours.
It is on in Development (which Docker Compose uses) and off elsewhere unless
`Demo:Enabled` is `true`. Real accounts keep their spots across restarts —
the Flask database on the `skincheck-data` volume is no longer wiped at
startup.

Accounts are stored in the web app's own SQLite file (`skincheck-users.db`,
created and migrated automatically on first run). Everything else about a
person — spots, checks, risk profile — is stored by the Flask API under their
account id, which the web app sends in an `X-User-Id` header.

**Shared secret between the two services.** When you run Flask locally its
port is reachable from your machine (Docker Compose doesn't publish it), so
that header is only trusted when it arrives with the secret in
`SKINCHECK_INTERNAL_KEY` (Flask) / `FlaskApi:InternalKey` (web). Set both to
the same random string — `.env` covers both under Docker Compose; locally use
`dotnet user-secrets set "FlaskApi:InternalKey" "..."` and export
`SKINCHECK_INTERNAL_KEY` before `python main.py`. Leaving it unset is allowed
in Development only (Flask logs a warning).

**API keys and the browser.** Only the Google Maps key ever reaches the browser
— Google's map script has to load there, so it can't be hidden, only locked
down. In [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
open that key and set **Application restrictions → Websites** to your site's
URL(s) (add `http://localhost:7001/*` and `https://localhost:7001/*` for local
runs) and **API restrictions → Maps JavaScript API** only; a key restricted this
way is useless to anyone who copies it. It is never in the shipped JavaScript or
HTML — the server hands it to the page at runtime. Everything else stays on the
server: the OpenAI key and `SKINCHECK_INTERNAL_KEY` live only in the Flask
process, and the Google sign-in secret and cookie keys only in the web process.
Under Docker Compose the Flask port isn't published to the host at all.

**Rate limits and input validation.** Both services throttle per account, so
one person can't run up the OpenAI bill or tie up the analysis service: the web
app allows 8 photo analyses, 4 explanations and 10 Find Care lookups a minute,
caps sign-in attempts at 10 a minute per address and data export/deletion at 5,
and Flask enforces slightly higher limits of its own (`main.py`,
`ENDPOINT_RATE_LIMITS`), plus a lockout for repeated wrong internal keys.
Uploads are checked by content, not just file name (real PNG/JPEG/BMP, under
25 megapixels), and every text field has a length cap (`InputLimits.cs` on the
web side, `validation.py` in Flask — keep the two in step). Limits are held in
memory, so they reset on restart and assume a single instance of each service.

## Setup with Docker (alternative)

Runs both services in containers, so you don't need Python or the .NET SDK
installed. Prerequisite: [Docker Desktop](https://www.docker.com/products/docker-desktop/),
open and running.

**1. Create your `.env`** in the repo root (it's gitignored, so keys stay local):

```powershell
Copy-Item .env.example .env    # macOS/Linux: cp .env.example .env
```

**2. Fill it in.** Only one value is required:

| Variable | Needed for |
|---|---|
| `SKINCHECK_INTERNAL_KEY` | **Required.** Any long random string — it's a shared password between the two containers, and you never have to remember it. Generate one with `[Convert]::ToBase64String((1..32 \| ForEach-Object { Get-Random -Maximum 256 }))` (PowerShell) or `openssl rand -base64 32`. |
| `Authentication__Google__ClientId` / `ClientSecret` | Real Google sign-in ([Setup step 3](#setup-quick), redirect URI `http://localhost:7001/signin-google`). Skip it to use **Try the demo** instead. |
| `GoogleMaps__ApiKey` | The Find Care map. |
| `OPENAI_API_KEY` | The optional "Get AI Explanation" button. |

**3. Build and start:**

```powershell
docker compose up --build
```

The first build takes a few minutes. When the logs settle, open
**`http://localhost:7001`** (plain `http`, not `https`) and click **Try the demo**.

**Everyday commands** (run from the repo root):

| Task | Command |
|---|---|
| Stop | `docker compose down` |
| Start again after code changes | `docker compose up --build` |
| Apply a changed `.env` | `docker compose down`, then `docker compose up` |
| See logs | `docker compose logs -f` |
| Delete all saved data too | `docker compose down -v` |

Your data lives in Docker volumes, so it survives restarts and rebuilds; only
`down -v` erases it.

**Troubleshooting**

- *Port already in use:* something else is on 7001 (or 5002 for a local
  `python main.py`), usually an earlier `dotnet run` or `python main.py`. Stop it, or find it with
  `Get-NetTCPConnection -LocalPort 7001,5002 -State Listen`.
- *"Missing or invalid internal API key":* `SKINCHECK_INTERNAL_KEY` is unset,
  or a container was started before you set it. Set it, then `docker compose down`
  and `docker compose up`. Don't mix Docker with a local `dotnet run` — the two
  sides end up with different keys.
- *The browser says you're offline:* nothing is answering on port 7001, so the
  site's offline page shows instead. Run `docker compose ps`; if `web` isn't
  `Up`, start it with `docker compose up -d`. If the offline page sticks after
  that, unregister the service worker (DevTools → Application → Service
  Workers) or hard-refresh.
- *"You're going a little fast" / HTTP 429:* you hit a rate limit (see above).
  Wait the number of seconds it says and try again.
- *The web container never starts:* it waits for the Flask API's health check;
  see `docker compose logs python-api`.

## How to use

Sign in (Google, or **Try the demo**), and the sidebar (top bar on narrow
screens) has five places:

1. **Home** (`/`) — a dashboard: how many spots you track, which are due for a
   re-check, a body map of where they are, and your latest checks. A short
   risk-profile questionnaire (`/onboarding`) is offered the first time.
2. **Spots** (`/spots`) — every mole or lesion you track, sortable by when it's
   next due, with a body map. Open one (`/spots/{id}`) for its timeline of
   photos, score trend and next check date; rename or archive it there.
3. **Check** (`/check`) — the guided flow for a new photo:
   pick a spot (or track a new one) → take or choose a photo (JPEG, PNG or
   BMP, up to 5 MB, under 25 megapixels) → confirm the photo → confirm the
   detected outline matches your spot → add symptoms and a note → results.
   Results show the risk score, the four ABCD factor bars and, from a spot's
   second photo on, how it changed since last time. Nothing is kept until you
   choose **Save**.
4. **Find care** (`/map`) — pick a Louisiana parish to see nearby dermatology
   providers from the NPI Registry on a Google map (needs `GoogleMaps__ApiKey`).
5. **Profile** (`/profile`) — who you're signed in as (with sign-out), the
   risk profile that tunes how often re-checks are suggested,
   privacy/notification toggles, app preferences, and **Data & account**:
   *Export my data* downloads everything held about you as JSON, *Delete
   account* erases it all and signs you out everywhere.

**All checks** (`/history`, reached from Home) lists every saved check as a
card; sort them, or turn on **Compare** to select up to two and view them side
by side. Click a card to open its full detail view (`/results/{id}`) — every
pipeline-stage image, the four ABCD visual explanations, and the AI explanation
button.

No test images on hand? A labeled set from the ISIC archive lives in
[`Images/Benign/`](Images/Benign/) and [`Images/Malignant/`](Images/Malignant/).

## Medical disclaimer

**Educational use only. This is NOT a substitute for professional medical
diagnosis. Consult a dermatologist.** The risk score is a heuristic derived
from classical image-processing measurements, not a validated diagnostic tool,
and has not been cleared or approved by any regulatory body. If you notice a
new, changing, or unusual mole, see a dermatologist regardless of what this
tool reports.

The disclaimer banner is shown on the Upload and Results (detail view) pages —
the screens that actually produce a risk read. It is not currently repeated on
History or Profile, which show no analysis output of their own.

## Limitations

- **Measured accuracy: 5/10 (50%) on a small labeled ISIC sample** — see
  `MelanomaDetection.Python/validate_accuracy.py`. This is a real, honest
  ceiling of the current heuristic approach, not a rounding error.
- **No trained ML model.** Every score comes from classical CV measurements,
  not a model trained on labeled data.
- **"Evolving" needs two photos of the same spot.** It compares a spot's photo
  with its previous one, so a first photo shows "First photo of this spot" instead
  of a score. It measures shape and color change on consumer phone photos (no
  fixed distance, lighting or registration), and growth only when both photos
  have hair-based size calibration — treat it as a prompt to review with a
  clinician, not a finding.
- **Diameter calibration depends on visible hair.** The mm measurement needs
  fine hair somewhere in the photo to calibrate a pixel-to-mm scale; if none is
  detected (e.g. a dermoscope with polarization that suppresses surface hair),
  diameter is reported as not measurable rather than guessed.
- **Full-resolution pipeline images are session-only.** Spots, checks, scores,
  and the risk profile (including its notification and privacy toggles) are
  all persisted to SQLite (see Setup), but the full-size intermediate
  visualizations (denoised, hair-removed, segmented, etc.) live in a plain
  Python dict in the Flask process — restarting `main.py` clears those
  (thumbnails shown in History are unaffected).
- **The AI explanation is grounded, not guaranteed.** Every claim must quote
  a booklet passage and the reply is validated in code, but code can confirm a
  quote is real, not that the model's paraphrase around it is faithful. The
  booklet is a 1999 publication (the explanation says so), and only its
  ABCD, self-exam and prevention excerpts are used, not its treatment content.
- **Flask API has no auth of its own.** It trusts the `X-User-Id` header sent
  by the web app, verified only via the shared internal secret (see Setup) —
  don't expose port 5002 to the public internet (Docker Compose keeps it
  internal by default).
- **Camera capture falls back to the browser's native file-input mode**
  (`capture="environment"`) on browsers/contexts where `getUserMedia` isn't
  available (no HTTPS, an old browser, or no camera) — otherwise it uses an
  in-app viewfinder with a live preview and front/rear switching.

## Future enhancements

- Replace or augment the heuristic scorer with a trained ML model
- Recalibrate ABCDE thresholds against a much larger labeled dataset than the
  10-image validation set used so far
- Persist full-resolution pipeline images too, so they survive a Flask restart
  (not just the thumbnails/scores, which already do)
