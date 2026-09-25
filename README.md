# Melanoma Detection

## What it does

A mobile-friendly skin-check app: photograph a lesion, tag its body location and
any symptoms, and get a preliminary ABCDE-based risk read in under a minute.
Track each spot over time: re-check it later and the app measures how it has
changed, suggests when to look again, and lets you compare past checks side by
side.

The current CV pipeline (**V5**, frozen) is: uploaded image → frozen YOLO
instance segmentation → raster lesion mask → ABCD + enhanced interpretable
features (17 total) → a frozen logistic-regression decision model → a fixed
0.25 threshold → **LOWER VISUAL CONCERN** / **ELEVATED VISUAL CONCERN**. This
replaced an earlier classical LAB/Otsu segmentation entirely (YOLO produces
substantially better lesion masks) and, separately, replaced a hand-tuned
rule-count decision layer with a trained model. See
[`docs/CV_PIPELINE.md`](docs/CV_PIPELINE.md) for the full history of how the
pipeline got here (original → YOLO transition → V2 → V3 → V4 → V5) and the
exact feature list.

A–D each also get a 0–10 display score shown alongside the headline verdict
(NOT what decides it: the frozen model's raw 17-feature vector does that):

- **A**symmetry: mirror-overlap analysis, aligned to the lesion's own
  principal axis
- **B**order irregularity: contour circularity, solidity, and boundary
  turning-angle variation
- **C**olor variation: measured against *this photo's own* sampled skin
  tone (color-distance CV, LAB channel spread, color entropy), with specific
  "dangerous color" detection (pink/red, blue-gray, black)
- **D**iameter: pixel-based only. **This pipeline has no physical scale
  reference (no ruler or similar object in the photo), so it never reports
  millimeters**, only a pixel measurement and a relative-to-photo-width
  percentage, both clearly labeled as such in the UI
- **E**volving: scored by comparing a spot's photo with its previous one
  (shape change, color change; size comparison is pixel-based, same caveat as
  D). A spot's first photo shows "First photo of this spot" rather than a
  made-up number

The model's decision score is a **decision score, not a melanoma or cancer
probability**, and YOLO's detection confidence (one of the 17 model inputs)
is a **detection confidence, not a melanoma probability** either. Neither
is presented to the user as one. When more than one lesion instance is
detected in a photo, all instances are kept separate (never unioned into one
mask); the highest-confidence instance is used for analysis, and the app
shows that more than one spot was found. The pipeline also removes the
dermoscope vignette, denoises, and removes hair before segmentation; the app
shows every intermediate image plus four annotated visualizations (one per
scored ABCDE criterion) explaining what each display score is based on. An
optional AI-generated plain-language explanation (OpenAI, safety-constrained
to never state a diagnosis) is available on demand. It is grounded in
excerpts of the National Cancer Institute booklet *What You Need To Know
About Melanoma* (`MelanomaDetection.Python/knowledge/`): the model only sees
passages for the features that were flagged, must cite and exactly quote one
for every claim, and its reply is checked in code (quotes, numbers, diagnosis
wording, unflagged features). If it can't produce a reply that passes after
one retry, an explanation built directly from the passages is shown instead.
The verdict itself never comes from the model.

This is a research/prototype visual-concern screening tool, not a melanoma
diagnosis. See [`docs/CV_PIPELINE.md`](docs/CV_PIPELINE.md) for evaluation
results and known integration TODOs.

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

The frozen YOLO segmentation weights (`models/yolo_melanoma_seg.pt`) and the
frozen V5 decision model (`pipeline_v5/frozen_model.pkl`) are both committed
to this repository. Nothing extra to download for a fresh clone to run the
CV pipeline. `YOLO_WEIGHTS_PATH` can override the weights location if you
need to point at a different checkpoint (`yolo_config.py`).

An OpenAI API key is only required for the optional "Get AI Explanation" and
"Ask a question" (textbook chat) features. Copy `.env.example` (repo root)
to `.env` and fill in `OPENAI_API_KEY` if you want those; everything else,
including the full image-analysis pipeline, works without it.

Leave this running. It serves the API on `http://localhost:5002`. Confirm it's
up:

```bash
curl http://localhost:5002/health
# {"status":"healthy"}
```

**Port 5002 must not already be in use by another Flask instance** (e.g. a
leftover process from a previous run you forgot to stop). Werkzeug's dev
server can appear to start successfully against an already-occupied port
without erroring, silently leaving the old process handling requests instead
of yours. If `/api/image/process` ever returns unexpected errors, check
`http://localhost:5002/health` is really answering from a freshly started
process, or check `netstat -ano | findstr :5002` (Windows) /
`lsof -i :5002` (macOS/Linux) for a stray listener before assuming the
application code is at fault.

**2. Start the Blazor app** (in a second terminal):

```bash
cd MelanomaDetection/MelanomaDetection.Web
dotnet run --launch-profile https
```

Open `https://localhost:7001` in a browser. Both processes need to stay
running: the Blazor app calls the Flask API over HTTP for every image analysis.

**3. Set up sign-in** (once):

Every page except sign-in, privacy and terms needs a signed-in person, and
Google is the only way to sign in. There are no passwords to manage. Google
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
   links. Those live at `/privacy` and `/terms` and need the bracketed
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
It is on in Development (which Docker Compose's local override uses) and off
elsewhere unless `Demo:Enabled` is `true`. Real accounts keep their spots
across restarts: the Flask database on the `skincheck-data` volume is no
longer wiped at startup.

Accounts are stored in the web app's own SQLite file (`skincheck-users.db`,
created and migrated automatically on first run). Everything else about a
person (spots, checks, risk profile) is stored by the Flask API under their
account id, which the web app sends in an `X-User-Id` header.

**Shared secret between the two services.** When you run Flask locally its
port is reachable from your machine (Docker Compose doesn't publish it), so
that header is only trusted when it arrives with the secret in
`SKINCHECK_INTERNAL_KEY` (Flask) / `FlaskApi:InternalKey` (web). Set both to
the same random string. `.env` covers both under Docker Compose; locally use
`dotnet user-secrets set "FlaskApi:InternalKey" "..."` and export
`SKINCHECK_INTERNAL_KEY` before `python main.py` (Flask also reads it from the
repo-root `.env`). Flask refuses to start without it; for throwaway local
development only, set `SKINCHECK_ALLOW_NO_KEY=1` to run without (it logs a
warning and trusts `X-User-Id` from any caller).

**API keys and the browser.** Only the Google Maps key ever reaches the browser:
Google's map script has to load there, so it can't be hidden, only locked
down. In [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
open that key and set **Application restrictions → Websites** to your site's
URL(s) (add `http://localhost:7001/*` and `https://localhost:7001/*` for local
runs) and **API restrictions → Maps JavaScript API** only; a key restricted this
way is useless to anyone who copies it. It is never in the shipped JavaScript or
HTML: the server hands it to the page at runtime. Everything else stays on the
server: the OpenAI key and `SKINCHECK_INTERNAL_KEY` live only in the Flask
process, and the Google sign-in secret and cookie keys only in the web process.
Under Docker Compose the Flask port isn't published to the host at all, and
both containers run as unprivileged users with a memory limit.

**Rate limits and input validation.** Both services throttle per account, so
one person can't run up the OpenAI bill or tie up the analysis service: the web
app allows 8 photo analyses, 4 explanations and 10 Find Care lookups a minute,
caps sign-in attempts at 10 a minute per address and data export/deletion at 5,
and Flask enforces slightly higher limits of its own (`main.py`,
`ENDPOINT_RATE_LIMITS`), plus a lockout for repeated wrong internal keys.
Uploads are checked by content, not just file name (real PNG/JPEG/BMP, under
25 megapixels), and every text field has a length cap (`InputLimits.cs` on the
web side, `validation.py` in Flask; keep the two in step). Limits are held in
memory, so they reset on restart and assume a single instance of each service.

**Browser hardening.** Every response carries a Content-Security-Policy (only
this site's scripts run; only Google Maps, Fonts and profile pictures load from
elsewhere), plus `X-Content-Type-Options`, `Referrer-Policy` and a
`Permissions-Policy` that allows the camera for this site only
(`Services/SecurityHeaders.cs`). If you add a script, stylesheet or image from a
new host, add it there or the browser will block it. `AllowedHosts` is
`localhost`; set it to your real hostname when deploying.

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
| `SKINCHECK_INTERNAL_KEY` | **Required.** Any long random string: it's a shared password between the two containers, and you never have to remember it. Generate one with `[Convert]::ToBase64String((1..32 \| ForEach-Object { Get-Random -Maximum 256 }))` (PowerShell) or `openssl rand -base64 32`. |
| `Authentication__Google__ClientId` / `ClientSecret` | Real Google sign-in ([Setup step 3](#setup-quick), redirect URI `http://localhost:7001/signin-google`). Skip it to use **Try the demo** instead. |
| `GoogleMaps__ApiKey` | The Find Care map. |
| `OPENAI_API_KEY` | The optional "Get AI Explanation" button. |

**3. Build and start:**

```powershell
docker compose up --build
```

This runs the **local** setup: `docker-compose.override.yml` (merged in
automatically) switches on Development mode and the **Try the demo** button.
The base `docker-compose.yml` on its own is the safe-by-default setup. See
[Deploying for real](#deploying-for-real).

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

Both containers restart on their own after Docker Desktop or your computer
restarts; `docker compose down` stops them until you run `up` again.

### Deploying for real

Don't use the local override for anything other people can reach. Start from the
base file only:

```powershell
docker compose -f docker-compose.yml up -d --build
```

That runs in Production: no developer error pages, demo mode off, and the app
refuses to start unless Google sign-in and `SKINCHECK_INTERNAL_KEY` are set.
Before you expose it:

- **HTTPS.** Sign-in cookies are HTTPS-only in Production, so put a
  TLS-terminating reverse proxy (Caddy, nginx, a cloud load balancer) in front
  of port 7001. Plain `http://localhost:7001` won't hold a session outside
  Development.
- **Trust the proxy, and only the proxy.** In `.env`, set
  `ReverseProxy__KnownNetworks__0` to the network the proxy is on (see
  `.env.example`). Without it every visitor looks like the proxy's address and
  shares one sign-in rate limit; with it, only that proxy may say who the client is.
- **Hostname.** Set `AllowedHosts` in `.env` to your domain.
- **Data at rest.** Health data (thumbnails, notes, profiles) sits unencrypted in
  SQLite on the Docker volumes, next to the cookie keys. Put those volumes on an
  encrypted disk and treat backups accordingly.
- **Google OAuth.** Add your real redirect URI (`https://your.domain/signin-google`).

**Troubleshooting**

- *Compose says "required variable SKINCHECK_INTERNAL_KEY is missing":* add it
  to `.env` (any long random string). Both containers refuse to run without it.
- *Port already in use:* something else is on 7001 (or 5002 for a local
  `python main.py`), usually an earlier `dotnet run` or `python main.py`. Stop it, or find it with
  `Get-NetTCPConnection -LocalPort 7001,5002 -State Listen`.
- *"Missing or invalid internal API key":* `SKINCHECK_INTERNAL_KEY` is unset,
  or a container was started before you set it. Set it, then `docker compose down`
  and `docker compose up`. Don't mix Docker with a local `dotnet run`: the two
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

1. **Home** (`/`) is a dashboard: how many spots you track, which are due for a
   re-check, a body map of where they are, and your latest checks. A short
   risk-profile questionnaire (`/onboarding`) is offered the first time.
2. **Spots** (`/spots`) lists every mole or lesion you track, sortable by when
   it's next due, with a body map. Open one (`/spots/{id}`) for its timeline of
   photos, score trend and next check date; rename or archive it there.
3. **Check** (`/check`) is the guided flow for a new photo:
   pick a spot (or track a new one) → take or choose a photo (JPEG, PNG or
   BMP, up to 5 MB, under 25 megapixels) → confirm the photo → confirm the
   detected outline matches your spot → add symptoms and a note → results.
   Results show the risk score, the four ABCD factor bars and, from a spot's
   second photo on, how it changed since last time. Nothing is kept until you
   choose **Save**.
4. **Find care** (`/map`) lets you enter a Louisiana zip code to see nearby
   dermatology providers from the NPI Registry on a Google map (needs
   `GoogleMaps__ApiKey`).
5. **Profile** (`/profile`) shows who you're signed in as (with sign-out), the
   risk profile that tunes how often re-checks are suggested,
   privacy/notification toggles, app preferences, and **Data & account**:
   *Export my data* downloads everything held about you as JSON, *Delete
   account* erases it all and signs you out everywhere.

**All checks** (`/history`, reached from Home) lists every saved check as a
card; sort them, or turn on **Compare** to select up to two and view them side
by side. Click a card to open its full detail view (`/results/{id}`): every
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

The disclaimer banner is shown on the Upload and Results (detail view) pages:
the screens that actually produce a risk read. It is not currently repeated on
History or Profile, which show no analysis output of their own.

## Limitations

See [`docs/CV_PIPELINE.md`](docs/CV_PIPELINE.md) for full evaluation results
(segmentation accuracy, dev/locked-test classification metrics across every
pipeline version) and current known integration TODOs. Headlines:

- **Prototype research results, not clinical validation.** The frozen V5
  model's locked-test balanced accuracy is ~0.71 on a specific ISIC-derived
  evaluation cohort. That's a real, measured number, not a rounding error, and
  not evidence this tool works on the general population or any individual case.
- **The decision layer is a trained logistic-regression model** (V5, frozen;
  see `pipeline_v5/`), not a hand-tuned rule, but its output is a decision
  score, not a melanoma or cancer probability.
- **"Evolving" needs two photos of the same spot.** It compares a spot's photo
  with its previous one, so a first photo shows "First photo of this spot" instead
  of a score. It measures shape and color change on consumer phone photos (no
  fixed distance, lighting or registration); size comparison is pixel-based
  only (see Diameter, below). Treat it as a prompt to review with a
  clinician, not a finding.
- **Diameter has no physical (mm) calibration.** This pipeline does not use
  hair-width or any other physical scale reference, so diameter is reported
  in pixels and as a percentage of the photo's width only, never
  millimeters. See `docs/CV_PIPELINE.md` for why, and for the known UI
  verification TODOs around this.
- **Full-resolution pipeline images are session-only.** Spots, checks, scores,
  and the risk profile (including its notification and privacy toggles) are
  all persisted to SQLite (see Setup), but the full-size intermediate
  visualizations (denoised, hair-removed, segmented, etc.) live in a plain
  in-memory store in the Flask process. Restarting `main.py` clears those, and
  even without a restart they expire after 4 hours and are capped at 12 per
  person (80 overall), oldest first, so memory can't fill up (thumbnails shown
  in History are unaffected).
- **The AI explanation is grounded, not guaranteed.** Every claim must quote
  a booklet passage and the reply is validated in code, but code can confirm a
  quote is real, not that the model's paraphrase around it is faithful. The
  booklet is a 1999 publication (the explanation says so), and only its
  ABCD, self-exam and prevention excerpts are used, not its treatment content.
- **Flask API has no auth of its own.** It trusts the `X-User-Id` header sent
  by the web app, verified only via the shared internal secret (see Setup).
  Don't expose port 5002 to the public internet (Docker Compose keeps it
  internal by default).
- **Camera capture falls back to the browser's native file-input mode**
  (`capture="environment"`) on browsers/contexts where `getUserMedia` isn't
  available (no HTTPS, an old browser, or no camera). Otherwise it uses an
  in-app viewfinder with a live preview and front/rear switching.

## Future enhancements

- Replace or augment the heuristic scorer with a trained ML model
- Recalibrate ABCDE thresholds against a much larger labeled dataset than the
  10-image validation set used so far
- Persist full-resolution pipeline images too, so they survive a Flask restart
  (not just the thumbnails/scores, which already do)
