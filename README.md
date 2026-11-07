# AttentionLens

**A browser-based prototype that estimates a learner's attention from a webcam while they watch a short lesson, adapts the session when attention drifts, and produces an evidence-based report linking attention to quiz performance and metacognitive confidence.**

Author: Md Shahid Afridi

| | |
|---|---|
| **Live demo** | https://attentionlens.vercel.app |
| **Source code** | https://github.com/ms-afridi/attention-lens |
| **Demo video** | https://www.youtube.com/watch?v=3pmC0SRynFY |

> The demo backend runs on a free hosting tier and sleeps when idle. The first request can take up to a minute to wake it. Please use desktop Chrome and allow camera access. A full run takes about 5 minutes.

---

## 1. What problem this addresses

Learners often lose attention (mind wandering) without noticing, and they often misjudge how well they have learned. AttentionLens is a small end-to-end system for studying both:

1. **Attention:** can multimodal behavioural signals from an ordinary webcam indicate when a learner's attention drifts, in real time?
2. **Metacognition:** do learners know what they know? Each quiz answer carries a confidence rating, so confident-but-wrong answers can be identified.
3. **Adaptive support:** can the learning environment respond to a drop in attention (here: pause and offer a replay)?

This is a **prototype**, not a validated instrument. Section 9 states its limits plainly.

---

## 2. What a participant experiences

1. **Start page.** Explains the study, links to this repository, and asks for consent (including an optional "blur my snapshots" setting).
2. **Sensor check (about 10 s).** Shows live frame rate and blink signals, then asks for three slow blinks. This measures camera quality and sets personal blink thresholds.
3. **Gaze calibration (about 30 s).** Nine dots appear in turn. The system learns the mapping from the participant's eyes to screen positions and reports its own validation error.
4. **Lesson.** A short story video plays (embedded from YouTube). While it plays:
   - Gaze, blinks, eye closure, head pose, posture and interaction events are tracked.
   - 1 to 4 **thought probes** ("were you thinking about something unrelated?") are asked, depending on video length. These are the self-reported reference labels.
   - If smoothed attention risk stays high, the video pauses and offers a 10-second replay.
   - Small **snapshots** are saved when attention drops and a visible cause exists.
5. **Quiz.** Five questions generated from the lesson transcript, one at a time. Answers are final once submitted. Each question requires a **confidence rating (1 to 5)**. The camera stays on, and the time spent on each question is recorded.
6. **Report.** Attention timeline, snapshots with reasons, gaze and blink charts, a per-question table (time spent, confidence, correctness, gaze on the question, attention risk), data-quality notes, and an AI-written summary. It can be downloaded as a **PDF**.

---

## 3. Signals measured

All signals are computed **in the browser** from a pretrained face-landmark model (MediaPipe Face Landmarker). Per 5-second window, 14 features are sent to the backend:

| Group | Features |
|---|---|
| Gaze | gaze-off-lesson ratio, gaze dispersion, saccade rate |
| Head and posture | yaw, pitch, head motion, leaning closer or further |
| Eyes | blink rate, eye-closure fraction (PERCLOS-style) |
| Face | face present, mouth open (possible yawn), brow furrow |
| Behaviour | tab hidden, pause and seek events |

**Gaze mapping.** The iris position inside each eye (from MediaPipe iris landmarks), plus head pose and face position, is mapped to a screen position by a **per-person ridge regression** trained in the 9-point calibration. Accuracy is checked by **leave-one-point-out validation**. If the error is too large, the app says so and falls back to relative gaze.

**Blinks.** Detected from two signals (the model's blink score and an eye-aspect ratio from the landmarks), normalised with each person's open-eye and closed-eye levels measured in the sensor check. Blink numbers are only reported if the sensor check detected at least two of three blinks and the frame rate was adequate.

---

## 4. How the attention estimate works

- The 14 window features go to a model in `backend/attention_model.py`, which returns a **mind-wandering risk** between 0 and 1. The risk is smoothed over time.
- **Current model: a transparent, hand-set heuristic** (a weighted sum passed through a sigmoid). It has not been fitted to real data yet.
- A **trainable pipeline** is included (`backend/train.py`). It learns from sessions exported by the app, using the thought probes as labels (windows in the 10 s before a probe take that probe's answer). It uses subject-wise cross-validation, compares logistic regression with gradient boosting, and runs leave-one-modality-out ablations. When a trained model exists, the backend uses it automatically.

**The language model never judges attention.** The LLM (Groq, `openai/gpt-oss-120b`, with fallbacks) is used only to write quiz questions from the transcript, grade short answers, and phrase the report summary.

---

## 5. Privacy and data handling

- Camera video is **never recorded or uploaded**. It is processed in the browser.
- Snapshots live only in the page's memory. They are deleted when the tab is closed or refreshed, or when the participant presses "Delete all my data". The PDF is generated on the participant's device.
- The server receives numeric features, quiz answers and answer times. **It stores nothing.** Quiz answers are processed by the LLM provider (Groq) for grading.
- The video is embedded using YouTube's privacy-enhanced mode. YouTube may still receive the IP address when the video loads.
- The downloadable session file contains numbers and metadata, not images.

---

## 6. Transparency about data quality

Webcam sensing is noisy, especially on weak hardware. The report therefore states its own reliability:

- the median **frame rate** of the session,
- the **gaze calibration error** (or that relative gaze was used),
- whether **blink detection** was reliable,
- how many seconds looked like an **advertisement** and were excluded (best-effort guard; ads cannot be blocked inside an embedded player),
- whether the quiz used the generated questions or the built-in backup set.

The AI summary is instructed to ignore small differences, keep lesson time and quiz time separate, and never interpret blink numbers when blink detection was unreliable.

---

## 7. Architecture

```
Browser (HTML, CSS, JavaScript)
  MediaPipe face landmarks, gaze calibration, blink detection,
  feature windows, snapshots (memory only), charts, PDF
        |  14 numeric features per 5 s window         |  quiz, grading, report text
        v                                             v
FastAPI backend (Render)  --->  attention_model.py    Groq LLM
  /api/attention  /api/questions  /api/grade  /api/report
```

| Part | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript (no build step), Chart.js, jsPDF |
| Face and iris landmarks | MediaPipe Face Landmarker (pretrained) |
| Backend | Python, FastAPI |
| Attention model | heuristic by default; scikit-learn pipeline for training |
| Language model | Groq (`openai/gpt-oss-120b`, with `gpt-oss-20b` and `llama-3.3-70b-versatile` as fallbacks) |
| Hosting | Vercel (frontend), Render (backend) |

**Reliability features.** Questions are generated once from the transcript and prompt, cached on disk, and pre-generated in the background while the lesson plays. If the LLM is unavailable, a built-in backup question set is used, short answers fall back to keyword grading, and the report falls back to a rule-based summary.

### Repository layout

```
attention-lens/
  frontend/
    index.html        page structure
    app.js            tracking, calibration, quiz, report, PDF
    gazemap.js        per-person gaze regression and validation
    styles.css
    config.js         backend URL, GitHub URL, YouTube video
  backend/
    main.py           API: attention, questions, grading, report
    attention_model.py  features, heuristic, trained-model loading
    train.py          training and evaluation
    format_transcript.py  converts a pasted YouTube transcript
    content/
      transcript.txt          lesson text with [mm:ss] timestamps
      question_prompt.txt     instructions for the question generator
      questions_fallback.json backup questions
    requirements.txt
    .env.example
```

---

## 8. Run it yourself

### Backend (Windows PowerShell)

```powershell
cd backend
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env      # then put your GROQ_API_KEY in .env
uvicorn main:app --reload        # http://localhost:8000
```

Check `http://localhost:8000/api/health` and `http://localhost:8000/api/selftest`.

### Frontend

```powershell
cd frontend
python -m http.server 5500       # open http://localhost:5500
```

The camera works on `localhost` or on an `https` address. In `frontend/config.js`, set `ATTENTIONLENS_API` to your backend address (no trailing slash).

### Change the lesson

1. Put the YouTube link in `config.js` (`ATTENTIONLENS_YOUTUBE`).
2. Put the matching transcript in `backend/content/transcript.txt`, one line per few seconds, in the form `[mm:ss] text`. `python format_transcript.py raw_transcript.txt` converts a transcript pasted from YouTube.
3. Edit `backend/content/question_prompt.txt` to say what to ask. Questions are regenerated automatically when the transcript or prompt changes.

### Train the attention model on real sessions

```powershell
cd backend
python train.py                            # synthetic data: tests the pipeline only
python train.py --real sessions\*.json     # sessions downloaded from the report page
```

Environment variables for the backend: `GROQ_API_KEY`, `GROQ_MODEL` (optional), `CORS_ORIGINS` (the frontend address).

### Deploy

Frontend on Vercel (root directory `frontend`, no build step). Backend on Render as a Python web service (root directory `backend`, start command `uvicorn main:app --host 0.0.0.0 --port $PORT`). Set `CORS_ORIGINS` to the frontend address.

---

## 9. Limitations (stated openly)

- **No validation study yet.** The attention model is a heuristic. The only synthetic-data runs test the training pipeline and say nothing about real validity.
- **Webcam gaze is approximate.** The gaze error is reported in each session and is typically a sizeable fraction of the screen. It cannot replace an eye tracker.
- **Hardware matters.** On weak laptops and cameras the frame rate can drop to a few frames per second, which makes blink and gaze measures unreliable. The app detects and reports this.
- **Few labels per session.** One to four thought probes per session is a thin reference. A real study needs more probes and many participants.
- **Snapshots are context, not proof.** They show what was visible when risk was high, and a snapshot does not prove mind wandering.
- **One short lesson.** The content is a single short story, and results from one session are anecdotal.
- **Ads cannot be blocked** inside an embedded video. The guard is best-effort.

---

## 10. Proposed research extensions

1. **Validate against a reference eye tracker** (fixations, saccades) to quantify webcam error.
2. **Run a user study** (for example n >= 30) with more probes per session, then fit and evaluate the model with subject-wise cross-validation.
3. **Personalised and temporal models** (per-learner baselines, sequence models over windows) compared with the current window-level approach.
4. **More modalities:** audio, interaction logs, and object detection (for example a phone in frame).
5. **Compare adaptive strategies** (replay, retrieval question, break) in a randomised experiment.
6. **Link attention to metacognition at scale:** how mind wandering relates to confidence calibration across lesson segments.

---

## 11. Acknowledgements

Built with MediaPipe, FastAPI, scikit-learn, Chart.js, jsPDF, and the Groq API.
