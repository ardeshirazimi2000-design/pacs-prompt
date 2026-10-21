# پایپ‌لاین هوش مصنوعی CT (ctpipeline) — نسخهٔ MVP

> ⚠️ **هشدار:** این نرم‌افزار نمونهٔ اولیهٔ پژوهشی است، **وسیلهٔ پزشکی نیست** و نباید برای تشخیص یا جایگزین پزشک استفاده شود. خروجی `DummyThresholdAdapter` کاملاً ساختگی است و در همهٔ خروجی‌ها با برچسب `SYNTHETIC` مشخص می‌شود.

این مخزن پیاده‌سازی برش اول سند [`PROJECT_SPEC_FA.md`](PROJECT_SPEC_FA.md) است:
دریافت مطالعهٔ CT با فرمت DICOM، اعتبارسنجی و مرتب‌سازی هندسی، پیش‌پردازش نسخه‌دار،
اجرای مدل از طریق adapter، پس‌پردازش و QC، و تولید DICOM SEG، DICOM SR (TID 1500) و Overlay PNG،
همراه با صف job، retry، لغو، بازیابی پس از restart و manifest بازتولیدپذیری.

## معماری

```text
POST /v1/studies ─► ذخیرهٔ ورودی ─► VALIDATING (ingestion) ─► QUEUED ─┐
                                                                    │  (جدول jobs = صف؛ claim اتمیک)
worker ◄────────────────────────────────────────────────────────────┘
  PREPROCESSING ─► INFERENCE (ModelAdapter) ─► POSTPROCESSING (QC) ─► EXPORTING ─► SUCCEEDED
                                                       │                  ├─ seg.dcm   (DICOM SEG)
                                                       │                  ├─ sr.dcm    (DICOM SR)
                                                       │                  ├─ overlay.png
                                                       │                  ├─ findings.json
                                                       └─ خطا ─► retry/backoff یا FAILED (+ dead-letter)
                                                                          └─ manifest.json
```

| مسیر | مسئولیت |
|---|---|
| `src/ctpipeline/ingestion/` | خواندن و اعتبارسنجی DICOM، ترتیب برش‌ها از هندسه، `VolumeGeometry` |
| `src/ctpipeline/preprocessing/` | تبدیل HU، windowing، normalization، resampling و نگاشت معکوس label |
| `src/ctpipeline/adapters/` | رابط `ModelAdapter`، رجیستری مدل‌ها، `DummyThresholdAdapter` |
| `src/ctpipeline/postprocessing/` | حذف اجزای کوچک، یافته‌ها با مختصات voxel و فیزیکی جدا، QC |
| `src/ctpipeline/outputs/` | DICOM SEG و SR با highdicom، Overlay PNG |
| `src/ctpipeline/jobs/` | مدل داده، وضعیت‌ها، صف مبتنی بر DB، runner، worker، manifest |
| `src/ctpipeline/api/` | FastAPI (بدون منطق پردازش تصویر) |
| `src/ctpipeline/storage/` | ذخیره‌سازی اشیا (فعلاً فایل‌سیستم/PVC) |
| `configs/models/*.yaml` | پروفایل نسخه‌دار هر مدل (پیش/پس‌پردازش، پارامترها) |
| `deploy/k8s/` | manifestهای Kubernetes (kustomize) |
| `docs/` | قراردادهای هندسه، دامنهٔ پشتیبانی DICOM، عملیات و نقشهٔ راه |

## اجرای محلی

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

# یک سری CT مصنوعی (بدون دادهٔ بیمار) بسازید
python -m ctpipeline.testing.synthetic ./data/synthetic-series

# API به‌همراه worker داخلی (فقط برای توسعه)
CTP_EMBEDDED_WORKER=true uvicorn ctpipeline.api.app:build_app --factory --port 8000

# ارسال مطالعه
curl -s -H "Idempotency-Key: demo-1" \
  $(for f in data/synthetic-series/*.dcm; do printf -- "-F files=@%s " "$f"; done) \
  http://localhost:8000/v1/studies
curl -s http://localhost:8000/v1/jobs/<job_id>
curl -s http://localhost:8000/v1/jobs/<job_id>/results
curl -s -o seg.dcm http://localhost:8000/v1/jobs/<job_id>/results/DICOM_SEG
```

در حالت عادی API و worker جدا اجرا می‌شوند: `ctpipeline-worker`. با Docker:

```bash
docker compose up --build      # API + CPU worker + PostgreSQL
```

تنظیمات از متغیرهای محیطی با پیشوند `CTP_` خوانده می‌شوند؛ نمونه در [`.env.example`](.env.example).

## API

| متد | مسیر | توضیح |
|---|---|---|
| `POST` | `/v1/studies` | آپلود multipart فایل‌های یک سری (`files`)، فیلدهای اختیاری `model`، `series_instance_uid`، `priority`؛ هدر `Idempotency-Key` |
| `GET` | `/v1/jobs/{id}` | وضعیت، تعداد تلاش، خطای ساخت‌یافته (`code`/`message`) |
| `GET` | `/v1/jobs/{id}/events` | تاریخچهٔ وضعیت (audit) |
| `GET` | `/v1/jobs/{id}/results` | فهرست خروجی‌ها با SHA-256 و manifest |
| `GET` | `/v1/jobs/{id}/results/{type}` | دانلود `DICOM_SEG`، `DICOM_SR`، `OVERLAY`، `FINDINGS`، `MANIFEST` |
| `POST` | `/v1/jobs/{id}/cancel` | لغو (فوری در صف؛ در حال اجرا بین مراحل) |
| `GET` | `/healthz`، `/readyz`، `/metrics` | سلامت، آمادگی (DB و storage)، متریک Prometheus |

اگر `CTP_API_KEY` تنظیم شود، همهٔ مسیرهای `/v1` هدر `X-API-Key` می‌خواهند.

## آزمون و کیفیت کد

```bash
python -m pytest          # 52 آزمون واحد و یکپارچه
ruff check src tests scripts && ruff format --check src tests scripts
mypy
```

آزمون‌ها فقط از دادهٔ مصنوعی استفاده می‌کنند (`ctpipeline.testing.synthetic`).

## افزودن مدل واقعی

1. کلاسی از `ModelAdapter` بسازید که `SegmentationPrediction` با کلاس‌های دارای **کد SNOMED/DCM معتبر** برگرداند.
2. آن را با `ctpipeline.adapters.registry.register()` ثبت کنید.
3. پروفایل `configs/models/<name>.yaml` با تنظیمات پیش/پس‌پردازش آن بسازید.

هیچ تغییری در API، صف یا exporterها لازم نیست.

## وضعیت و محدودیت‌ها

آنچه انجام شده و آنچه مانده در [`docs/roadmap.md`](docs/roadmap.md) آمده است. مهم‌ترین موارد باقی‌مانده:
اعتبارسنجی مستقل SEG/SR (آزمون CI آماده است ولی هنوز informational است)، باز کردن خروجی در viewer هدف،
صف RabbitMQ و object storage سازگار با S3، adapter مدل واقعی روی GPU، migrationهای پایگاه داده و Helm chart.
