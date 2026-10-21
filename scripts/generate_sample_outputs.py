"""Run the full pipeline on a synthetic series and write SEG/SR/overlay to a folder.

Used by the CI ``dicom-conformance`` job to feed an independent validator.
Usage: python scripts/generate_sample_outputs.py OUT_DIR
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from ctpipeline.common.config import Settings
from ctpipeline.jobs.service import submit_study
from ctpipeline.jobs.worker import Worker, build_context
from ctpipeline.testing.synthetic import make_ct_series


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "sample-outputs")
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp())
    settings = Settings(
        database_url=f"sqlite:///{work / 'db.sqlite'}",
        storage_root=work / "objects",
        configs_dir=Path(__file__).resolve().parents[1] / "configs",
    )
    ctx = build_context(settings)
    paths = make_ct_series(out / "source")
    job, _ = submit_study(ctx, [p.read_bytes() for p in paths], model_name="dummy-threshold")
    Worker(ctx).run_once()
    job = ctx.store.get(job.id)
    if job.status != "SUCCEEDED":
        print(f"Job ended in {job.status}: {job.error_code}", file=sys.stderr)
        return 1
    for art in ctx.store.artifacts(job.id):
        (out / art.key.rsplit("/", 1)[-1]).write_bytes(ctx.objects.get(art.key))
    print(f"Wrote outputs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
