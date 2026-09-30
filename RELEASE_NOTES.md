# Arch-EPVC research artifact release v0.1.0

Research data, paired model checkpoints, original training scripts, and platform source for school-building annual and hourly energy/PV/carbon assessment.

## Downloads

- `Arch-EPVC-source-and-samples.zip`: repository snapshot, model weights, ONNX models, numerical data, geometry package and eight hourly examples.
- `hourly-data-01.zip` through `hourly-data-07.zip`: 3,356 original building-level hourly workbooks. Each ZIP is independently extractable; combine their contents in a single directory.
- `hourly_archive_manifest.csv`: building-to-archive mapping and workbook SHA-256 checksums.
- `archive_checksums.csv`: checksums for the seven hourly archives.

Platform: https://arch-epvc.vercel.app

Project page: https://chevalier1chen.github.io/Arch-EPVC/

This is an artifact release, not a newly validated model benchmark. Original training scripts require local path configuration and additional source caches. The platform snapshot has missing resources in the existing-building hourly inference path; see PUBLICATION_STATUS.md. Numerical quality and 8,760-hour completeness have not been re-audited during packaging. Reuse license and paper author metadata remain pending author finalization.
