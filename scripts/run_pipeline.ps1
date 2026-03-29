param(
  [string]$Config = "configs/default.yaml",
  [string]$DefectType = "good"
)

python -m defectfill.cli --config $Config synthesize
python -m defectfill.cli --config $Config build-memory
python -m defectfill.cli --config $Config infer --defect-type $DefectType
