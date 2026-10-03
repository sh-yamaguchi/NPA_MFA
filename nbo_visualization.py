from lib.pymfa import NBOFieldAnalysis

datasets = [
    "R1major",
    "R1minor",
    "R2major",
    "R2minor",
    "R3major",
    "R3minor",
]

analysis = NBOFieldAnalysis(
    grid_x=8,
    grid_y=8,
    grid_z=8,
    unitcell_size=1.0,
    threshold=0.01,
    require_nonzero_q=True
)

for name in datasets:
    analysis.visualization(
        f"coefficient_{name}.txt",
        f"mol_name_{name}.txt",
        Directory=f"vis{name}"
    )