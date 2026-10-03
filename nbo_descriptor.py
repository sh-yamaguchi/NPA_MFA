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
    unitcell_size=1.0
)

for name in datasets:
    mol_name = f"mol_name_{name}.txt"
    descriptor_file = f"descriptor{name}.txt"

    print(f"Processing: {name}")

    analysis.descriptor(
        mol_name,
        output_file=descriptor_file
    )