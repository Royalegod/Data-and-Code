model_name = 'Rock3D_WithProbe'
rock_part_name = 'RockPart'

X_length, Y_length, Z_thickness = 12.0, 12.0, 0.8   # mm
seed_size = 0.08

# ------------------ label-to-phase mapping ------------------
label_to_phase = {
    # key   : integer label from image segmentation/classification
    # value : mineral phase name
}

# ------------------ material properties ------------------
mat_props = {
    # 'PhaseName': {
    #     'E'   : Young's modulus,
    #     'nu'  : Poisson's ratio,
    #     'rho' : density,
    #     'H'   : hardness(define in inp file)
    # }
}
mat_names = [label_to_phase[i] for i in sorted(label_to_phase.keys())]

# ==========================================================
# Read the phase label matrix
# ==========================================================
if not os.path.isfile(txt_path):
    raise IOError('Label file not found: %s' % txt_path)

labels = np.loadtxt(txt_path, dtype=int)
H, W = labels.shape

uniq = np.unique(labels)
for u in uniq:
    if u not in label_to_phase:
        raise ValueError('Undefined label %d in the label matrix' % u)

# ==========================================================
# Create a new model
# ==========================================================
if model_name in mdb.models.keys():
    del mdb.models[model_name]
model = mdb.Model(name=model_name)

# ==========================================================
# Create rock geometry
# ==========================================================
sk = model.ConstrainedSketch(
    name='RockSketch',
    sheetSize=max(X_length, Y_length) * 2.0
)
sk.rectangle(point1=(0.0, 0.0), point2=(X_length, Y_length))

rock_part = model.Part(
    name=rock_part_name,
    dimensionality=THREE_D,
    type=DEFORMABLE_BODY
)
rock_part.BaseSolidExtrude(sketch=sk, depth=Z_thickness)

# ==========================================================
# Create materials and sections
# ==========================================================
for name, p in mat_props.items():
    m = model.Material(name=name)
    m.Density(table=((p['rho'],),))
    m.Elastic(table=((p['E'], p['nu']),))
    model.HomogeneousSolidSection(
        name=name + '_Sec',
        material=name,
        thickness=None
    )

# Assign a default section first
rock_part.SectionAssignment(
    region=regionToolset.Region(cells=rock_part.cells[:]),
    sectionName=mat_names[0] + '_Sec'
)

# ==========================================================
# Mesh the rock
# ==========================================================
rock_part.seedPart(size=seed_size, deviationFactor=0.1, minSizeFactor=0.1)
et = mesh.ElemType(elemCode=C3D8R, elemLibrary=STANDARD)
rock_part.setElementType(regions=(rock_part.cells[:],), elemTypes=(et,))
rock_part.generateMesh()

# ==========================================================
# Map element centroids to image pixels
# ==========================================================
phase_map = dict((pid, []) for pid in label_to_phase.keys())

for e in rock_part.elements:
    conn = e.connectivity
    pts = [rock_part.nodes[n - 1].coordinates for n in conn]

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = sum(xs) / float(len(xs))
    cy = sum(ys) / float(len(ys))

    ii = int(math.floor(cx / (X_length / float(W))))
    jj = int(math.floor(cy / (Y_length / float(H))))

    if ii < 0:
        ii = 0
    if ii > W - 1:
        ii = W - 1
    if jj < 0:
        jj = 0
    if jj > H - 1:
        jj = H - 1

    # The label matrix uses a top-left origin, while Abaqus uses a bottom-left origin.
    pid = int(labels[H - 1 - jj, ii])
    phase_map[pid].append(e.label)

# ==========================================================
# Assign sections phase by phase
# ==========================================================
for pid, labs in phase_map.items():
    if len(labs) == 0:
        continue

    mat = label_to_phase[pid]
    seq = rock_part.elements.sequenceFromLabels(labs)
    set_name = 'Set_' + mat

    rock_part.Set(name=set_name, elements=seq)
    rock_part.SectionAssignment(
        region=rock_part.sets[set_name],
        sectionName=mat + '_Sec'
    )