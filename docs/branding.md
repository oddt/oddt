# Logo structure and interactions

The ODDT logo uses inhibitor M15 (chain A, residue 402) bound to human aldose
reductase in [PDB 4LB3](https://www.rcsb.org/structure/4LB3), a 0.80 Å crystal
structure. Its 24-heavy-atom ligand provides a richer map than the previous
trypsin–benzamidine example, with eight displayed contacts across three types.
All classifications and distances were calculated with ODDT's RDKit backend.

Gold dashed lines show hydrogen bonds, violet dashed lines show the iodine
halogen bond, and teal dotted lines show representative hydrophobic contacts.

| Type | M15 atom | Protein atom | Distance in 3D |
| --- | --- | --- | --- |
| Hydrogen bond | O20 | His A110 NE2 | 2.68 Å |
| Hydrogen bond | O21 | Trp A111 NE1 | 3.02 Å |
| Halogen bond | I23 | Thr A113 OG1 | 2.95 Å |
| Hydrophobic contact | C13 | Trp A20 CZ2 | 3.84 Å |
| Hydrophobic contact | C1 | Trp A111 CZ2 | 3.35 Å |
| Hydrophobic contact | C12 | Phe A122 CZ | 3.75 Å |
| Hydrophobic contact | C1 | Leu A300 CB | 3.79 Å |
| Hydrophobic contact | C4 | Tyr A309 CE1 | 3.86 Å |

Hydrogen bonds pass `oddt.interactions.hbonds`' strict angular mask with a
3.5 Å cutoff and 30° angular tolerance; donors must carry hydrogens. Displayed
hydrogen bonds are restricted to conventional N/O acceptors. ODDT also flags
an N···F contact to M15 F24; that candidate is recorded in the source data but
omitted from the logo. The halogen bond passes `oddt.interactions.halogenbonds`'
strict mask with a 4.0 Å cutoff and 30° angular tolerance. Hydrophobic contacts
use `oddt.interactions.hydrophobic_contacts` with a 4.0 Å cutoff; only the closest
classified atom pair per protein residue is shown.

Protein ATOM records with blank/A alternate conformations were used. Polar
hydrogens were added with RDKit. Waters, NADP+, other hetero compounds, and
crystallographic symmetry mates were excluded. Ligand bond orders and formal
charges follow the [M15 Chemical Component Dictionary](https://files.rcsb.org/ligands/download/M15.cif).

The drawing uses a compact RDKit 2D chemical layout so both aromatic rings remain
legible. Protein contact nodes are arranged around it schematically; their
display positions do not represent a projection of the crystal coordinates.
The protein silhouette is decorative rather than a molecular surface.

The [source data](./_static/oddt-logo-source.json) records original atom coordinates,
PDB atom names and serial numbers, diagram positions, analysis settings, excluded
contacts, and source-file checksums. Each SVG embeds a source summary and titles
on its interaction markers.

Vector assets: [light logo](./_static/oddt-logo.svg),
[dark logo](./_static/oddt-logo-dark.svg),
[light icon](./_static/oddt-icon.svg), and
[dark icon](./_static/oddt-icon-dark.svg).
