### Development version (Git)


### Version 0.6 (2018-02-28)
* Virtual Screening module has ben rewritten to use batches of ligands
* Implement scoring functions based on PLEC fingerprints (linear, nn and rf)
* Add `mol.calccharges` method to RDKit backend
* Introduce `resid` and `resnum` uniformly to both toolkits (index and PDB number)
* generators of diverse conformers for OB and RDKit backends
* PLEC fingerprint bug fixes
* testing is now done by Pytest


### Version 0.5 (2017-12-14)
* Protein-Ligand Extended Connectivity (PLEC) fingerprint
* PDBQT reader and writer for RDKit + Vina support
* DUD-e database wrappers
* standard deviation (SD) metric used in CASF
* RIE and BEDROC metrics
* minor bug fixes


### Version 0.4.1 (2017-08-09)
* Issue warning if maximum allowed neighbor count is reached in `atom_dict`


### Version 0.4 (2017-07-17)
* Interaction Fingerprints (IFP and SIFP);
    by Michał Kukiełka (@mkukielka)
* Ultrafast Shape Recognition methods (USR, USRCAT, Electroshape);
    by Paulina Knut (@pknut) and Kinga Szyman (@kinga322)
* Extended connectivity fingerprints ECFP (internal implementation)
* Python 3.6 support
