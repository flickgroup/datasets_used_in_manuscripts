# Figure inset artwork

Structure renders that the figures overlay on the plots. They are artwork rather than data;
`pmbd_bilayer.add_schematic` places each one at the position the published figures use, and
prints a note and carries on if a file is absent.

| File | Used by | Shows |
| --- | --- | --- |
| `hbn_aa.png`, `hbn_ab.png`, `hbn_ab1.png`, `hbn_aap.png` | `main-script/plot_fig1.py` | hBN stackings, viewed along z |
| `graphene_aa.png`, `graphene_ab.png` | `main-script/plot_fig1.py` | graphene stackings, viewed along z |
| `hbn_aap.png`, `graphene_ab.png` | `SI/plot_figS3.py` | the same |
| `graphene_ab.png` | `SI/plot_figS4.py` | the same |
| `hbn_interlayer_schematic.png`, `hbn_lbm_schematic.png` | `main-script/plot_fig2.py` | the two hBN layers with R<sub>0</sub> marked, and with the breathing-mode arrows |
| `graphene_interlayer_schematic.png`, `graphene_lbm_schematic.png` | `SI/plot_figS2.py` | the same for graphene |
| `graphene_unitcell.png`, `graphene_5x5_supercell.png` | `SI/plot_figS1.py` | the graphene unit cell and a 5 × 5 supercell |
